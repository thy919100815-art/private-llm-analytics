"""90-Day Privacy Epoch and Workload-Driven RDP Budget Allocation (v1.0).

Implements:
1. Fixed 90-day Privacy Epoch tracking.
2. Workload-driven budget allocation (business weights -> RDP vectors -> noise calibration).
3. Authoritative RDP accounting ensuring sum_i rho_i(alpha) <= R_epoch(alpha) and eps_epoch(delta) <= 4.0.
4. Lifetime Exposure Monitor tracking cumulative RDP across epochs.
5. Persistent serialization to SQLite.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compose_rdp_vectors,
    compute_gaussian_rdp,
    rdp_to_epsilon_delta,
    to_decimal,
)

# Equivalent noise parameter for global epoch cap (eps=4.0, delta=1e-6)
# Under Gaussian RDP: R_target(alpha) = alpha / (2 * SIGMA_EQUIV_CAP^2)
SIGMA_EQUIV_CAP = 1.404987
TARGET_VARIANCE_FACTOR = 1.0 / (SIGMA_EQUIV_CAP**2)  # ~0.506589

DEFAULT_WORKLOAD_WEIGHTS = {
    "interaction_volume": 0.25,
    "success_rate": 0.20,
    "user_prevalence": 0.15,
    "user_normalized_share": 0.10,
    "top_k_tasks": 0.10,
    "continual": 0.10,
    "reserve": 0.10,
}

WORKLOAD_DISPLAY_NAMES = {
    "interaction_volume": "Requests (调用量)",
    "success_rate": "Success Rate (成功率)",
    "user_prevalence": "Prevalence (渗透率)",
    "user_normalized_share": "Share (归一化占比)",
    "top_k_tasks": "Top-K (热门任务)",
    "continual": "Continual (持续观测)",
    "reserve": "Reserve (预留机动)",
}


@dataclass
class WorkloadQuota:
    """Allocated RDP budget and consumption status for a specific analytics workload."""

    name: str
    display_name: str
    weight: float
    is_supported: bool
    allocated_rdp: Dict[Decimal, Decimal]
    spent_rdp: Dict[Decimal, Decimal] = field(default_factory=dict)
    release_count: int = 0

    def __post_init__(self):
        if not self.spent_rdp:
            self.spent_rdp = {alpha: Decimal("0.0") for alpha in self.allocated_rdp}

    @property
    def spent_ratio(self) -> float:
        """Ratio of consumed RDP relative to allocated RDP at order alpha=4.0 (or first non-zero)."""
        ref_alpha = Decimal("4.0") if Decimal("4.0") in self.allocated_rdp else next(iter(self.allocated_rdp))
        alloc = self.allocated_rdp.get(ref_alpha, Decimal("0.0"))
        if alloc <= Decimal("0.0"):
            return 0.0
        spent = self.spent_rdp.get(ref_alpha, Decimal("0.0"))
        return float(min(Decimal("1.0"), max(Decimal("0.0"), spent / alloc)))

    @property
    def spent_percentage(self) -> float:
        return round(self.spent_ratio * 100.0, 1)

    @property
    def allocated_epsilon(self) -> float:
        """Single-release equivalent effective epsilon for this workload quota."""
        if not any(v > Decimal("0.0") for v in self.allocated_rdp.values()):
            return 0.0
        _, eps = rdp_to_epsilon_delta(self.allocated_rdp, Decimal("1e-6"))
        return round(float(eps), 4)

    @property
    def spent_epsilon(self) -> float:
        """Equivalent effective epsilon of currently spent RDP for this workload."""
        if not any(v > Decimal("0.0") for v in self.spent_rdp.values()):
            return 0.0
        _, eps = rdp_to_epsilon_delta(self.spent_rdp, Decimal("1e-6"))
        return round(float(eps), 4)

    def can_absorb(self, candidate_rdp: Dict[Decimal, Decimal], tolerance: Decimal = Decimal("1e-9")) -> bool:
        """Check if candidate RDP vector fits within this workload's remaining quota."""
        for alpha, cost in candidate_rdp.items():
            dec_alpha = to_decimal(alpha)
            dec_cost = to_decimal(cost)
            curr = self.spent_rdp.get(dec_alpha, Decimal("0.0"))
            alloc = self.allocated_rdp.get(dec_alpha, Decimal("0.0"))
            if curr + dec_cost > alloc + tolerance:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "weight": self.weight,
            "is_supported": self.is_supported,
            "allocated_epsilon": self.allocated_epsilon,
            "spent_epsilon": self.spent_epsilon,
            "spent_percentage": self.spent_percentage,
            "release_count": self.release_count,
            "allocated_rdp": {str(k): str(v) for k, v in self.allocated_rdp.items()},
            "spent_rdp": {str(k): str(v) for k, v in self.spent_rdp.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadQuota":
        alloc = {to_decimal(k): to_decimal(v) for k, v in data.get("allocated_rdp", {}).items()}
        spent = {to_decimal(k): to_decimal(v) for k, v in data.get("spent_rdp", {}).items()}
        return cls(
            name=data["name"],
            display_name=data.get("display_name", WORKLOAD_DISPLAY_NAMES.get(data["name"], data["name"])),
            weight=float(data.get("weight", 0.0)),
            is_supported=bool(data.get("is_supported", True)),
            allocated_rdp=alloc,
            spent_rdp=spent,
            release_count=int(data.get("release_count", 0)),
        )


@dataclass
class PrivacyEpoch:
    """90-day Privacy Epoch tracking authoritative RDP composition and workload quotas."""

    epoch_id: str
    epoch_start: str  # YYYY-MM-DD
    epoch_end: str    # YYYY-MM-DD
    epoch_days: int = 90
    epsilon_cap: Decimal = Decimal("4.0")
    delta_global: Decimal = Decimal("1e-6")
    rdp_orders: List[Decimal] = field(default_factory=lambda: list(DEFAULT_RDP_ORDERS))
    total_rdp: Dict[Decimal, Decimal] = field(default_factory=dict)
    workloads: Dict[str, WorkloadQuota] = field(default_factory=dict)
    reserve: WorkloadQuota = field(init=False)
    release_ids: List[str] = field(default_factory=list)
    status: str = "ACTIVE"  # 'ACTIVE' or 'CLOSED'
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    closed_at: Optional[str] = None

    def __post_init__(self):
        self.epsilon_cap = to_decimal(self.epsilon_cap)
        self.delta_global = to_decimal(self.delta_global)
        if not self.total_rdp:
            self.total_rdp = {alpha: Decimal("0.0") for alpha in self.rdp_orders}

        # Initialize workloads if empty
        if not self.workloads:
            self._init_default_workloads()

        if "reserve" in self.workloads:
            self.reserve = self.workloads["reserve"]
        else:
            self.reserve = WorkloadQuota(
                name="reserve",
                display_name=WORKLOAD_DISPLAY_NAMES["reserve"],
                weight=0.10,
                is_supported=True,
                allocated_rdp={alpha: Decimal("0.0") for alpha in self.rdp_orders},
            )
            self.workloads["reserve"] = self.reserve

    def _init_default_workloads(self, weights: Optional[Dict[str, float]] = None):
        """Construct default workload allocations ensuring total composed RDP <= epsilon_cap."""
        raw_weights = dict(weights or DEFAULT_WORKLOAD_WEIGHTS)
        tot = sum(raw_weights.values())
        norm_weights = {k: v / tot for k, v in raw_weights.items()}

        for name, w in norm_weights.items():
            # R_w(alpha) = w * (alpha * TARGET_VARIANCE_FACTOR / 2)
            alloc_rdp = {
                alpha: to_decimal(w * float(alpha) * TARGET_VARIANCE_FACTOR / 2.0)
                for alpha in self.rdp_orders
            }
            # Continual is marked unsupported by default if not explicitly enabled
            is_supp = name != "continual"  # verified by manager based on config
            self.workloads[name] = WorkloadQuota(
                name=name,
                display_name=WORKLOAD_DISPLAY_NAMES.get(name, name),
                weight=w,
                is_supported=is_supp,
                allocated_rdp=alloc_rdp,
            )

    @property
    def current_day(self) -> int:
        """Compute the current day (1 to 90) of the epoch."""
        try:
            start_dt = datetime.strptime(self.epoch_start[:10], "%Y-%m-%d").date()
            today = datetime.now(timezone.utc).date()
            diff = (today - start_dt).days + 1
            return max(1, min(self.epoch_days, diff))
        except Exception:
            return 1

    @property
    def effective_epsilon(self) -> float:
        """Current effective epsilon under authoritative RDP composition."""
        if not any(v > Decimal("0.0") for v in self.total_rdp.values()):
            return 0.0
        _, eps = rdp_to_epsilon_delta(self.total_rdp, self.delta_global)
        return round(float(eps), 4)

    @property
    def remaining_headroom(self) -> float:
        """Remaining epsilon headroom until reaching the 4.0 cap."""
        eff = self.effective_epsilon
        cap = float(self.epsilon_cap)
        return round(max(0.0, cap - eff), 4)

    @property
    def reserve_percentage_remaining(self) -> float:
        """Remaining percentage of the reserve budget."""
        if "reserve" in self.workloads:
            return round(100.0 - self.workloads["reserve"].spent_percentage, 1)
        return 100.0

    def _normalize_rdp(self, rdp: Dict[Any, Any]) -> Dict[Decimal, Decimal]:
        """Normalize candidate RDP vector to contain all orders in self.rdp_orders."""
        dec_dict = {to_decimal(k): to_decimal(v) for k, v in rdp.items()}
        return {alpha: dec_dict.get(alpha, Decimal("0.0")) for alpha in self.rdp_orders}

    def can_execute_release(
        self,
        workload_name: str,
        candidate_rdp: Dict[Decimal, Decimal],
    ) -> Tuple[bool, str, str]:
        """Evaluate query execution priority under Epoch Workload Budgeting.

        Returns (is_allowed, allocation_source, reason):
          - allocation_source: 'WORKLOAD_QUOTA', 'RESERVE', or 'BLOCKED'
        """
        if self.status != "ACTIVE":
            return False, "BLOCKED", f"Epoch '{self.epoch_id}' is closed (status: {self.status})."

        cand = self._normalize_rdp(candidate_rdp)

        # Check total composed RDP against Epoch Cap
        composed_total = compose_rdp_vectors([self.total_rdp, cand], orders=self.rdp_orders)
        _, eps_after = rdp_to_epsilon_delta(composed_total, self.delta_global)
        if eps_after > self.epsilon_cap + Decimal("1e-5"):
            return (
                False,
                "BLOCKED",
                f"Release would exceed Epoch ε cap ({eps_after:.4f} > {self.epsilon_cap}). Epoch budget exhausted.",
            )

        # Priority 3: Check if fits in designated planned workload quota
        wl = self.workloads.get(workload_name)
        if wl and wl.is_supported and wl.can_absorb(cand):
            return True, "WORKLOAD_QUOTA", f"Allocated from planned workload '{workload_name}' quota."

        # Priority 4: Fall back to Reserve
        res = self.workloads.get("reserve")
        if res and res.can_absorb(cand):
            return True, "RESERVE", f"Workload quota exceeded or unallocated; consumed from Reserve."

        return (
            False,
            "BLOCKED",
            f"Workload '{workload_name}' quota and Reserve are both exhausted for this candidate RDP.",
        )

    def record_release_spend(
        self,
        workload_name: str,
        release_id: str,
        candidate_rdp: Dict[Decimal, Decimal],
        source: str = "WORKLOAD_QUOTA",
    ) -> None:
        """Atomically record RDP consumption into workload and epoch total."""
        cand = self._normalize_rdp(candidate_rdp)
        target_wl_name = workload_name if source == "WORKLOAD_QUOTA" and workload_name in self.workloads else "reserve"
        wl = self.workloads.get(target_wl_name)
        if wl:
            wl.spent_rdp = compose_rdp_vectors([wl.spent_rdp, cand], orders=self.rdp_orders)
            wl.release_count += 1

        self.total_rdp = compose_rdp_vectors([self.total_rdp, cand], orders=self.rdp_orders)
        if release_id not in self.release_ids:
            self.release_ids.append(release_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "epoch_start": self.epoch_start,
            "epoch_end": self.epoch_end,
            "epoch_days": self.epoch_days,
            "current_day": self.current_day,
            "epsilon_cap": str(self.epsilon_cap),
            "effective_epsilon": self.effective_epsilon,
            "remaining_headroom": self.remaining_headroom,
            "delta_global": str(self.delta_global),
            "status": self.status,
            "release_count": len(self.release_ids),
            "release_ids": self.release_ids,
            "reserve_percentage_remaining": self.reserve_percentage_remaining,
            "workloads": {k: v.to_dict() for k, v in self.workloads.items()},
            "total_rdp": {str(k): str(v) for k, v in self.total_rdp.items()},
            "created_at": self.created_at,
            "closed_at": self.closed_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PrivacyEpoch":
        orders = [to_decimal(k) for k in d.get("total_rdp", {}).keys()] if d.get("total_rdp") else list(DEFAULT_RDP_ORDERS)
        tot_rdp = {to_decimal(k): to_decimal(v) for k, v in d.get("total_rdp", {}).items()}
        wls = {k: WorkloadQuota.from_dict(v) for k, v in d.get("workloads", {}).items()}

        epoch = cls(
            epoch_id=d["epoch_id"],
            epoch_start=d["epoch_start"],
            epoch_end=d["epoch_end"],
            epoch_days=int(d.get("epoch_days", 90)),
            epsilon_cap=to_decimal(d.get("epsilon_cap", "4.0")),
            delta_global=to_decimal(d.get("delta_global", "1e-6")),
            rdp_orders=orders,
            total_rdp=tot_rdp,
            workloads=wls,
            release_ids=list(d.get("release_ids", [])),
            status=d.get("status", "ACTIVE"),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            closed_at=d.get("closed_at"),
        )
        return epoch


@dataclass
class LifetimeExposureMonitor:
    """Tracks cumulative privacy exposure across all past and current epochs."""

    total_epochs_count: int = 1
    lifetime_rdp: Dict[Decimal, Decimal] = field(default_factory=dict)
    delta_global: Decimal = Decimal("1e-6")
    notice: str = "新 Epoch 会重新开放本周期业务预算，但历史隐私损失不会消失，Lifetime Exposure 将持续累计。"

    @property
    def lifetime_effective_epsilon(self) -> float:
        if not any(v > Decimal("0.0") for v in self.lifetime_rdp.values()):
            return 0.0
        _, eps = rdp_to_epsilon_delta(self.lifetime_rdp, self.delta_global)
        return round(float(eps), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_epochs_count": self.total_epochs_count,
            "lifetime_effective_epsilon": self.lifetime_effective_epsilon,
            "lifetime_delta": float(self.delta_global),
            "accumulated_rdp_summary": {str(k): round(float(v), 6) for k, v in self.lifetime_rdp.items()},
            "notice": self.notice,
        }


class EpochBudgetPlanner:
    """Planner for Workload-driven automatic RDP and noise calibration."""

    def __init__(
        self,
        epoch_cap: Union[float, Decimal] = 4.0,
        delta_global: Union[float, Decimal] = 1e-6,
        orders: Optional[List[Decimal]] = None,
        weights: Optional[Dict[str, float]] = None,
        continual_enabled: bool = False,
    ):
        self.epoch_cap = to_decimal(epoch_cap)
        self.delta_global = to_decimal(delta_global)
        self.orders = orders or list(DEFAULT_RDP_ORDERS)
        self.continual_enabled = continual_enabled

        # Resolve weights: if continual is disabled, its weight moves to reserve
        raw_weights = dict(weights or DEFAULT_WORKLOAD_WEIGHTS)
        if not self.continual_enabled:
            continual_w = raw_weights.get("continual", 0.10)
            raw_weights["continual"] = 0.0
            raw_weights["reserve"] = raw_weights.get("reserve", 0.10) + continual_w

        tot = sum(raw_weights.values())
        self.weights = {k: v / tot for k, v in raw_weights.items()}

    def create_epoch(
        self,
        epoch_id: str,
        start_date: str,
        epoch_days: int = 90,
    ) -> PrivacyEpoch:
        """Create a new 90-day Privacy Epoch with pre-calibrated workload allocations."""
        start_dt = datetime.strptime(start_date[:10], "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=epoch_days - 1)
        end_date = end_dt.strftime("%Y-%m-%d")

        workloads: Dict[str, WorkloadQuota] = {}
        for name, w in self.weights.items():
            alloc_rdp = {
                alpha: to_decimal(w * float(alpha) * TARGET_VARIANCE_FACTOR / 2.0)
                for alpha in self.orders
            }
            is_supp = True
            if name == "continual" and not self.continual_enabled:
                is_supp = False

            workloads[name] = WorkloadQuota(
                name=name,
                display_name=WORKLOAD_DISPLAY_NAMES.get(name, name),
                weight=w,
                is_supported=is_supp,
                allocated_rdp=alloc_rdp,
            )

        return PrivacyEpoch(
            epoch_id=epoch_id,
            epoch_start=start_date[:10],
            epoch_end=end_date,
            epoch_days=epoch_days,
            epsilon_cap=self.epoch_cap,
            delta_global=self.delta_global,
            rdp_orders=self.orders,
            workloads=workloads,
            status="ACTIVE",
        )

    def calibrate_workload_noise(
        self,
        workload_name: str,
        sensitivity: Union[float, Decimal],
    ) -> Tuple[float, float, Dict[Decimal, Decimal]]:
        """Calibrate Gaussian noise sigma, equivalent release epsilon, and candidate RDP for a planned workload.

        Returns (noise_sigma, effective_epsilon, candidate_rdp).
        """
        w = self.weights.get(workload_name, self.weights.get("reserve", 0.10))
        if w <= 0.0:
            # Fall back to reserve if workload weight is 0 (e.g. disabled)
            w = self.weights.get("reserve", 0.10)

        sens_val = float(sensitivity)
        V_w = w * TARGET_VARIANCE_FACTOR
        noise_sigma = sens_val / (V_w ** 0.5)

        candidate_rdp = compute_gaussian_rdp(
            sigma=noise_sigma,
            sensitivity=sens_val,
            orders=self.orders,
        )
        _, eff_eps = rdp_to_epsilon_delta(candidate_rdp, self.delta_global)
        return float(noise_sigma), float(eff_eps), candidate_rdp


class EpochBudgetManager:
    """Coordinates persistent 90-day Privacy Epochs, Workload Budgeting, and Lifetime Exposure."""

    def __init__(
        self,
        state_manager: Any,
        epoch_cap: Union[float, Decimal] = 4.0,
        delta_global: Union[float, Decimal] = 1e-6,
        epoch_days: int = 90,
        continual_enabled: bool = False,
        planner: Optional[EpochBudgetPlanner] = None,
    ):
        self.state_manager = state_manager
        self.epoch_cap = to_decimal(epoch_cap)
        self.delta_global = to_decimal(delta_global)
        self.epoch_days = int(epoch_days)
        self.continual_enabled = bool(continual_enabled)
        self.planner = planner or EpochBudgetPlanner(
            epoch_cap=self.epoch_cap,
            delta_global=self.delta_global,
            continual_enabled=self.continual_enabled,
        )

    def _epoch_to_record(self, epoch: PrivacyEpoch) -> Dict[str, Any]:
        return {
            "epoch_id": epoch.epoch_id,
            "epoch_start": epoch.epoch_start,
            "epoch_end": epoch.epoch_end,
            "epoch_days": epoch.epoch_days,
            "epsilon_cap": str(epoch.epsilon_cap),
            "delta": str(epoch.delta_global),
            "rdp_orders_json": json.dumps([str(o) for o in epoch.rdp_orders]),
            "rdp_vector_json": json.dumps({str(k): str(v) for k, v in epoch.total_rdp.items()}),
            "workload_allocations_json": json.dumps({
                k: {str(a): str(b) for a, b in v.allocated_rdp.items()}
                for k, v in epoch.workloads.items()
            }),
            "workload_spent_json": json.dumps({
                k: {
                    "spent_rdp": {str(a): str(b) for a, b in v.spent_rdp.items()},
                    "release_count": v.release_count,
                }
                for k, v in epoch.workloads.items()
            }),
            "reserve_json": json.dumps({
                "allocated_rdp": {str(a): str(b) for a, b in epoch.reserve.allocated_rdp.items()},
                "spent_rdp": {str(a): str(b) for a, b in epoch.reserve.spent_rdp.items()},
                "release_count": epoch.reserve.release_count,
            }),
            "release_ids_json": json.dumps(epoch.release_ids),
            "status": epoch.status,
            "created_at": epoch.created_at,
            "closed_at": epoch.closed_at,
        }

    def _record_to_epoch(self, rec: Dict[str, Any]) -> PrivacyEpoch:
        orders_raw = json.loads(rec.get("rdp_orders_json", "[]"))
        orders = [to_decimal(o) for o in orders_raw] if orders_raw else list(DEFAULT_RDP_ORDERS)
        tot_rdp_raw = json.loads(rec.get("rdp_vector_json", "{}"))
        total_rdp = {to_decimal(k): to_decimal(v) for k, v in tot_rdp_raw.items()}

        allocs_raw = json.loads(rec.get("workload_allocations_json", "{}"))
        spent_raw = json.loads(rec.get("workload_spent_json", "{}"))

        workloads: Dict[str, WorkloadQuota] = {}
        for name, w in self.planner.weights.items():
            alloc_dict = {to_decimal(a): to_decimal(b) for a, b in allocs_raw.get(name, {}).items()}
            if not alloc_dict:
                alloc_dict = {
                    alpha: to_decimal(w * float(alpha) * TARGET_VARIANCE_FACTOR / 2.0)
                    for alpha in orders
                }

            sp_item = spent_raw.get(name, {})
            spent_dict = {to_decimal(a): to_decimal(b) for a, b in sp_item.get("spent_rdp", {}).items()}
            rel_cnt = int(sp_item.get("release_count", 0))

            is_supp = True
            if name == "continual" and not self.continual_enabled:
                is_supp = False

            workloads[name] = WorkloadQuota(
                name=name,
                display_name=WORKLOAD_DISPLAY_NAMES.get(name, name),
                weight=w,
                is_supported=is_supp,
                allocated_rdp=alloc_dict,
                spent_rdp=spent_dict,
                release_count=rel_cnt,
            )

        rel_ids = json.loads(rec.get("release_ids_json", "[]"))

        return PrivacyEpoch(
            epoch_id=rec["epoch_id"],
            epoch_start=rec["epoch_start"],
            epoch_end=rec["epoch_end"],
            epoch_days=int(rec.get("epoch_days", 90)),
            epsilon_cap=to_decimal(rec.get("epsilon_cap", "4.0")),
            delta_global=to_decimal(rec.get("delta", "1e-6")),
            rdp_orders=orders,
            total_rdp=total_rdp,
            workloads=workloads,
            release_ids=rel_ids,
            status=rec.get("status", "ACTIVE"),
            created_at=rec.get("created_at", datetime.now(timezone.utc).isoformat()),
            closed_at=rec.get("closed_at"),
        )

    def get_or_create_active_epoch(self, default_start_date: str = "2026-10-01") -> PrivacyEpoch:
        """Load currently active epoch from database, or create initial 90-day epoch."""
        rec = self.state_manager.get_active_epoch_record()
        if rec:
            return self._record_to_epoch(rec)

        # Create initial epoch
        epoch = self.planner.create_epoch(
            epoch_id="epoch_001",
            start_date=default_start_date,
            epoch_days=self.epoch_days,
        )
        self.state_manager.save_epoch_record(self._epoch_to_record(epoch))
        return epoch

    def advance_to_new_epoch(self, new_start_date: Optional[str] = None) -> PrivacyEpoch:
        """Close active epoch, advance lifetime epochs count, and create fresh 90-day epoch."""
        active = self.get_or_create_active_epoch()
        now_iso = datetime.now(timezone.utc).isoformat()
        self.state_manager.close_epoch_record(active.epoch_id, closed_at=now_iso)

        # Advance lifetime epochs count
        lt_rec = self.state_manager.get_lifetime_exposure_record()
        cur_cnt = int(lt_rec.get("total_epochs_count", 1))
        self.state_manager.update_lifetime_exposure_record(
            lifetime_rdp_json=lt_rec.get("lifetime_rdp_json", "{}"),
            total_epochs_count=cur_cnt + 1,
        )

        # Determine new start date
        if new_start_date:
            start_d = new_start_date[:10]
        else:
            try:
                prev_end = datetime.strptime(active.epoch_end[:10], "%Y-%m-%d").date()
                start_d = (prev_end + timedelta(days=1)).strftime("%Y-%m-%d")
            except Exception:
                start_d = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        all_epochs = self.state_manager.list_all_epochs()
        next_idx = len(all_epochs) + 1
        new_epoch = self.planner.create_epoch(
            epoch_id=f"epoch_{next_idx:03d}",
            start_date=start_d,
            epoch_days=self.epoch_days,
        )
        self.state_manager.save_epoch_record(self._epoch_to_record(new_epoch))
        return new_epoch

    def record_spend(
        self,
        workload_name: str,
        release_id: str,
        candidate_rdp: Dict[Decimal, Decimal],
        source: str = "WORKLOAD_QUOTA",
    ) -> PrivacyEpoch:
        """Atomically record spend to both active epoch and lifetime exposure monitor."""
        epoch = self.get_or_create_active_epoch()
        cand = epoch._normalize_rdp(candidate_rdp)
        epoch.record_release_spend(
            workload_name=workload_name,
            release_id=release_id,
            candidate_rdp=cand,
            source=source,
        )
        self.state_manager.save_epoch_record(self._epoch_to_record(epoch))

        # Accumulate into lifetime exposure
        lt_rec = self.state_manager.get_lifetime_exposure_record()
        lt_rdp_raw = json.loads(lt_rec.get("lifetime_rdp_json", "{}"))
        lt_rdp = {to_decimal(k): to_decimal(v) for k, v in lt_rdp_raw.items()}
        if not lt_rdp:
            lt_rdp = {alpha: Decimal("0.0") for alpha in epoch.rdp_orders}
        lt_rdp = epoch._normalize_rdp(lt_rdp)

        new_lt_rdp = compose_rdp_vectors([lt_rdp, cand], orders=epoch.rdp_orders)
        self.state_manager.update_lifetime_exposure_record(
            lifetime_rdp_json=json.dumps({str(k): str(v) for k, v in new_lt_rdp.items()}),
            total_epochs_count=int(lt_rec.get("total_epochs_count", 1)),
        )
        return epoch

    def get_lifetime_exposure(self) -> LifetimeExposureMonitor:
        """Retrieve authoritative lifetime exposure across all epochs."""
        lt_rec = self.state_manager.get_lifetime_exposure_record()
        lt_rdp_raw = json.loads(lt_rec.get("lifetime_rdp_json", "{}"))
        lt_rdp = {to_decimal(k): to_decimal(v) for k, v in lt_rdp_raw.items()}
        if not lt_rdp:
            lt_rdp = {alpha: Decimal("0.0") for alpha in DEFAULT_RDP_ORDERS}

        return LifetimeExposureMonitor(
            total_epochs_count=int(lt_rec.get("total_epochs_count", 1)),
            lifetime_rdp=lt_rdp,
            delta_global=self.delta_global,
        )

