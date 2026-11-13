"""Daily Sealing Job for SDG Additive Channels.

Executes SEAL(t) at day t + L (default L = 2 days lateness grace period).
Enforces:
1. Data-independent per-user daily clipping.
2. Exact Discrete Gaussian perturbation N_Z(0, sigma^2).
3. Immutable append to SealedStore.
4. Automatic registration in UnifiedPrivacyLedger.
5. Performance latency benchmarking (execution duration in ms).
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from private_llm_analytics.sdg_mh.clipping import clip_daily_additive
from private_llm_analytics.sdg_mh.discrete_gaussian import sample_discrete_gaussian
from private_llm_analytics.sdg_mh.ledger import PrivacyAdmissionError, ReleaseRecord, UnifiedPrivacyLedger
from private_llm_analytics.sdg_mh.sealed_store import DailySealedRecord, SealedStore


class SealJob:
    """Orchestrates daily SEAL(t) executions with exact zCDP budget calibration."""

    def __init__(
        self,
        sealed_store: SealedStore,
        ledger: UnifiedPrivacyLedger,
        C_r: float = 50.0,
        C_f: float = 10.0,
        sigma_r: Optional[float] = None,
        sigma_f: Optional[float] = None,
        C_p: Optional[float] = None,
        sigma_p: Optional[float] = None,
        providers_list: Optional[Sequence[str]] = None,
        lateness_grace_days: int = 2,
        daily_d: Optional[float] = None,
        eta: float = 25.0 / 26.0,  # Default isotropic split C_r^2 / (C_r^2 + C_f^2)
        eta_p: float = 0.0,
    ):
        self.sealed_store = sealed_store
        self.ledger = ledger
        self.C_r = float(C_r)
        self.C_f = float(C_f)
        self.C_p = float(C_p) if C_p is not None else None
        self.providers_list = list(providers_list) if providers_list is not None else None
        self.lateness_grace_days = int(lateness_grace_days)
        self.daily_d = float(daily_d) if daily_d is not None else None
        self.eta = float(eta)
        self.eta_p = float(eta_p)
        self.execution_times_ms: List[float] = []

        # Budget-driven noise calibration: d = rho_r + rho_f (+ rho_p)
        if self.daily_d is not None:
            if not (0.0 < self.eta < 1.0):
                raise ValueError(f"eta must be in (0, 1), got {self.eta}")
            rem = 1.0 - self.eta - self.eta_p
            if rem <= 0:
                raise ValueError(f"eta + eta_p must be < 1, got {self.eta + self.eta_p}")

            self.rho_r = self.eta * self.daily_d
            self.rho_f = rem * self.daily_d
            self.sigma_r = float(self.C_r / math.sqrt(2.0 * max(1e-12, self.rho_r)))
            self.sigma_f = float(self.C_f / math.sqrt(2.0 * max(1e-12, self.rho_f)))

            if self.C_p is not None and self.eta_p > 0:
                self.rho_p = self.eta_p * self.daily_d
                self.sigma_p = float(self.C_p / math.sqrt(2.0 * max(1e-12, self.rho_p)))
            else:
                self.rho_p = 0.0
                self.sigma_p = None
        else:
            # Fallback to direct sigma arguments (backwards compatibility)
            self.sigma_r = float(sigma_r if sigma_r is not None else 2500.0)
            self.sigma_f = float(sigma_f if sigma_f is not None else 700.0)
            self.sigma_p = float(sigma_p) if sigma_p is not None else None
            self.rho_r = float((self.C_r ** 2) / (2.0 * self.sigma_r ** 2))
            self.rho_f = float((self.C_f ** 2) / (2.0 * self.sigma_f ** 2))
            self.rho_p = float((self.C_p ** 2) / (2.0 * self.sigma_p ** 2)) if (self.C_p and self.sigma_p) else 0.0
            self.daily_d = self.rho_r + self.rho_f + self.rho_p

    @classmethod
    def from_budget(
        cls,
        sealed_store: SealedStore,
        ledger: UnifiedPrivacyLedger,
        daily_d: float,
        eta: float = 25.0 / 26.0,
        C_r: float = 50.0,
        C_f: float = 10.0,
        C_p: Optional[float] = None,
        eta_p: float = 0.0,
        providers_list: Optional[Sequence[str]] = None,
        lateness_grace_days: int = 2,
    ) -> SealJob:
        """Factory creating SealJob strictly calibrated from daily zCDP budget d."""
        return cls(
            sealed_store=sealed_store,
            ledger=ledger,
            C_r=C_r,
            C_f=C_f,
            C_p=C_p,
            providers_list=providers_list,
            lateness_grace_days=lateness_grace_days,
            daily_d=daily_d,
            eta=eta,
            eta_p=eta_p,
        )

    def seal_day(
        self,
        day_index: int,
        date_str: str,
        df_day: pd.DataFrame,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> DailySealedRecord:
        """Execute SEAL(t) for day_index with timestamped telemetry records."""
        t0 = time.perf_counter()

        # 1. Idempotency check: if day is already sealed in store, return existing record
        existing = self.sealed_store.get_day(day_index)
        if existing is not None:
            return existing

        req_id = f"sdg_additive_req_{day_index}"
        fail_id = f"sdg_additive_fail_{day_index}"
        prov_id = f"sdg_additive_prov_{day_index}" if (self.C_p is not None and self.sigma_p is not None) else None

        # 2. Privacy admission pre-check (Fail-closed: check before clipping & noise generation)
        cand_req = ReleaseRecord(
            release_id=req_id,
            days_accessed=frozenset([day_index]),
            mechanism_type="gaussian",
            params={"sigma": self.sigma_r},
            sensitivity=self.C_r,
            rho_cost=self.rho_r,
            tag="sdg_requests",
        )
        cand_fail = ReleaseRecord(
            release_id=fail_id,
            days_accessed=frozenset([day_index]),
            mechanism_type="gaussian",
            params={"sigma": self.sigma_f},
            sensitivity=self.C_f,
            rho_cost=self.rho_f,
            tag="sdg_failures",
        )
        cands = [cand_req, cand_fail]
        if prov_id and self.C_p is not None and self.sigma_p is not None:
            cands.append(
                ReleaseRecord(
                    release_id=prov_id,
                    days_accessed=frozenset([day_index]),
                    mechanism_type="gaussian",
                    params={"sigma": self.sigma_p},
                    sensitivity=self.C_p,
                    rho_cost=self.rho_p,
                    tag="sdg_providers",
                )
            )

        for cand in cands:
            admitted, reason = self.ledger.check_admission(cand)
            if not admitted:
                raise PrivacyAdmissionError(f"SealJob rejected day {day_index}: {reason}")

        # 3. User daily contribution clipping
        c_req, c_fail, c_prov, diags = clip_daily_additive(
            df_day=df_day,
            C_r=self.C_r,
            C_f=self.C_f,
            providers_list=self.providers_list,
            C_p=self.C_p,
        )

        # 4. Add Discrete Gaussian noise
        noise_r = sample_discrete_gaussian(sigma=self.sigma_r, rng=rng, seed=seed)
        noise_f = sample_discrete_gaussian(
            sigma=self.sigma_f,
            rng=rng,
            seed=None if seed is None else seed + 1000003,
        )

        y_r = int(round(c_req)) + int(noise_r)
        y_f = int(round(c_fail)) + int(noise_f)

        y_prov_list = None
        if c_prov is not None and self.sigma_p is not None:
            noise_p = sample_discrete_gaussian(
                sigma=self.sigma_p,
                size=len(c_prov),
                rng=rng,
                seed=None if seed is None else seed + 2000003,
            )
            y_prov_list = (c_prov + noise_p).tolist()

        # 5. Create immutable record
        now_iso = datetime.now(timezone.utc).isoformat()
        record = DailySealedRecord(
            day_index=day_index,
            date_str=date_str,
            y_requests=y_r,
            y_failures=y_f,
            y_providers=y_prov_list,
            sigma_r=self.sigma_r,
            sigma_f=self.sigma_f,
            sigma_p=self.sigma_p,
            C_r=self.C_r,
            C_f=self.C_f,
            sealed_at=now_iso,
        )

        # 6. Register in Unified Ledger
        for cand in cands:
            self.ledger.record_gaussian_release(
                release_id=cand.release_id,
                days_accessed=list(cand.days_accessed),
                sensitivity=cand.sensitivity,
                sigma=cand.params["sigma"],
                tag=cand.tag,
                enforce_admission=False,  # Already pre-checked
            )

        # 7. Append to SealedStore (with atomic rollback on store failure)
        try:
            self.sealed_store.append(record)
        except Exception:
            for cand in cands:
                self.ledger.rollback_release(cand.release_id)
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.execution_times_ms.append(elapsed_ms)
        return record

    def get_benchmarks(self) -> Dict[str, float]:
        """Compute latency statistics in milliseconds."""
        if not self.execution_times_ms:
            return {"count": 0, "p50_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
        arr = np.array(self.execution_times_ms)
        return {
            "count": len(arr),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
        }
