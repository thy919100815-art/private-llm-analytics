"""Longitudinal Privacy Composition Data Structures and Utilities."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional
from private_llm_analytics.privacy.rdp import (
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
    rdp_vector_from_json,
    rdp_vector_to_json,
    to_decimal,
)


@dataclass
class RDPReleaseCost:
    """Historical or candidate release RDP cost metadata."""
    release_key: str
    mechanism: str
    sensitivity: float
    noise_scale: float
    requested_epsilon: float
    calibration_delta: float
    orders: List[Decimal]
    rdp_vector: Dict[Decimal, Decimal]
    budget_scope_id: str = "global_external"
    recipient: str = ""
    recipient_role: str = "external"

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "release_key": self.release_key,
            "mechanism": self.mechanism,
            "sensitivity": self.sensitivity,
            "noise_scale": self.noise_scale,
            "requested_epsilon": self.requested_epsilon,
            "calibration_delta": self.calibration_delta,
            "orders": [str(o) for o in self.orders],
            "rdp_vector": {str(k): str(v) for k, v in self.rdp_vector.items()},
            "budget_scope_id": self.budget_scope_id,
            "recipient": self.recipient,
            "recipient_role": self.recipient_role,
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "RDPReleaseCost":
        orders = [to_decimal(o) for o in data["orders"]]
        rdp_vector = {to_decimal(k): to_decimal(v) for k, v in data["rdp_vector"].items()}
        return cls(
            release_key=data["release_key"],
            mechanism=data["mechanism"],
            sensitivity=float(data["sensitivity"]),
            noise_scale=float(data["noise_scale"]),
            requested_epsilon=float(data["requested_epsilon"]),
            calibration_delta=float(data["calibration_delta"]),
            orders=orders,
            rdp_vector=rdp_vector,
            budget_scope_id=data.get("budget_scope_id", "global_external"),
            recipient=data.get("recipient", ""),
            recipient_role=data.get("recipient_role", "external"),
        )


@dataclass
class PrivacyView:
    """Authoritative reconstructed privacy view for a given budget scope or recipient."""
    budget_scope_id: str
    release_count: int
    total_rdp_vector: Dict[Decimal, Decimal]
    effective_epsilon: Decimal
    best_alpha: Optional[Decimal]
    delta_global: Decimal
    epsilon_cap: Decimal
    remaining_epsilon: Decimal
    accountant_mode: str = "rdp"
    orders: Optional[List[Decimal]] = None

    @property
    def total_releases(self) -> int:
        return self.release_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "budget_scope_id": self.budget_scope_id,
            "accountant_mode": self.accountant_mode,
            "release_count": self.release_count,
            "total_releases": self.release_count,
            "effective_epsilon": float(self.effective_epsilon),
            "effective_epsilon_decimal": str(self.effective_epsilon),
            "best_alpha": float(self.best_alpha) if self.best_alpha is not None else None,
            "best_alpha_decimal": str(self.best_alpha) if self.best_alpha is not None else None,
            "delta_global": float(self.delta_global),
            "epsilon_cap": float(self.epsilon_cap),
            "remaining_epsilon": float(self.remaining_epsilon),
            "remaining_epsilon_decimal": str(self.remaining_epsilon),
        }


def resolve_budget_scope_id(
    recipient: str,
    recipient_role: str = "external",
    mode: str = "global_external",
    groups: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Resolve authoritative budget scope ID from recipient and composition policy.

    Policies:
      - Direct scope IDs ('global_external', 'internal', 'recipient:*', 'collusion:*') pass through
      - 'internal' or trusted role -> 'internal' (trusted curator model, not subject to external DP cap)
      - 'global_external' -> 'global_external' (all external recipients share single global view)
      - 'per_recipient' -> 'recipient:<name>' (isolated view per recipient)
      - 'collusion_group' -> 'collusion:<group>' (members in a collusion group share a common view)
    """
    if recipient in ("global_external", "internal") or recipient.startswith(("recipient:", "collusion:")):
        return recipient

    if recipient_role == "trusted" or recipient == "internal":
        return "internal"

    norm_mode = mode.lower()
    if norm_mode == "global_external":
        return "global_external"
    elif norm_mode == "per_recipient":
        return f"recipient:{recipient}"
    elif norm_mode == "collusion_group":
        if groups:
            matched_groups = [grp_name for grp_name, members in groups.items() if recipient in members]
            if len(matched_groups) > 1:
                raise ValueError(
                    f"Collusion group overlap detected: recipient '{recipient}' belongs to multiple groups: {matched_groups}. "
                    "Overlapping collusion groups are strictly forbidden."
                )
            if matched_groups:
                return f"collusion:{matched_groups[0]}"
        return f"collusion:{recipient}"
    else:
        raise ValueError(f"Unknown recipient composition mode: '{mode}'")

