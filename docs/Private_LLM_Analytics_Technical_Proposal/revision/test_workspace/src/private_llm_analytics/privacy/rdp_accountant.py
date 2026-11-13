"""Longitudinal Rényi Differential Privacy (RDP) Accountant.

Implements order-wise RDP accounting, candidate preview, lifetime composition,
and (epsilon, delta) conversion against global privacy policy caps.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from private_llm_analytics.privacy.accountant import PrivacyAccountant
from private_llm_analytics.privacy.budget import PrivacyBudget, PrivacySpent
from private_llm_analytics.privacy.composition import PrivacyView, RDPReleaseCost
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    compute_laplace_rdp_bound,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
    to_decimal,
    validate_delta,
    validate_orders,
)
from private_llm_analytics.privacy.state import PrivacyStateManager


class RDPAccountant(PrivacyAccountant):
    """Longitudinal RDP accountant managing persistent composition across releases."""

    def __init__(
        self,
        epsilon_cap: Union[float, Decimal] = 4.0,
        delta_global: Union[float, Decimal] = 1e-6,
        orders: Optional[Sequence[Union[float, Decimal]]] = None,
        state_manager: Optional[PrivacyStateManager] = None,
    ):
        self.epsilon_cap = to_decimal(epsilon_cap)
        self.delta_global = validate_delta(delta_global)
        raw_orders = orders if orders is not None else DEFAULT_RDP_ORDERS
        self.orders = validate_orders(raw_orders)
        self.state_manager = state_manager or PrivacyStateManager()
        self._manual_total_rdp: Optional[Dict[Decimal, Decimal]] = None

    def compute_release_rdp(
        self,
        mechanism: str,
        sensitivity: Union[float, Decimal],
        noise_scale: Union[float, Decimal],
        requested_epsilon: Union[float, Decimal],
        calibration_delta: Optional[Union[float, Decimal]] = None,
    ) -> Dict[Decimal, Decimal]:
        """Compute exact RDP vector for an executed or planned release.

        For Gaussian mechanism: computes alpha * Delta^2 / (2 * sigma^2) from actual sigma.
        For Laplace mechanism: computes conservative pure-DP bound (RDP(alpha) <= epsilon).
        """
        mech_lower = mechanism.lower()
        if "gaussian" in mech_lower:
            return compute_gaussian_rdp(
                sigma=noise_scale,
                sensitivity=sensitivity,
                orders=self.orders,
            )
        elif "laplace" in mech_lower:
            return compute_laplace_rdp_bound(
                epsilon=requested_epsilon,
                orders=self.orders,
            )
        else:
            raise ValueError(f"RDP accountant does not support mechanism '{mechanism}'")

    def preview_composition(
        self,
        current_total_rdp: Optional[Dict[Decimal, Decimal]],
        candidate_rdp: Dict[Decimal, Decimal],
    ) -> Dict[str, Any]:
        """Preview resulting effective epsilon and decision without committing state."""
        base_rdp = current_total_rdp or {alpha: Decimal("0.0") for alpha in self.orders}
        composed = compose_rdp_vectors([base_rdp, candidate_rdp], orders=self.orders)

        best_alpha_before, eps_before = (
            rdp_to_epsilon_delta(base_rdp, self.delta_global)
            if any(v > Decimal("0.0") for v in base_rdp.values())
            else (self.orders[0], Decimal("0.0"))
        )
        best_alpha_after, eps_after = rdp_to_epsilon_delta(composed, self.delta_global)

        allowed = eps_after <= self.epsilon_cap

        return {
            "allowed": allowed,
            "epsilon_cap": str(self.epsilon_cap),
            "delta_global": str(self.delta_global),
            "epsilon_before": str(eps_before),
            "best_alpha_before": str(best_alpha_before),
            "epsilon_after": str(eps_after),
            "best_alpha_after": str(best_alpha_after),
            "candidate_rdp": {str(k): str(v) for k, v in candidate_rdp.items()},
        }

    def preview_candidate_rdp(
        self,
        candidate_rdp: Dict[Decimal, Decimal],
        budget_scope_id: str = "global_external",
    ) -> Dict[str, Any]:
        """Preview resulting effective epsilon and decision for a candidate RDP vector against current scope state."""
        view = self.state_manager.get_privacy_view(
            budget_scope_id=budget_scope_id,
            delta_global=self.delta_global,
            orders=self.orders,
            epsilon_cap=self.epsilon_cap,
            accountant_mode="rdp",
        )
        return self.preview_composition(
            current_total_rdp=view.total_rdp_vector,
            candidate_rdp=candidate_rdp,
        )

    @property
    def spent_epsilon(self) -> float:
        """Cumulative committed effective epsilon from authoritative state.

        If external releases exist under any active budget scope, computes the maximum
        effective epsilon across all external budget scopes under RDP composition.
        If no RDP releases exist, falls back to legacy state manager scalar spent.
        """
        active_scopes = self.state_manager.get_active_budget_scopes(exclude_internal=True)
        if not active_scopes:
            active_scopes = ["global_external"]

        max_eps = 0.0
        for scope in active_scopes:
            view = self.state_manager.get_privacy_view(
                budget_scope_id=scope,
                delta_global=self.delta_global,
                orders=self.orders,
                epsilon_cap=self.epsilon_cap,
                accountant_mode="rdp",
            )
            if view.release_count > 0:
                max_eps = max(max_eps, float(view.effective_epsilon))
        if max_eps > 0.0:
            return max_eps
        total_eps, _ = self.state_manager.get_total_spent()
        return float(total_eps)

    @property
    def spent_delta(self) -> float:
        """Global delta policy parameter."""
        active_scopes = self.state_manager.get_active_budget_scopes(exclude_internal=True)
        if not active_scopes:
            active_scopes = ["global_external"]
        for scope in active_scopes:
            view = self.state_manager.get_privacy_view(
                budget_scope_id=scope,
                delta_global=self.delta_global,
                orders=self.orders,
                epsilon_cap=self.epsilon_cap,
                accountant_mode="rdp",
            )
            if view.release_count > 0:
                return float(self.delta_global)
        _, total_delta = self.state_manager.get_total_spent()
        return float(total_delta)

    # Backward compatibility with PrivacyAccountant interface
    def check_budget_available(self, budget_needed: PrivacyBudget) -> bool:
        """Fail-closed: RDP composition cannot be performed from scalar PrivacyBudget."""
        raise NotImplementedError(
            "RDPAccountant cannot evaluate budget availability solely from scalar (epsilon, delta) PrivacyBudget. "
            "Use preview_candidate_rdp() or state_manager.reserve_or_get_release() with candidate RDP vector."
        )

    def consume_budget(self, budget: PrivacyBudget, query_id: str) -> None:
        """Fail-closed: RDP budget consumption cannot be performed as a non-atomic scalar no-op."""
        raise NotImplementedError(
            "RDPAccountant does not support non-atomic scalar consume_budget(). "
            "All privacy accounting must be performed atomically via PrivacyStateManager."
        )

    def get_total_spent(self, budget_scope_id: Optional[str] = None) -> PrivacySpent:
        """Return effective epsilon and delta_global from current persistent state.

        If budget_scope_id is specified, queries that specific scope.
        Otherwise, evaluates across active external budget scopes (maximum effective epsilon).
        Returns PrivacySpent allowing exact zero spend (epsilon = Decimal('0.0')).
        """
        if budget_scope_id is not None:
            view = self.state_manager.get_privacy_view(
                budget_scope_id=budget_scope_id,
                delta_global=self.delta_global,
                orders=self.orders,
                epsilon_cap=self.epsilon_cap,
                accountant_mode="rdp",
            )
            return PrivacySpent(
                epsilon=to_decimal(view.effective_epsilon),
                delta=to_decimal(self.delta_global),
            )

        active_scopes = self.state_manager.get_active_budget_scopes(exclude_internal=True)
        if not active_scopes:
            active_scopes = ["global_external"]

        max_eps = Decimal("0.0")
        for scope in active_scopes:
            view = self.state_manager.get_privacy_view(
                budget_scope_id=scope,
                delta_global=self.delta_global,
                orders=self.orders,
                epsilon_cap=self.epsilon_cap,
                accountant_mode="rdp",
            )
            max_eps = max(max_eps, to_decimal(view.effective_epsilon))

        return PrivacySpent(
            epsilon=max_eps,
            delta=to_decimal(self.delta_global),
        )
