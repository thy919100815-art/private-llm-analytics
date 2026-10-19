"""Recipient access policy filtering and security enforcement in v0.1.1."""

from typing import Any, Dict, List, Optional
from private_llm_analytics.config.models import RecipientPolicy
from private_llm_analytics.metrics.base import DPReleaseResult


from private_llm_analytics.privacy.exceptions import RecipientSecurityViolation



def filter_release_for_recipient(
    release_result: DPReleaseResult,
    recipient_name: str,
    policy: RecipientPolicy,
    include_exact: bool = False,
    exact_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Filter DP release statistics to enforce recipient isolation and user privacy."""
    # 1. Enforce DP results authorization
    if not policy.dp_results:
        raise RecipientSecurityViolation(
            f"Recipient '{recipient_name}' is not authorized to receive DP results (dp_results=False)."
        )

    # 2. Enforce exact ground truth authorization
    if include_exact:
        if policy.role == "external" or not policy.exact_results:
            raise RecipientSecurityViolation(
                f"Recipient '{recipient_name}' is strictly forbidden from receiving exact results (exact_results=False)."
            )

    allowed_providers = set(policy.providers)
    wildcard_provider = "*" in allowed_providers

    output: Dict[str, Any] = {
        "metric": release_result.metric_name,
        "recipient": recipient_name,
        "epsilon": release_result.epsilon,
        "delta": release_result.delta,
        "mechanism": release_result.mechanism,
        "noise_scale": release_result.noise_scale,
    }

    # CRITICAL FIX C1: Do NOT release exact active user count to external recipients!
    if policy.role == "trusted":
        output["active_user_count"] = release_result.n_users
        if release_result.public_population_size is not None:
            output["public_population_size"] = release_result.public_population_size
    else:
        # External recipients only receive public population size if defined
        if release_result.public_population_size is not None:
            output["public_population_size"] = release_result.public_population_size

    # Post-processed statistics
    pp = release_result.post_processed
    filtered_stats: Dict[str, Any] = {}

    for k, v in pp.items():
        if k in ("dp_counts", "prevalence_rates", "cell_volumes"):
            if wildcard_provider:
                filtered_stats[k] = v
            else:
                filtered_stats[k] = {
                    cell_k: cell_v for cell_k, cell_v in v.items()
                    if any(cell_k.startswith(f"{p}:") for p in allowed_providers)
                }
        elif k == "cell_shares":
            # If cross_provider_comparison is False and external recipient:
            # Re-normalize to 100% within the allowed provider's tasks!
            if wildcard_provider or policy.cross_provider_comparison:
                filtered_stats[k] = v
            else:
                prov_cells = {
                    cell_k: cell_v for cell_k, cell_v in v.items()
                    if any(cell_k.startswith(f"{p}:") for p in allowed_providers)
                }
                tot_p = sum(prov_cells.values())
                if tot_p > 0:
                    filtered_stats[k] = {ck: cv / tot_p for ck, cv in prov_cells.items()}
                else:
                    filtered_stats[k] = {ck: 1.0 / max(len(prov_cells), 1) for ck in prov_cells}
        elif k == "provider_conditional_task_distribution":
            # P(Task | Provider) is always safe because it's normalized to 100% within each provider
            if wildcard_provider:
                filtered_stats[k] = v
            else:
                filtered_stats[k] = {p: v[p] for p in allowed_providers if p in v}
        elif k == "provider_distribution":
            # CRITICAL FIX H2: Provider distribution P(Provider) reveals company-wide market share!
            # Only include if wildcard_provider OR cross_provider_comparison is True!
            if wildcard_provider or policy.cross_provider_comparison:
                filtered_stats[k] = v
            # Otherwise omitted to prevent competitor volume leakage
        elif k == "task_distribution":
            # Overall task distribution across all providers
            if wildcard_provider or policy.cross_provider_comparison:
                filtered_stats[k] = v
        elif k in ("clamped_volume_array", "share_distribution_array", "grid_3x3", "clamped_counts_array"):
            if wildcard_provider:
                filtered_stats[k] = v.tolist() if hasattr(v, "tolist") else v
        elif k == "total_estimated_volume":
            if wildcard_provider or policy.cross_provider_comparison:
                filtered_stats[k] = v

    output["results"] = filtered_stats

    # Include exact data only if recipient is trusted and authorized
    if policy.role == "trusted" and include_exact and exact_data is not None:
        output["exact_ground_truth"] = exact_data

    return output
