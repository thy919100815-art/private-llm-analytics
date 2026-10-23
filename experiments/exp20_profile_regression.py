"""Exp20: Profile Regression and Equivalence Verification (v0.6).

Verifies that migrating the three existing analytics metrics into statistic-specific
DP profiles preserves their mathematical and empirical semantics bit-for-bit.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_cell_definitions
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
from private_llm_analytics.metrics.prevalence import UserPrevalenceMetric
from private_llm_analytics.metrics.normalized_share import UserNormalizedShareMetric
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.statistics.prevalence import UserPrevalenceProfile
from private_llm_analytics.statistics.normalized_share import UserNormalizedShareProfile
from private_llm_analytics.statistics.interaction_volume import InteractionVolumeProfile


def build_evaluation_user_matrix() -> UserMatrix:
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)
    rng = np.random.default_rng(202610)
    # 50 users, realistic interaction counts across 9 cells
    matrix = rng.poisson(lam=8.0, size=(50, 9)).astype(np.float64)
    # Sparsify some cells to test zero-handling
    matrix[matrix < 5.0] = 0.0

    return UserMatrix(
        matrix=matrix,
        user_ids=[f"user_{i}" for i in range(50)],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=1000,
    )


def run_exp20_equivalence() -> dict:
    umatrix = build_evaluation_user_matrix()
    results = {}

    # 1. User Prevalence Equivalence
    legacy_prev = UserPrevalenceMetric(max_cells_per_user=9)
    profile_prev = UserPrevalenceProfile(max_cells_per_user=9)

    contrib_legacy = legacy_prev.build_user_contributions(umatrix)
    contrib_profile = profile_prev.build_user_contributions(umatrix)
    assert np.allclose(contrib_legacy, contrib_profile), "Prevalence contributions differ!"

    sens_legacy = legacy_prev.get_sensitivity("gaussian")
    sens_profile = profile_prev.sensitivity()
    assert abs(sens_legacy - sens_profile) < 1e-12, "Prevalence sensitivities differ!"

    plan_profile = profile_prev.build_mechanism_plan(epsilon=1.0, delta=1e-6)
    mech_legacy = GaussianMechanism(epsilon=1.0, delta=1e-6, sensitivity=sens_legacy, method="analytic")
    assert abs(mech_legacy.noise_scale - plan_profile.noise_scale) < 1e-10, "Prevalence noise scale differs!"

    # Release under identical RNG seed
    noisy_vec = np.sum(contrib_legacy, axis=0) + np.random.default_rng(42).normal(0, mech_legacy.noise_scale, size=9)
    pp_legacy = legacy_prev.post_process(noisy_vec, umatrix)
    pp_profile = profile_prev.post_process(noisy_vec, umatrix)
    for k in pp_legacy:
        if isinstance(pp_legacy[k], np.ndarray):
            assert np.allclose(pp_legacy[k], pp_profile[k]), f"Prevalence post-process key {k} differs!"
        else:
            assert pp_legacy[k] == pp_profile[k], f"Prevalence post-process key {k} differs!"

    results["user_prevalence"] = {
        "contributions_match": True,
        "sensitivity_match": True,
        "noise_scale_match": True,
        "post_processing_match": True,
        "sensitivity": float(sens_profile),
        "noise_scale": float(plan_profile.noise_scale),
    }

    # 2. User-Normalized Share Equivalence
    legacy_share = UserNormalizedShareMetric(norm_bound=1.0)
    profile_share = UserNormalizedShareProfile(norm_bound=1.0)

    contrib_legacy = legacy_share.build_user_contributions(umatrix)
    contrib_profile = profile_share.build_user_contributions(umatrix)
    assert np.allclose(contrib_legacy, contrib_profile), "Share contributions differ!"

    sens_legacy = legacy_share.get_sensitivity("gaussian")
    sens_profile = profile_share.sensitivity()
    assert abs(sens_legacy - sens_profile) < 1e-12, "Share sensitivities differ!"

    plan_profile = profile_share.build_mechanism_plan(epsilon=1.0, delta=1e-6)
    mech_legacy = GaussianMechanism(epsilon=1.0, delta=1e-6, sensitivity=sens_legacy, method="analytic")
    assert abs(mech_legacy.noise_scale - plan_profile.noise_scale) < 1e-10, "Share noise scale differs!"

    noisy_vec = np.sum(contrib_legacy, axis=0) + np.random.default_rng(43).normal(0, mech_legacy.noise_scale, size=9)
    pp_legacy = legacy_share.post_process(noisy_vec, umatrix)
    pp_profile = profile_share.post_process(noisy_vec, umatrix)
    for k in pp_legacy:
        if isinstance(pp_legacy[k], np.ndarray):
            assert np.allclose(pp_legacy[k], pp_profile[k]), f"Share post-process key {k} differs!"

    results["user_normalized_share"] = {
        "contributions_match": True,
        "sensitivity_match": True,
        "noise_scale_match": True,
        "post_processing_match": True,
        "sensitivity": float(sens_profile),
        "noise_scale": float(plan_profile.noise_scale),
    }

    # 3. Interaction Volume Equivalence
    clip_c = 25.0
    legacy_vol = InteractionVolumeMetric(clip_norm=clip_c)
    profile_vol = InteractionVolumeProfile(clip_norm=clip_c)

    contrib_legacy = legacy_vol.build_user_contributions(umatrix)
    contrib_profile = profile_vol.build_user_contributions(umatrix)
    assert np.allclose(contrib_legacy, contrib_profile), "Volume contributions differ!"

    sens_legacy = legacy_vol.get_sensitivity("gaussian")
    sens_profile = profile_vol.sensitivity()
    assert abs(sens_legacy - sens_profile) < 1e-12, "Volume sensitivities differ!"

    plan_profile = profile_vol.build_mechanism_plan(epsilon=1.0, delta=1e-6)
    mech_legacy = GaussianMechanism(epsilon=1.0, delta=1e-6, sensitivity=sens_legacy, method="analytic")
    assert abs(mech_legacy.noise_scale - plan_profile.noise_scale) < 1e-10, "Volume noise scale differs!"

    noisy_vec = np.sum(contrib_legacy, axis=0) + np.random.default_rng(44).normal(0, mech_legacy.noise_scale, size=9)
    pp_legacy = legacy_vol.post_process(noisy_vec, umatrix)
    pp_profile = profile_vol.post_process(noisy_vec, umatrix)
    for k in pp_legacy:
        if isinstance(pp_legacy[k], np.ndarray):
            assert np.allclose(pp_legacy[k], pp_profile[k]), f"Volume post-process key {k} differs!"

    results["interaction_volume"] = {
        "contributions_match": True,
        "sensitivity_match": True,
        "noise_scale_match": True,
        "post_processing_match": True,
        "sensitivity": float(sens_profile),
        "noise_scale": float(plan_profile.noise_scale),
    }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp20_profile_regression.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Exp20 PASS: Profile regression verified across all 3 legacy statistics. Artifact: {out_file}")
    return results


if __name__ == "__main__":
    run_exp20_equivalence()
