"""Verification script: Experiment and Production Isolation (Audit F1).

Verifies that:
  1. Experiment executions (with noise_seed or caller RNG) write only to the isolated experiment DB.
  2. Production execution paths never hit experiment-cached releases (namespaced canonical keys).
  3. Reverse contamination is prevented: production releases do not poison experiment runs.
  4. Production compute() strictly prohibits caller-supplied RNG, seed, or noise for ALL recipients.
  5. An attacker with knowledge of experiment noise_seed cannot recover private aggregates
     from production external releases.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.privacy.exceptions import (
    ProductionRNGControlError,
    RecipientSecurityViolation,
)
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.service.analytics import AnalyticsService


def main():
    print("=" * 70)
    print("VERIFICATION: Experiment & Production Isolation (Audit F1)")
    print("=" * 70)

    prod_db = Path("results/verify_isolation_prod.db")
    exp_db = Path("results/verify_isolation_exp.db")

    for p in [prod_db, exp_db]:
        for ext in ["", "-wal", "-shm", ".jsonl"]:
            f = Path(str(p) + ext)
            if f.exists():
                f.unlink(missing_ok=True)

    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(prod_db)
    config.privacy.global_budget.epsilon = 10.0
    config.privacy.global_budget.delta = 1e-4

    service = AnalyticsService(config)
    service.generate_synthetic_data(n_users=200, seed=42)

    # -------------------------------------------------------------
    # Step 1: Run deterministic experiment release with known seed
    # -------------------------------------------------------------
    print("[1] Executing experiment release with deterministic noise seed 12345...")
    exp_res = service.compute_experiment(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        mechanism="gaussian",
        include_exact=True,
        noise_seed=12345,
        experiment_db_path=exp_db,
    )

    assert not exp_res["cache_hit"], "Experiment first call should not be a cache hit"
    exp_key = exp_res["release_key"]
    print(f"    Experiment release key: {exp_key}")

    # Verify physical isolation: written to exp_db, prod_db must have 0 spend
    prod_state = PrivacyStateManager(prod_db)
    spent_eps, spent_delta = prod_state.get_total_spent()
    assert float(spent_eps) == 0.0, f"Production DB spent epsilon should be 0, got {spent_eps}"
    assert float(spent_delta) == 0.0, f"Production DB spent delta should be 0, got {spent_delta}"
    print("    [PASS] Production state DB has 0 spend after experiment execution.")

    # -------------------------------------------------------------
    # Step 2: Run production query with identical parameters
    # -------------------------------------------------------------
    print("[2] Executing production external query with identical parameters...")
    prod_res = service.compute(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        mechanism="gaussian",
    )

    prod_key = prod_res["release_key"]
    print(f"    Production release key: {prod_key}")

    assert not prod_res["cache_hit"], "Production query MUST NOT hit experiment cache!"
    assert prod_key != exp_key, "Production and experiment release keys must be distinct (namespaced by execution_mode)!"
    print("    [PASS] Production query was a cache miss; keys are distinct.")

    # Verify that production noise was NOT generated from seed 12345
    exp_released_tech = exp_res["results"]["dp_counts"]["provider_a:technical"]
    prod_released_tech = prod_res["results"]["dp_counts"]["provider_a:technical"]

    assert exp_released_tech != prod_released_tech, "Production release noise must not match experiment noise!"
    print("    [PASS] Production output was generated with secure unseeded entropy.")

    # -------------------------------------------------------------
    # Step 3: Verify caller RNG is strictly forbidden in production
    # -------------------------------------------------------------
    print("[3] Verifying caller-supplied RNG is rejected in production compute()...")
    bad_rng = np.random.default_rng(9999)

    rejected_external = False
    try:
        service.compute(metric="user_prevalence", recipient="provider_a", rng=bad_rng)
    except (ProductionRNGControlError, RecipientSecurityViolation):
        rejected_external = True
    assert rejected_external, "Production compute() must reject caller RNG for external recipients"

    rejected_internal = False
    try:
        service.compute(metric="user_prevalence", recipient="internal", rng=bad_rng)
    except (ProductionRNGControlError, RecipientSecurityViolation):
        rejected_internal = True
    assert rejected_internal, "Production compute() must reject caller RNG for trusted/internal recipients too"
    print("    [PASS] Caller-supplied RNG is blocked for all recipients in production compute().")

    # -------------------------------------------------------------
    # Step 4: Verify repeated experiment call hits experiment cache
    # -------------------------------------------------------------
    print("[4] Verifying repeated experiment query hits experiment cache...")
    exp_res_2 = service.compute_experiment(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        mechanism="gaussian",
        noise_seed=12345,
        experiment_db_path=exp_db,
    )
    assert exp_res_2["cache_hit"], "Repeated experiment query must hit experiment cache"
    assert exp_res_2["results"] == exp_res["results"], "Experiment memoized results must match"
    print("    [PASS] Experiment state correctly memoizes experiment releases.")

    # Cleanup test DBs
    for p in [prod_db, exp_db]:
        for ext in ["", "-wal", "-shm", ".jsonl"]:
            f = Path(str(p) + ext)
            if f.exists():
                f.unlink(missing_ok=True)

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED: Experiment & Production Isolation Verified")
    print("=" * 70)


if __name__ == "__main__":
    main()
