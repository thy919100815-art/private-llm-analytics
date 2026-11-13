"""Verification script: 500 Independent Noise Trials & Evaluation Accounting Mode (v0.1.6 Audit Fix A).

Verifies that:
  1. ExperimentRunner in default EVALUATION accounting mode can execute 500+ independent
     noise trials (seeds 0..499) without raising PrivacyBudgetExceededError.
  2. All 500 trials produce distinct release keys and distinct noisy estimates.
  3. Re-running with an already evaluated seed hits the experiment release cache.
  4. Production state DB remains 100% untouched (epsilon spent = 0, releases = 0).
  5. ExperimentRunner in BOUNDED mode fails closed with PrivacyBudgetExceededError
     once the experiment budget is exhausted.
"""

from decimal import Decimal
from pathlib import Path
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.validator import ExperimentAccountingMode
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.service.experiment import ExperimentRunner


import gc


def cleanup_db(db_path: Path):
    gc.collect()
    for ext in ["", "-wal", "-shm", ".jsonl"]:
        f = Path(str(db_path) + ext)
        try:
            if f.exists():
                f.unlink(missing_ok=True)
        except PermissionError:
            pass


def main():
    print("=" * 70)
    print("VERIFICATION: 500 Independent Noise Trials & Evaluation Mode (Fix A)")
    print("=" * 70)

    prod_db = Path("results/verify_500_prod.db")
    exp_db = Path("results/verify_500_exp.db")
    bounded_db = Path("results/verify_500_bounded.db")

    for p in [prod_db, exp_db, bounded_db]:
        cleanup_db(p)

    # 1. Setup base configuration
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(prod_db)
    config.privacy.global_budget.epsilon = 1.0
    config.privacy.global_budget.delta = 1e-5

    # Initialize production manager (empty)
    prod_state = PrivacyStateManager(db_path=prod_db)

    # Synthetic test data
    df = pd.DataFrame({
        "interaction_id": [f"i_{i}" for i in range(20)],
        "user_id": [f"u_{i % 5}" for i in range(20)],
        "timestamp": ["2026-01-01T00:00:00Z"] * 20,
        "provider": ["provider_a"] * 10 + ["provider_b"] * 10,
        "task": ["technical"] * 10 + ["analysis"] * 10,
    })

    # -------------------------------------------------------------
    # Step 1: Run 500 independent trials in EVALUATION mode
    # -------------------------------------------------------------
    print("[1] Executing 500 noise trials with ExperimentRunner (accounting_mode=EVALUATION)...")
    runner_eval = ExperimentRunner(
        config=config,
        state_db_path=exp_db,
        accounting_mode=ExperimentAccountingMode.EVALUATION,
    )
    runner_eval.set_data(df)

    # Run trials using convenience method
    n_trials = 500
    results = runner_eval.run_trials(
        metric="user_prevalence",
        recipient="internal",
        epsilon=0.1,
        delta=1e-6,
        n_trials=n_trials,
        base_seed=0,
    )

    assert len(results) == n_trials, f"Expected {n_trials} results, got {len(results)}"
    print(f"    Successfully completed {len(results)} trials without budget exhaustion.")

    # -------------------------------------------------------------
    # Step 2: Verify distinct release keys and noise variation
    # -------------------------------------------------------------
    print("[2] Verifying trial diversity and release keys...")
    release_keys = set()
    noisy_estimates = []
    for r in results:
        assert not r["cache_hit"], "First run of distinct seeds must be cache misses"
        release_keys.add(r["release_key"])
        # Take the prevalence value for 'provider_a:technical'
        val = r["results"]["prevalence_rates"]["provider_a:technical"]
        noisy_estimates.append(val)

    assert len(release_keys) == n_trials, (
        f"Expected {n_trials} distinct release keys, found {len(release_keys)}"
    )
    distinct_estimates = set(noisy_estimates)
    assert len(distinct_estimates) > 100, (
        f"Expected high noise variance across 500 trials, got {len(distinct_estimates)} distinct values"
    )
    print(f"    Generated {len(release_keys)} unique release keys and {len(distinct_estimates)} unique estimates.")

    # -------------------------------------------------------------
    # Step 3: Verify repeatability and cache hit for seed=0
    # -------------------------------------------------------------
    print("[3] Verifying cache hit on repeated trial with seed=0...")
    repeat_res = runner_eval.run_trial(
        metric="user_prevalence",
        recipient="internal",
        epsilon=0.1,
        delta=1e-6,
        seed=0,
    )
    assert repeat_res["cache_hit"] is True, "Repeat run with seed=0 MUST hit experiment cache"
    assert repeat_res["release_key"] == results[0]["release_key"], "Release key must match seed=0 key"
    assert repeat_res["results"] == results[0]["results"], "Cached result must match exactly"
    print("    [PASS] Cache hit confirmed on repeated experiment run.")

    # -------------------------------------------------------------
    # Step 4: Verify production state DB is untouched (100% isolation)
    # -------------------------------------------------------------
    print("[4] Verifying production state DB isolation...")
    spent_eps, spent_delta = prod_state.get_total_spent_decimal()
    prod_releases = prod_state.list_releases()
    assert spent_eps == Decimal("0.0"), f"Production epsilon spent must be 0, got {spent_eps}"
    assert spent_delta == Decimal("0.0"), f"Production delta spent must be 0, got {spent_delta}"
    assert len(prod_releases) == 0, f"Production releases must be 0, got {len(prod_releases)}"
    print("    [PASS] Production state DB is 100% clean (0 spent, 0 releases).")

    # -------------------------------------------------------------
    # Step 5: Verify BOUNDED mode enforces budget limits
    # -------------------------------------------------------------
    print("[5] Verifying BOUNDED mode enforces budget limit...")
    runner_bounded = ExperimentRunner(
        config=config,
        state_db_path=bounded_db,
        accounting_mode=ExperimentAccountingMode.BOUNDED,
    )
    runner_bounded.set_data(df)

    # Budget is 1.0 epsilon. Each trial spends 0.3 epsilon.
    # Trial 1: 0.3 (total 0.3 <= 1.0) -> OK
    # Trial 2: 0.3 (total 0.6 <= 1.0) -> OK
    # Trial 3: 0.3 (total 0.9 <= 1.0) -> OK
    # Trial 4: 0.3 (total 1.2 > 1.0) -> Must fail with PrivacyBudgetExceededError
    runner_bounded.run_trial(metric="user_prevalence", recipient="internal", epsilon=0.3, delta=1e-6, seed=100)
    runner_bounded.run_trial(metric="user_prevalence", recipient="internal", epsilon=0.3, delta=1e-6, seed=101)
    runner_bounded.run_trial(metric="user_prevalence", recipient="internal", epsilon=0.3, delta=1e-6, seed=102)

    exceeded = False
    try:
        runner_bounded.run_trial(metric="user_prevalence", recipient="internal", epsilon=0.3, delta=1e-6, seed=103)
    except PrivacyBudgetExceededError:
        exceeded = True

    assert exceeded, "BOUNDED mode MUST raise PrivacyBudgetExceededError when budget is exhausted!"
    print("    [PASS] BOUNDED mode correctly rejected trial exceeding budget.")

    # Cleanup
    for p in [prod_db, exp_db, bounded_db]:
        cleanup_db(p)

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED: 500 Noise Trials & Evaluation Mode")
    print("=" * 70)


if __name__ == "__main__":
    main()
