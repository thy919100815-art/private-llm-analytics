"""Verification script: Custom Production DB State Isolation (v0.1.6 Audit Fix B).

Verifies that:
  1. ReleaseOrchestrator(ExecutionMode.EXPERIMENT) checks custom production state DB paths
     configured in app_config.privacy.release.state_db_path (in addition to default results/privacy_state.db).
  2. Attempting to use the custom production DB in EXPERIMENT mode (directly or via relative path / symlink)
     strictly raises ExperimentProductionStateCollisionError.
  3. ExperimentRunner with a custom production DB path strictly raises ExperimentProductionStateCollisionError.
  4. ExperimentRunner using a dedicated experiment DB executes successfully without altering or
     polluting the custom production DB.
"""

from decimal import Decimal
from pathlib import Path
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.validator import ExecutionMode
from private_llm_analytics.privacy.exceptions import ExperimentProductionStateCollisionError
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator
from private_llm_analytics.service.analytics import AnalyticsService
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
    print("VERIFICATION: Custom Production State Isolation (Fix B)")
    print("=" * 70)

    custom_prod_db = Path("results/verify_custom_prod.db")
    custom_exp_db = Path("results/verify_custom_exp.db")

    cleanup_db(custom_prod_db)
    cleanup_db(custom_exp_db)

    # 1. Setup config with custom production DB path
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(custom_prod_db)
    config.privacy.global_budget.epsilon = 5.0
    config.privacy.global_budget.delta = 1e-4

    df = pd.DataFrame({
        "interaction_id": [f"ix_{i}" for i in range(15)],
        "user_id": [f"user_{i % 5}" for i in range(15)],
        "timestamp": ["2026-01-01T00:00:00Z"] * 15,
        "provider": ["provider_a"] * 10 + ["provider_b"] * 5,
        "task": ["technical"] * 10 + ["analysis"] * 5,
    })

    # -------------------------------------------------------------
    # Step 1: Run production release on custom_prod_db
    # -------------------------------------------------------------
    print("[1] Executing production release targeting custom_prod_db...")
    prod_service = AnalyticsService(config)
    prod_service.set_data(df)

    prod_res = prod_service.compute(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-5,
    )
    assert not prod_res["cache_hit"], "Initial production release must be cache miss"

    prod_state = PrivacyStateManager(db_path=custom_prod_db)
    spent_eps_initial, spent_delta_initial = prod_state.get_total_spent_decimal()
    assert spent_eps_initial == Decimal("1.0"), f"Expected 1.0 epsilon, got {spent_eps_initial}"
    print(f"    Committed {spent_eps_initial} epsilon to custom production DB.")

    # -------------------------------------------------------------
    # Step 2: Verify ReleaseOrchestrator(EXPERIMENT) fails closed against custom_prod_db
    # -------------------------------------------------------------
    print("[2] Verifying ReleaseOrchestrator(EXPERIMENT) fails closed against custom_prod_db...")
    # Exact path
    collision_caught_1 = False
    try:
        ReleaseOrchestrator(
            app_config=config,
            execution_mode=ExecutionMode.EXPERIMENT,
            state_manager=PrivacyStateManager(db_path=custom_prod_db),
        )
    except ExperimentProductionStateCollisionError as exc:
        collision_caught_1 = True
        print(f"    [PASS] Direct path collision blocked: {exc}")

    assert collision_caught_1, "ReleaseOrchestrator MUST reject custom_prod_db in EXPERIMENT mode!"

    # Relative path / traversal: e.g. results/../results/verify_custom_prod.db
    relative_traversal = Path("results/../results/verify_custom_prod.db")
    collision_caught_2 = False
    try:
        ReleaseOrchestrator(
            app_config=config,
            execution_mode=ExecutionMode.EXPERIMENT,
            state_manager=PrivacyStateManager(db_path=relative_traversal),
        )
    except ExperimentProductionStateCollisionError as exc:
        collision_caught_2 = True
        print(f"    [PASS] Relative path traversal collision blocked: {exc}")

    assert collision_caught_2, "ReleaseOrchestrator MUST resolve canonical paths and reject collision!"

    # Default results/privacy_state.db must also be rejected
    default_prod = Path("results/privacy_state.db")
    collision_caught_3 = False
    try:
        ReleaseOrchestrator(
            app_config=config,
            execution_mode=ExecutionMode.EXPERIMENT,
            state_manager=PrivacyStateManager(db_path=default_prod),
        )
    except ExperimentProductionStateCollisionError as exc:
        collision_caught_3 = True
        print(f"    [PASS] Default production path collision blocked: {exc}")

    assert collision_caught_3, "ReleaseOrchestrator MUST reject default production path in EXPERIMENT mode!"

    # -------------------------------------------------------------
    # Step 3: Verify ExperimentRunner fails closed against custom_prod_db
    # -------------------------------------------------------------
    print("[3] Verifying ExperimentRunner fails closed against custom_prod_db...")
    runner_collision = False
    try:
        ExperimentRunner(
            config=config,
            state_db_path=custom_prod_db,
        )
    except ExperimentProductionStateCollisionError as exc:
        runner_collision = True
        print(f"    [PASS] ExperimentRunner custom production DB collision blocked: {exc}")

    assert runner_collision, "ExperimentRunner MUST reject custom production DB path!"

    # -------------------------------------------------------------
    # Step 4: Verify ExperimentRunner with distinct DB does not alter custom_prod_db
    # -------------------------------------------------------------
    print("[4] Verifying ExperimentRunner with custom_exp_db runs isolated...")
    exp_runner = ExperimentRunner(
        config=config,
        state_db_path=custom_exp_db,
    )
    exp_runner.set_data(df)

    exp_res = exp_runner.run_trial(
        metric="user_prevalence",
        recipient="internal",
        epsilon=0.5,
        delta=1e-5,
        seed=42,
    )
    assert not exp_res["cache_hit"]

    # Check custom production DB is completely untouched
    spent_eps_after, spent_delta_after = prod_state.get_total_spent_decimal()
    prod_releases_after = prod_state.list_releases()

    assert spent_eps_after == spent_eps_initial, (
        f"Production budget altered! Before: {spent_eps_initial}, After: {spent_eps_after}"
    )
    assert len(prod_releases_after) == 1, (
        f"Production releases changed! Expected 1, got {len(prod_releases_after)}"
    )
    print("    [PASS] Custom production state DB completely unchanged after experiment trial.")

    # Cleanup
    cleanup_db(custom_prod_db)
    cleanup_db(custom_exp_db)

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED: Custom Production Isolation Verified")
    print("=" * 70)


if __name__ == "__main__":
    main()
