"""Verification script: Experiment Seed Identity and Monte Carlo Disambiguation (Audit v0.1.5 Finding B).

Verifies that:
  1. In experiment mode, distinct noise_seed values produce distinct canonical semantics and release keys.
  2. Distinct noise seeds do NOT hit each other's release cache.
  3. Re-running the exact same noise_seed on the same data produces the exact same release key and hits cache.
  4. trial_id disambiguates trials when noise_seed is not specified.
"""

from pathlib import Path
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService


def main():
    print("=" * 70)
    print("VERIFICATION: Experiment Seed Identity & Cache Isolation (Finding B)")
    print("=" * 70)

    test_exp_db = Path("results/verify_exp_seed.db")
    for ext in ["", "-wal", "-shm", ".jsonl"]:
        f = Path(str(test_exp_db) + ext)
        if f.exists():
            f.unlink(missing_ok=True)

    config = load_app_config("configs")
    config.privacy.global_budget.epsilon = 10.0
    config.privacy.global_budget.delta = 1e-4

    service = AnalyticsService(config)

    df = pd.DataFrame({
        "interaction_id": [f"ix_{i}" for i in range(20)],
        "user_id": [f"usr_{i % 5}" for i in range(20)],
        "timestamp": ["2026-01-01T00:00:00Z"] * 20,
        "provider": ["provider_a", "provider_b"] * 10,
        "task": ["technical", "analysis"] * 10,
    })
    service.set_data(df)

    # 1. Run with noise_seed=1
    print("[1] Running query with noise_seed=1...")
    res_s1 = service.compute_experiment(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        noise_seed=1,
        experiment_db_path=test_exp_db,
    )
    key_s1 = res_s1["release_key"]
    assert not res_s1["cache_hit"], "Initial seed=1 run must be a cache miss"
    print(f"    Release key for seed=1: {key_s1}")

    # 2. Run with noise_seed=2
    print("[2] Running query with noise_seed=2...")
    res_s2 = service.compute_experiment(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        noise_seed=2,
        experiment_db_path=test_exp_db,
    )
    key_s2 = res_s2["release_key"]
    print(f"    Release key for seed=2: {key_s2}")

    assert not res_s2["cache_hit"], "Distinct seed=2 MUST NOT hit seed=1 cache (Finding B)!"
    assert key_s1 != key_s2, "Distinct noise seeds MUST generate distinct canonical release keys!"
    assert res_s1["results"] != res_s2["results"], "Distinct seeds must generate distinct noise vectors!"
    print("    [PASS] Distinct noise seeds generate distinct release keys and independent noise.")

    # 3. Re-run with noise_seed=1 (should cache hit)
    print("[3] Re-running query with noise_seed=1...")
    res_s1_repeat = service.compute_experiment(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        noise_seed=1,
        experiment_db_path=test_exp_db,
    )
    assert res_s1_repeat["cache_hit"], "Same noise_seed=1 MUST hit cache!"
    assert res_s1_repeat["release_key"] == key_s1
    assert res_s1_repeat["results"] == res_s1["results"]
    print("    [PASS] Re-running same noise_seed correctly hits cache and reproduces output.")

    # 4. Run with trial_id
    print("[4] Testing trial_id disambiguation...")
    res_t1 = service.compute_experiment(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        trial_id="mc_trial_100",
        experiment_db_path=test_exp_db,
    )
    res_t2 = service.compute_experiment(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        trial_id="mc_trial_101",
        experiment_db_path=test_exp_db,
    )
    assert res_t1["release_key"] != res_t2["release_key"]
    assert not res_t2["cache_hit"]
    print("    [PASS] Distinct trial_id values produce distinct release identities.")

    # Cleanup
    for ext in ["", "-wal", "-shm", ".jsonl"]:
        f = Path(str(test_exp_db) + ext)
        if f.exists():
            f.unlink(missing_ok=True)

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED: Experiment Seed Identity Verified")
    print("=" * 70)


if __name__ == "__main__":
    main()
