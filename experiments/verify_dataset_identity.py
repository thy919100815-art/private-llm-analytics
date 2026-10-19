"""Verification script: Dataset Identity and Fingerprint Invalidation (Audit F4).

Verifies that:
  1. set_data() without explicit dataset_version automatically computes a SHA-256 multiset content fingerprint.
  2. Modifying in-memory dataframe contents changes the fingerprint and release key.
  3. Changing datasets prevents stale cache hits: different data generates a fresh release.
  4. Row order permutation preserves the multiset content fingerprint (stable cache hit under reordering).
"""

from pathlib import Path
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.ingestion.fingerprint import compute_dataset_fingerprint
from private_llm_analytics.service.analytics import AnalyticsService


def main():
    print("=" * 70)
    print("VERIFICATION: Dataset Identity & Content Fingerprinting (Audit F4)")
    print("=" * 70)

    test_db = Path("results/verify_dataset_identity.db")
    for ext in ["", "-wal", "-shm", ".jsonl"]:
        f = Path(str(test_db) + ext)
        if f.exists():
            f.unlink(missing_ok=True)

    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(test_db)
    config.privacy.global_budget.epsilon = 10.0
    config.privacy.global_budget.delta = 1e-4

    service = AnalyticsService(config)

    # -------------------------------------------------------------
    # Step 1: Create Dataset A and query prevalence
    # -------------------------------------------------------------
    df_a = pd.DataFrame({
        "interaction_id": [f"a_{i}" for i in range(10)],
        "user_id": [f"user_{i}" for i in range(10)],
        "timestamp": ["2026-01-01T00:00:00Z"] * 10,
        "provider": ["provider_a"] * 5 + ["provider_b"] * 5,
        "task": ["technical"] * 10,
    })

    print("[1] Setting Dataset A (auto-fingerprint)...")
    service.set_data(df_a)
    fp_a = service._dataset_version
    print(f"    Dataset A fingerprint: {fp_a}")
    assert fp_a == compute_dataset_fingerprint(df_a)

    res_a = service.compute(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
    )
    assert not res_a["cache_hit"], "Dataset A first release should be cache miss"
    key_a = res_a["release_key"]
    print(f"    Release key for A: {key_a}")

    # -------------------------------------------------------------
    # Step 2: Create Dataset B (different data) and query same metric
    # In v0.1.3, this resulted in a stale cache hit reusing Dataset A's release!
    # -------------------------------------------------------------
    df_b = pd.DataFrame({
        "interaction_id": [f"b_{i}" for i in range(10)],
        "user_id": [f"user_{i}" for i in range(10)],
        "timestamp": ["2026-01-01T00:00:00Z"] * 10,
        "provider": ["provider_c"] * 10,  # Entirely different provider
        "task": ["document"] * 10,
    })

    print("[2] Setting Dataset B (different content)...")
    service.set_data(df_b)
    fp_b = service._dataset_version
    print(f"    Dataset B fingerprint: {fp_b}")
    assert fp_b != fp_a, "Different datasets must have different fingerprints!"

    res_b = service.compute(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
    )
    key_b = res_b["release_key"]
    print(f"    Release key for B: {key_b}")

    assert not res_b["cache_hit"], "Dataset B MUST NOT hit Dataset A's cache (Audit F4)!"
    assert key_b != key_a, "Release keys must differ when dataset content differs!"
    print("    [PASS] Dataset change successfully invalidated release cache key.")

    # -------------------------------------------------------------
    # Step 3: Shuffled Dataset B (row permutation invariance)
    # -------------------------------------------------------------
    print("[3] Setting shuffled Dataset B (multiset row order permutation)...")
    df_b_shuffled = df_b.sample(frac=1.0, random_state=42).reset_index(drop=True)
    service.set_data(df_b_shuffled)
    fp_b_shuffled = service._dataset_version
    print(f"    Shuffled B fingerprint: {fp_b_shuffled}")

    assert fp_b_shuffled == fp_b, "Multiset fingerprint must be invariant under row permutation!"

    res_b_shuffled = service.compute(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
    )
    assert res_b_shuffled["cache_hit"], "Identical multiset data must hit release cache!"
    assert res_b_shuffled["release_key"] == key_b, "Release key must match for identical multiset!"
    print("    [PASS] Row permutation invariance correctly verified.")

    # -------------------------------------------------------------
    # Step 4: Explicit label override is bound to content fingerprint (Audit v0.1.5 Finding C)
    # Caller supplies same human label "benchmark_v1" for two different datasets
    # -------------------------------------------------------------
    print("[4] Testing explicit dataset_version label override binding to content...")
    service.set_data(df_a, dataset_version="benchmark_v1")
    fp_label_a = service._dataset_version
    print(f"    Dataset A with label: {fp_label_a}")
    assert fp_label_a == f"benchmark_v1@{compute_dataset_fingerprint(df_a)}"

    service.set_data(df_b, dataset_version="benchmark_v1")
    fp_label_b = service._dataset_version
    print(f"    Dataset B with label: {fp_label_b}")
    assert fp_label_b == f"benchmark_v1@{compute_dataset_fingerprint(df_b)}"
    assert fp_label_a != fp_label_b, "Same human label on different content MUST yield different dataset versions!"
    print("    [PASS] Human label safely combined with multiset content fingerprint.")

    # -------------------------------------------------------------
    # Step 5: Synthetic dataset seed=0 preservation (Audit v0.1.5 Finding D)
    # -------------------------------------------------------------
    print("[5] Testing synthetic dataset generator seed=0 vs seed=42...")
    df_syn_0 = service.generate_synthetic_data(n_users=10, seed=0)
    ver_syn_0 = service._dataset_version
    print(f"    Synthetic seed=0 version: {ver_syn_0}")
    assert "synthetic_seed_0_" in ver_syn_0, "seed=0 must NOT be treated as falsy or replaced with default seed!"

    df_syn_42 = service.generate_synthetic_data(n_users=10, seed=42)
    ver_syn_42 = service._dataset_version
    print(f"    Synthetic seed=42 version: {ver_syn_42}")
    assert "synthetic_seed_42_" in ver_syn_42
    assert ver_syn_0 != ver_syn_42, "seed=0 and seed=42 must have distinct version identities!"
    print("    [PASS] Synthetic seed=0 correctly preserved and distinct.")

    # Cleanup test DB
    for ext in ["", "-wal", "-shm", ".jsonl"]:
        f = Path(str(test_db) + ext)
        if f.exists():
            f.unlink(missing_ok=True)

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED: Dataset Identity Invalidation Verified")
    print("=" * 70)


if __name__ == "__main__":
    main()
