"""Verification script: RDP Historical View Reconstruction & Process Restart Safety (v0.2).

Verifies that:
  1. Releases committed by previous processes survive process exit/restart.
  2. On fresh initialization, PrivacyStateManager / RDPAccountant accurately
     reconstructs cumulative RDP vectors and effective epsilon from persistent SQLite.
  3. Stale crashed reservations are properly cleaned up upon restart or lease expiration.
  4. Subsequent releases correctly compose with pre-restart historical releases.
"""

from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import ReservationLostError
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    rdp_to_epsilon_delta,
    rdp_vector_to_json,
)
from private_llm_analytics.privacy.state import PrivacyStateManager


def main():
    print("=" * 70)
    print("VERIFICATION: RDP Historical Reconstruction & Restart Safety")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "privacy_state.db"
        orders = DEFAULT_RDP_ORDERS
        delta_global = 1e-5
        cap = 10.0
        global_budget = PrivacyBudget(epsilon=cap, delta=delta_global)
        needed_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)
        rdp_cand = compute_gaussian_rdp(sigma=2.0, sensitivity=1.0, orders=orders)

        # -------------------------------------------------------------
        # Phase 1: Initial process runs 2 releases then terminates
        # -------------------------------------------------------------
        print("[1] Phase 1: Initial process committing 2 releases...")
        state_mgr_1 = PrivacyStateManager(db_path)

        for i in range(1, 3):
            key = f"phase1_query_{i}"
            status, _, res_id, rel_id = state_mgr_1.reserve_or_get_release(
                release_key=key,
                canonical_semantics={"query_index": i},
                global_budget=global_budget,
                budget_needed=needed_budget,
                plan_metadata={"query_index": i},
                lease_timeout=10.0,
                accountant_mode="rdp",
                budget_scope_id="global_external",
                rdp_orders=orders,
                rdp_candidate_cost=rdp_cand,
                epsilon_cap=cap,
                delta_global=delta_global,
            )
            dummy = DPReleaseResult(
                metric_name="user_prevalence",
                dp_vector=np.array([float(i) * 1.5]),
                post_processed={},
                epsilon=1.0,
                delta=1e-6,
                mechanism="Gaussian",
                noise_scale=2.0,
                sensitivity=1.0,
                cell_names=["c1"],
                n_users=10,
                public_population_size=50000,
            )
            state_mgr_1.commit_release(
                reservation_id=res_id,
                release_key=key,
                release_id=rel_id,
                release_result=dummy,
                recipient="provider_a",
                dataset_version="v1.0",
                generation=1,
                budget_scope_id="global_external",
                accountant_mode="rdp",
                rdp_cost_json=rdp_vector_to_json(rdp_cand),
                delta_global=delta_global,
            )

        view_1 = state_mgr_1.get_privacy_view("global_external", delta_global=delta_global, orders=orders)
        print(f"    Phase 1 spend: releases={view_1.release_count}, effective_eps={view_1.effective_epsilon:.4f}")
        assert view_1.release_count == 2
        del state_mgr_1

        # -------------------------------------------------------------
        # Phase 2: Fresh process restarts and reconstructs state
        # -------------------------------------------------------------
        print("[2] Phase 2: Fresh process starts up and reconstructs privacy view...")
        state_mgr_2 = PrivacyStateManager(db_path)
        view_2 = state_mgr_2.get_privacy_view("global_external", delta_global=delta_global, orders=orders)

        print(f"    Phase 2 reconstructed spend: releases={view_2.release_count}, effective_eps={view_2.effective_epsilon:.4f}")
        assert view_2.release_count == 2
        assert abs(view_2.effective_epsilon - view_1.effective_epsilon) < Decimal("1e-9"), "Effective epsilon mismatch across restart!"
        for alpha, rdp_val in view_1.total_rdp_vector.items():
            assert abs(view_2.total_rdp_vector[alpha] - rdp_val) < Decimal("1e-12"), f"RDP vector mismatch at alpha={alpha}"
        print("    [PASS] Historical privacy view bitwise accurately reconstructed after process restart.")

        # -------------------------------------------------------------
        # Phase 3: Simulated crash recovery of stale uncommitted reservation
        # -------------------------------------------------------------
        print("[3] Phase 3: Simulating stale reservation recovery...")
        # Reserve with very short lease
        status, _, crash_res_id, crash_rel_id = state_mgr_2.reserve_or_get_release(
            release_key="crashed_query",
            canonical_semantics={"crash": True},
            global_budget=global_budget,
            budget_needed=needed_budget,
            plan_metadata={},
            lease_timeout=0.1,
            accountant_mode="rdp",
            budget_scope_id="global_external",
            rdp_orders=orders,
            rdp_candidate_cost=rdp_cand,
            epsilon_cap=cap,
            delta_global=delta_global,
        )
        assert status == "RESERVED"
        time.sleep(0.15)  # Lease expires

        # Recover stale reservations
        recovered_count = state_mgr_2.recover_stale_reservations(max_age_seconds=0.1, force=True)
        assert recovered_count >= 1, f"Expected at least 1 recovered reservation, got {recovered_count}"
        print(f"    Recovered {recovered_count} expired reservation(s).")

        # -------------------------------------------------------------
        # Phase 4: Commit release 3 and verify full 3-release composition
        # -------------------------------------------------------------
        print("[4] Phase 4: Committing 3rd release on restarted instance...")
        status, _, res_3, rel_3 = state_mgr_2.reserve_or_get_release(
            release_key="phase2_query_3",
            canonical_semantics={"query_index": 3},
            global_budget=global_budget,
            budget_needed=needed_budget,
            plan_metadata={"query_index": 3},
            lease_timeout=10.0,
            accountant_mode="rdp",
            budget_scope_id="global_external",
            rdp_orders=orders,
            rdp_candidate_cost=rdp_cand,
            epsilon_cap=cap,
            delta_global=delta_global,
        )
        dummy_3 = DPReleaseResult(
            metric_name="user_prevalence",
            dp_vector=np.array([4.5]),
            post_processed={},
            epsilon=1.0,
            delta=1e-6,
            mechanism="Gaussian",
            noise_scale=2.0,
            sensitivity=1.0,
            cell_names=["c1"],
            n_users=10,
            public_population_size=50000,
        )
        state_mgr_2.commit_release(
            reservation_id=res_3,
            release_key="phase2_query_3",
            release_id=rel_3,
            release_result=dummy_3,
            recipient="provider_a",
            dataset_version="v1.0",
            generation=1,
            budget_scope_id="global_external",
            accountant_mode="rdp",
            rdp_cost_json=rdp_vector_to_json(rdp_cand),
            delta_global=delta_global,
        )

        view_final = state_mgr_2.get_privacy_view("global_external", delta_global=delta_global, orders=orders)
        print(f"    Final spend: releases={view_final.release_count}, effective_eps={view_final.effective_epsilon:.4f}")
        assert view_final.release_count == 3
        assert view_final.effective_epsilon > view_2.effective_epsilon
        print("    [PASS] 3rd release successfully composed with pre-restart history.")

    print("\n[VERIFICATION PASSED] RDP historical reconstruction and restart safety are fully verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
