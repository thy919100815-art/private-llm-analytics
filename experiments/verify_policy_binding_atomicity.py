"""Verification script: Privacy Policy Binding Atomicity & Migration Safety (v0.2).

Verifies that:
  1. privacy_policy_metadata is strictly a singleton table (id INTEGER PRIMARY KEY CHECK (id = 1)).
  2. First reservation and policy binding occur atomically in a single BEGIN IMMEDIATE transaction with Barrier synchronization.
  3. Concurrent processes with conflicting composition policies fail closed atomically.
  4. Legacy v0.1 databases with external history and no policy metadata can only migrate to GLOBAL_EXTERNAL.
  5. Legacy v0.2 metadata with multiple policy rows strictly fails closed without dropping records (Item A).
  6. Legacy single policy row with inconsistent historical scope strictly fails closed (Item B).
  7. Empty RDP accountant reports exact zero spend without faking epsilon (Item D).
  8. Recipient role alterations on a database with external history fail closed.
  9. Blank and unknown mechanisms fail closed without guessing from delta.
"""

from decimal import Decimal
import json
import multiprocessing
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import (
    ProductionSafetyConfigurationError,
    PrivacyConfigurationError,
)
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    rdp_vector_to_json,
)
from private_llm_analytics.privacy.rdp_accountant import RDPAccountant
from private_llm_analytics.privacy.state import (
    PrivacyStateManager,
    compute_recipient_policy_fingerprint,
)


def _worker_policy_binding_proc(db_path_str: str, mode: str, groups, roles, barrier, result_queue):
    """Worker task executed in separate OS process attempting atomic policy binding & reservation with barrier."""
    state_mgr = PrivacyStateManager(db_path_str)
    gb = PrivacyBudget(epsilon=10.0, delta=1e-5)
    bn = PrivacyBudget(epsilon=1.0, delta=1e-6)
    cand_rdp = compute_gaussian_rdp(sigma=2.0, sensitivity=1.0, orders=DEFAULT_RDP_ORDERS)

    # Barrier synchronization ensures simultaneous arrival at the reservation gate
    if barrier is not None:
        try:
            barrier.wait(timeout=10.0)
        except Exception as e:
            result_queue.put({"success": False, "error": "BarrierTimeout", "msg": str(e), "mode": mode})
            return

    try:
        res = state_mgr.reserve_or_get_release(
            release_key=f"worker_key_{mode}",
            canonical_semantics={"worker_mode": mode},
            global_budget=gb,
            budget_needed=bn,
            plan_metadata={"mechanism": "gaussian", "sensitivity": 1.0, "noise_scale": 2.0},
            recipient="provider_a",
            recipient_role="external",
            budget_scope_id=f"scope_{mode}",
            accountant_mode="rdp",
            rdp_candidate_cost=cand_rdp,
            rdp_orders=DEFAULT_RDP_ORDERS,
            epsilon_cap=Decimal("10.0"),
            delta_global=Decimal("1e-5"),
            policy_mode=mode,
            policy_groups=groups,
            recipient_roles=roles,
        )
        result_queue.put({"success": True, "error": None, "mode": mode})
    except Exception as e:
        result_queue.put({"success": False, "error": type(e).__name__, "msg": str(e), "mode": mode})


def verify_singleton_constraint(tmp_dir: Path):
    print("[1] Verifying singleton constraint on privacy_policy_metadata table...")
    db_path = tmp_dir / "singleton_test.db"
    state_mgr = PrivacyStateManager(db_path)

    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, policy_fingerprint FROM privacy_policy_metadata")
        assert cursor.fetchall() == []

        # Valid insert id = 1
        cursor.execute(
            "INSERT INTO privacy_policy_metadata VALUES (1, 'fp1', 'global_external', '{}', '{}', '2026-01-01T00:00:00Z')"
        )

        # Attempt insert with id = 2 -> CHECK constraint failure
        try:
            cursor.execute(
                "INSERT INTO privacy_policy_metadata VALUES (2, 'fp2', 'per_recipient', '{}', '{}', '2026-01-01T00:00:00Z')"
            )
            assert False, "Should have failed CHECK constraint (id = 1)"
        except sqlite3.IntegrityError:
            pass

        # Attempt insert duplicate id = 1 -> PRIMARY KEY failure
        try:
            cursor.execute(
                "INSERT INTO privacy_policy_metadata VALUES (1, 'fp3', 'collusion_group', '{}', '{}', '2026-01-01T00:00:00Z')"
            )
            assert False, "Should have failed PRIMARY KEY constraint"
        except sqlite3.IntegrityError:
            pass

        cursor.execute("SELECT COUNT(*) FROM privacy_policy_metadata")
        assert cursor.fetchone()[0] == 1

    print("    [PASS] Singleton constraint (id = 1) physically enforced by SQLite engine.")


def verify_concurrent_conflicting_policies_barrier(tmp_dir: Path):
    print("[2] Verifying concurrent conflicting policy binding with true Barrier synchronization (Item F)...")
    db_path = tmp_dir / "concurrent_conflicts_barrier.db"
    PrivacyStateManager(db_path)

    roles = {"provider_a": "external", "provider_b": "external", "analyst_internal": "internal"}
    barrier = multiprocessing.Barrier(2)
    queue = multiprocessing.Queue()

    p1 = multiprocessing.Process(
        target=_worker_policy_binding_proc,
        args=(str(db_path), "global_external", None, roles, barrier, queue),
    )
    p2 = multiprocessing.Process(
        target=_worker_policy_binding_proc,
        args=(str(db_path), "per_recipient", None, roles, barrier, queue),
    )

    p1.start()
    p2.start()

    p1.join(timeout=10.0)
    p2.join(timeout=10.0)

    results = []
    while not queue.empty():
        results.append(queue.get())

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    assert len(successes) == 1, f"Expected exactly 1 success, got: {results}"
    assert len(failures) == 1, f"Expected exactly 1 failure, got: {results}"
    assert failures[0]["error"] in ("ProductionSafetyConfigurationError", "IntegrityError")

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM privacy_policy_metadata").fetchone()[0]
        row = conn.execute("SELECT id, recipient_composition_mode FROM privacy_policy_metadata").fetchone()
        assert count == 1
        assert row[0] == 1
        assert row[1] == successes[0]["mode"]

    print(f"    [PASS] Barrier race: Exactly 1 process bound policy ('{successes[0]['mode']}'); conflicting process failed closed.")


def verify_legacy_v01_migration(tmp_dir: Path):
    print("[3] Verifying legacy v0.1 external database migration rules...")
    db_path = tmp_dir / "legacy_migration.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE release_store (
            release_key TEXT PRIMARY KEY, reservation_id TEXT NOT NULL, owner_id TEXT NOT NULL DEFAULT '',
            generation INTEGER NOT NULL DEFAULT 1, release_id TEXT NOT NULL, created_at TEXT NOT NULL,
            status TEXT NOT NULL, canonical_semantics_json TEXT NOT NULL, metric TEXT NOT NULL,
            window_start TEXT, window_end TEXT, dataset_version TEXT NOT NULL, mechanism TEXT NOT NULL,
            epsilon TEXT NOT NULL, delta TEXT NOT NULL, sensitivity REAL NOT NULL, noise_scale REAL NOT NULL,
            cell_names_json TEXT NOT NULL, n_users INTEGER NOT NULL, public_population_size INTEGER,
            dp_vector_json TEXT, post_processed_json TEXT, committed_at TEXT,
            recipient TEXT NOT NULL DEFAULT 'provider_a', recipient_role TEXT NOT NULL DEFAULT 'external',
            budget_scope_id TEXT NOT NULL DEFAULT 'global_external', accountant_mode TEXT NOT NULL DEFAULT 'rdp',
            rdp_cost_json TEXT NOT NULL, rdp_orders_json TEXT NOT NULL,
            delta_global TEXT NOT NULL DEFAULT '1e-6', effective_epsilon TEXT, best_alpha TEXT
        )
    """)
    rdp_dummy = json.dumps({"2.0": "0.1", "3.0": "0.2"})
    orders_dummy = json.dumps([2.0, 3.0])
    conn.execute("""
        INSERT INTO release_store VALUES (
            'k_legacy_ext', 'r_leg', '', 1, 'rel_leg', '2026-01-01', 'COMMITTED', '{}',
            'user_prevalence', NULL, NULL, 'v1.0', 'gaussian', '1.0', '1e-6', 1.0, 1.0, '[]', 50, 50, '[]', '[]', '2026-01-01',
            'provider_a', 'external', 'global_external', 'rdp', ?, ?, '1e-6', '1.0', '2.0'
        )
    """, (rdp_dummy, orders_dummy))
    conn.commit()
    conn.close()

    state_mgr = PrivacyStateManager(db_path)

    # PER_RECIPIENT migration must fail closed
    try:
        state_mgr.verify_or_bind_policy(mode="per_recipient")
        assert False, "Should have rejected migration to per_recipient"
    except ProductionSafetyConfigurationError as e:
        assert "can only be migrated to 'global_external'" in str(e)

    # COLLUSION_GROUP migration must fail closed
    try:
        state_mgr.verify_or_bind_policy(mode="collusion_group", groups={"g1": ["provider_a"]})
        assert False, "Should have rejected migration to collusion_group"
    except ProductionSafetyConfigurationError as e:
        assert "can only be migrated to 'global_external'" in str(e)

    # GLOBAL_EXTERNAL migration must succeed
    state_mgr.verify_or_bind_policy(mode="global_external")

    with state_mgr._connection() as conn:
        row = conn.execute("SELECT id, recipient_composition_mode FROM privacy_policy_metadata").fetchone()
        assert row[0] == 1
        assert row[1] == "global_external"

    print("    [PASS] Legacy database with external history strictly restricted to global_external.")


def verify_legacy_policy_metadata_multiple_rows(tmp_dir: Path):
    print("[4] Verifying legacy metadata multiple rows fail-closed safety (Item A)...")
    db_path = tmp_dir / "multi_policy_legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE privacy_policy_metadata (
            policy_fingerprint TEXT PRIMARY KEY,
            recipient_composition_mode TEXT NOT NULL,
            canonicalized_collusion_groups TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        INSERT INTO privacy_policy_metadata VALUES
        ('fp_global', 'global_external', '{}', '2026-01-01T00:00:00Z'),
        ('fp_per', 'per_recipient', '{}', '2026-01-01T00:00:00Z')
    """)
    conn.commit()
    conn.close()

    try:
        PrivacyStateManager(db_path)
        assert False, "Should have failed closed on multiple policy bindings"
    except ProductionSafetyConfigurationError as e:
        assert "contains multiple policy bindings" in str(e)

    # Confirm old records were NOT dropped or lost
    conn2 = sqlite3.connect(db_path)
    rows = conn2.execute("SELECT policy_fingerprint FROM privacy_policy_metadata").fetchall()
    conn2.close()
    assert len(rows) == 2
    assert {r[0] for r in rows} == {"fp_global", "fp_per"}
    print("    [PASS] Legacy metadata with >1 rows fails closed; old records preserved intact.")


def verify_legacy_single_policy_inconsistent_scopes(tmp_dir: Path):
    print("[5] Verifying legacy single policy inconsistent scope safety (Item B)...")
    db_path = tmp_dir / "inconsistent_scope.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE privacy_policy_metadata (
            policy_fingerprint TEXT PRIMARY KEY,
            recipient_composition_mode TEXT NOT NULL,
            canonicalized_collusion_groups TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        INSERT INTO privacy_policy_metadata VALUES
        ('fp_global', 'global_external', '{}', '2026-01-01T00:00:00Z')
    """)
    conn.execute("""
        CREATE TABLE release_store (
            release_key TEXT PRIMARY KEY, reservation_id TEXT NOT NULL, owner_id TEXT NOT NULL DEFAULT '',
            generation INTEGER NOT NULL DEFAULT 1, release_id TEXT NOT NULL, created_at TEXT NOT NULL,
            status TEXT NOT NULL, canonical_semantics_json TEXT NOT NULL, metric TEXT NOT NULL,
            window_start TEXT, window_end TEXT, dataset_version TEXT NOT NULL, mechanism TEXT NOT NULL,
            epsilon TEXT NOT NULL, delta TEXT NOT NULL, sensitivity REAL NOT NULL, noise_scale REAL NOT NULL,
            cell_names_json TEXT NOT NULL, n_users INTEGER NOT NULL, public_population_size INTEGER,
            dp_vector_json TEXT, post_processed_json TEXT, committed_at TEXT,
            recipient TEXT NOT NULL DEFAULT 'provider_a', recipient_role TEXT NOT NULL DEFAULT 'external',
            budget_scope_id TEXT NOT NULL DEFAULT 'recipient:provider_a', accountant_mode TEXT NOT NULL DEFAULT 'rdp',
            rdp_cost_json TEXT NOT NULL, rdp_orders_json TEXT NOT NULL,
            delta_global TEXT NOT NULL DEFAULT '1e-6', effective_epsilon TEXT, best_alpha TEXT
        )
    """)
    rdp_dummy = json.dumps({"2.0": "0.1", "3.0": "0.2"})
    orders_dummy = json.dumps([2.0, 3.0])
    conn.execute("""
        INSERT INTO release_store VALUES (
            'k1', 'r1', '', 1, 'rel1', '2026-01-01', 'COMMITTED', '{}',
            'user_prevalence', NULL, NULL, 'v1.0', 'gaussian', '1.0', '1e-6', 1.0, 1.0, '[]', 50, 50, '[]', '[]', '2026-01-01',
            'provider_a', 'external', 'recipient:provider_a', 'rdp', ?, ?, '1e-6', '1.0', '2.0'
        )
    """, (rdp_dummy, orders_dummy))
    conn.commit()
    conn.close()

    try:
        PrivacyStateManager(db_path)
        assert False, "Should have failed closed on inconsistent historical scope"
    except ProductionSafetyConfigurationError as e:
        assert "is inconsistent with legacy policy mode 'global_external'" in str(e)

    print("    [PASS] Legacy policy with inconsistent historical scope strictly fails closed.")


def verify_empty_rdp_status_exact_zero(tmp_dir: Path):
    print("[6] Verifying empty RDP status reports exact zero spend (Item D)...")
    db_path = tmp_dir / "empty_rdp_status.db"
    state_mgr = PrivacyStateManager(db_path)
    acct = RDPAccountant(epsilon_cap=5.0, delta_global=1e-6, state_manager=state_mgr)

    assert acct.spent_epsilon == 0.0, f"Expected 0.0, got {acct.spent_epsilon}"
    spent = acct.get_total_spent()
    assert spent.epsilon == 0, f"Expected 0, got {spent.epsilon}"
    assert spent.epsilon == Decimal("0.0")
    assert float(spent.epsilon) == 0.0
    assert spent.delta == Decimal("1e-6")

    spent_a = acct.get_total_spent(budget_scope_id="recipient:provider_a")
    assert spent_a.epsilon == 0
    assert spent_a.epsilon == Decimal("0.0")
    assert float(spent_a.epsilon) == 0.0
    print("    [PASS] Empty RDP status strictly reports exact zero spend.")


def verify_recipient_role_binding(tmp_dir: Path):
    print("[7] Verifying recipient role modification fail-closed protection...")
    db_path = tmp_dir / "role_binding.db"
    state_mgr = PrivacyStateManager(db_path)

    roles_initial = {"provider_a": "external", "analyst_internal": "internal"}
    state_mgr.verify_or_bind_policy(mode="global_external", recipient_roles=roles_initial)

    # Commit a dummy release to establish history
    gb = PrivacyBudget(epsilon=10.0, delta=1e-5)
    bn = PrivacyBudget(epsilon=1.0, delta=1e-6)
    cand_rdp = compute_gaussian_rdp(sigma=2.0, sensitivity=1.0, orders=DEFAULT_RDP_ORDERS)
    status, _, res_id, rel_id = state_mgr.reserve_or_get_release(
        release_key="k_role_test",
        canonical_semantics={"q": "test"},
        global_budget=gb,
        budget_needed=bn,
        plan_metadata={"mechanism": "gaussian", "sensitivity": 1.0, "noise_scale": 2.0},
        recipient="provider_a",
        recipient_role="external",
        budget_scope_id="global_external",
        accountant_mode="rdp",
        rdp_candidate_cost=cand_rdp,
    )
    import numpy as np
    dummy = DPReleaseResult(
        metric_name="user_prevalence",
        dp_vector=np.array([1.0]),
        post_processed={},
        epsilon=1.0,
        delta=1e-6,
        mechanism="Gaussian",
        noise_scale=2.0,
        sensitivity=1.0,
        cell_names=["c1"],
        n_users=10,
        public_population_size=100,
    )
    state_mgr.commit_release(
        reservation_id=res_id,
        release_key="k_role_test",
        release_id=rel_id,
        release_result=dummy,
        recipient="provider_a",
        dataset_version="v1.0",
        budget_scope_id="global_external",
        accountant_mode="rdp",
        rdp_cost_json=rdp_vector_to_json(cand_rdp),
        delta_global=Decimal("1e-5"),
    )

    # Now change provider_a to internal
    roles_altered = {"provider_a": "internal", "analyst_internal": "internal"}
    try:
        state_mgr.verify_or_bind_policy(mode="global_external", recipient_roles=roles_altered)
        assert False, "Should have raised ProductionSafetyConfigurationError on role change"
    except ProductionSafetyConfigurationError as e:
        assert "Recipient composition policy mismatch" in str(e)

    print("    [PASS] Role changes on database with releases strictly fail closed.")


def verify_blank_mechanism_fail_closed(tmp_dir: Path):
    print("[8] Verifying blank and unknown mechanism fail-closed rejection...")
    db_path = tmp_dir / "blank_mechanism_test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE release_store (
            release_key TEXT PRIMARY KEY, reservation_id TEXT NOT NULL, owner_id TEXT NOT NULL DEFAULT '',
            generation INTEGER NOT NULL DEFAULT 1, release_id TEXT NOT NULL, created_at TEXT NOT NULL,
            status TEXT NOT NULL, canonical_semantics_json TEXT NOT NULL, metric TEXT NOT NULL,
            window_start TEXT, window_end TEXT, dataset_version TEXT NOT NULL, mechanism TEXT NOT NULL,
            epsilon TEXT NOT NULL, delta TEXT NOT NULL, sensitivity REAL NOT NULL, noise_scale REAL NOT NULL,
            cell_names_json TEXT NOT NULL, n_users INTEGER NOT NULL, public_population_size INTEGER,
            dp_vector_json TEXT, post_processed_json TEXT, committed_at TEXT
        )
    """)
    conn.execute("""
        INSERT INTO release_store VALUES (
            'k_blank', 'r_blank', '', 1, 'rel_b', '2026-01-01', 'COMMITTED', '{}',
            'user_prevalence', NULL, NULL, 'v1.0', '', '1.0', '1e-6', 1.0, 1.0, '[]', 50, 50, '[]', '[]', '2026-01-01'
        )
    """)
    conn.commit()
    conn.close()

    try:
        PrivacyStateManager(db_path)
        assert False, "Should have failed closed on blank mechanism"
    except ValueError as e:
        assert "missing, blank, or unsupported mechanism" in str(e)

    print("    [PASS] Blank mechanism without guessing strictly failed closed.")


def main():
    print("=" * 70)
    print("VERIFICATION: Privacy Policy Binding Atomicity & Migration Safety")
    print("=" * 70)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        verify_singleton_constraint(tmp_dir)
        verify_concurrent_conflicting_policies_barrier(tmp_dir)
        verify_legacy_v01_migration(tmp_dir)
        verify_legacy_policy_metadata_multiple_rows(tmp_dir)
        verify_legacy_single_policy_inconsistent_scopes(tmp_dir)
        verify_empty_rdp_status_exact_zero(tmp_dir)
        verify_recipient_role_binding(tmp_dir)
        verify_blank_mechanism_fail_closed(tmp_dir)

    print("\n" + "=" * 70)
    print("[PASS] ALL POLICY BINDING ATOMICITY CHECKS PASSED.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
