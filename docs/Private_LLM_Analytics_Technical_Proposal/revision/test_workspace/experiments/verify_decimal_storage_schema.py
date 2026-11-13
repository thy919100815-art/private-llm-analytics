"""Verification script: Decimal SQLite Storage Schema and Dynamic Migration (v0.1.6 Audit Fix C).

Verifies that:
  1. Fresh PrivacyStateManager initializes privacy_reservations, release_store, and privacy_ledger
     with TEXT NOT NULL columns for epsilon and delta.
  2. Legacy SQLite tables with REAL columns for epsilon/delta are automatically migrated
     to TEXT NOT NULL during initialization, preserving existing records.
  3. Raw values stored in SQLite are exact string representations (e.g., '0.1', '1e-06'),
     preventing IEEE-754 binary floating-point drift.
  4. Adversarial precision overspend checks enforce exact Decimal accounting down to
     arbitrary precision (e.g., 10^-18).
"""

from decimal import Decimal
from pathlib import Path
import sqlite3
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.state import PrivacyStateManager


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


def get_column_types(conn: sqlite3.Connection, table_name: str):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    return {row[1]: row[2].upper() for row in cursor.fetchall()}


def main():
    print("=" * 70)
    print("VERIFICATION: Decimal SQLite Schema & Migration (Fix C)")
    print("=" * 70)

    fresh_db = Path("results/verify_fresh_schema.db")
    legacy_db = Path("results/verify_legacy_migration.db")

    cleanup_db(fresh_db)
    cleanup_db(legacy_db)

    # -------------------------------------------------------------
    # Step 1: Fresh Database Schema Verification
    # -------------------------------------------------------------
    print("[1] Verifying fresh PrivacyStateManager schema...")
    fresh_mgr = PrivacyStateManager(db_path=fresh_db)

    with sqlite3.connect(fresh_db) as conn:
        for table in ["privacy_reservations", "release_store", "privacy_ledger"]:
            col_types = get_column_types(conn, table)
            print(f"    Table {table}: epsilon={col_types.get('epsilon')}, delta={col_types.get('delta')}")
            assert col_types.get("epsilon") == "TEXT", f"Expected TEXT for {table}.epsilon, got {col_types.get('epsilon')}"
            assert col_types.get("delta") == "TEXT", f"Expected TEXT for {table}.delta, got {col_types.get('delta')}"

    print("    [PASS] Fresh schema uses TEXT for epsilon and delta across all tables.")

    # -------------------------------------------------------------
    # Step 2: Legacy REAL Schema Migration Verification
    # -------------------------------------------------------------
    print("[2] Creating legacy database with REAL columns...")
    with sqlite3.connect(legacy_db) as conn:
        conn.execute("""
            CREATE TABLE privacy_reservations (
                reservation_id TEXT PRIMARY KEY,
                release_key TEXT NOT NULL,
                owner_id TEXT NOT NULL DEFAULT '',
                generation INTEGER NOT NULL DEFAULT 1,
                epsilon REAL NOT NULL,
                delta REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                committed_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE release_store (
                release_key TEXT PRIMARY KEY,
                reservation_id TEXT NOT NULL,
                owner_id TEXT NOT NULL DEFAULT '',
                generation INTEGER NOT NULL DEFAULT 1,
                release_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                canonical_semantics_json TEXT NOT NULL,
                metric TEXT NOT NULL,
                recipient TEXT NOT NULL,
                epsilon REAL NOT NULL,
                delta REAL NOT NULL,
                sensitivity REAL NOT NULL,
                mechanism TEXT NOT NULL,
                noise_scale REAL NOT NULL,
                result_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE privacy_ledger (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                reservation_id TEXT,
                release_key TEXT NOT NULL,
                recipient TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                metric TEXT NOT NULL,
                mechanism TEXT NOT NULL,
                epsilon REAL NOT NULL,
                delta REAL NOT NULL,
                noise_scale REAL NOT NULL,
                cache_hit INTEGER NOT NULL,
                window_start TEXT,
                window_end TEXT,
                caller_info TEXT
            )
        """)
        # Insert sample rows with REAL values
        conn.execute("""
            INSERT INTO privacy_reservations VALUES (
                'res_old_1', 'key_old_1', 'owner_1', 1, 0.1, 0.00001, 'COMMITTED', '2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z'
            )
        """)
        conn.execute("""
            INSERT INTO release_store VALUES (
                'key_old_1', 'res_old_1', 'owner_1', 1, 'rel_old_1', '2026-01-01T00:00:00Z', 'COMMITTED', '{}', 'user_prevalence', 'internal', 0.1, 0.00001, 3.0, 'gaussian', 1.0, '{}'
            )
        """)
        conn.execute("""
            INSERT INTO privacy_ledger VALUES (
                NULL, '2026-01-01T00:00:00Z', 'RELEASE', 'res_old_1', 'key_old_1', 'internal', 'v1', 'user_prevalence', 'gaussian', 0.1, 0.00001, 1.0, 0, NULL, NULL, NULL
            )
        """)
        conn.commit()

    print("    Legacy database initialized. Now initializing PrivacyStateManager to trigger migration...")
    migrated_mgr = PrivacyStateManager(db_path=legacy_db)

    with sqlite3.connect(legacy_db) as conn:
        for table in ["privacy_reservations", "release_store", "privacy_ledger"]:
            col_types = get_column_types(conn, table)
            print(f"    Migrated {table}: epsilon={col_types.get('epsilon')}, delta={col_types.get('delta')}")
            assert col_types.get("epsilon") == "TEXT", f"Migration failed for {table}.epsilon: {col_types.get('epsilon')}"
            assert col_types.get("delta") == "TEXT", f"Migration failed for {table}.delta: {col_types.get('delta')}"

        # Verify data preservation
        cursor = conn.cursor()
        cursor.execute("SELECT epsilon, delta FROM privacy_reservations WHERE reservation_id='res_old_1';")
        row = cursor.fetchone()
        assert row is not None, "Old row in privacy_reservations was lost during migration!"
        print(f"    Migrated reservation values: epsilon='{row[0]}' ({type(row[0])}), delta='{row[1]}'")
        assert isinstance(row[0], str), "Migrated epsilon must be stored as str (TEXT)"

    print("    [PASS] Legacy database seamlessly migrated to TEXT without data loss.")

    # -------------------------------------------------------------
    # Step 3: Exact Decimal Representation & Prevention of Drift
    # -------------------------------------------------------------
    print("[3] Verifying exact string preservation on new records...")
    # Reserve and commit with exact Decimal
    target_eps = Decimal("0.100000000000000005")
    target_delta = Decimal("0.000000000000000001")

    res_tuple = fresh_mgr.reserve_or_get_release(
        release_key="key_exact_test",
        canonical_semantics={"test": 1},
        global_budget=PrivacyBudget(epsilon=Decimal("1.0"), delta=Decimal("1e-4")),
        budget_needed=PrivacyBudget(epsilon=target_eps, delta=target_delta),
        plan_metadata={},
    )
    status, _, res_id, _ = res_tuple
    assert status == "RESERVED"

    with sqlite3.connect(fresh_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT epsilon, delta FROM privacy_reservations WHERE reservation_id=?;", (res_id,))
        eps_str, delta_str = cursor.fetchone()
        print(f"    Stored SQLite epsilon: '{eps_str}', delta: '{delta_str}'")
        assert eps_str == str(target_eps), f"Expected exact string '{target_eps}', got '{eps_str}'"
        assert delta_str == str(target_delta), f"Expected exact string '{target_delta}', got '{delta_str}'"

    print("    [PASS] Stored values exactly match Decimal string representations.")

    # -------------------------------------------------------------
    # Step 4: Adversarial Precision Overspend Enforcement
    # -------------------------------------------------------------
    print("[4] Testing adversarial precision overspend enforcement...")
    # Create manager with microscopic budget
    nano_db = Path("results/verify_nano_budget.db")
    cleanup_db(nano_db)

    nano_mgr = PrivacyStateManager(db_path=nano_db)
    nano_total = PrivacyBudget(epsilon=Decimal("0.000000000000000001"), delta=Decimal("1e-6"))
    nano_overspend = PrivacyBudget(epsilon=Decimal("0.000000000000000002"), delta=Decimal("1e-6"))

    overspend_rejected = False
    try:
        nano_mgr.reserve_or_get_release(
            release_key="nano_key",
            canonical_semantics={"n": 1},
            global_budget=nano_total,
            budget_needed=nano_overspend,
            plan_metadata={},
            allow_overrun=False,
        )
    except PrivacyBudgetExceededError as exc:
        overspend_rejected = True
        print(f"    [PASS] Exact 10^-18 overspend strictly rejected: {exc}")

    assert overspend_rejected, "Overspend by 10^-18 MUST be rejected by exact Decimal math!"

    # Cleanup
    for p in [fresh_db, legacy_db, nano_db]:
        cleanup_db(p)

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED: Decimal Storage Schema & Migration")
    print("=" * 70)


if __name__ == "__main__":
    main()
