"""Verification script: Exact Budget Accounting and Precision (Audit F3).

Verifies that:
  1. Global delta = 0.0 strictly rejects any positive requested delta (no fixed +1e-12 tolerance slack).
  2. Very small global epsilon (e.g. 1e-8) is strictly enforced (no fixed +1e-9 tolerance slack).
  3. Exact decimal composition maintains zero floating-point drift up to the budget cap.
  4. Once total spend equals the global budget, any additional non-zero request is rejected.
"""

from decimal import Decimal
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.state import PrivacyStateManager


def main():
    print("=" * 70)
    print("VERIFICATION: Exact Budget Accounting & Precision (Audit F3)")
    print("=" * 70)

    test_db = Path("results/verify_budget_precision.db")
    for ext in ["", "-wal", "-shm", ".jsonl"]:
        f = Path(str(test_db) + ext)
        if f.exists():
            f.unlink(missing_ok=True)

    state_mgr = PrivacyStateManager(test_db)

    # -------------------------------------------------------------
    # Test 1: Global delta = 0.0 rejects positive delta
    # -------------------------------------------------------------
    print("[1] Testing global delta = 0.0 rejects requested delta = 1e-13...")
    global_b1 = PrivacyBudget(epsilon=4.0, delta=0.0)
    req_b1 = PrivacyBudget(epsilon=1.0, delta=1e-13)

    rejected_delta = False
    try:
        state_mgr.reserve_or_get_release(
            release_key="test_delta_zero_key",
            canonical_semantics={"test": "delta_zero"},
            global_budget=global_b1,
            budget_needed=req_b1,
            plan_metadata={},
            allow_overrun=False,
        )
    except PrivacyBudgetExceededError as exc:
        rejected_delta = True
        print(f"    Correctly rejected: {exc}")

    assert rejected_delta, "Global delta=0.0 MUST reject requested delta=1e-13 (Audit F3)!"
    print("    [PASS] Zero-slack delta comparison strictly enforced.")

    # -------------------------------------------------------------
    # Test 2: Very small global epsilon without tolerance drift
    # -------------------------------------------------------------
    print("[2] Testing small global epsilon = 1e-8 without 1e-9 tolerance overspend...")
    global_b2 = PrivacyBudget(epsilon=1e-8, delta=0.0)

    # Step 2a: Request 6e-9 (within 1e-8)
    req_b2a = PrivacyBudget(epsilon=6e-9, delta=0.0)
    status, _, res_id1, rel_id1 = state_mgr.reserve_or_get_release(
        release_key="test_small_eps_1",
        canonical_semantics={"test": "small_eps_1"},
        global_budget=global_b2,
        budget_needed=req_b2a,
        plan_metadata={},
        allow_overrun=False,
    )
    assert status == "RESERVED"

    # Step 2b: Request second 6e-9 (sum = 1.2e-8 > 1.0e-8)
    # In v0.1.3 this was accepted because 1.2e-8 <= 1.0e-8 + 1e-9 was True!
    req_b2b = PrivacyBudget(epsilon=6e-9, delta=0.0)
    rejected_eps = False
    try:
        state_mgr.reserve_or_get_release(
            release_key="test_small_eps_2",
            canonical_semantics={"test": "small_eps_2"},
            global_budget=global_b2,
            budget_needed=req_b2b,
            plan_metadata={},
            allow_overrun=False,
        )
    except PrivacyBudgetExceededError as exc:
        rejected_eps = True
        print(f"    Correctly rejected: {exc}")

    assert rejected_eps, "Small epsilon overspend MUST be rejected without fixed tolerance slack!"
    print("    [PASS] Strict decimal inequality prevents tolerance-based overspend.")

    # Clean up test 1 & 2
    for ext in ["", "-wal", "-shm", ".jsonl"]:
        f = Path(str(test_db) + ext)
        if f.exists():
            f.unlink(missing_ok=True)

    # -------------------------------------------------------------
    # Test 3: Exact composition up to the global cap
    # -------------------------------------------------------------
    print("[3] Testing exact composition up to cap (4 x 1.0 = 4.0)...")
    state_mgr_2 = PrivacyStateManager(test_db)
    global_b3 = PrivacyBudget(epsilon=4.0, delta=4e-6)
    step_b = PrivacyBudget(epsilon=1.0, delta=1e-6)

    for i in range(4):
        st, _, r_id, rel_id = state_mgr_2.reserve_or_get_release(
            release_key=f"cap_step_{i}",
            canonical_semantics={"step": i},
            global_budget=global_b3,
            budget_needed=step_b,
            plan_metadata={},
            allow_overrun=False,
        )
        assert st == "RESERVED"

    spent_eps, spent_delta = state_mgr_2.get_total_reserved_or_spent()
    print(f"    Cumulative spend at cap: epsilon={spent_eps}, delta={spent_delta}")
    assert float(spent_eps) == 4.0
    assert abs(float(spent_delta) - 4e-6) < 1e-15

    # Request 5th release with minuscule epsilon 1e-12
    req_overflow = PrivacyBudget(epsilon=1e-12, delta=0.0)
    rejected_overflow = False
    try:
        state_mgr_2.reserve_or_get_release(
            release_key="cap_step_overflow",
            canonical_semantics={"overflow": True},
            global_budget=global_b3,
            budget_needed=req_overflow,
            plan_metadata={},
            allow_overrun=False,
        )
    except PrivacyBudgetExceededError:
        rejected_overflow = True

    assert rejected_overflow, "Exceeding cap even by 1e-12 MUST be rejected!"
    print("    [PASS] Zero slack at budget cap strictly verified.")

    # -------------------------------------------------------------
    # Test 4: Adversarial IEEE 754 Precision (Audit v0.1.5 Finding A)
    # 0.01 + 0.06 in float is 0.06999999999999999 < 0.07.
    # An additional 5e-18 would falsely pass under float summation!
    # Decimal summation correctly calculates 0.070000000000000005 > 0.07.
    # -------------------------------------------------------------
    print("[4] Testing adversarial IEEE 754 float precision (Finding A)...")
    for ext in ["", "-wal", "-shm", ".jsonl"]:
        f = Path(str(test_db) + ext)
        if f.exists():
            f.unlink(missing_ok=True)

    state_mgr_4 = PrivacyStateManager(test_db)
    global_b4 = PrivacyBudget(epsilon=0.07, delta=0.0)

    # Step 4a: reserve 0.01
    st1, _, _, _ = state_mgr_4.reserve_or_get_release(
        release_key="step_float_1",
        canonical_semantics={"step": 1},
        global_budget=global_b4,
        budget_needed=PrivacyBudget(epsilon=0.01, delta=0.0),
        plan_metadata={},
        allow_overrun=False,
    )
    assert st1 == "RESERVED"

    # Step 4b: reserve 0.06
    st2, _, _, _ = state_mgr_4.reserve_or_get_release(
        release_key="step_float_2",
        canonical_semantics={"step": 2},
        global_budget=global_b4,
        budget_needed=PrivacyBudget(epsilon=0.06, delta=0.0),
        plan_metadata={},
        allow_overrun=False,
    )
    assert st2 == "RESERVED"

    # Step 4c: reserve 5e-18 (exceeds 0.07 in exact Decimal arithmetic)
    rejected_float_drift = False
    try:
        state_mgr_4.reserve_or_get_release(
            release_key="step_float_3",
            canonical_semantics={"step": 3},
            global_budget=global_b4,
            budget_needed=PrivacyBudget(epsilon=5e-18, delta=0.0),
            plan_metadata={},
            allow_overrun=False,
        )
    except PrivacyBudgetExceededError:
        rejected_float_drift = True

    assert rejected_float_drift, "Adversarial 0.01 + 0.06 + 5e-18 <= 0.07 MUST be rejected by exact Decimal accounting!"
    print("    [PASS] Adversarial IEEE 754 precision drift prevented by exact Decimal accounting.")

    # Cleanup
    for ext in ["", "-wal", "-shm", ".jsonl"]:
        f = Path(str(test_db) + ext)
        if f.exists():
            f.unlink(missing_ok=True)

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED: Budget Precision & Zero Slack Verified")
    print("=" * 70)


if __name__ == "__main__":
    main()
