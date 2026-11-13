"""Verification script: Exp02 Population Semantics & Reproduction Verification.

Verifies:
  1. Exp02 canonical dataset (data/interactions_n50000.parquet) has exactly 50,000 unique active users.
  2. Public population size configured for Exp02 is 50,000 (satisfying active_users <= public_population).
  3. run_all_experiments.py specifies n_users=50000 for Exp02.
  4. Non-destructive smoke run with n_trials=1, epsilons=(1.0,), outputting to results/verification/
     succeeds without overwriting official research results in results/tables/.
"""

import inspect
from pathlib import Path
import shutil
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp02_epsilon import run_epsilon_sweep
import run_all_experiments


def main():
    print("=" * 70)
    print("VERIFICATION: Exp02 N=50,000 Population & Reproduction (Gate Fix B)")
    print("=" * 70)

    canonical_path = Path("data/interactions_n50000.parquet")
    assert canonical_path.is_file(), f"Expected {canonical_path} to exist!"

    # 1. Check dataset active users
    df = pd.read_parquet(canonical_path)
    active_users = int(df["user_id"].nunique())
    print(f"exp02 input dataset active users: {active_users}")
    assert active_users == 50000, f"Expected 50000 active users, got {active_users}"

    # 2. Check exp02 default public population
    sig = inspect.signature(run_epsilon_sweep)
    default_n_users = sig.parameters["n_users"].default
    default_data_path = sig.parameters["data_path"].default
    print(f"configured public population: {default_n_users}")
    print(f"configured data_path: {default_data_path}")
    assert default_n_users == 50000, f"Expected default n_users=50000, got {default_n_users}"
    assert "n50000" in default_data_path, f"Expected n50000 in data_path, got {default_data_path}"

    # 3. Check run_all_experiments.py configuration for exp02
    src_code = inspect.getsource(run_all_experiments.run_all_experiments)
    assert "n_users=50000" in src_code, "run_all_experiments must explicitly set n_users=50000 for Exp02"
    assert "interactions_n50000.parquet" in src_code, "run_all_experiments must reference interactions_n50000.parquet"
    print("run_all_experiments exp02 n_users: 50000")

    # 4. Run rapid non-destructive smoke run (n_trials=1, 1 epsilon)
    verify_out = Path("results/verification")
    verify_out.mkdir(parents=True, exist_ok=True)

    print("Running rapid non-destructive smoke run with n_trials=1, epsilons=(1.0,)...")
    res_df = run_epsilon_sweep(
        n_users=50000,
        epsilons=(1.0,),
        delta=1e-6,
        n_trials=1,
        seed=42,
        data_path="data/interactions_n50000.parquet",
        output_dir=str(verify_out),
        save_artifacts=True,
    )

    assert res_df is not None
    assert len(res_df) > 0
    assert (verify_out / "tables" / "exp02_epsilon_summary.csv").is_file()
    print("Smoke run completed successfully. Verification artifacts written to results/verification/")

    # Clean up verification output
    try:
        shutil.rmtree(verify_out, ignore_errors=True)
    except Exception:
        pass

    # Confirm official research results remain intact with N=50,000
    official_table = Path("results/tables/exp02_epsilon_summary.csv")
    if official_table.is_file():
        off_df = pd.read_csv(official_table)
        assert off_df["n_users"].iloc[0] == 50000, f"Official table corrupted! Found n_users={off_df['n_users'].iloc[0]}"
        print("Official research results verified intact (n_users=50,000).")

    print("PASS")


if __name__ == "__main__":
    main()
