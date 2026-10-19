"""Verification script: Smoke-Mode Non-Destructive Execution Verification.

Verifies:
  1. All official research artifacts (results/tables/*, results/raw/*, results/figures/*)
     are byte-identical (same SHA-256 hash) before and after running 'run_all_experiments.py --smoke'.
  2. Smoke execution writes exclusively to results/verification/ (tables, raw, figures).
  3. No official research files are overwritten, modified, or truncated.
"""

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def get_file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect_official_hashes(results_dir: Path) -> dict:
    hashes = {}
    for sub in ["tables", "raw", "figures"]:
        folder = results_dir / sub
        if folder.is_dir():
            for f in sorted(folder.iterdir()):
                if f.is_file():
                    hashes[f.relative_to(results_dir)] = get_file_hash(f)
    return hashes


def main():
    print("=" * 70)
    print("VERIFICATION: Smoke-Mode Non-Destructive Artifact Safety")
    print("=" * 70)

    results_dir = Path("results")
    verification_dir = results_dir / "verification"

    # 1. Collect hashes of all official research artifacts
    print("[1] Recording SHA-256 hashes of all official research artifacts...")
    hashes_before = collect_official_hashes(results_dir)
    assert len(hashes_before) > 0, "No official artifacts found in results/ to protect!"
    print(f"    Protected {len(hashes_before)} official artifact files.")

    # 2. Reset smoke directory before running to ensure fresh generation
    if verification_dir.exists():
        shutil.rmtree(verification_dir, ignore_errors=True)

    # 3. Execute smoke run via subprocess to test actual entrypoint
    print("\n[2] Executing 'run_all_experiments.py --smoke'...")
    proc = subprocess.run(
        [sys.executable, "run_all_experiments.py", "--smoke"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("Smoke run FAILED!")
        print(proc.stdout)
        print(proc.stderr)
        sys.exit(1)

    print("    Smoke run completed successfully.")

    # 4. Verify all official research artifacts remain byte-identical
    print("\n[3] Verifying official research artifact byte preservation...")
    hashes_after = collect_official_hashes(results_dir)

    mismatches = []
    missing = []
    for rel_path, h_before in hashes_before.items():
        if rel_path not in hashes_after:
            missing.append(str(rel_path))
        elif hashes_after[rel_path] != h_before:
            mismatches.append(str(rel_path))

    assert not missing, f"Official artifacts missing after smoke test: {missing}"
    assert not mismatches, f"Official artifacts modified by smoke test: {mismatches}"
    print(f"    All {len(hashes_before)} official artifacts remain 100% byte-identical.")

    # 5. Verify smoke artifacts were generated inside results/verification/
    print("\n[4] Verifying smoke artifacts generated in results/verification/...")
    smoke_tables = list((verification_dir / "tables").glob("*.csv")) if (verification_dir / "tables").is_dir() else []
    smoke_raw = list((verification_dir / "raw").glob("*.csv")) if (verification_dir / "raw").is_dir() else []
    smoke_figures = list((verification_dir / "figures").glob("*.*")) if (verification_dir / "figures").is_dir() else []

    assert len(smoke_tables) >= 4, f"Expected at least 4 smoke tables, found {len(smoke_tables)}"
    assert len(smoke_raw) >= 4, f"Expected at least 4 smoke raw trial files, found {len(smoke_raw)}"
    assert len(smoke_figures) >= 4, f"Expected at least 4 smoke figures, found {len(smoke_figures)}"

    print(f"    Smoke tables generated:  {[f.name for f in smoke_tables]}")
    print(f"    Smoke raw trials generated: {[f.name for f in smoke_raw]}")
    print(f"    Smoke figures generated: {[f.name for f in smoke_figures]}")

    print("\n" + "=" * 70)
    print("Official artifacts unchanged: PASS")
    print("Smoke artifacts generated separately: PASS")
    print("PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
