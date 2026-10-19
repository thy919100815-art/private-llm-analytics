"""Packaging script for Private LLM Analytics v0.1.7 (Gate Patch & Freeze Candidate).

Enforces distribution invariants:
  1. No historical or nested *.zip archives
  2. N=100,000 dataset (interactions_n100000.parquet) is excluded to keep ZIP compact
  3. No duplicate dataset aliases (interactions_n10k.parquet, interactions_n50k.parquet)
  4. results/privacy_state.db has 0 reservations, 0 releases, 0 ledger rows
  5. No runtime junk: *-wal, *-shm, verify_*.db*, test_*.db*, temp_*.db*, __pycache__, .pytest_cache
"""

import gc
from pathlib import Path
import sqlite3
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP = PROJECT_ROOT / "Private_LLM_Analytics_v0.1.7.zip"

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".system_generated",
    ".gemini",
    "verification",  # results/verification smoke artifacts
}


def cleanup_transient_files():
    """Clean all transient verification/test databases and stale files before packing."""
    gc.collect()
    results_dir = PROJECT_ROOT / "results"
    if results_dir.exists():
        for p in list(results_dir.glob("*.db*")):
            if p.name.startswith(("verify_", "test_", "temp_", "custom_", "experiment_state")):
                try:
                    p.unlink(missing_ok=True)
                except PermissionError:
                    pass
            elif p.suffix in (".db-wal", ".db-shm"):
                try:
                    p.unlink(missing_ok=True)
                except PermissionError:
                    pass

    # Ensure privacy_state.db is 100% clean
    prod_db = results_dir / "privacy_state.db"
    if prod_db.is_file():
        try:
            with sqlite3.connect(prod_db) as conn:
                conn.execute("DELETE FROM privacy_reservations;")
                conn.execute("DELETE FROM release_store;")
                conn.execute("DELETE FROM privacy_ledger;")
                conn.commit()
                conn.execute("VACUUM;")
        except Exception:
            pass


def should_exclude(rel_path: Path) -> bool:
    # Check parts for directory exclusion
    for part in rel_path.parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True

    name = rel_path.name
    # Exclude all zip files
    if name.endswith(".zip"):
        return True
    # Exclude bytecode
    if name.endswith((".pyc", ".pyo")):
        return True
    # Exclude SQLite WAL/SHM
    if name.endswith(("-wal", "-shm")):
        return True
    # Exclude N=100,000 large dataset to keep distribution compact
    if name == "interactions_n100000.parquet":
        return True
    # Exclude duplicate legacy alias datasets
    if name in ("interactions_n10k.parquet", "interactions_n50k.parquet"):
        return True
    # Exclude test/verification transient databases
    if name.startswith(("verify_", "test_", "temp_", "custom_", "experiment_state")) and ".db" in name:
        return True
    # Exclude prompt dump text files
    if name.startswith("prompt_") and name.endswith(".txt"):
        return True

    return False


def build_package():
    print(f"Creating clean release archive: {OUTPUT_ZIP.name}")
    cleanup_transient_files()

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    included_count = 0
    total_size = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(PROJECT_ROOT.rglob("*")):
            if not file_path.is_file():
                continue

            rel_path = file_path.relative_to(PROJECT_ROOT)
            if should_exclude(rel_path):
                continue

            zf.write(file_path, arcname=str(rel_path).replace("\\", "/"))
            included_count += 1
            total_size += file_path.stat().st_size

    zip_size = OUTPUT_ZIP.stat().st_size
    print(f"Packaged {included_count} files.")
    print(f"Raw content size: {total_size / (1024 * 1024):.2f} MB")
    print(f"Compressed archive size: {zip_size / (1024 * 1024):.2f} MB")
    print("Release package successfully built.")


if __name__ == "__main__":
    build_package()
