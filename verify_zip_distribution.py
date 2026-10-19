"""Clean temporary directory verification of Private_LLM_Analytics_v0.1.7.zip.

Performs complete isolated verification:
  1. Inspects ZIP archive: 0 nested ZIPs, 0 cache junk, 0 WAL/SHM, 0 duplicate datasets,
     n100000 excluded.
  2. Extracts into fresh isolated temporary directory.
  3. Verifies machine-parseable data/checksums.sha256 bit-for-bit.
  4. Runs full pytest suite (159 tests) inside temp directory.
  5. Dynamically discovers and runs ALL experiments/verify_*.py (>= 14 scripts).
"""

import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

ZIP_PATH = Path("Private_LLM_Analytics_v0.1.7.zip").resolve()


def main():
    print("=" * 70)
    print("VERIFICATION: Fresh ZIP Archive Extraction & Execution Test (v0.1.7)")
    print("=" * 70)

    assert ZIP_PATH.exists(), f"ZIP archive not found at {ZIP_PATH}"
    print(f"Archive file: {ZIP_PATH.name} ({ZIP_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # 1. Inspect archive contents
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        namelist = zf.namelist()
        nested_zips = [f for f in namelist if f.endswith(".zip")]
        assert len(nested_zips) == 0, f"Found nested zips: {nested_zips}"

        duplicate_aliases = [f for f in namelist if "interactions_n10k.parquet" in f or "interactions_n50k.parquet" in f]
        assert len(duplicate_aliases) == 0, f"Found duplicate dataset aliases: {duplicate_aliases}"

        excluded_large_data = [f for f in namelist if "interactions_n100000.parquet" in f]
        assert len(excluded_large_data) == 0, f"interactions_n100000.parquet must be excluded from distribution archive!"

        runtime_junk = [
            f for f in namelist
            if f.endswith(("-wal", "-shm"))
            or "__pycache__" in f
            or ".pytest_cache" in f
            or (Path(f).name.startswith(("verify_", "test_", "temp_", "custom_")) and f.endswith(".db"))
        ]
        assert len(runtime_junk) == 0, f"Found runtime junk: {runtime_junk}"

        print(f"[PASS] Archive verified clean: {len(namelist)} files, 0 nested zips, 0 duplicate datasets, 0 cache junk.")

    # 2. Extract into temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="pla_v017_verify_"))
    print(f"Extracting to isolated temporary directory: {temp_dir}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(temp_dir)

    python_exe = sys.executable

    # 3. Verify checksums file in extracted directory
    print("\n--- Verifying Checksums in Extracted Directory ---")
    checksum_file = temp_dir / "data" / "checksums.sha256"
    assert checksum_file.is_file(), "data/checksums.sha256 missing from archive"
    lines = [line.strip() for line in checksum_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 6

    pattern = re.compile(r"^[0-9a-f]{64}  [a-zA-Z0-9_.-]+$")
    for line in lines:
        assert pattern.match(line), f"Invalid checksum format: '{line}'"
        expected_hash, fname = line.split("  ", 1)
        file_path = temp_dir / "data" / fname
        assert file_path.is_file(), f"File {fname} in checksums.sha256 not found in extracted data/"
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {fname}"
        print(f"  [OK] {fname}")
    print("[PASS] Checksums verification passed.")

    # 4. Run pytest inside temporary directory
    print("\n--- Running Pytest in Extracted Directory ---")
    pytest_proc = subprocess.run(
        [python_exe, "-m", "pytest", "tests", "-q"],
        cwd=str(temp_dir),
        capture_output=True,
        text=True,
    )
    print(pytest_proc.stdout)
    if pytest_proc.stderr:
        print(pytest_proc.stderr)
    assert pytest_proc.returncode == 0, f"Pytest failed with exit code {pytest_proc.returncode}"
    assert "passed" in pytest_proc.stdout
    print("[PASS] All unit and regression tests passed in extracted clean directory.")

    # 5. Dynamically discover all verify_*.py scripts
    verify_scripts = sorted(temp_dir.glob("experiments/verify_*.py"))
    assert len(verify_scripts) >= 16, f"Expected >= 16 verify scripts, found {len(verify_scripts)}"
    print(f"\n--- Running All {len(verify_scripts)} Verification Scripts in Extracted Directory ---")

    for script_path in verify_scripts:
        rel_script = script_path.relative_to(temp_dir)
        print(f"Running {rel_script}...")
        proc = subprocess.run(
            [python_exe, str(rel_script)],
            cwd=str(temp_dir),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"FAILED: {rel_script}")
            print(proc.stdout)
            print(proc.stderr)
            raise RuntimeError(f"Script {rel_script} failed with return code {proc.returncode}")
        print(f"  -> PASS: {rel_script.name}")

    print("\n" + "=" * 70)
    print(f"ALL {len(verify_scripts)} VERIFICATIONS COMPLETED SUCCESSFULLY IN CLEAN EXTRACTED ARCHIVE")
    print("=" * 70)


if __name__ == "__main__":
    main()
