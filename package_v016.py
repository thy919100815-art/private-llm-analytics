"""Packaging script for Private LLM Analytics v0.1.6 (Audit Fix E).

Excludes:
  - Any *.zip files (especially historical v0.1..v0.1.5 archives)
  - Python bytecode cache (__pycache__, *.pyc, *.pyo)
  - Pytest cache (.pytest_cache)
  - Git internals (.git, .gitignore)
  - SQLite temporary/runtime files (*-wal, *-shm, verify_*.db, test_*.db, custom_*.db)
  - Temporary folders / scratch files
"""

from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP = PROJECT_ROOT / "Private_LLM_Analytics_v0.1.6.zip"

EXCLUDE_PATTERNS = [
    "*.zip",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".git",
    "*-wal",
    "*-shm",
    "verify_*.db",
    "test_*.db",
    "custom_*.db",
    ".system_generated",
    ".gemini",
]

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".system_generated",
    ".gemini",
}


def should_exclude(rel_path: Path) -> bool:
    # Check parts for directory exclusion
    for part in rel_path.parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True

    name = rel_path.name
    # Exclude zip files
    if name.endswith(".zip"):
        return True
    # Exclude bytecode
    if name.endswith((".pyc", ".pyo")):
        return True
    # Exclude SQLite WAL/SHM
    if name.endswith(("-wal", "-shm")):
        return True
    # Exclude test/verification transient databases
    if name.startswith(("verify_", "test_", "custom_")) and name.endswith(".db"):
        return True

    return False


def build_package():
    print(f"Creating clean release archive: {OUTPUT_ZIP.name}")
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
