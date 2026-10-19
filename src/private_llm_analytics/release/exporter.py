"""Exporter for publishing release outputs to JSON or CSV."""

import json
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd


def export_release_json(release_output: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Export release output as JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(release_output, f, indent=2)


def export_release_csv(release_output: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Export release statistics table to CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    results = release_output.get("results", {})
    # Look for tabular cell data
    cell_data = None
    for key in ("cell_shares", "dp_counts", "prevalence_rates", "cell_volumes"):
        if key in results and isinstance(results[key], dict):
            cell_data = results[key]
            value_col = key
            break

    if cell_data is not None:
        rows = []
        for cell_name, val in cell_data.items():
            parts = cell_name.split(":")
            provider = parts[0] if len(parts) > 1 else ""
            task = parts[1] if len(parts) > 1 else ""
            rows.append({
                "cell": cell_name,
                "provider": provider,
                "task": task,
                value_col: val,
            })
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    else:
        # Fallback to key-value export
        df = pd.DataFrame(list(results.items()), columns=["metric_field", "value"])
        df.to_csv(path, index=False)
