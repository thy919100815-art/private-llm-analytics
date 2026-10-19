"""Cell indexing and coordinate grouping utilities."""

from typing import Dict, List, Tuple

DEFAULT_PROVIDERS = ["provider_a", "provider_b", "provider_c"]
DEFAULT_TASKS = ["technical", "document", "analysis"]


def build_cell_definitions(
    providers: List[str] = DEFAULT_PROVIDERS,
    tasks: List[str] = DEFAULT_TASKS,
) -> Tuple[List[str], Dict[Tuple[str, str], int]]:
    """Build standardized cell names and mapping dictionary.

    Returns:
        cell_names: List of 9 cell strings in format 'provider:task'.
        cell_map: Dict mapping (provider, task) to cell index (0..8).
    """
    cell_names: List[str] = []
    cell_map: Dict[Tuple[str, str], int] = {}
    idx = 0
    for p in providers:
        for t in tasks:
            name = f"{p}:{t}"
            cell_names.append(name)
            cell_map[(p, t)] = idx
            idx += 1
    return cell_names, cell_map
