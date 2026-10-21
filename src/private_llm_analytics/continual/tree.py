"""Binary Continual Observation Tree and Interval Decomposition (v0.4).

Implements:
- Binary time tree topology over arbitrary epoch_days with next-power-of-two padding.
- Deterministic canonical dyadic interval decomposition (canonical_cover).
- Prefix and sliding-window interval covers.
- Deterministic canonical node identity hashing.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ContinualNodeInfo:
    """Metadata for a dyadic interval node in the binary time tree."""

    level: int
    node_index: int
    interval_start: int  # 1-based inclusive start day
    interval_end: int    # 1-based inclusive end day
    is_padding: bool     # True if entirely beyond real epoch_days
    contains_padding: bool  # True if partially or entirely beyond real epoch_days


def compute_continual_node_id(
    stream_fingerprint: str,
    epoch_id: str,
    level: int,
    node_index: int,
) -> str:
    """Compute canonical deterministic SHA-256 identity for a tree node (Section 10 & 24).

    Node identity depends strictly on (stream_fingerprint, epoch_id, level, node_index)
    and NEVER on recipient, scope, or query ID.
    """
    canonical_payload = {
        "stream_fingerprint": str(stream_fingerprint),
        "epoch_id": str(epoch_id),
        "level": int(level),
        "node_index": int(node_index),
    }
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class BinaryTimeTree:
    """Binary time tree over an epoch horizon for continual observation."""

    def __init__(self, epoch_days: int):
        if epoch_days <= 0:
            raise ValueError(f"epoch_days must be positive, got {epoch_days}")
        self.epoch_days = int(epoch_days)
        self.tree_size = 1 if self.epoch_days == 1 else 2 ** int(math.ceil(math.log2(self.epoch_days)))
        self.num_levels = int(math.log2(self.tree_size)) + 1

    def node_interval(self, level: int, node_index: int) -> Tuple[int, int]:
        """Return 1-based inclusive interval (start_day, end_day) for node (level, node_index)."""
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Level {level} out of range [0, {self.num_levels - 1}]")
        block_size = 2 ** level
        num_nodes_at_level = self.tree_size // block_size
        if node_index < 0 or node_index >= num_nodes_at_level:
            raise ValueError(
                f"Node index {node_index} out of range [0, {num_nodes_at_level - 1}] on level {level}"
            )
        start_day = node_index * block_size + 1
        end_day = (node_index + 1) * block_size
        return start_day, end_day

    def get_node(self, level: int, node_index: int) -> ContinualNodeInfo:
        """Get ContinualNodeInfo for node (level, node_index)."""
        start_day, end_day = self.node_interval(level, node_index)
        is_padding = start_day > self.epoch_days
        contains_padding = end_day > self.epoch_days
        return ContinualNodeInfo(
            level=level,
            node_index=node_index,
            interval_start=start_day,
            interval_end=end_day,
            is_padding=is_padding,
            contains_padding=contains_padding,
        )

    def canonical_cover(self, start_day: int, end_day: int) -> List[Tuple[int, int]]:
        """Decompose [start_day, end_day] into canonical minimal dyadic interval nodes.

        Args:
            start_day: 1-based inclusive start day.
            end_day: 1-based inclusive end day.

        Returns:
            List of (level, node_index) tuples representing pairwise disjoint dyadic nodes
            whose union is exactly [start_day, end_day], sorted chronologically.
        """
        if not (1 <= start_day <= end_day <= self.epoch_days):
            raise ValueError(
                f"Query interval [{start_day}, {end_day}] must satisfy 1 <= start_day <= end_day <= epoch_days ({self.epoch_days})"
            )

        result: List[Tuple[int, int]] = []

        def _traverse(level: int, node_idx: int):
            u, v = self.node_interval(level, node_idx)
            # Case 1: completely outside query range
            if v < start_day or u > end_day:
                return
            # Case 2: completely inside query range
            if u >= start_day and v <= end_day:
                result.append((level, node_idx))
                return
            # Case 3: partial overlap -> descend to children
            if level > 0:
                left_child = (level - 1, 2 * node_idx)
                right_child = (level - 1, 2 * node_idx + 1)
                _traverse(left_child[0], left_child[1])
                _traverse(right_child[0], right_child[1])

        # Start traversal from root (highest level)
        root_level = self.num_levels - 1
        _traverse(root_level, 0)
        return result

    def prefix_cover(self, end_day: int) -> List[Tuple[int, int]]:
        """Canonical cover for prefix query [1, end_day]."""
        return self.canonical_cover(1, end_day)

    def sliding_window_cover(self, end_day: int, window_days: int) -> List[Tuple[int, int]]:
        """Canonical cover for sliding-window query ending at end_day with duration window_days."""
        if window_days <= 0:
            raise ValueError(f"window_days must be positive, got {window_days}")
        start_day = max(1, end_day - window_days + 1)
        return self.canonical_cover(start_day, end_day)
