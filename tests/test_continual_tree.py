"""Unit tests for binary time tree and canonical interval decomposition (Section 17, 18)."""

import math
import pytest

from private_llm_analytics.continual.tree import (
    BinaryTimeTree,
    compute_continual_node_id,
)


@pytest.mark.parametrize("T", [1, 2, 3, 5, 8, 16, 31, 32])
def test_binary_tree_interval_cover_exhaustive(T: int):
    """Exhaustive interval decomposition test (Section 18).

    For every 1 <= l <= r <= T:
      1. union(nodes) == [l, r]
      2. intersection(nodes_i, nodes_j) == empty
      3. node_count <= 2 * ceil(log2(T)) + 2
      4. nodes sorted chronologically
    """
    tree = BinaryTimeTree(epoch_days=T)
    max_expected_nodes = 2 * int(math.ceil(math.log2(max(2, T)))) + 2

    for l in range(1, T + 1):
        for r in range(l, T + 1):
            cover = tree.canonical_cover(l, r)
            assert len(cover) > 0, f"Empty cover for [{l}, {r}]"
            assert len(cover) <= max_expected_nodes, (
                f"Cover size {len(cover)} exceeds bound {max_expected_nodes} for [{l}, {r}] in T={T}"
            )

            # Extract intervals
            intervals = [tree.node_interval(level, node_idx) for level, node_idx in cover]

            # 1. Check pairwise disjointness
            for i in range(len(intervals) - 1):
                cur_end = intervals[i][1]
                next_start = intervals[i + 1][0]
                assert cur_end < next_start, f"Intervals overlap or unsorted: {intervals[i]} vs {intervals[i+1]}"
                # Also adjacent since they cover the contiguous range
                assert cur_end + 1 == next_start, f"Gap between intervals: {intervals[i]} and {intervals[i+1]}"

            # 2. Check union exact equality
            assert intervals[0][0] == l, f"Start day {intervals[0][0]} != {l}"
            assert intervals[-1][1] == r, f"End day {intervals[-1][1]} != {r}"

            # Check exact set of points
            covered_points = set()
            for u, v in intervals:
                covered_points.update(range(u, v + 1))
            assert covered_points == set(range(l, r + 1))


def test_prefix_cover():
    """Verify prefix covers match [1, t] exactly."""
    T = 90
    tree = BinaryTimeTree(epoch_days=T)

    for t in [1, 2, 7, 14, 30, 45, 60, 89, 90]:
        prefix_nodes = tree.prefix_cover(t)
        canonical_nodes = tree.canonical_cover(1, t)
        assert prefix_nodes == canonical_nodes
        # Check interval
        intervals = [tree.node_interval(lvl, idx) for lvl, idx in prefix_nodes]
        assert intervals[0][0] == 1
        assert intervals[-1][1] == t


def test_sliding_window_cover():
    """Verify sliding-window covers for 7, 30, 90 day windows."""
    T = 180
    tree = BinaryTimeTree(epoch_days=T)

    for end_day in [10, 30, 60, 90, 120, 180]:
        for window in [7, 14, 30, 90]:
            sw_nodes = tree.sliding_window_cover(end_day, window)
            expected_start = max(1, end_day - window + 1)
            canonical_nodes = tree.canonical_cover(expected_start, end_day)
            assert sw_nodes == canonical_nodes

            intervals = [tree.node_interval(lvl, idx) for lvl, idx in sw_nodes]
            assert intervals[0][0] == expected_start
            assert intervals[-1][1] == end_day


def test_deterministic_node_id():
    """Verify compute_continual_node_id is deterministic and collision-free across components."""
    id1 = compute_continual_node_id("stream_fp_abc", "epoch_1", level=2, node_index=5)
    id2 = compute_continual_node_id("stream_fp_abc", "epoch_1", level=2, node_index=5)
    assert id1 == id2
    assert len(id1) == 64  # sha256 hex string

    # Change level
    id_diff_level = compute_continual_node_id("stream_fp_abc", "epoch_1", level=3, node_index=5)
    assert id_diff_level != id1

    # Change node_index
    id_diff_node = compute_continual_node_id("stream_fp_abc", "epoch_1", level=2, node_index=6)
    assert id_diff_node != id1

    # Change epoch
    id_diff_epoch = compute_continual_node_id("stream_fp_abc", "epoch_2", level=2, node_index=5)
    assert id_diff_epoch != id1

    # Change stream
    id_diff_stream = compute_continual_node_id("stream_fp_xyz", "epoch_1", level=2, node_index=5)
    assert id_diff_stream != id1


@pytest.mark.parametrize("epoch_days,expected_size,expected_levels", [
    (1, 1, 1),
    (7, 8, 4),
    (30, 32, 6),
    (60, 64, 7),
    (90, 128, 8),
    (180, 256, 9),
    (365, 512, 10),
])
def test_arbitrary_epoch_days(epoch_days: int, expected_size: int, expected_levels: int):
    """Verify arbitrary epoch_days produces correct next_power_of_two and level count."""
    tree = BinaryTimeTree(epoch_days=epoch_days)
    assert tree.tree_size == expected_size
    assert tree.num_levels == expected_levels

    # Root node interval
    root_interval = tree.node_interval(tree.num_levels - 1, 0)
    assert root_interval == (1, expected_size)


def test_padding_node_identification():
    """Verify padding node detection (Section 9)."""
    epoch_days = 90
    tree = BinaryTimeTree(epoch_days=epoch_days)
    # tree_size is 128

    # Leaf day 90: real
    node_90 = tree.get_node(0, 89)
    assert not node_90.is_padding
    assert not node_90.contains_padding

    # Leaf day 91: padding
    node_91 = tree.get_node(0, 90)
    assert node_91.is_padding
    assert node_91.contains_padding

    # Root node: interval [1, 128] -> contains padding but start_day is real
    root_node = tree.get_node(tree.num_levels - 1, 0)
    assert not root_node.is_padding
    assert root_node.contains_padding


def test_invalid_interval_bounds():
    """Verify invalid interval boundaries raise ValueError."""
    tree = BinaryTimeTree(epoch_days=30)

    with pytest.raises(ValueError):
        tree.canonical_cover(0, 10)  # start_day < 1

    with pytest.raises(ValueError):
        tree.canonical_cover(1, 31)  # end_day > epoch_days

    with pytest.raises(ValueError):
        tree.canonical_cover(15, 10) # start_day > end_day
