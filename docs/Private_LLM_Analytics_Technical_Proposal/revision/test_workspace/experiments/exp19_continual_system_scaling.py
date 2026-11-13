#!/usr/bin/env python3
"""Experiment 19: Long-Horizon Engineering and System Scaling (Section 67, 98).

Formal Evaluation:
Benchmarks system scaling metrics across long horizons T in {30, 90, 180, 365, 730}:
1. Tree topology: tree_size, num_levels, canonical_node_capacity
2. Persistent storage: SQLite database file size (KB)
3. Latency:
   - First-time node materialization latency (ms)
   - Cached interval query latency (ms)
Confirms query time is decoupled from total telemetry volume and scales with O(log T).
"""

import argparse
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.temporal_truth import generate_rate_consistent_temporal_truth
from private_llm_analytics.continual.models import ContinualNoiseConfig, ContinualStreamConfig
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.continual.state import ContinualStateManager
from private_llm_analytics.continual.tree import BinaryTimeTree


def run_exp19(smoke: bool = False, output_dir: Optional[Path] = None):
    print(f"=== Starting Exp19 (smoke={smoke}) ===")
    out_dir = output_dir or (Path("results/verification") if smoke else Path("results/tables"))
    fig_dir = Path("results/verification") if smoke else Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    horizons = [30, 90] if smoke else [30, 90, 180, 365, 730]
    N = 200 if smoke else 500

    roster = [f"user_{i}" for i in range(N)]
    rows: List[Dict[str, Any]] = []

    for T in horizons:
        tree = BinaryTimeTree(epoch_days=T)
        num_levels = tree.num_levels
        tree_size = tree.tree_size
        total_tree_nodes = 2 * tree_size - 1

        # Generate interactions for horizon T
        df = generate_rate_consistent_temporal_truth(
            n_users=N,
            horizon_days=T,
            reference_days=30,
            reference_mean=60.0,
            reference_std=20.0,
            seed=42 + T,
        )
        roster = sorted(df["user_id"].unique().tolist())

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / f"bench_{T}.db"
            state_mgr = ContinualStateManager(db_path=db_path)
            engine = ContinualQueryEngine(state_manager=state_mgr, composition_mode="global_external")

            cfg = ContinualStreamConfig(
                stream_id=f"stream_bench_{T}",
                epoch_id="epoch_001",
                epoch_start="2026-01-01",
                epoch_days=T,
                daily_l2_bound=10.0,
                epoch_group_l12_bound=30.0,
                noise=ContinualNoiseConfig(target_epoch_epsilon=2.0),
            )

            # 1. Benchmark 30-day interval query [1, 30] materialization latency
            t0 = time.perf_counter()
            res1 = engine.query_interval(
                stream_config=cfg,
                start_day=1,
                end_day=min(30, T),
                recipient="Benchmarker",
                interactions=df,
                public_user_roster=roster,
                finalized_up_to_day=T,
                current_wall_clock_day=T + 1,
            )
            t1 = time.perf_counter()
            mat_latency_ms = (t1 - t0) * 1000.0

            # 2. Benchmark repeated cached query (zero raw data access)
            query_latencies = []
            for _ in range(20):
                t_start = time.perf_counter()
                _ = engine.query_interval(
                    stream_config=cfg,
                    start_day=1,
                    end_day=min(30, T),
                    recipient="Benchmarker",
                    interactions=None,
                    public_user_roster=None,
                )
                t_end = time.perf_counter()
                query_latencies.append((t_end - t_start) * 1000.0)

            avg_query_latency_ms = float(np.mean(query_latencies))
            p95_query_latency_ms = float(np.percentile(query_latencies, 95))

            # 3. Database file size on disk
            db_size_kb = float(db_path.stat().st_size / 1024.0) if db_path.exists() else 0.0

            row = {
                "horizon_T": T,
                "tree_size": tree_size,
                "num_levels": num_levels,
                "total_tree_nodes_capacity": total_tree_nodes,
                "db_size_kb": round(db_size_kb, 2),
                "materialization_latency_ms": round(mat_latency_ms, 2),
                "cached_query_latency_ms_avg": round(avg_query_latency_ms, 3),
                "cached_query_latency_ms_p95": round(p95_query_latency_ms, 3),
            }
            rows.append(row)
            print(f"  T={T}: Levels={num_levels}, DB={db_size_kb:.1f}KB, MatLat={mat_latency_ms:.1f}ms, CachedQuery={avg_query_latency_ms:.3f}ms")

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "exp19_system_scaling.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(df_out["horizon_T"], df_out["cached_query_latency_ms_avg"], "o-", color="#0275D8", label="Cached Query Latency (ms)")
    ax2.plot(df_out["horizon_T"], df_out["db_size_kb"], "s--", color="#5CB85C", label="SQLite DB Size (KB)")

    ax1.set_xlabel("Horizon T (Days)")
    ax1.set_ylabel("Query Latency (ms)", color="#0275D8")
    ax2.set_ylabel("DB Size (KB)", color="#5CB85C")
    plt.title("Exp19: Long-Horizon System Scaling & Query Latency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(fig_dir / "exp19_system_scaling.png", dpi=150)
    plt.savefig(fig_dir / "exp19_system_scaling.pdf")
    plt.close()

    print("Exp19 completed successfully.")
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    out_path = Path(args.output_dir) if args.output_dir else None
    run_exp19(smoke=args.smoke, output_dir=out_path)
