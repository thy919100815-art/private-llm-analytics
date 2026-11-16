"""Generate authoritative markdown tables directly from true_e2e_benchmark_results.csv.

Enforces zero manual drift between CSV authoritative data and engineering reports.
"""

from pathlib import Path
import pandas as pd


CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "calendar_planning" / "true_e2e_benchmark_results.csv"


def load_and_validate_csv():
    assert CSV_PATH.exists(), f"Benchmark CSV missing: {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 16, f"Expected exactly 16 benchmark rows, found {len(df)}"
    return df


def generate_16_row_full_table(df: pd.DataFrame) -> str:
    lines = [
        "| 工作负载 | 承诺曲线 | 组别 | 求解状态 | 相位 $\\vec\\phi$ | 规划 $d$ | 实际单日 $\\rho$ | Req 噪声 RMSE (理论 $\\sigma$) | Req 端到端 RMSE | Fail 噪声 RMSE (理论 $\\sigma$) | Fail 端到端 RMSE | 7d Distinct MAPE | 30d Distinct MAPE | 90d Distinct MAPE (N=1) | 规划器抽象目标 |",
        "|---|---|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in df.iterrows():
        wl = f"**{row['workload']}**"
        curve = str(row['commitment_curve'])
        grp = str(row['group'])
        status = f"**{row['status']}**" if row['status'] == "INFEASIBLE" else str(row['status'])
        phases = str(row['phases'])
        if grp == "D_JointPlanning" and phases in ("(0, 4, 0)", "(0, 3, 0)"):
            phases = f"**{phases}**"

        d_val = f"{row['daily_d']:.6f}" if pd.notna(row['daily_d']) else "—"
        actual_rho = f"**{float(row['actual_daily_rho_sum']):.6f}**" if pd.notna(row['actual_daily_rho_sum']) else "—"

        if row['status'] == "INFEASIBLE":
            req_noise = "—"
            req_total = "—"
            fail_noise = "—"
            fail_total = "—"
            dist_7 = "—"
            dist_30 = "—"
            dist_90 = "—"
            obj_str = "$\\infty$"
        else:
            req_noise = f"{row['requests_raw_noise_rmse']:.2f} ({row['requests_analytical_sigma']:.2f})"
            req_total = f"{row['requests_total_rmse']:.2f}"
            fail_noise = f"{row['failures_raw_noise_rmse']:.2f} ({row['failures_analytical_sigma']:.2f})"
            fail_total = f"{row['failures_total_rmse']:.2f}"
            dist_7 = f"{row['distinct_7d_mape_pct']:.2f}%"
            dist_30 = f"{row['distinct_30d_mape_pct']:.2f}%"
            dist_90 = f"{row['distinct_90d_mape_pct']:.2f}%"

            obj_val = float(row['planner_abstract_objective'])
            obj_str = f"{obj_val:.2f}"
            if grp in ("B_BudgetOnly", "D_JointPlanning"):
                if grp == "D_JointPlanning" and obj_val < 1000 and "DistinctHeavy" in wl:
                    obj_str = f"**{obj_val:.2f}**"
                else:
                    obj_str = f"**{obj_val:.2f}**"

        lines.append(
            f"| {wl} | {curve} | {grp} | {status} | {phases} | {d_val} | {actual_rho} | "
            f"{req_noise} | {req_total} | {fail_noise} | {fail_total} | {dist_7} | {dist_30} | {dist_90} | {obj_str} |"
        )

    return "\n".join(lines)


def generate_statistical_separation_table(df: pd.DataFrame) -> str:
    lines = [
        "| 工作负载 | 曲线 | 组别 | 指标项 | 理论解析 $\\sigma$ | 纯机制噪声 RMSE | 裁剪偏差 (Bias) | 后处理重构偏差 | 最终端到端 RMSE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in df.iterrows():
        if row['status'] == "INFEASIBLE":
            continue
        wl = str(row['workload'])
        curve = str(row['commitment_curve'])
        grp = str(row['group'])

        # Requests
        lines.append(
            f"| {wl} | {curve} | {grp} | Requests | {row['requests_analytical_sigma']:.2f} | "
            f"{row['requests_raw_noise_rmse']:.2f} | {row['requests_clipping_bias']:.2f} | "
            f"{row['requests_postproc_bias']:.2f} | {row['requests_total_rmse']:.2f} |"
        )
        # Failures
        lines.append(
            f"| {wl} | {curve} | {grp} | Failures | {row['failures_analytical_sigma']:.2f} | "
            f"{row['failures_raw_noise_rmse']:.2f} | {row['failures_clipping_bias']:.2f} | "
            f"{row['failures_postproc_bias']:.2f} | {row['failures_total_rmse']:.2f} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    df = load_and_validate_csv()
    print("=== AUTHORITATIVE 16-ROW TABLE ===")
    print(generate_16_row_full_table(df))
    print("\n=== STATISTICAL SEPARATION TABLE ===")
    print(generate_statistical_separation_table(df))
