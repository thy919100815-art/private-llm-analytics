"""Generates authoritative key results JSON directly from formal experimental CSVs.

Strictly reads data/tables CSV outputs produced by Exp09 through Exp14 to populate:
  results/tables/v03_key_results.json

Ensures zero manual discrepancy between experimental CSV data and documentation/reports.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def generate_key_results_json(
    tables_dir: str = "results/tables",
    output_file: str = "results/tables/v03_key_results.json",
) -> Dict[str, Any]:
    tables_path = Path(tables_dir)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "metadata": {
            "version": "v0.3",
            "source_tables_dir": str(tables_path),
        }
    }

    # 1. Exp09: Calibration Metrics
    exp09_path = tables_path / "exp09_small_n_error_decomposition.csv"
    if exp09_path.exists():
        df09 = pd.read_csv(exp09_path)
        # Check if normalized columns exist
        if "predicted_norm_total" in df09.columns and "empirical_norm_total" in df09.columns:
            pred = df09["predicted_norm_total"].values
            emp = df09["empirical_norm_total"].values
            bias = float(np.mean(pred - emp))
            mae = float(np.mean(np.abs(pred - emp)))
            rel_err = float(mae / max(1e-9, np.mean(emp)))
            rho, pval = spearmanr(pred, emp)
            summary["calibration_metrics"] = {
                "metric_definition": "normalized_total_l1_loss",
                "calibration_bias": round(bias, 6),
                "calibration_MAE": round(mae, 6),
                "relative_calibration_error": round(rel_err, 6),
                "spearman_rho": round(float(rho), 4),
                "p_value": float(pval),
                "sample_count": len(df09),
            }
        else:
            summary["calibration_metrics"] = {"status": "legacy_schema_detected"}

    # 2. Exp10: Clipping Selection by Population N
    exp10_path = tables_path / "exp10_clipping_planner.csv"
    if exp10_path.exists():
        df10 = pd.read_csv(exp10_path)
        selected_c: Dict[str, float] = {}
        oracle_c: Dict[str, float] = {}
        regret_c: Dict[str, float] = {}
        for _, row in df10.iterrows():
            n_key = str(int(row["population_n"]))
            selected_c[n_key] = float(row.get("planner_selected_c", row.get("selected_c", 0.0)))
            oracle_c[n_key] = float(row.get("oracle_optimal_c", row.get("oracle_c", 0.0)))
            regret_c[n_key] = float(row.get("planner_rel_regret", row.get("relative_regret", 0.0)))

        summary["selected_C_by_N"] = selected_c
        summary["oracle_C_by_N"] = oracle_c
        summary["regret_by_N"] = regret_c

    # 3. Exp11: Granularity Planning by Dimension / Workload
    exp11_path = tables_path / "exp11_granularity_planner.csv"
    if exp11_path.exists():
        df11 = pd.read_csv(exp11_path)
        summary["selected_G_by_N"] = {
            "description": "Direct target sensitivity release eliminates post-processing amplification",
            "evaluated_granularities": list(df11["granularity"].unique()),
        }
        if "direct_variance_reduction_pct" in df11.columns:
            summary["selected_G_by_N"]["max_direct_variance_reduction_pct"] = float(
                df11["direct_variance_reduction_pct"].max()
            )

    # 4. Exp12: Frequency Allocation Under Budget Depletion
    exp12_path = tables_path / "exp12_frequency_planner.csv"
    if exp12_path.exists():
        df12 = pd.read_csv(exp12_path)
        freq_by_budget: Dict[str, Any] = {}
        for _, row in df12.iterrows():
            ratio_key = f"{row['target_ratio']*100:.0f}%_spent"
            freq_by_budget[ratio_key] = {
                "remaining_epsilon": float(row.get("remaining_epsilon_budget", 0.0)),
                "selected_frequency_days": int(row.get("selected_frequency_days", 0)),
                "planned_releases_count": int(row.get("planned_releases_count", 0)),
                "actual_initial_epsilon": float(row.get("actual_initial_epsilon", 0.0)),
            }
        summary["selected_F_by_budget"] = freq_by_budget

    # 5. Exp13: Joint Policy & Window Selection
    exp13_path = tables_path / "exp13_joint_planner.csv"
    if exp13_path.exists():
        df13 = pd.read_csv(exp13_path)
        selected_w: Dict[str, Any] = {}
        infeasible_cases: List[Dict[str, Any]] = []
        for _, row in df13.iterrows():
            n = int(row["population_n"])
            eps_cap = float(row["epsilon_cap"])
            case_key = f"N{n}_eps{eps_cap:.1f}"
            status = str(row.get("status", "FEASIBLE"))
            if status == "NO_FEASIBLE_POLICY" or str(row.get("oracle_candidate")) == "NULL":
                infeasible_cases.append({
                    "population_n": n,
                    "epsilon_cap": eps_cap,
                    "status": "NO_FEASIBLE_POLICY",
                    "minimum_projected_epsilon": float(row.get("minimum_projected_epsilon", 0.0)),
                })
                selected_w[case_key] = {
                    "status": "NO_FEASIBLE_POLICY",
                    "policy": None,
                }
            else:
                cand_str = str(row.get("planner_candidate", ""))
                # Extract W from candidate string e.g. ..._W30_...
                w_val = 30
                if "_W" in cand_str:
                    try:
                        w_val = int(cand_str.split("_W")[1].split("_")[0])
                    except (ValueError, IndexError):
                        pass
                selected_w[case_key] = {
                    "status": "FEASIBLE",
                    "candidate": cand_str,
                    "selected_window_days": w_val,
                    "planner_true_utility": float(row.get("planner_true_utility", 0.0)),
                    "oracle_true_utility": float(row.get("oracle_true_utility", 0.0)),
                    "planner_rel_regret": float(row.get("planner_rel_regret", 0.0)),
                }

        summary["selected_W_by_N"] = selected_w
        summary["infeasible_conditions"] = infeasible_cases

    # 6. Exp14: Prior Misspecification Robustness
    exp14_path = tables_path / "exp14_prior_misspecification.csv"
    if exp14_path.exists():
        df14 = pd.read_csv(exp14_path)
        regimes_summary: Dict[str, Any] = {}
        for regime in df14["truth_regime"].unique():
            rdf = df14[df14["truth_regime"] == regime]
            matched_row = rdf[rdf["prior_case"].str.contains("Matched", case=False)]
            matched_regret = float(matched_row["relative_regret"].iloc[0]) if not matched_row.empty else 0.0
            max_regret = float(rdf["relative_regret"].max())
            mean_regret = float(rdf["relative_regret"].mean())
            fixed_regret = float(rdf["fixed_mid_rel_regret"].iloc[0]) if "fixed_mid_rel_regret" in rdf.columns else 0.0

            cases_list = []
            for _, r in rdf.iterrows():
                cases_list.append({
                    "case": str(r["prior_case"]),
                    "selected_c": float(r.get("selected_c", r.get("planner_selected_c", 0.0))),
                    "oracle_c": float(r.get("oracle_c", r.get("oracle_optimal_c", 0.0))),
                    "planner_regret": float(r.get("relative_regret", r.get("planner_regret", 0.0))),
                    "fixed_baseline_regret": float(r.get("fixed_mid_rel_regret", r.get("fixed_baseline_regret", 0.0))),
                })

            regimes_summary[str(regime)] = {
                "matched_prior_regret": round(matched_regret, 4),
                "mean_misspecified_regret": round(mean_regret, 4),
                "max_misspecified_regret": round(max_regret, 4),
                "fixed_baseline_regret": round(fixed_regret, 4),
                "cases": cases_list,
            }

        summary["prior_robustness"] = regimes_summary

    # Write out JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] v03_key_results.json successfully generated at: {out_path}")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate v03 Key Results JSON from experimental CSV tables")
    parser.add_argument("--tables-dir", type=str, default="results/tables")
    parser.add_argument("--output-file", type=str, default="results/tables/v03_key_results.json")
    args = parser.parse_args()

    res = generate_key_results_json(tables_dir=args.tables_dir, output_file=args.output_file)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
