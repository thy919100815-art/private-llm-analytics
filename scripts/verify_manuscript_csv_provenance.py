"""Structured LaTeX Table & Joint-Key Manuscript-to-CSV Provenance Verifier for PVLDB 2027 Paper.

Replaces naive global substring matching (`val in tex`) with structured LaTeX table parsing:
1. Extracts each LaTeX table environment by its exact `\\label{...}` (`tab:curves`, `tab:planner_bounds`,
   `tab:paired_v3`, `tab:baselines_v3`, plus any real-data / banded-MF tables when present).
2. Parses every row and column cell-by-cell after stripping LaTeX formatting macros.
3. Matches each row using its composite primary key (e.g., `(workload, curve, comparison, metric)`
   in Table 4, `(method)` in Table 5, `(workload, curve, group)` in Table 3, `(curve, row_type)` in Table 2)
   and verifies base_mean, trt_mean, paired_diff, CI95, raw p_t, Holm p_holm, Wilcoxon p, seed counts,
   calibration scope, and units (`%` vs unitless RMSE) directly against the authoritative frozen CSVs
   and source code.
4. Supports passing `tex_text` directly for automated mutation testing (`test_p0_manuscript_provenance_mutation_failures`),
   guaranteeing that mutating a single MAPE diff, `p_holm`, baseline method label/scope, or
   service/task dimension deterministically fails the corresponding structured check.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from private_llm_analytics.sdg_mh.calendar_accounting import (  # noqa: E402
    ORIGINAL_PLANNER_CURVE,
    WEEK1_V2_CURVE,
)
from private_llm_analytics.sdg_mh.ledger import UnifiedPrivacyLedger  # noqa: E402


def _clean_latex_cell(cell: str) -> str:
    """Strip LaTeX styling macros while preserving numbers, signs, %, ±, and text identifiers."""
    s = cell.strip()
    s = re.sub(r"\\allowbreak\s*", "", s)
    s = re.sub(r"\\multirow\{\d+\}\{\*\}\{([^}]*)\}", r"\1", s)
    for _ in range(4):
        s = re.sub(r"\\(?:textbf|texttt|mathbf|mathrm|emph)\{([^{}]*)\}", r"\1", s)
    s = s.replace(r"\_", "_").replace(r"\%", "%").replace("$", "")
    s = s.replace(r"\pm", "±").replace("---", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_tabular_rows_by_label(tex: str, label: str) -> List[List[str]]:
    """Locate the table block containing `\\label{label}` and parse its tabular data rows into cells."""
    pattern = re.compile(
        r"\\begin\{table\*?\}(?:(?!\\end\{table\*?\}).)*?\\label\{"
        + re.escape(label)
        + r"\}(?:(?!\\end\{table\*?\}).)*?\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\}",
        re.DOTALL,
    )
    m = pattern.search(tex)
    if not m:
        raise AssertionError(f"Could not find tabular block for label '{label}'")
    body = m.group(1)
    # Remove comments
    body = re.sub(r"(?<!\\)%.*", "", body)
    raw_rows = body.split(r"\\")
    parsed_rows: List[List[str]] = []
    for raw in raw_rows:
        cleaned = re.sub(
            r"\\(?:toprule|midrule|bottomrule|cmidrule(?:\([lr]\))?\{\d+-\d+\}|addlinespace(?:\[[^\]]*\])?)",
            "",
            raw,
        ).strip()
        if not cleaned or "&" not in cleaned:
            continue
        cells = [_clean_latex_cell(c) for c in cleaned.split("&")]
        parsed_rows.append(cells)
    return parsed_rows


def verify_manuscript_provenance(
    tex_text: Optional[str] = None,
    write_report: bool = True,
) -> Dict[str, Any]:
    tex_path = ROOT / "docs" / "pvlbd_2027_paper" / "main.tex"
    tex = tex_text if tex_text is not None else tex_path.read_text(encoding="utf-8")

    checks: List[Dict[str, Any]] = []

    def record_check(name: str, passed: bool, source: str, expected: str, actual: str = "") -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "source": source,
                "expected": expected,
                "actual": actual,
            }
        )
        if not passed:
            raise AssertionError(
                f"Provenance check failed: {name} | expected={expected} | actual={actual} | source={source}"
            )

    # -------------------------------------------------------------------------
    # 1. Ban non-v3 / unverified experimental strings anywhere in manuscript
    # -------------------------------------------------------------------------
    banned_tokens = [
        "N=180",
        "180 users",
        "M=9 cells",
        "9 provider-by-task cells",
        "10 independent data seeds",
        "S=10",
        "20.05\\%",
        "39.95\\%",
        "35.61",
        "39.65",
        "41.61",
        "43.82",
        "50.09\\%",
        "79.96\\%",
    ]
    for tok in banned_tokens:
        record_check(
            name=f"ban_non_v3_token:{tok}",
            passed=(tok not in tex),
            source="docs/pvlbd_2027_paper/main.tex",
            expected="absent",
            actual="present" if tok in tex else "absent",
        )

    # -------------------------------------------------------------------------
    # 2. Structured System Dimension & v3 Experimental Setup Checks
    # -------------------------------------------------------------------------
    m_serv = re.search(r"\|\s*\\mathcal\{S\}\s*\|\s*=\s*m\s*=\s*(\d+)", tex)
    record_check(
        "sys_dim_services_m_eq_5",
        bool(m_serv and int(m_serv.group(1)) == 5),
        "src/private_llm_analytics/sdg_mh/rolling_state.py",
        "m=5",
        m_serv.group(0) if m_serv else "missing",
    )

    m_task = re.search(r"\|\s*\\mathcal\{A\}\s*\|\s*=\s*(\d+)", tex)
    record_check(
        "sys_dim_tasks_A_eq_20",
        bool(m_task and int(m_task.group(1)) == 20),
        "src/private_llm_analytics/sdg_mh/rolling_state.py",
        "|A|=20",
        m_task.group(0) if m_task else "missing",
    )

    m_sens = re.search(r"\\Delta_\{\\text\{distinct\}\}\^2\s*=\s*m\s*\+\s*2k\s*=\s*(\d+)\s*\+\s*2\((\d+)\)\s*=\s*(\d+)", tex)
    record_check(
        "sys_dim_distinct_sensitivity_5_plus_2x3_eq_11",
        bool(m_sens and (int(m_sens.group(1)), int(m_sens.group(2)), int(m_sens.group(3))) == (5, 3, 11)),
        "src/private_llm_analytics/sdg_mh/calendar_job.py",
        "5 + 2(3) = 11",
        m_sens.group(0) if m_sens else "missing",
    )

    m_setup = re.search(
        r"K\s*=\s*(\d+)\$?\s+independent\s+longitudinal\s+dataset\s+seeds.*?\\times\$?\s+\$?M\s*=\s*(\d+)\$?\s+MC\s+trials\s+\(\$(\d+)\$\s+runs",
        tex,
        re.DOTALL,
    )
    record_check(
        "exp_v3_K8_M5_40runs",
        bool(m_setup and (int(m_setup.group(1)), int(m_setup.group(2)), int(m_setup.group(3))) == (8, 5, 40)),
        "results/sdg_mh/long_horizon_420d_v3_final/AUDIT_V3_REPORT.md",
        "K=8, M=5, 40 runs",
        m_setup.group(0) if m_setup else "missing",
    )

    # -------------------------------------------------------------------------
    # 3. Structured Parsing of Table 2 (`tab:curves`)
    # -------------------------------------------------------------------------
    rows_t2 = _extract_tabular_rows_by_label(tex, "tab:curves")
    # Data rows are rows_t2[1..6] (3 rows for Original_Planner, 3 rows for Week1_v2)
    drift_summary = UnifiedPrivacyLedger.commitment_curve_drift_summary()
    curve_specs = [
        ("Original_Planner", ORIGINAL_PLANNER_CURVE, rows_t2[1:4], "3.2193 \\times 10^{-11}"),
        ("Week1_v2", WEEK1_V2_CURVE, rows_t2[4:7], "2.9122 \\times 10^{-11}"),
    ]
    for cname, cobj, crows, exp_drift_tex in curve_specs:
        eps_row, rho_row, slope_row = crows[0], crows[1], crows[2]
        record_check(f"tab2_{cname}_name", eps_row[0] == cname, "calendar_accounting.py", cname, eps_row[0])
        for idx, h in enumerate(cobj.horizons):
            exp_e = f"{cobj.epsilons[idx]:.4f}"
            exp_r = f"{cobj.rhos[idx]:.6f}"
            record_check(f"tab2_{cname}_eps_h{h}", eps_row[2 + idx] == exp_e, "calendar_accounting.py", exp_e, eps_row[2 + idx])
            record_check(f"tab2_{cname}_rho_h{h}", rho_row[2 + idx] == exp_r, "calendar_accounting.py", exp_r, rho_row[2 + idx])
        # Check secant slopes
        for idx in range(len(cobj.horizons) - 1):
            dh = cobj.horizons[idx + 1] - cobj.horizons[idx]
            dr = cobj.rhos[idx + 1] - cobj.rhos[idx]
            exp_s = f"{dr / dh:.7f}"
            act_s = slope_row[3 + idx].replace(r"\uparrow", "").strip()
            record_check(f"tab2_{cname}_slope_{idx}", act_s == exp_s, "calendar_accounting.py", exp_s, act_s)
        # Check affine tail beta_tail, a_tail, min_rho, drift
        record_check(f"tab2_{cname}_beta_tail", eps_row[7] == f"{cobj.beta_tail:.8f}", "calendar_accounting.py", f"{cobj.beta_tail:.8f}", eps_row[7])
        record_check(f"tab2_{cname}_a_tail", eps_row[8] == f"{cobj.a_tail:.8f}", "calendar_accounting.py", f"{cobj.a_tail:.8f}", eps_row[8])
        record_check(f"tab2_{cname}_min_rho", eps_row[9] == f"{cobj.rhos[0]:.8f}", "calendar_accounting.py", f"{cobj.rhos[0]:.8f}", eps_row[9])
        record_check(f"tab2_{cname}_drift", exp_drift_tex in eps_row[10], "ledger.py", exp_drift_tex, eps_row[10])

    # -------------------------------------------------------------------------
    # 4. Structured Parsing of Table 3 (`tab:planner_bounds`) by Joint Key (workload, curve, group)
    # -------------------------------------------------------------------------
    rows_t3 = _extract_tabular_rows_by_label(tex, "tab:planner_bounds")
    t3_map: Dict[Tuple[str, str, str], List[str]] = {}
    for r in rows_t3[1:]:
        if len(r) >= 11:
            t3_map[(r[0], r[1], r[2])] = r

    solver_csv = ROOT / "results" / "sdg_mh" / "long_horizon_420d_v3_final" / "solver_runtime_and_dual_bounds.csv"
    group_short = {"A_Baseline": "A_Base", "B_BudgetOnly": "B_Budg", "D_JointPlanning": "D_Joint"}
    wl_short = {"Default": "Default", "DistinctHeavy": "DistHeavy"}
    with solver_csv.open("r", encoding="utf-8") as f:
        for csv_row in csv.DictReader(f):
            key = (wl_short[csv_row["workload"]], csv_row["commitment_curve"], group_short[csv_row["group"]])
            record_check(f"tab3_row_exists_{key}", key in t3_map, str(solver_csv.relative_to(ROOT)), str(key), str(list(t3_map.keys())))
            t_row = t3_map[key]
            exp_phi = csv_row["phases"].replace(" ", "")
            record_check(f"tab3_{key}_phases", t_row[3].replace(" ", "") == exp_phi, str(solver_csv.relative_to(ROOT)), exp_phi, t_row[3])
            exp_time = f"{float(csv_row['solve_seconds']):.3f}"
            record_check(f"tab3_{key}_v3_time", exp_time in t_row[4], str(solver_csv.relative_to(ROOT)), exp_time, t_row[4])
            exp_p = f"{float(csv_row['primal_obj_certified']):.4f}"
            record_check(f"tab3_{key}_primal", t_row[5] == exp_p, str(solver_csv.relative_to(ROOT)), exp_p, t_row[5])
            if math.isfinite(float(csv_row["dual_lower_bound"])):
                exp_dlb = f"{float(csv_row['dual_lower_bound']):.4f}"
                record_check(f"tab3_{key}_dual_lb", t_row[6].startswith(exp_dlb), str(solver_csv.relative_to(ROOT)), exp_dlb, t_row[6])
            exp_d = f"{float(csv_row['daily_d']) * 1e4:.4f}"
            record_check(f"tab3_{key}_daily_d", t_row[7] == exp_d, str(solver_csv.relative_to(ROOT)), exp_d, t_row[7])
            exp_sadd = f"{float(csv_row['sigma_add_eff']):.1f}"
            record_check(f"tab3_{key}_sigma_add", t_row[9] == exp_sadd, str(solver_csv.relative_to(ROOT)), exp_sadd, t_row[9])

    # -------------------------------------------------------------------------
    # 5. Structured Parsing of Table 4 (`tab:paired_v3`) by Joint Key (workload/curve, contrast, metric)
    # -------------------------------------------------------------------------
    rows_t4 = _extract_tabular_rows_by_label(tex, "tab:paired_v3")
    t4_map: Dict[Tuple[str, str, str], List[str]] = {}
    for r in rows_t4[1:]:
        if len(r) >= 10:
            metric_tok = r[2].split(" ")[0].strip()
            t4_map[(r[0], r[1], metric_tok)] = r

    paired_csv = ROOT / "results" / "sdg_mh" / "long_horizon_420d_v3_final" / "audit_v3_paired_with_holm_wilcoxon.csv"
    with paired_csv.open("r", encoding="utf-8") as f:
        csv_paired = {
            (
                f"{wl_short[r['workload']]} / {r['commitment_curve']}",
                "D - B" if "B_BudgetOnly" in r["comparison"] else "D - A",
                r["metric"],
            ): r
            for r in csv.DictReader(f)
        }

    record_check("tab4_row_count", len(t4_map) == 22, str(paired_csv.relative_to(ROOT)), "22", str(len(t4_map)))
    for key, t_row in t4_map.items():
        record_check(f"tab4_key_in_csv_{key}", key in csv_paired, str(paired_csv.relative_to(ROOT)), str(key), "found" if key in csv_paired else "missing")
        c_row = csv_paired[key]
        met = key[2]
        is_mape = "mape" in met
        fmt = ".4f" if ("rmse" not in met or "drift_rmse" in met) else ".2f"
        unit = "%" if is_mape else ""

        exp_base = f"{format(float(c_row['base_mean']), fmt)}{unit}"
        exp_trt = f"{format(float(c_row['trt_mean']), fmt)}{unit}"
        diff_val = float(c_row["paired_diff_mean"])
        sign = "+" if diff_val >= 0 else ""
        exp_diff = f"{sign}{format(diff_val, fmt)}{unit}"
        exp_ci = f"± {format(float(c_row['paired_diff_ci95']), fmt)}{unit}"
        exp_pt = f"{float(c_row['paired_t_p_value']):.4f}"
        exp_holm = f"{float(c_row['paired_t_p_holm']):.4f}" if c_row["paired_t_p_holm"].strip() != "" else "diag"
        exp_wilc = f"{float(c_row['wilcoxon_p_value']):.4f}"
        exp_seeds = c_row["seeds_lower_error_count"].strip()

        record_check(f"tab4_{key}_base", t_row[3] == exp_base, str(paired_csv.relative_to(ROOT)), exp_base, t_row[3])
        record_check(f"tab4_{key}_trt", t_row[4] == exp_trt, str(paired_csv.relative_to(ROOT)), exp_trt, t_row[4])
        record_check(f"tab4_{key}_diff", t_row[5].startswith(exp_diff), str(paired_csv.relative_to(ROOT)), exp_diff, t_row[5])
        record_check(f"tab4_{key}_ci95", exp_ci in t_row[5], str(paired_csv.relative_to(ROOT)), exp_ci, t_row[5])
        record_check(f"tab4_{key}_raw_p", t_row[6] == exp_pt, str(paired_csv.relative_to(ROOT)), exp_pt, t_row[6])
        record_check(f"tab4_{key}_holm_p", t_row[7] == exp_holm, str(paired_csv.relative_to(ROOT)), exp_holm, t_row[7])
        record_check(f"tab4_{key}_wilcoxon_p", t_row[8] == exp_wilc, str(paired_csv.relative_to(ROOT)), exp_wilc, t_row[8])
        record_check(f"tab4_{key}_seeds", t_row[9].startswith(exp_seeds), str(paired_csv.relative_to(ROOT)), exp_seeds, t_row[9])

    # -------------------------------------------------------------------------
    # 6. Structured Parsing of Table 5 (`tab:baselines_v3`) by Method Label & Scope
    # -------------------------------------------------------------------------
    rows_t5 = _extract_tabular_rows_by_label(tex, "tab:baselines_v3")
    t5_map: Dict[str, List[str]] = {}
    for r in rows_t5[1:]:
        if len(r) >= 10:
            t5_map[r[0]] = r

    base_csv = ROOT / "results" / "sdg_mh" / "long_horizon_420d_v3_final" / "audit_v3_competitive_baselines_summary.csv"
    csv_base: Dict[Tuple[str, str], Dict[str, str]] = {}
    with base_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            csv_base[(r["method"], r["metric"])] = r

    # Also load Banded-MF v4 summary CSV if present
    banded_csv = ROOT / "results" / "sdg_mh" / "banded_mf_baseline_v4" / "banded_mf_competitive_summary.csv"
    if banded_csv.exists():
        with banded_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                csv_base[(r["method"], r["metric"])] = r

    method_label_map = {
        "SDG-MH (D_Joint)": ("SDG-MH (D_JointPlanning)", "Infinite", "Supported"),
        "Banded-MF (b=8, MH-RemEnv)": ("Banded-MF (b=8, MH-RemEnv)", "Finite Toeplitz", "N/A"),
        "Banded-MF (b=30, MH-RemEnv)": ("Banded-MF (b=30, MH-RemEnv)", "Finite Toeplitz", "N/A"),
        "Tree-2l (MH-RemEnv)": ("Tree-2l (MH-RemEnv)", "Finite Dyadic", "N/A"),
        "Tree-2l (Daily-d)": ("Tree-2l (Daily-d Parity)", "Finite Dyadic", "N/A"),
        "Tree-U (MH-RemEnv)": ("Tree-U (MH-RemEnv)", "Finite Dyadic", "N/A"),
        "W-BD-30 (MH-Calib)": ("W-BD-30 (MH-Calibrated)", "Rolling Finite", "N/A"),
    }

    col_metrics = [
        (2, "requests_1d_rmse"),
        (3, "requests_7d_rmse"),
        (4, "requests_30d_rmse"),
        (5, "requests_90d_rmse"),
        (6, "failures_1d_rmse"),
        (7, "failures_30d_rmse"),
        (8, "failures_90d_rmse"),
    ]
    for tex_label, (csv_method, exp_scope_prefix, exp_distinct_prefix) in method_label_map.items():
        record_check(f"tab5_method_present_{tex_label}", tex_label in t5_map, str(base_csv.relative_to(ROOT)), tex_label, str(list(t5_map.keys())))
        t_row = t5_map[tex_label]
        record_check(f"tab5_{tex_label}_scope", t_row[1].startswith(exp_scope_prefix), str(base_csv.relative_to(ROOT)), exp_scope_prefix, t_row[1])
        record_check(f"tab5_{tex_label}_distinct_col", t_row[9].startswith(exp_distinct_prefix), str(base_csv.relative_to(ROOT)), exp_distinct_prefix, t_row[9])
        for col_idx, m_name in col_metrics:
            c_row = csv_base[(csv_method, m_name)]
            exp_cell = f"{float(c_row['mean']):.1f}± {float(c_row['seed_level_ci95']):.1f}"
            act_cell = t_row[col_idx].replace(" ± ", "± ")
            record_check(f"tab5_{tex_label}_{m_name}", act_cell == exp_cell, str(base_csv.relative_to(ROOT)), exp_cell, act_cell)

    # Reject any unknown/unrecognized baseline rows in Table 5
    for k in t5_map:
        record_check(f"tab5_authorized_method_{k}", k in method_label_map, "Table 5 method whitelist", k, k)

    # -------------------------------------------------------------------------
    # 7. Verify Section 5.4 (C_r Clipping Ablation) against cr_clipping_ablation_420d.csv
    # -------------------------------------------------------------------------
    abl_csv = ROOT / "results" / "sdg_mh" / "long_horizon_420d" / "cr_clipping_ablation_420d.csv"
    with abl_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cr_int = int(float(r["C_r"]))
            rmse_str = f"{float(r['requests_total_rmse_mean']):.2f}"
            bias_str = f"{float(r['requests_clipping_bias_mean']):.2f}"
            record_check(f"ablation_Cr_{cr_int}_rmse", rmse_str in tex, str(abl_csv.relative_to(ROOT)), rmse_str)
            record_check(f"ablation_Cr_{cr_int}_bias", bias_str in tex, str(abl_csv.relative_to(ROOT)), bias_str)

    # -------------------------------------------------------------------------
    # 8. Structured Parsing of Table 6 (`tab:wildchat_v4`) against WildChat v4 CSVs
    # -------------------------------------------------------------------------
    wc_paired_csv = ROOT / "results" / "sdg_mh" / "wildchat_real_validation_v4" / "wildchat_k8_cohorts_paired_d_vs_b.csv"
    wc_audit_csv = ROOT / "results" / "sdg_mh" / "wildchat_real_validation_v4" / "wildchat_dataset_audit_summary.csv"
    rows_t6 = _extract_tabular_rows_by_label(tex, "tab:wildchat_v4")
    t6_map: Dict[Tuple[str, str, str], List[str]] = {}
    for r in rows_t6[1:]:
        if len(r) >= 10:
            m_clean = r[2].split(" ")[0].strip()
            t6_map[(r[0], r[1], m_clean)] = r

    with wc_paired_csv.open("r", encoding="utf-8") as f:
        csv_wc_paired = {
            (
                f"{wl_short[r['workload']]} / {r['curve']}",
                "D - B" if "B_BudgetOnly" in r["contrast"] else "D - A",
                r["metric_key"],
            ): r
            for r in csv.DictReader(f)
        }

    record_check("tab6_row_count", len(t6_map) == 13, str(wc_paired_csv.relative_to(ROOT)), "13", str(len(t6_map)))
    for key, t_row in t6_map.items():
        record_check(f"tab6_key_in_csv_{key}", key in csv_wc_paired, str(wc_paired_csv.relative_to(ROOT)), str(key), "found" if key in csv_wc_paired else "missing")
        c_row = csv_wc_paired[key]
        met = key[2]
        is_mape = "mape" in met
        unit = "%" if is_mape else ""
        exp_base = f"{float(c_row['base_mean']):.2f}{unit}"
        exp_trt = f"{float(c_row['trt_mean']):.2f}{unit}"
        diff_val = float(c_row["paired_diff"])
        sign = "+" if diff_val >= 0 else ""
        exp_diff = f"{sign}{diff_val:.2f}{unit}"
        ci_half = 0.5 * (float(c_row["ci95_high"]) - float(c_row["ci95_low"]))
        exp_ci = f"± {ci_half:.2f}{unit}"
        exp_wilc = f"{float(c_row['p_wilcoxon']):.4f}"
        exp_cohorts = f"{int(c_row['cohorts_d_lt_base'])}/8"

        record_check(f"tab6_{key}_base", t_row[3] == exp_base, str(wc_paired_csv.relative_to(ROOT)), exp_base, t_row[3])
        record_check(f"tab6_{key}_trt", t_row[4] == exp_trt, str(wc_paired_csv.relative_to(ROOT)), exp_trt, t_row[4])
        record_check(f"tab6_{key}_diff", t_row[5].startswith(exp_diff), str(wc_paired_csv.relative_to(ROOT)), exp_diff, t_row[5])
        record_check(f"tab6_{key}_ci95", exp_ci in t_row[5], str(wc_paired_csv.relative_to(ROOT)), exp_ci, t_row[5])
        record_check(f"tab6_{key}_wilcoxon", t_row[8] == exp_wilc, str(wc_paired_csv.relative_to(ROOT)), exp_wilc, t_row[8])
        record_check(f"tab6_{key}_cohorts", t_row[9].startswith(exp_cohorts), str(wc_paired_csv.relative_to(ROOT)), exp_cohorts, t_row[9])

    with wc_audit_csv.open("r", encoding="utf-8") as f:
        wc_aud = next(csv.DictReader(f))
        for field_key, fmt_fn in [
            ("mean_dedup_reduction_7d_pct", lambda v: f"{float(v):.2f}%"),
            ("mean_dedup_reduction_30d_pct", lambda v: f"{float(v):.2f}%"),
            ("mean_dedup_reduction_90d_pct", lambda v: f"{float(v):.2f}%"),
            ("mean_daily_req_clip_bias_pct", lambda v: f"{float(v):.2f}%"),
            ("mean_cat30d_k3_clip_bias_pct", lambda v: f"{float(v):.2f}%"),
        ]:
            val_str = fmt_fn(wc_aud[field_key]).replace("%", r"\%")
            record_check(f"wildchat_audit_{field_key}", val_str in tex, str(wc_audit_csv.relative_to(ROOT)), val_str)

    report = {
        "manuscript": "docs/pvlbd_2027_paper/main.tex",
        "verifier_mode": "STRUCTURED_LATEX_TABLE_JOINT_KEY_PARSER",
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c["passed"]),
        "failed_checks": sum(1 for c in checks if not c["passed"]),
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
    if write_report:
        out_path = ROOT / "results" / "sdg_mh" / "canonical_7_orbits_benchmark" / "manuscript_csv_provenance_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import math
    rep = verify_manuscript_provenance()
    print(f"Structured Manuscript-to-CSV Provenance Check: {rep['passed_checks']}/{rep['total_checks']} passed.")
