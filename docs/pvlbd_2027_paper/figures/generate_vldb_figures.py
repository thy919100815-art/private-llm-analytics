"""Generate crisp, publication-grade vector PDF figures for the PVLDB 2027 paper.

Reads exclusively from the verified experimental CSVs without modifying any data.
Produces three 2-panel full-width (figure*) vector PDF figures with >=9.5pt
readable typography and explicit percentage-point (pp) units on MAPE diffs:
  1. fig_privacy_envelope.pdf
  2. fig_joint_planning_eval.pdf
  3. fig_baselines_and_clipping.pdf
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from private_llm_analytics.sdg_mh.calendar_accounting import (  # noqa: E402
    ORIGINAL_PLANNER_CURVE,
    WEEK1_V2_CURVE,
)

OUT_DIR = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.5,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 8.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_NAVY = "#1B4F72"
C_TEAL = "#117A65"
C_AMBER = "#B9770E"
C_CORAL = "#A93226"
C_SLATE = "#5D6D7E"
C_PURPLE = "#6C3483"


def make_fig_privacy_envelope() -> None:
    df = pd.read_csv(ROOT / "results" / "sdg_mh" / "long_horizon_420d" / "long_horizon_420d_cross_period_ledger.csv")
    sub = df[
        (df["workload"] == "DistinctHeavy")
        & (df["commitment_curve"] == "Week1_v2")
        & (df["group"] == "D_JointPlanning")
        & (df["checkpoint_day"] == 420)
    ].groupby("horizon_h", as_index=False)[
        ["observed_ledger_max_rho", "theoretical_L_max_575", "commitment_rho_bar"]
    ].mean().sort_values("horizon_h")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.35), constrained_layout=True)

    hs = sub["horizon_h"].to_numpy()
    ax1.plot(hs, sub["commitment_rho_bar"], color=C_NAVY, lw=2.0, marker="o", ms=4.0, label=r"Commitment $\bar{\rho}(h)$")
    ax1.plot(hs, sub["theoretical_L_max_575"], color=C_TEAL, lw=1.7, ls="--", marker="s", ms=3.8, label=r"575d Bound $L_{\max}(h)$")
    ax1.plot(hs, sub["observed_ledger_max_rho"], color=C_AMBER, lw=1.7, ls=":", marker="^", ms=4.2, label=r"Observed Ledger $\rho_{\mathrm{ledger}}(h)$")
    ax1.axvline(365, color=C_SLATE, lw=1.0, ls="-.")
    ax1.text(358, 0.05, "Affine tail\n($h \\geq 365$)", fontsize=8.0, color=C_SLATE, ha="right")
    ax1.set_xlabel("Horizon $h$ (days)")
    ax1.set_ylabel(r"Cumulative zCDP ($\rho$)")
    ax1.set_title("(a) 420-Day Ledger vs. Commitment")
    ax1.legend(frameon=False, loc="upper left")

    labels = ["[1, 7]", "[7, 30]", "[30, 90]", "[90, 365]"]
    x = np.arange(len(labels))
    w = 0.34
    s_std = [s[2] * 1e4 for s in WEEK1_V2_CURVE.knot_rho_slopes()]
    s_bal = [s[2] * 1e4 for s in ORIGINAL_PLANNER_CURVE.knot_rho_slopes()]
    ax2.bar(x - w / 2, s_std, width=w, color=C_NAVY, alpha=0.88, label="Standard Schedule")
    ax2.bar(x + w / 2, s_bal, width=w, color=C_SLATE, alpha=0.78, label="Balanced Schedule")
    ax2.annotate(
        "Slope rises (non-concave)",
        xy=(2.85, s_std[3] + 0.15),
        xytext=(1.05, 11.2),
        arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=1.2),
        fontsize=8.2,
        color=C_CORAL,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Horizon Interval $[h_k, h_{k+1}]$")
    ax2.set_ylabel(r"Secant Slope $\Delta\bar{\rho}/\Delta h$ ($\times 10^{-4}$)")
    ax2.set_title("(b) Non-Concavity of Commitment Slopes")
    ax2.set_ylim(0, 15.6)
    ax2.legend(frameon=False, loc="upper center", ncol=2)

    fig.savefig(OUT_DIR / "fig_privacy_envelope.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig_joint_planning_eval() -> None:
    df_syn = pd.read_csv(ROOT / "results" / "sdg_mh" / "long_horizon_420d_v3_final" / "audit_v3_paired_with_holm_wilcoxon.csv")
    df_wc = pd.read_csv(ROOT / "results" / "sdg_mh" / "wildchat_real_validation_v5" / "wildchat_k8_cohorts_paired_v5.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.45), constrained_layout=True)

    # Panel (a): Synthetic DistinctHeavy / Standard
    sub_s = df_syn[(df_syn["workload"] == "DistinctHeavy") & (df_syn["commitment_curve"] == "Week1_v2")]
    syn_specs = [
        ("rel_7d_mape_pct", "7d (Release)"),
        ("common_daily_30d_served_mape", "30d (Daily)"),
        ("common_sunday_30d_served_mape", "30d (Sunday)"),
        ("rel_90d_mape_pct", "90d (Release)"),
    ]
    y_pos = np.arange(len(syn_specs))[::-1]
    diffs_s, cis_s, pvals_s = [], [], []
    for mkey, _ in syn_specs:
        row = sub_s[sub_s["metric"] == mkey].iloc[0]
        diffs_s.append(float(row["paired_diff_mean"]))
        cis_s.append(float(row["paired_diff_ci95"]))
        pvals_s.append(float(row["paired_t_p_holm"]))

    ax1.axvline(0.0, color="black", lw=0.9, ls="--")
    for y, d, ci, p in zip(y_pos, diffs_s, cis_s, pvals_s):
        col = C_TEAL if (p < 0.05 and d < 0) else C_SLATE
        ax1.errorbar(d, y, xerr=ci, fmt="o", color=col, capsize=3.5, ms=5.5, lw=1.6)
        tag = f"{d:+.3f} pp (sig.)" if p < 0.05 else f"{d:+.3f} pp (n.s.)"
        x_txt = d - 0.008 if d < -0.02 else 0.008
        ha = "right" if d < -0.02 else "left"
        ax1.text(x_txt, y + 0.20, tag, ha=ha, va="bottom", fontsize=7.8, color=col)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([lbl for _, lbl in syn_specs])
    ax1.set_ylim(-0.55, len(syn_specs) - 0.25)
    ax1.set_xlim(-0.135, 0.115)
    ax1.set_xlabel("Paired MAPE Diff (pp) [Scale: $\\pm 0.13$ pp]")
    ax1.set_title("(a) Synthetic Workload ($K=8$ seeds, $df=7$)")

    # Panel (b): WildChat Real Trace DistinctHeavy / Standard
    sub_w = df_wc[(df_wc["workload"] == "DistinctHeavy") & (df_wc["curve"] == "Week1_v2")]
    wc_specs = [
        ("distinct_7d_active_mape", "7d (Release)"),
        ("distinct_30d_active_mape", "30d (Release)"),
        ("distinct_30d_latest_active_mape", "30d (All-Day)"),
        ("distinct_90d_active_mape", "90d (Release)"),
    ]
    y_pos_w = np.arange(len(wc_specs))[::-1]
    diffs_w, cis_w = [], []
    for mkey, _ in wc_specs:
        row = sub_w[sub_w["metric_key"] == mkey].iloc[0]
        diffs_w.append(float(row["paired_diff"]))
        cis_w.append(0.5 * (float(row["ci95_high"]) - float(row["ci95_low"])))

    ax2.axvline(0.0, color="black", lw=0.9, ls="--")
    for y, d, ci in zip(y_pos_w, diffs_w, cis_w):
        col = C_CORAL if d > 0 else C_TEAL
        ax2.errorbar(d, y, xerr=ci, fmt="s", color=col, capsize=3.5, ms=5.5, lw=1.6)
        x_txt = d - 0.10 if d < 0 else d + 0.10
        ha = "right" if d < 0 else "left"
        ax2.text(x_txt, y + 0.20, f"{d:+.2f} pp", ha=ha, va="bottom", fontsize=7.8, color=col)
    ax2.set_yticks(y_pos_w)
    ax2.set_yticklabels([lbl for _, lbl in wc_specs])
    ax2.set_ylim(-0.55, len(wc_specs) - 0.25)
    ax2.set_xlim(-2.4, 2.9)
    ax2.set_xlabel("Paired Active MAPE Diff (pp) [Scale: $\\pm 2.6$ pp]")
    ax2.set_title("(b) WildChat-4.8M Trace ($K=8$ cohorts, cond.)")

    fig.savefig(OUT_DIR / "fig_joint_planning_eval.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig_baselines_and_clipping() -> None:
    df_v3 = pd.read_csv(ROOT / "results" / "sdg_mh" / "long_horizon_420d_v3_final" / "audit_v3_competitive_baselines_summary.csv")
    df_bmf = pd.read_csv(ROOT / "results" / "sdg_mh" / "banded_mf_baseline_v5" / "banded_mf_competitive_summary_v5.csv")
    df_abl = pd.read_csv(ROOT / "results" / "sdg_mh" / "long_horizon_420d" / "cr_clipping_ablation_420d.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.45), constrained_layout=True)

    windows = [1, 7, 30, 90]
    x = np.arange(len(windows))
    methods = [
        ("SDG-MH (Joint-Opt)", df_v3, "SDG-MH (D_JointPlanning)", C_NAVY, "-o"),
        ("Banded-MF (b=8)", df_bmf, "Banded-MF (b=8, MH-RemEnv)", C_TEAL, "--s"),
        ("Banded-MF (b=30)", df_bmf, "Banded-MF (b=30, MH-RemEnv)", C_PURPLE, ":d"),
        ("Tree-2l (GLS)", df_v3, "Tree-2l (MH-RemEnv)", C_AMBER, "-.^"),
        ("W-BD-30", df_v3, "W-BD-30 (MH-Calibrated)", C_CORAL, "--v"),
    ]
    for label, src_df, mname, col, fmt in methods:
        means, cis = [], []
        for w in windows:
            r = src_df[(src_df["method"] == mname) & (src_df["metric"] == f"requests_{w}d_rmse")].iloc[0]
            means.append(float(r["mean"]))
            cis.append(float(r["seed_level_ci95"]))
        ax1.errorbar(x, means, yerr=cis, fmt=fmt, color=col, lw=1.6, ms=4.8, capsize=2.5, label=label)

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["1d", "7d", "30d", "90d"])
    ax1.set_xlabel("Query Window $w$")
    ax1.set_ylabel("Requests RMSE (log scale)")
    ax1.set_title("(a) Additive Error Under $[1,420]$ Envelope")
    ax1.legend(frameon=False, loc="lower right", ncol=1)

    crs = df_abl["C_r"].to_numpy()
    tot_rmse = df_abl["requests_total_rmse_mean"].to_numpy()
    tot_ci = df_abl["requests_total_rmse_ci95"].to_numpy()
    abs_bias = np.abs(df_abl["requests_clipping_bias_mean"].to_numpy())
    noise_rmse = df_abl["requests_raw_noise_rmse_mean"].to_numpy()

    ax2.errorbar(crs, tot_rmse, yerr=tot_ci, fmt="-o", color=C_NAVY, lw=1.8, ms=5.0, capsize=3, label="Total RMSE")
    ax2.plot(crs, abs_bias, ls="--", marker="s", ms=4.2, color=C_CORAL, lw=1.5, label=r"$|$Clipping Bias$|$")
    ax2.plot(crs, noise_rmse, ls=":", marker="^", ms=4.5, color=C_TEAL, lw=1.5, label="Noise RMSE")
    ax2.axvline(50, color=C_SLATE, lw=1.0, ls="-.")
    ax2.text(55, 4200, "Optimum\n$C_r=50$", fontsize=8.0, color=C_NAVY)
    ax2.set_xticks(crs)
    ax2.set_ylim(0, 7400)
    ax2.set_xlabel("Daily Per-User Clipping Threshold $C_r$")
    ax2.set_ylabel("Daily Requests Error")
    ax2.set_title("(b) Clipping Bias--Variance Trade-off")
    ax2.legend(frameon=False, loc="upper left")

    fig.savefig(OUT_DIR / "fig_baselines_and_clipping.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_fig_privacy_envelope()
    make_fig_joint_planning_eval()
    make_fig_baselines_and_clipping()
    print("Regenerated 3 high-readability vector PDF figures.")


if __name__ == "__main__":
    main()
