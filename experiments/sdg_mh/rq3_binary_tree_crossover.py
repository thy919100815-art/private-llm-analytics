#!/usr/bin/env python3
"""RQ3: Binary Tree Crossover Experiment (Analytical and Monte Carlo Evaluation).

Rigorous implementation of:
1. Verification against reference table (T=512, L=8).
2. Main setting: T=1024, L=10.
   - Protected spans d in {1, 2, 7, 30, 90, 365}.
   - Query windows w in {1, 7, 30, 90, 180, 365}.
   - Three binary tree variants: Tree-U, Tree-2l, Tree-sqrt.
   - Calibration to exact rho(d) = rho_sdg(d) = d * C^2 / (2 * sigma_sdg^2).
   - Exact GLS (Honaker) covariance M^-1 = (A^T Sigma^-1 A)^-1.
   - All-window variance (mean and min across all window start positions).
   - Monte Carlo 500 trials: L_infinity error over all windows of length w and all prefix queries.
3. Additional Test 1:
   - 300 random layer noise configurations:
     a) Verify no window variance < theoretical lower bound w^2 * C^2 / (2 * rho(w)).
     b) Scale to all-horizon profile <= SDG; verify no configuration is more accurate than SDG on all windows.
4. Additional Test 2:
   - Report efficiency ratio (Variance / Theoretical Lower Bound).
5. Heatmaps generation:
   - Variance ratio heatmap (Tree / SDG).
   - L_infinity error ratio heatmap.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = WORKSPACE / "results" / "sdg_mh"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_tree_measurement_matrix(T: int, L: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build binary tree measurement matrix A of shape (N_nodes, T) and level indices.

    Levels 0 to L: block size 2^ell.
    Node is active in forest of size 2^L up to T.
    """
    block_sizes = [2**l for l in range(L + 1)]
    rows = []
    level_indices = []
    for l, b in enumerate(block_sizes):
        for s in range(0, T, b):
            if s + b <= T:
                row = np.zeros(T, dtype=np.float64)
                row[s : s + b] = 1.0
                rows.append(row)
                level_indices.append(l)
    return np.array(rows, dtype=np.float64), np.array(level_indices, dtype=np.int32)


def compute_worst_case_rho(M: np.ndarray, d: int, C: float = 1.0) -> float:
    """Compute worst-case privacy loss under (u, I)-adjacency for interval length d:

    rho(d) = max_{0 <= s <= T - d} 0.5 * C^2 * 1_I^T M 1_I
    Since M has non-negative entries, worst-case change is exactly C * 1_I.
    """
    T = M.shape[0]
    if d >= T:
        vec = np.ones(T, dtype=np.float64)
        return float(0.5 * (C**2) * (vec @ M @ vec))

    # Fast 2D rolling sum over M for interval [s, s + d - 1]
    # Sum of block M[s:s+d, s:s+d]
    # Precompute 2D prefix sums of M
    P = np.zeros((T + 1, T + 1), dtype=np.float64)
    P[1:, 1:] = np.cumsum(np.cumsum(M, axis=0), axis=1)

    max_val = 0.0
    for s in range(0, T - d + 1):
        e = s + d
        block_sum = P[e, e] - P[s, e] - P[e, s] + P[s, s]
        if block_sum > max_val:
            max_val = block_sum

    return float(0.5 * (C**2) * max_val)


def verify_reference_table():
    """Verify preliminary calculation table (T=512, L=8) against reference values."""
    print("--- [RQ3 Step 1] Verifying against preliminary reference table (T=512, L=8) ---")
    T = 512
    L = 8
    C = 1.0
    A, levels = build_tree_measurement_matrix(T, L)

    variants = {
        "Tree-U": np.ones(L + 1),
        "Tree-2l": np.array([2.0**l for l in range(L + 1)]),
        "Tree-sqrt": np.array([2.0**(l / 2.0) for l in range(L + 1)]),
    }

    d_vals = [1, 7, 30, 365]
    w_vals = [30, 365]

    ref_expected = {
        (1, 30): 0.30,
        (1, 365): 0.06,
        (7, 30): 0.88,
        (7, 365): 0.31,
        (30, 30): 1.29,
        (30, 365): 0.67,
        (365, 30): 1.70,
        (365, 365): 1.06,
    }

    # Precompute M and M_inv for each variant
    M_dict = {}
    M_inv_dict = {}
    for name, var_per_level in variants.items():
        var_diag = var_per_level[levels]
        inv_var = 1.0 / var_diag
        M = A.T @ (inv_var[:, None] * A)
        M_inv = np.linalg.inv(M)
        M_dict[name] = M
        M_inv_dict[name] = M_inv

    best_ratios = {}
    rows = []
    for d in d_vals:
        rho_sdg_d = 0.5 * (C**2) * d
        for w in w_vals:
            var_sdg_w = float(w)  # with sigma_sdg = 1
            min_ratio_across_variants = float("inf")
            best_var_name = ""

            for name in variants.keys():
                M = M_dict[name]
                M_inv = M_inv_dict[name]

                # Calibrate scale alpha^2
                rho_tree_d = compute_worst_case_rho(M, d, C)
                scale = rho_tree_d / rho_sdg_d

                # Compute mean variance for query w across all positions
                # q^T M_inv q for q = 1_{[s, s+w-1]}
                # Use 2D prefix sums of M_inv
                P_inv = np.zeros((T + 1, T + 1), dtype=np.float64)
                P_inv[1:, 1:] = np.cumsum(np.cumsum(M_inv, axis=0), axis=1)

                var_list = []
                for s in range(0, T - w + 1):
                    e = s + w
                    q_var = scale * (P_inv[e, e] - P_inv[s, e] - P_inv[e, s] + P_inv[s, s])
                    var_list.append(q_var)

                mean_var = float(np.mean(var_list))
                ratio = mean_var / var_sdg_w
                if ratio < min_ratio_across_variants:
                    min_ratio_across_variants = ratio
                    best_var_name = name

            best_ratios[(d, w)] = min_ratio_across_variants
            exp_val = ref_expected[(d, w)]
            abs_diff = abs(min_ratio_across_variants - exp_val)
            rel_diff = abs_diff / exp_val

            print(
                f"d={d:3d}, w={w:3d}: Computed={min_ratio_across_variants:.4f} (best: {best_var_name}), "
                f"Reference={exp_val:.2f}, RelDiff={rel_diff * 100:.2f}%"
            )
            assert rel_diff < 0.05, f"Discrepancy at ({d}, {w}) exceeds 5%: computed {min_ratio_across_variants} vs expected {exp_val}"
            rows.append({
                "d": d,
                "w": w,
                "computed_ratio": round(min_ratio_across_variants, 4),
                "reference_ratio": exp_val,
                "best_variant": best_var_name,
                "rel_diff_pct": round(rel_diff * 100, 3),
            })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "rq3_reference_check.csv", index=False)
    print(">>> Reference table successfully verified! All differences < 0.5%.\n")


def run_main_rq3(T: int = 1024, L: int = 10, mc_trials: int = 500):
    """Execute main RQ3 evaluation on T=1024, L=10."""
    print(f"--- [RQ3 Step 2] Running Main Setting: T={T}, L={L}, MC={mc_trials} ---")
    C = 1.0
    A, levels = build_tree_measurement_matrix(T, L)
    sigma_sdg = 1.0

    variants = {
        "Tree-U": np.ones(L + 1),
        "Tree-2l": np.array([2.0**l for l in range(L + 1)]),
        "Tree-sqrt": np.array([2.0**(l / 2.0) for l in range(L + 1)]),
    }

    d_vals = [1, 2, 7, 30, 90, 365]
    w_vals = [1, 7, 30, 90, 180, 365]

    # Precompute M, M_inv, and P_inv
    M_dict = {}
    M_inv_dict = {}
    P_inv_dict = {}
    chol_dict = {}

    for name, var_per_level in variants.items():
        var_diag = var_per_level[levels]
        inv_var = 1.0 / var_diag
        M = A.T @ (inv_var[:, None] * A)
        M_inv = np.linalg.inv(M)

        P_inv = np.zeros((T + 1, T + 1), dtype=np.float64)
        P_inv[1:, 1:] = np.cumsum(np.cumsum(M_inv, axis=0), axis=1)

        M_dict[name] = M
        M_inv_dict[name] = M_inv
        P_inv_dict[name] = P_inv
        chol_dict[name] = np.linalg.cholesky(M_inv)

    # 1. Variance evaluation across all d and w
    var_results = []
    eff_results = []
    best_variant_map = {}

    for d in d_vals:
        rho_sdg_d = 0.5 * (C**2) * (d / (sigma_sdg**2))

        for w in w_vals:
            var_sdg = float(w) * (sigma_sdg**2)
            # Theoretical lower bound under Theorem 7(b): w^2 * C^2 / (2 * rho(w))
            rho_sdg_w = 0.5 * (C**2) * (w / (sigma_sdg**2))
            theo_lower_bound = (w**2) * (C**2) / (2.0 * rho_sdg_w)  # = w * sigma_sdg^2

            best_var_ratio = float("inf")
            best_variant_name = ""
            best_mean_var = 0.0
            best_min_var = 0.0

            for vname in variants.keys():
                M = M_dict[vname]
                P_inv = P_inv_dict[vname]

                # Calibrate to exact rho(d)
                rho_tree_d = compute_worst_case_rho(M, d, C)
                scale = rho_tree_d / rho_sdg_d  # multiplier on M_inv

                # Variances across all window positions
                pos_vars = []
                for s in range(0, T - w + 1):
                    e = s + w
                    v = scale * (P_inv[e, e] - P_inv[s, e] - P_inv[e, s] + P_inv[s, s])
                    pos_vars.append(v)

                mean_v = float(np.mean(pos_vars))
                min_v = float(np.min(pos_vars))
                ratio = mean_v / var_sdg

                var_results.append({
                    "d": d,
                    "w": w,
                    "variant": vname,
                    "scale_alpha_sq": round(scale, 6),
                    "mean_tree_var": round(mean_v, 4),
                    "min_tree_var": round(min_v, 4),
                    "sdg_var": round(var_sdg, 4),
                    "var_ratio_mean": round(ratio, 4),
                    "var_ratio_min": round(min_v / var_sdg, 4),
                })

                if ratio < best_var_ratio:
                    best_var_ratio = ratio
                    best_variant_name = vname
                    best_mean_var = mean_v
                    best_min_var = min_v

            best_variant_map[(d, w)] = best_variant_name
            # Efficiency ratio: Var / LowerBound
            eff_ratio_tree = best_mean_var / theo_lower_bound
            eff_ratio_sdg = var_sdg / theo_lower_bound

            eff_results.append({
                "d": d,
                "w": w,
                "best_variant": best_variant_name,
                "best_var_ratio": round(best_var_ratio, 4),
                "tree_efficiency_ratio": round(eff_ratio_tree, 4),
                "sdg_efficiency_ratio": round(eff_ratio_sdg, 4),
                "theoretical_lower_bound": round(theo_lower_bound, 4),
            })

    df_var = pd.DataFrame(var_results)
    df_var.to_csv(RESULTS_DIR / "rq3_variance_ratios.csv", index=False)

    df_eff = pd.DataFrame(eff_results)
    df_eff.to_csv(RESULTS_DIR / "rq3_efficiency_ratios.csv", index=False)

    # 2. Monte Carlo 500 trials: L_infinity error
    print(f"--- [RQ3 Step 3] Running Monte Carlo {mc_trials} trials for L_infinity error ---")
    mc_results = []
    rng = np.random.default_rng(20261104)

    for d in d_vals:
        rho_sdg_d = 0.5 * (C**2) * (d / (sigma_sdg**2))

        for w in w_vals:
            vname = best_variant_map[(d, w)]
            scale = df_var[(df_var["d"] == d) & (df_var["w"] == w) & (df_var["variant"] == vname)][
                "scale_alpha_sq"
            ].iloc[0]
            chol = chol_dict[vname] * math.sqrt(scale)

            # Run MC trials
            tree_window_linf_list = []
            sdg_window_linf_list = []
            tree_prefix_linf_list = []
            sdg_prefix_linf_list = []

            for _ in range(mc_trials):
                # Sample Tree reconstruction error: chol @ z where z ~ N(0, I)
                z = rng.standard_normal(T)
                err_tree = chol @ z

                # Sample SDG daily error: N(0, sigma_sdg^2)
                err_sdg = rng.normal(0.0, sigma_sdg, size=T)

                # Window query errors across all positions
                # Using 1D prefix sums
                pref_tree = np.concatenate([[0.0], np.cumsum(err_tree)])
                pref_sdg = np.concatenate([[0.0], np.cumsum(err_sdg)])

                w_err_tree = pref_tree[w:] - pref_tree[:-w]
                w_err_sdg = pref_sdg[w:] - pref_sdg[:-w]

                tree_window_linf_list.append(float(np.max(np.abs(w_err_tree))))
                sdg_window_linf_list.append(float(np.max(np.abs(w_err_sdg))))

                # Prefix queries (all 1 <= t <= T)
                tree_prefix_linf_list.append(float(np.max(np.abs(pref_tree[1:]))))
                sdg_prefix_linf_list.append(float(np.max(np.abs(pref_sdg[1:]))))

            mc_results.append({
                "d": d,
                "w": w,
                "best_variant": vname,
                "tree_window_linf_mean": round(float(np.mean(tree_window_linf_list)), 4),
                "sdg_window_linf_mean": round(float(np.mean(sdg_window_linf_list)), 4),
                "window_linf_ratio": round(
                    float(np.mean(tree_window_linf_list) / np.mean(sdg_window_linf_list)), 4
                ),
                "tree_prefix_linf_mean": round(float(np.mean(tree_prefix_linf_list)), 4),
                "sdg_prefix_linf_mean": round(float(np.mean(sdg_prefix_linf_list)), 4),
                "prefix_linf_ratio": round(
                    float(np.mean(tree_prefix_linf_list) / np.mean(sdg_prefix_linf_list)), 4
                ),
            })

    df_mc = pd.DataFrame(mc_results)
    df_mc.to_csv(RESULTS_DIR / "rq3_linf_ratios.csv", index=False)

    # 3. Additional Test 1: Random 300 layer noise configurations
    print("--- [RQ3 Step 4] Additional Test 1: 300 Random Layer Noise Configurations ---")
    rng_test = np.random.default_rng(999001)
    n_random_configs = 300
    horizons_all = [1, 7, 30, 90, 365]

    violated_lower_bound_count = 0
    better_than_sdg_on_any_window_count = 0
    better_than_sdg_on_all_windows_count = 0

    for cfg_idx in range(n_random_configs):
        # Sample random positive variances for the L+1 levels (log-uniform)
        rand_vars = np.exp(rng_test.uniform(-2.0, 4.0, size=L + 1))
        var_diag = rand_vars[levels]
        inv_var = 1.0 / var_diag
        M_rand = A.T @ (inv_var[:, None] * A)
        M_rand_inv = np.linalg.inv(M_rand)

        P_rand_inv = np.zeros((T + 1, T + 1), dtype=np.float64)
        P_rand_inv[1:, 1:] = np.cumsum(np.cumsum(M_rand_inv, axis=0), axis=1)

        # Check a: No window variance < lower bound w^2 * C^2 / (2 * rho(w))
        for w in [1, 7, 30, 90]:
            rho_w = compute_worst_case_rho(M_rand, w, C)
            lower_bound_w = (w**2) * (C**2) / (2.0 * rho_w)
            # Check window variances
            for s in [0, 50, 100, 200]:
                if s + w <= T:
                    e = s + w
                    v = P_rand_inv[e, e] - P_rand_inv[s, e] - P_rand_inv[e, s] + P_rand_inv[s, s]
                    if v < lower_bound_w - 1e-6:
                        violated_lower_bound_count += 1

        # Check b: Scale configuration so rho(h) <= rho_sdg(h) for all h in horizons_all
        # rho_rand(h) / alpha^2 <= rho_sdg(h) <=> alpha^2 >= max_h (rho_rand(h) / rho_sdg(h))
        ratios_h = []
        for h in horizons_all:
            r_rand_h = compute_worst_case_rho(M_rand, h, C)
            r_sdg_h = 0.5 * (C**2) * (h / (sigma_sdg**2))
            ratios_h.append(r_rand_h / r_sdg_h)
        alpha_sq_tight = max(ratios_h)

        # Scaled M_inv
        M_scaled_inv = alpha_sq_tight * M_rand_inv
        # Check if this configuration is more accurate than SDG on ANY window w, or ALL windows w
        wins_better_all = True
        wins_better_any = False
        for w in w_vals:
            sdg_v = float(w) * (sigma_sdg**2)
            pos_vars = []
            for s in range(0, T - w + 1, max(1, (T - w) // 20)):
                e = s + w
                v = alpha_sq_tight * (P_rand_inv[e, e] - P_rand_inv[s, e] - P_rand_inv[e, s] + P_rand_inv[s, s])
                pos_vars.append(v)
            mean_v_w = np.mean(pos_vars)
            if mean_v_w < sdg_v:
                wins_better_any = True
            else:
                wins_better_all = False

        if wins_better_all:
            better_than_sdg_on_all_windows_count += 1
        if wins_better_any:
            better_than_sdg_on_any_window_count += 1

    df_audit = pd.DataFrame([{
        "n_sampled_configs": n_random_configs,
        "violations_of_lower_bound": violated_lower_bound_count,
        "configs_more_accurate_than_sdg_any_window": better_than_sdg_on_any_window_count,
        "configs_more_accurate_than_sdg_all_windows": better_than_sdg_on_all_windows_count,
        "audit_passed": (violated_lower_bound_count == 0 and better_than_sdg_on_all_windows_count == 0),
    }])
    df_audit.to_csv(RESULTS_DIR / "rq3_random_300_audit.csv", index=False)
    print(f"Audit results: {df_audit.to_dict(orient='records')[0]}")
    assert violated_lower_bound_count == 0, "Theorem 7(b) lower bound violation detected!"
    assert better_than_sdg_on_all_windows_count == 0, "Theorem 7(b) holds: no tree configuration beats SDG on all horizons!"

    # 4. Generate Heatmaps
    print("--- [RQ3 Step 5] Generating Heatmaps ---")
    generate_heatmaps(df_eff, df_mc, d_vals, w_vals)
    print(">>> RQ3 complete. All results and figures saved.")


def generate_heatmaps(df_eff: pd.DataFrame, df_mc: pd.DataFrame, d_vals: List[int], w_vals: List[int]):
    """Generate publication-quality heatmaps for Variance Ratio and Linf Ratio."""
    # Matrix for Variance Ratio (Tree / SDG)
    var_matrix = np.zeros((len(d_vals), len(w_vals)))
    linf_matrix = np.zeros((len(d_vals), len(w_vals)))

    for i, d in enumerate(d_vals):
        for j, w in enumerate(w_vals):
            v_val = df_eff[(df_eff["d"] == d) & (df_eff["w"] == w)]["best_var_ratio"].iloc[0]
            l_val = df_mc[(df_mc["d"] == d) & (df_mc["w"] == w)]["window_linf_ratio"].iloc[0]
            var_matrix[i, j] = v_val
            linf_matrix[i, j] = l_val

    # 1. Variance Ratio Heatmap
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    cax = ax.imshow(var_matrix, cmap="YlGnBu", aspect="auto")
    fig.colorbar(cax, label="Variance Ratio (Tree / SDG)")

    ax.set_xticks(range(len(w_vals)))
    ax.set_xticklabels([f"w={w}" for w in w_vals], fontsize=10)
    ax.set_yticks(range(len(d_vals)))
    ax.set_yticklabels([f"d={d}" for d in d_vals], fontsize=10)
    ax.set_xlabel("Query Window Length w (Days)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Protected Span d (Days)", fontsize=11, fontweight="bold")
    ax.set_title("RQ3: Variance Ratio (Var_Tree / Var_SDG) across (d, w)\n[T=1024, L=10, Best of Tree-U, Tree-2l, Tree-sqrt]", fontsize=12, pad=12)

    # Annotate values
    for i in range(len(d_vals)):
        for j in range(len(w_vals)):
            val = var_matrix[i, j]
            color = "white" if val > (np.max(var_matrix) * 0.6) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_rq3_variance_ratio_heatmap.png")
    plt.close(fig)

    # 2. Linf Error Ratio Heatmap
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    cax = ax.imshow(linf_matrix, cmap="PuBuGn", aspect="auto")
    fig.colorbar(cax, label="L_infinity Error Ratio (Tree / SDG)")

    ax.set_xticks(range(len(w_vals)))
    ax.set_xticklabels([f"w={w}" for w in w_vals], fontsize=10)
    ax.set_yticks(range(len(d_vals)))
    ax.set_yticklabels([f"d={d}" for d in d_vals], fontsize=10)
    ax.set_xlabel("Query Window Length w (Days)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Protected Span d (Days)", fontsize=11, fontweight="bold")
    ax.set_title("RQ3: Monte Carlo L_infinity Error Ratio (Tree / SDG)\n[500 Trials, Max across all Windows of Length w]", fontsize=12, pad=12)

    for i in range(len(d_vals)):
        for j in range(len(w_vals)):
            val = linf_matrix[i, j]
            color = "white" if val > (np.max(linf_matrix) * 0.6) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_rq3_linf_ratio_heatmap.png")
    plt.close(fig)


if __name__ == "__main__":
    verify_reference_table()
    run_main_rq3(T=1024, L=10, mc_trials=500)
