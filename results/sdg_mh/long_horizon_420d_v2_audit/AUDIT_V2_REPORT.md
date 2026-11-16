# SDG-MH Paper-Grade Statistical, Dual Certificate & Competitive Baseline Audit (v2)

- **Run Mode**: `FULL_AUDIT_V2`
- **Design**: `2000` initial users, `420` days, **`K=8` independent data seeds** (`[42, 1042, 2042, 3042, 4042, 5042, 6042, 7042]`) $\times$ `5` CRN MC draws = `40` runs/config
- **Inference Unit**: Seed-level independent unit ($df = 7$)
- **Execution Cost**: Total wall time `1073.54s` (`532.19s`/seed avg across parallel workers)

## 1. Optimization Cost & Directed-Rounding Dual Bounds (`DECIMAL100_DIRECTED_ROUNDING`)

| Workload | Curve | Group | Phases | Solve Time (s) | Primal $f^*$ | Dual LB (`Rel Gap`) | $\sigma_{\text{add,eff}}$ | $\sigma_7$ | $\sigma_{30}$ | $\sigma_{90}$ | Tier |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Default | Week1_v2 | A_Baseline | `(0, 0, 0)` | 0.019s | 2047.7860 | -inf (`inf`) | 2319.65 | 59.96 | 59.96 | 59.96 | `FEASIBLE_CERTIFIED` |
| Default | Week1_v2 | D_JointPlanning | `(0, 0, 0)` | 1.521s | 1433.9182 | 1433.9182 (`2.04e-11`) | 1661.12 | 74.71 | 74.71 | 65.29 | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` |
| DistinctHeavy | Week1_v2 | B_BudgetOnly | `(0, 0, 0)` | 0.025s | 932.6324 | 932.6324 (`1.00e-11`) | 2024.73 | 59.02 | 61.77 | 68.98 | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` |
| DistinctHeavy | Week1_v2 | D_JointPlanning | `(0, 4, 0)` | 1.376s | 922.9791 | 922.9791 (`7.45e-09`) | 2037.67 | 58.43 | 61.39 | 66.82 | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` |
| DistinctHeavy | Original | B_BudgetOnly | `(0, 0, 0)` | 0.026s | 940.8881 | 940.8881 (`1.00e-11`) | 2013.55 | 59.11 | 62.43 | 70.93 | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` |
| DistinctHeavy | Original | D_JointPlanning | `(0, 4, 0)` | 1.273s | 930.3260 | 930.3260 (`1.02e-11`) | 2031.01 | 58.54 | 61.90 | 68.43 | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` |

## 2. Stratified Hierarchical Inference (Seed-Level 95% CI vs PR#5 Invalid Pooled CI)

| Workload | Curve | Group | Metric | Mean | Seed CI95 ($df=7$) | Between-Pop SD | Within-MC SD | ICC | Invalid Pooled CI95 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Default | Week1_v2 | A_Baseline | `requests_clipping_bias` | -1437.7045 | $\pm$ 18.3962 | 22.0045 | 0.0000 | 1.000 | $\pm$ 6.6667 |
| Default | Week1_v2 | A_Baseline | `requests_1d_total_rmse` | 2800.2892 | $\pm$ 31.1624 | 0.0000 | 88.7713 | 0.000 | $\pm$ 28.0870 |
| Default | Week1_v2 | A_Baseline | `failures_1d_total_rmse` | 1595.7297 | $\pm$ 26.6207 | 7.1898 | 69.3625 | 0.011 | $\pm$ 22.2899 |
| Default | Week1_v2 | A_Baseline | `dist_7d_mape_pct` | 2.8752 | $\pm$ 0.0647 | 0.0558 | 0.1199 | 0.178 | $\pm$ 0.0419 |
| Default | Week1_v2 | A_Baseline | `dist_30d_mape_pct` | 2.4311 | $\pm$ 0.0572 | 0.0520 | 0.0992 | 0.216 | $\pm$ 0.0354 |
| Default | Week1_v2 | A_Baseline | `dist_90d_mape_pct` | 2.3446 | $\pm$ 0.0967 | 0.0804 | 0.1859 | 0.158 | $\pm$ 0.0642 |
| Default | Week1_v2 | D_JointPlanning | `requests_clipping_bias` | -1437.7045 | $\pm$ 18.3962 | 22.0045 | 0.0000 | 1.000 | $\pm$ 6.6667 |
| Default | Week1_v2 | D_JointPlanning | `requests_1d_total_rmse` | 2310.5192 | $\pm$ 29.7056 | 20.3946 | 65.0613 | 0.089 | $\pm$ 21.7057 |
| Default | Week1_v2 | D_JointPlanning | `failures_1d_total_rmse` | 1176.5563 | $\pm$ 20.2006 | 7.0725 | 51.6635 | 0.018 | $\pm$ 16.6611 |
| Default | Week1_v2 | D_JointPlanning | `dist_7d_mape_pct` | 3.5800 | $\pm$ 0.0835 | 0.0739 | 0.1504 | 0.194 | $\pm$ 0.0531 |
| Default | Week1_v2 | D_JointPlanning | `dist_30d_mape_pct` | 3.0265 | $\pm$ 0.0729 | 0.0668 | 0.1254 | 0.221 | $\pm$ 0.0449 |
| Default | Week1_v2 | D_JointPlanning | `dist_90d_mape_pct` | 2.5510 | $\pm$ 0.1025 | 0.0772 | 0.2131 | 0.116 | $\pm$ 0.0720 |
| DistinctHeavy | Week1_v2 | B_BudgetOnly | `requests_clipping_bias` | -1437.7045 | $\pm$ 18.3962 | 22.0045 | 0.0000 | 1.000 | $\pm$ 6.6667 |
| DistinctHeavy | Week1_v2 | B_BudgetOnly | `requests_1d_total_rmse` | 2575.0135 | $\pm$ 30.2514 | 9.1247 | 78.2972 | 0.013 | $\pm$ 25.1928 |
| DistinctHeavy | Week1_v2 | B_BudgetOnly | `failures_1d_total_rmse` | 1408.5118 | $\pm$ 24.0066 | 7.3513 | 62.0695 | 0.014 | $\pm$ 19.9754 |
| DistinctHeavy | Week1_v2 | B_BudgetOnly | `dist_7d_mape_pct` | 2.8311 | $\pm$ 0.0642 | 0.0552 | 0.1193 | 0.176 | $\pm$ 0.0417 |
| DistinctHeavy | Week1_v2 | B_BudgetOnly | `dist_30d_mape_pct` | 2.5043 | $\pm$ 0.0609 | 0.0563 | 0.1032 | 0.229 | $\pm$ 0.0371 |
| DistinctHeavy | Week1_v2 | B_BudgetOnly | `dist_90d_mape_pct` | 2.6942 | $\pm$ 0.1142 | 0.0996 | 0.2090 | 0.185 | $\pm$ 0.0733 |
| DistinctHeavy | Week1_v2 | D_JointPlanning | `requests_clipping_bias` | -1437.7045 | $\pm$ 18.3962 | 22.0045 | 0.0000 | 1.000 | $\pm$ 6.6667 |
| DistinctHeavy | Week1_v2 | D_JointPlanning | `requests_1d_total_rmse` | 2584.7426 | $\pm$ 30.2239 | 8.2964 | 78.6812 | 0.011 | $\pm$ 25.2887 |
| DistinctHeavy | Week1_v2 | D_JointPlanning | `failures_1d_total_rmse` | 1416.7724 | $\pm$ 24.1729 | 7.6066 | 62.3767 | 0.015 | $\pm$ 20.0817 |
| DistinctHeavy | Week1_v2 | D_JointPlanning | `dist_7d_mape_pct` | 2.7988 | $\pm$ 0.0595 | 0.0486 | 0.1163 | 0.149 | $\pm$ 0.0400 |
| DistinctHeavy | Week1_v2 | D_JointPlanning | `dist_30d_mape_pct` | 2.4811 | $\pm$ 0.0257 | 0.0000 | 0.1145 | 0.000 | $\pm$ 0.0344 |
| DistinctHeavy | Week1_v2 | D_JointPlanning | `dist_90d_mape_pct` | 2.6030 | $\pm$ 0.1079 | 0.0893 | 0.2083 | 0.155 | $\pm$ 0.0719 |
| DistinctHeavy | Original | B_BudgetOnly | `requests_clipping_bias` | -1437.7045 | $\pm$ 18.3962 | 22.0045 | 0.0000 | 1.000 | $\pm$ 6.6667 |
| DistinctHeavy | Original | B_BudgetOnly | `requests_1d_total_rmse` | 2566.6511 | $\pm$ 30.4312 | 10.8874 | 77.6669 | 0.019 | $\pm$ 25.0571 |
| DistinctHeavy | Original | B_BudgetOnly | `failures_1d_total_rmse` | 1402.0221 | $\pm$ 23.5629 | 6.7840 | 61.1696 | 0.012 | $\pm$ 19.6707 |
| DistinctHeavy | Original | B_BudgetOnly | `dist_7d_mape_pct` | 2.8346 | $\pm$ 0.0641 | 0.0557 | 0.1176 | 0.183 | $\pm$ 0.0412 |
| DistinctHeavy | Original | B_BudgetOnly | `dist_30d_mape_pct` | 2.5292 | $\pm$ 0.0559 | 0.0478 | 0.1045 | 0.173 | $\pm$ 0.0364 |
| DistinctHeavy | Original | B_BudgetOnly | `dist_90d_mape_pct` | 2.7792 | $\pm$ 0.1180 | 0.1032 | 0.2152 | 0.187 | $\pm$ 0.0756 |
| DistinctHeavy | Original | D_JointPlanning | `requests_clipping_bias` | -1437.7045 | $\pm$ 18.3962 | 22.0045 | 0.0000 | 1.000 | $\pm$ 6.6667 |
| DistinctHeavy | Original | D_JointPlanning | `requests_1d_total_rmse` | 2580.2302 | $\pm$ 30.3248 | 9.3431 | 78.3715 | 0.014 | $\pm$ 25.2238 |
| DistinctHeavy | Original | D_JointPlanning | `failures_1d_total_rmse` | 1413.1638 | $\pm$ 24.2516 | 8.1726 | 62.2371 | 0.017 | $\pm$ 20.0578 |
| DistinctHeavy | Original | D_JointPlanning | `dist_7d_mape_pct` | 2.8025 | $\pm$ 0.0609 | 0.0499 | 0.1187 | 0.150 | $\pm$ 0.0409 |
| DistinctHeavy | Original | D_JointPlanning | `dist_30d_mape_pct` | 2.5042 | $\pm$ 0.0252 | 0.0000 | 0.1181 | 0.000 | $\pm$ 0.0354 |
| DistinctHeavy | Original | D_JointPlanning | `dist_90d_mape_pct` | 2.6683 | $\pm$ 0.1024 | 0.0743 | 0.2178 | 0.104 | $\pm$ 0.0732 |

## 3. Common Random Numbers (CRN) Paired Comparison: `D_JointPlanning` vs Baseline/BudgetOnly

| Workload | Curve | Contrast | Metric | Base Mean | Trt Mean | Paired Diff (Trt - Base) | Rel % | Paired 95% CI | Two-Sided $p$ | Seeds Favoring Lower Err |
|---|---|---|---|---:|---:|---:|---:|---:|---:|:---:|
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `requests_1d_total_rmse` | 2575.0135 | 2584.7426 | **+9.7292** | +0.38% | $\pm$ 0.2269 | `0.0000` | `0/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `failures_1d_total_rmse` | 1408.5118 | 1416.7724 | **+8.2606** | +0.59% | $\pm$ 0.2478 | `0.0000` | `0/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `dist_7d_raw_rmse` | 58.8688 | 58.2101 | **-0.6587** | -1.12% | $\pm$ 0.2268 | `0.0002` | `8/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `dist_7d_mape_pct` | 2.8311 | 2.7988 | **-0.0323** | -1.14% | $\pm$ 0.0096 | `0.0001` | `8/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `dist_7d_mean_gt` | 1669.1417 | 1669.1417 | **+0.0000** | +0.00% | $\pm$ 0.0000 | `1.0000` | `0/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `dist_30d_raw_rmse` | 61.6083 | 61.3177 | **-0.2906** | -0.47% | $\pm$ 1.2496 | `0.5995` | `6/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `dist_30d_mape_pct` | 2.5043 | 2.4811 | **-0.0232** | -0.93% | $\pm$ 0.0528 | `0.3345` | `6/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `dist_30d_mean_gt` | 1965.9188 | 1972.1116 | **+6.1929** | +0.32% | $\pm$ 0.5061 | `0.0000` | `8/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `dist_90d_raw_rmse` | 68.8664 | 66.4464 | **-2.4200** | -3.51% | $\pm$ 0.3800 | `0.0000` | `8/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `dist_90d_mape_pct` | 2.6942 | 2.6030 | **-0.0913** | -3.39% | $\pm$ 0.0113 | `0.0000` | `8/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `dist_90d_mean_gt` | 2049.3500 | 2049.3500 | **+0.0000** | +0.00% | $\pm$ 0.0000 | `1.0000` | `0/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `requests_1d_total_rmse` | 2566.6511 | 2580.2302 | **+13.5791** | +0.53% | $\pm$ 0.8137 | `0.0000` | `0/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `failures_1d_total_rmse` | 1402.0221 | 1413.1638 | **+11.1418** | +0.80% | $\pm$ 0.9935 | `0.0000` | `0/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `dist_7d_raw_rmse` | 58.9365 | 58.2816 | **-0.6548** | -1.11% | $\pm$ 0.1993 | `0.0001` | `8/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `dist_7d_mape_pct` | 2.8346 | 2.8025 | **-0.0322** | -1.14% | $\pm$ 0.0091 | `0.0001` | `8/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `dist_7d_mean_gt` | 1669.1417 | 1669.1417 | **+0.0000** | +0.00% | $\pm$ 0.0000 | `1.0000` | `0/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `dist_30d_raw_rmse` | 62.2438 | 61.8709 | **-0.3729** | -0.60% | $\pm$ 1.1887 | `0.4824` | `6/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `dist_30d_mape_pct` | 2.5292 | 2.5042 | **-0.0250** | -0.99% | $\pm$ 0.0479 | `0.2580` | `6/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `dist_30d_mean_gt` | 1965.9188 | 1972.1116 | **+6.1929** | +0.32% | $\pm$ 0.5061 | `0.0000` | `8/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `dist_90d_raw_rmse` | 71.0579 | 68.1291 | **-2.9288** | -4.12% | $\pm$ 0.7445 | `0.0000` | `8/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `dist_90d_mape_pct` | 2.7792 | 2.6683 | **-0.1109** | -3.99% | $\pm$ 0.0270 | `0.0000` | `8/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `dist_90d_mean_gt` | 2049.3500 | 2049.3500 | **+0.0000** | +0.00% | $\pm$ 0.0000 | `1.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `requests_1d_total_rmse` | 2800.2892 | 2310.5192 | **-489.7699** | -17.49% | $\pm$ 7.9123 | `0.0000` | `8/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `failures_1d_total_rmse` | 1595.7297 | 1176.5563 | **-419.1734** | -26.27% | $\pm$ 7.0162 | `0.0000` | `8/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `dist_7d_raw_rmse` | 59.7620 | 74.4450 | **+14.6830** | +24.57% | $\pm$ 0.3028 | `0.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `dist_7d_mape_pct` | 2.8752 | 3.5800 | **+0.7049** | +24.52% | $\pm$ 0.0197 | `0.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `dist_7d_mean_gt` | 1669.1417 | 1669.1417 | **+0.0000** | +0.00% | $\pm$ 0.0000 | `1.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `dist_30d_raw_rmse` | 59.7747 | 74.4538 | **+14.6791** | +24.56% | $\pm$ 0.3559 | `0.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `dist_30d_mape_pct` | 2.4311 | 3.0265 | **+0.5954** | +24.49% | $\pm$ 0.0165 | `0.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `dist_30d_mean_gt` | 1965.9188 | 1965.9188 | **+0.0000** | +0.00% | $\pm$ 0.0000 | `1.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `dist_90d_raw_rmse` | 59.9345 | 65.0478 | **+5.1132** | +8.53% | $\pm$ 0.3982 | `0.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `dist_90d_mape_pct` | 2.3446 | 2.5510 | **+0.2064** | +8.80% | $\pm$ 0.0149 | `0.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `dist_90d_mean_gt` | 2049.3500 | 2049.3500 | **+0.0000** | +0.00% | $\pm$ 0.0000 | `1.0000` | `0/8` |

## 4. Fair Time-Series DP Competitive Baselines (Identical `Week1_v2` Remaining Additive Envelope, `C_r=50, C_f=10`)

> **Non-Comparability Disclosure**:
> 1. **Non-Additive Distinct Histograms (`7d/30d/90d` with $k=3$ Task Cap)**: Binary Tree (`Tree-2l`, `Tree-U`) and $w$-event (`W-BD-30`) mechanisms operate only on additive numeric counts and **cannot** compute non-additive user-deduplicated Distinct counts across overlapping multi-day windows (`N/A`).
> 2. **$w$-Event Adaptive Privacy Mismatch**: Naive single-horizon `W-BD-30` calibrated only at $\bar\rho_{\text{add}}(30)$ has $h=1$ spend ratio `0.65x` of budget limit and **breaches** `Week1_v2` at $h=365$ by `1.35x` (tightest breach ratio at `h=32`). Enforcing $\rho_{\text{WBD}}(h) \le \bar\rho_{\text{add}}(h)$ for all $h \in [1, 420]$ shrinks `rho_w` to `0.012359` (`tight_h=32`).

| Method | Calibration Scope | Req 1d RMSE $\pm$ CI95 | Req 7d RMSE $\pm$ CI95 | Req 30d RMSE $\pm$ CI95 | Req 90d RMSE $\pm$ CI95 | Fail 30d RMSE $\pm$ CI95 | Distinct Tasks (7/30/90d) |
|---|---|---:|---:|---:|---:|---:|:---:|
| **SDG-MH (D_JointPlanning)** | Exact Multi-Horizon rem_add_envelope (All h in 1..420) | 2310.5 $\pm$ 29.7 | 10919.2 $\pm$ 324.2 | 43235.7 $\pm$ 1485.3 | 125238.1 $\pm$ 4161.6 | 7667.4 $\pm$ 510.3 | Supported (Native Calendar) |
| **Tree-2l (MH-RemEnv)** | Exact Multi-Horizon rem_add_envelope (tight_h=394) | 4378.6 $\pm$ 61.7 | 12736.0 $\pm$ 199.4 | 45308.8 $\pm$ 616.4 | 129162.9 $\pm$ 2309.3 | 9682.6 $\pm$ 777.3 | N/A (Non-Additive Union) |
| **Tree-2l (Daily-d Parity)** | Exact Multi-Horizon h*d_SDG (tight_h=384) | 4406.9 $\pm$ 62.2 | 12771.8 $\pm$ 200.6 | 45338.5 $\pm$ 618.9 | 129195.6 $\pm$ 2325.0 | 9748.7 $\pm$ 784.3 | N/A (Non-Additive Union) |
| **Tree-U (MH-RemEnv)** | Exact Multi-Horizon rem_add_envelope (tight_h=420) | 27742.4 $\pm$ 447.3 | 50365.7 $\pm$ 928.6 | 72781.3 $\pm$ 1850.0 | 143671.2 $\pm$ 5460.3 | 42092.5 $\pm$ 1118.8 | N/A (Non-Additive Union) |
| **W-BD-30 (MH-Calibrated)** | Shrunk to all-h Multi-Horizon envelope (tight_h=32, rho_w=0.012359) | 3565.6 $\pm$ 236.3 | 15477.0 $\pm$ 1310.2 | 49154.3 $\pm$ 2555.8 | 125582.2 $\pm$ 5147.2 | 20410.1 $\pm$ 1192.8 | N/A (Non-Additive Union) |

## 5. Figures
- [fig1_paired_diff_d_vs_b_seed_ci.png](figures/fig1_paired_diff_d_vs_b_seed_ci.png)
- [fig2_competitive_baseline_additive_rmse.png](figures/fig2_competitive_baseline_additive_rmse.png)