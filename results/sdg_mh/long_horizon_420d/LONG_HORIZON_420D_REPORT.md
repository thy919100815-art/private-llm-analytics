# SDG-MH 420-Day Cross-Period True E2E & $C_r$ Ablation Report

- **Run Mode**: `FULL`
- **Population / Horizon**: `2000` initial users, `420` days (`1..420` valid ISO-8601 calendar dates `2025-01-01` .. `2026-02-24`)
- **Statistical Replication**: Data seeds `[42, 1042, 2042]` $\times$ `10` Monte Carlo noise draws = `30` runs per configuration
- **Total Wall Time**: `826.19s`

## 1. Reproducible Commands
```powershell
# Smoke Test:
.\.venv\Scripts\python.exe experiments/sdg_mh/run_long_horizon_420d_and_ablation.py --smoke
# Official Full Experiment:
.\.venv\Scripts\python.exe experiments/sdg_mh/run_long_horizon_420d_and_ablation.py --full
```

## 2. 420-Day True E2E Results (Mean $\pm$ 95% CI)

> **Metric Separation Note**: Optimizer Abstract Objective ($\sum c_j / v_j$), Primal/Dual Optimality Gap, and Certification Tier characterize convex calendar planning; Requests/Failures RMSE and Distinct 7d/30d/90d MAPE reflect actual end-to-end executed pipeline noise and clipping.

| Workload | Curve | Group | Phases | Abstract Obj ($f^*$) | Dual LB / Rel Gap | Tier | Req Total RMSE | Fail Total RMSE | 7d MAPE (60w) | 30d MAPE (56w) | 90d MAPE (12w) |
|---|---|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| Default | Week1_v2 | A_Baseline | `(0, 0, 0)` | 2047.79 | -inf (`inf`) | `FEASIBLE_CERTIFIED` | 2796.08 $\pm$ 10.73 | 1560.44 $\pm$ 42.90 | 2.89% $\pm$ 0.05% | 2.44% $\pm$ 0.04% | 2.33% $\pm$ 0.10% |
| Default | Week1_v2 | D_JointPlanning | `(0, 0, 0)` | 1433.92 | 1433.92 (`0.00e+00`) | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` | 2355.85 $\pm$ 22.99 | 1253.37 $\pm$ 8.74 | 3.58% $\pm$ 0.07% | 3.03% $\pm$ 0.05% | 2.57% $\pm$ 0.10% |
| DistinctHeavy | Week1_v2 | B_BudgetOnly | `(0, 0, 0)` | 932.63 | 932.63 (`0.00e+00`) | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` | 2602.23 $\pm$ 16.99 | 1423.19 $\pm$ 34.20 | 2.88% $\pm$ 0.05% | 2.55% $\pm$ 0.05% | 2.78% $\pm$ 0.07% |
| DistinctHeavy | Week1_v2 | D_JointPlanning | `(0, 4, 0)` | 922.98 | 922.98 (`0.00e+00`) | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` | 2573.74 $\pm$ 12.17 | 1351.43 $\pm$ 17.66 | 2.79% $\pm$ 0.05% | 2.46% $\pm$ 0.04% | 2.61% $\pm$ 0.08% |
| DistinctHeavy | Original | B_BudgetOnly | `(0, 0, 0)` | 940.89 | 940.89 (`0.00e+00`) | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` | 2551.90 $\pm$ 18.16 | 1389.84 $\pm$ 18.53 | 2.83% $\pm$ 0.05% | 2.52% $\pm$ 0.04% | 2.75% $\pm$ 0.12% |
| DistinctHeavy | Original | D_JointPlanning | `(0, 4, 0)` | 930.33 | 930.33 (`0.00e+00`) | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` | 2595.37 $\pm$ 13.75 | 1345.18 $\pm$ 9.03 | 2.87% $\pm$ 0.04% | 2.55% $\pm$ 0.04% | 2.78% $\pm$ 0.07% |

## 3. Cross-Period Ledger Trajectory Checkpoints ($t = 210, 365, 420$)

| Curve | Group | Checkpoint Day | Horizon $h$ | Ledger $\max \rho(h)$ | Theo 575d $L_{\max}(h)$ | Commitment $\bar\rho(h)$ | Headroom | 575d Cert Secure |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| Original | B_BudgetOnly | 210 | 90 | 0.081447 | 0.081447 | 0.084106 | 0.002660 | PASS |
| Original | B_BudgetOnly | 210 | 210 | 0.164549 | 0.176460 | 0.179343 | 0.014793 | PASS |
| Original | B_BudgetOnly | 365 | 90 | 0.081447 | 0.081447 | 0.084106 | 0.002660 | PASS |
| Original | B_BudgetOnly | 365 | 210 | 0.176460 | 0.176460 | 0.179343 | 0.002883 | PASS |
| Original | B_BudgetOnly | 365 | 365 | 0.285394 | 0.298397 | 0.302356 | 0.016963 | PASS |
| Original | B_BudgetOnly | 420 | 90 | 0.081447 | 0.081447 | 0.084106 | 0.002660 | PASS |
| Original | B_BudgetOnly | 420 | 210 | 0.176460 | 0.176460 | 0.179343 | 0.002883 | PASS |
| Original | B_BudgetOnly | 420 | 365 | 0.296211 | 0.298397 | 0.302356 | 0.006146 | PASS |
| Original | B_BudgetOnly | 420 | 420 | 0.329099 | 0.341009 | 0.346006 | 0.016908 | PASS |
| Original | D_JointPlanning | 210 | 90 | 0.082284 | 0.082284 | 0.084106 | 0.001823 | PASS |
| Original | D_JointPlanning | 210 | 210 | 0.165616 | 0.177922 | 0.179343 | 0.013726 | PASS |
| Original | D_JointPlanning | 365 | 90 | 0.082284 | 0.082284 | 0.084106 | 0.001823 | PASS |
| Original | D_JointPlanning | 365 | 210 | 0.177922 | 0.177922 | 0.179343 | 0.001420 | PASS |
| Original | D_JointPlanning | 365 | 365 | 0.287227 | 0.300708 | 0.302356 | 0.015130 | PASS |
| Original | D_JointPlanning | 420 | 90 | 0.082284 | 0.082284 | 0.084106 | 0.001823 | PASS |
| Original | D_JointPlanning | 420 | 210 | 0.177922 | 0.177922 | 0.179343 | 0.001420 | PASS |
| Original | D_JointPlanning | 420 | 365 | 0.298358 | 0.300708 | 0.302356 | 0.003998 | PASS |
| Original | D_JointPlanning | 420 | 420 | 0.331232 | 0.343539 | 0.346006 | 0.014774 | PASS |
| Week1_v2 | B_BudgetOnly | 210 | 90 | 0.082090 | 0.082090 | 0.084831 | 0.002741 | PASS |
| Week1_v2 | B_BudgetOnly | 210 | 210 | 0.165307 | 0.177562 | 0.180314 | 0.015007 | PASS |
| Week1_v2 | B_BudgetOnly | 365 | 90 | 0.082090 | 0.082090 | 0.084831 | 0.002741 | PASS |
| Week1_v2 | B_BudgetOnly | 365 | 210 | 0.177562 | 0.177562 | 0.180314 | 0.002753 | PASS |
| Week1_v2 | B_BudgetOnly | 365 | 365 | 0.286695 | 0.300106 | 0.303646 | 0.016951 | PASS |
| Week1_v2 | B_BudgetOnly | 420 | 90 | 0.082090 | 0.082090 | 0.084831 | 0.002741 | PASS |
| Week1_v2 | B_BudgetOnly | 420 | 210 | 0.177562 | 0.177562 | 0.180314 | 0.002753 | PASS |
| Week1_v2 | B_BudgetOnly | 420 | 365 | 0.297794 | 0.300106 | 0.303646 | 0.005852 | PASS |
| Week1_v2 | B_BudgetOnly | 420 | 420 | 0.330614 | 0.342869 | 0.347409 | 0.016795 | PASS |
| Week1_v2 | D_JointPlanning | 210 | 90 | 0.082931 | 0.082931 | 0.084831 | 0.001901 | PASS |
| Week1_v2 | D_JointPlanning | 210 | 210 | 0.166479 | 0.179082 | 0.180314 | 0.013835 | PASS |
| Week1_v2 | D_JointPlanning | 365 | 90 | 0.082931 | 0.082931 | 0.084831 | 0.001901 | PASS |
| Week1_v2 | D_JointPlanning | 365 | 210 | 0.179082 | 0.179082 | 0.180314 | 0.001232 | PASS |
| Week1_v2 | D_JointPlanning | 365 | 365 | 0.288713 | 0.302548 | 0.303646 | 0.014934 | PASS |
| Week1_v2 | D_JointPlanning | 420 | 90 | 0.082931 | 0.082931 | 0.084831 | 0.001901 | PASS |
| Week1_v2 | D_JointPlanning | 420 | 210 | 0.179082 | 0.179082 | 0.180314 | 0.001232 | PASS |
| Week1_v2 | D_JointPlanning | 420 | 365 | 0.300084 | 0.302548 | 0.303646 | 0.003562 | PASS |
| Week1_v2 | D_JointPlanning | 420 | 420 | 0.332958 | 0.345561 | 0.347409 | 0.014451 | PASS |

## 4. $C_r \in \{30, 50, 100, 200\}$ Clipping Sensitivity Ablation (Fixed Week1_v2 Budget $d$)

| $C_r$ | Isotropic $\eta$ | $\sigma_r$ [Analytic] | Clipped User-Days (%) | Clipping Bias (Mean $\pm$ 95% CI) | DP Mechanism Noise RMSE (Mean $\pm$ 95% CI) | Postproc RMSE | Total End-to-End RMSE (Mean $\pm$ 95% CI) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.900000 | 1030.18 | 13.59% | -2866.45 $\pm$ 2.58 | 1022.16 $\pm$ 9.21 | 1022.16 | **3337.54 $\pm$ 9.92** |
| 50 | 0.961500 | 1661.12 | 5.37% | -1439.56 $\pm$ 0.18 | 1703.37 $\pm$ 4.98 | 1697.60 | **2346.22 $\pm$ 23.90** |
| 100 | 0.990100 | 3273.98 | 1.05% | -417.07 $\pm$ 2.80 | 3193.06 $\pm$ 19.99 | 3131.48 | **3175.34 $\pm$ 22.78** |
| 200 | 0.997500 | 6523.60 | 0.13% | -81.99 $\pm$ 0.92 | 6515.98 $\pm$ 64.64 | 6197.14 | **6195.86 $\pm$ 35.92** |

## 5. Figures
- [fig1_420d_cross_period_ledger_trajectory.png](figures/fig1_420d_cross_period_ledger_trajectory.png)
- [fig2_cr_clipping_pareto_ablation.png](figures/fig2_cr_clipping_pareto_ablation.png)
- [fig3_420d_distinct_mape_comparison.png](figures/fig3_420d_distinct_mape_comparison.png)