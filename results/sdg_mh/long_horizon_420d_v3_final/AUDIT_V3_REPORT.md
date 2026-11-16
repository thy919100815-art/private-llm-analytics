# SDG-MH Final Consistency Audit v3 Report (Post-PR #6)

- **Run Mode**: `FULL_AUDIT_V3`
- **Design**: `2000` initial users, `420` days, **`K=8` independent data seeds** (`[42, 1042, 2042, 3042, 4042, 5042, 6042, 7042]`) $\times$ `5` CRN MC trials (`df=7`)
- **128-bit Seed Collision Audit**: `50520` seeds checked (including `CalendarJob +500003` task offset) -> **`50520` unique, `0` collisions**
- **Total Wall Time**: `876.72s` (`431.70s`/seed avg across parallel workers)

## 1. Optimization Cost & Directed-Rounding Dual Certificate (`DECIMAL100_DIRECTED_ROUNDING`)

> **Assumption Clarification**: `DECIMAL100_DIRECTED_ROUNDING` bounds all dual-function rounding using `prec=100` `ROUND_FLOOR`/`ROUND_CEILING` relative to the operational `float64` RHS curve `rho_bars` (plus exact rational `1/7, 1/30` lower bounds), conditional on active constraint representation and $c_j \ge 0, l_j > 0$. `CONSERVATIVE_BACKOFF_REL = 1e-11` ensures primal feasibility in both IEEE-754 and real arithmetic.

| Workload | Curve | Group | Phases | Solve Time | Primal $f^*$ | Dual LB (`Rel Gap`) | Tier |
|---|---|---|---|---:|---:|---:|:---:|
| Default | Week1_v2 | A_Baseline | `(0, 0, 0)` | 0.024s | 2047.7860 | -inf (`inf`) | `FEASIBLE_CERTIFIED` |
| Default | Week1_v2 | D_JointPlanning | `(0, 0, 0)` | 1.540s | 1433.9182 | 1433.9182 (`2.04e-11`) | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` |
| DistinctHeavy | Week1_v2 | B_BudgetOnly | `(0, 0, 0)` | 0.026s | 932.6324 | 932.6324 (`1.00e-11`) | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` |
| DistinctHeavy | Week1_v2 | D_JointPlanning | `(0, 4, 0)` | 1.576s | 922.9791 | 922.9791 (`7.45e-09`) | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` |
| DistinctHeavy | Original | B_BudgetOnly | `(0, 0, 0)` | 0.028s | 940.8881 | 940.8881 (`1.00e-11`) | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` |
| DistinctHeavy | Original | D_JointPlanning | `(0, 4, 0)` | 1.661s | 930.3260 | 930.3260 (`1.02e-11`) | `NEAR_GLOBAL_OPTIMAL_CERTIFIED` |

## 2. Same-Query-Date Comparison (`B_BudgetOnly` vs `D_JointPlanning`, CRN + Holm-Bonferroni + Wilcoxon)

- **Regime A (`rel_*`)**: Evaluated on each schedule's own release dates (0 staleness; for `30d`, B uses `day%7==0` Sunday while D uses `day%7==4` Thursday).
- **Regime B (`common_daily_*`)**: Evaluated on **identical daily dashboard query dates** (`w=7`: `7..420`, `w=30`: `35..420`, `w=90`: `90..420`) against **identical ground truth `gt(w,d)`**, cleanly isolating latest-available served error vs mean staleness (`~3.0d` for 7d/30d).
- **Regime C (`common_sunday_*`)**: Evaluated on **identical fixed Sunday report dates (`d%7==0, d in 35..420`)**, isolating phase-shift staleness lag (`B: 0.0d` vs `D: 3.0d` for `30d`).

| Workload | Curve | Contrast | Metric | Base Mean | Trt Mean | Paired Diff (`95% CI`) | Raw $p_t$ | Holm $p_{\text{holm}}$ | Exact Wilcoxon $p$ | Seeds `<0` |
|---|---|---|---|---:|---:|---:|---:|---:|---:|:---:|
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `requests_1d_total_rmse` | 2577.9054 | 2587.7536 | **+9.8482** ($\pm$0.8047) | `0.0000` | `0.0000` | `0.0078` | `0/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `failures_1d_total_rmse` | 1389.5270 | 1397.7640 | **+8.2370** ($\pm$0.2539) | `0.0000` | `0.0000` | `0.0078` | `0/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `rel_7d_mape_pct` | 2.8137 | 2.7866 | **-0.0271** ($\pm$0.0049) | `0.0000` | `0.0000` | `0.0078` | `8/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `rel_30d_mape_pct` | 2.4907 | 2.4666 | **-0.0241** ($\pm$0.0835) | `0.5166` | `1.0000` | `0.6406` | `5/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `rel_90d_mape_pct` | 2.6957 | 2.6096 | **-0.0861** ($\pm$0.0119) | `0.0000` | `0.0000` | `0.0078` | `8/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `common_daily_7d_served_mape` | 2.9309 | 2.9039 | **-0.0270** ($\pm$0.0042) | `0.0000` | `0.0000` | `0.0078` | `8/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `common_daily_30d_served_mape` | 2.4919 | 2.4748 | **-0.0171** ($\pm$0.0800) | `0.6293` | `1.0000` | `0.7422` | `5/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `common_daily_30d_served_rmse` | 61.3500 | 61.2399 | **-0.1101** ($\pm$1.2732) | `0.8437` | diag | `0.8438` | `3/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `common_daily_30d_drift_rmse` | 6.2368 | 4.9317 | **-1.3051** ($\pm$0.1750) | `0.0000` | diag | `0.0078` | `8/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `common_daily_30d_staleness_d` | 2.9922 | 3.0000 | **+0.0078** ($\pm$0.0000) | `1.0000` | diag | `0.0078` | `0/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `common_daily_90d_served_mape` | 2.6961 | 2.6070 | **-0.0891** ($\pm$0.0092) | `0.0000` | `0.0000` | `0.0078` | `8/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `common_sunday_30d_served_mape` | 2.4907 | 2.4902 | **-0.0005** ($\pm$0.0826) | `0.9886` | `1.0000` | `0.8438` | `3/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `common_sunday_30d_staleness_d` | 0.0000 | 3.0000 | **+3.0000** ($\pm$0.0000) | `1.0000` | diag | `0.0078` | `0/8` |
| DistinctHeavy | Week1_v2 | `D_JointPlanning - B_BudgetOnly` | `common_sunday_30d_drift_rmse` | 0.0000 | 7.5459 | **+7.5459** ($\pm$0.3936) | `0.0000` | diag | `0.0078` | `0/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `requests_1d_total_rmse` | 2569.6445 | 2582.8758 | **+13.2313** ($\pm$0.8956) | `0.0000` | `0.0000` | `0.0078` | `0/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `failures_1d_total_rmse` | 1382.5630 | 1393.8386 | **+11.2757** ($\pm$0.3701) | `0.0000` | `0.0000` | `0.0078` | `0/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `rel_7d_mape_pct` | 2.8185 | 2.7936 | **-0.0248** ($\pm$0.0050) | `0.0000` | `0.0001` | `0.0078` | `8/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `rel_30d_mape_pct` | 2.5194 | 2.4895 | **-0.0299** ($\pm$0.0742) | `0.3721` | `1.0000` | `0.5469` | `5/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `rel_90d_mape_pct` | 2.7815 | 2.6771 | **-0.1044** ($\pm$0.0187) | `0.0000` | `0.0000` | `0.0078` | `8/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `common_daily_7d_served_mape` | 2.9354 | 2.9090 | **-0.0265** ($\pm$0.0045) | `0.0000` | `0.0000` | `0.0078` | `8/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `common_daily_30d_served_mape` | 2.5202 | 2.4971 | **-0.0231** ($\pm$0.0710) | `0.4671` | `1.0000` | `0.5469` | `5/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `common_daily_30d_served_rmse` | 62.0662 | 61.8146 | **-0.2516** ($\pm$1.1008) | `0.6057` | diag | `0.4609` | `5/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `common_daily_30d_drift_rmse` | 6.2368 | 4.9317 | **-1.3051** ($\pm$0.1750) | `0.0000` | diag | `0.0078` | `8/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `common_daily_30d_staleness_d` | 2.9922 | 3.0000 | **+0.0078** ($\pm$0.0000) | `1.0000` | diag | `0.0078` | `0/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `common_daily_90d_served_mape` | 2.7796 | 2.6759 | **-0.1037** ($\pm$0.0179) | `0.0000` | `0.0000` | `0.0078` | `8/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `common_sunday_30d_served_mape` | 2.5194 | 2.5121 | **-0.0073** ($\pm$0.0739) | `0.8223` | `1.0000` | `0.8438` | `4/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `common_sunday_30d_staleness_d` | 0.0000 | 3.0000 | **+3.0000** ($\pm$0.0000) | `1.0000` | diag | `0.0078` | `0/8` |
| DistinctHeavy | Original | `D_JointPlanning - B_BudgetOnly` | `common_sunday_30d_drift_rmse` | 0.0000 | 7.5459 | **+7.5459** ($\pm$0.3936) | `0.0000` | diag | `0.0078` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `requests_1d_total_rmse` | 2802.0587 | 2314.5460 | **-487.5127** ($\pm$12.7569) | `0.0000` | `0.0000` | `0.0078` | `8/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `failures_1d_total_rmse` | 1572.9925 | 1159.3726 | **-413.6198** ($\pm$12.8118) | `0.0000` | `0.0000` | `0.0078` | `8/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `rel_7d_mape_pct` | 2.8584 | 3.5646 | **+0.7062** ($\pm$0.0197) | `0.0000` | `0.0000` | `0.0078` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `rel_30d_mape_pct` | 2.4184 | 3.0155 | **+0.5971** ($\pm$0.0162) | `0.0000` | `0.0000` | `0.0078` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `rel_90d_mape_pct` | 2.3478 | 2.5492 | **+0.2015** ($\pm$0.0188) | `0.0000` | `0.0000` | `0.0078` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `common_daily_7d_served_mape` | 2.9726 | 3.6571 | **+0.6845** ($\pm$0.0196) | `0.0000` | `0.0000` | `0.0078` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `common_daily_30d_served_mape` | 2.4198 | 3.0128 | **+0.5930** ($\pm$0.0163) | `0.0000` | `0.0000` | `0.0078` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `common_daily_30d_served_rmse` | 59.5641 | 74.1733 | **+14.6091** ($\pm$0.2953) | `0.0000` | diag | `0.0078` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `common_daily_30d_drift_rmse` | 6.2368 | 6.2368 | **+0.0000** ($\pm$0.0000) | `1.0000` | diag | `1.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `common_daily_30d_staleness_d` | 2.9922 | 2.9922 | **+0.0000** ($\pm$0.0000) | `1.0000` | diag | `1.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `common_daily_90d_served_mape` | 2.3506 | 2.5471 | **+0.1965** ($\pm$0.0181) | `0.0000` | `0.0000` | `0.0078` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `common_sunday_30d_served_mape` | 2.4184 | 3.0155 | **+0.5971** ($\pm$0.0162) | `0.0000` | `0.0000` | `0.0078` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `common_sunday_30d_staleness_d` | 0.0000 | 0.0000 | **+0.0000** ($\pm$0.0000) | `1.0000` | diag | `1.0000` | `0/8` |
| Default | Week1_v2 | `D_JointPlanning - A_Baseline` | `common_sunday_30d_drift_rmse` | 0.0000 | 0.0000 | **+0.0000** ($\pm$0.0000) | `1.0000` | diag | `1.0000` | `0/8` |

## 3. Uniformly Post-Processed Competitive Baselines (Identical `[1, 420]` Additive Scope, `C_r=50, C_f=10`)

> **Finite vs Infinite Scope Note**: Performance comparison is strictly restricted to the **shared `[1, 420]` finite span**, **shared remaining additive Multi-Horizon curve $\bar\rho_{\text{add}}(h)$ on $h \in [1, 420]$**, and **shared additive rolling queries $w \in \{1,7,30,90\}$**. `SDG-MH` simultaneously holds an **infinite-horizon ($h \ge 1$) certificate**, whereas `Tree-2l / Tree-U (L=9)` covers only finite $T \le 512$ and `W-BD-30` is calibrated over `[1, 420]`. Non-additive rolling Distinct queries remain `N/A` for baselines.

| Method | Privacy Certificate Scope | Req 1d RMSE | Req 7d RMSE | Req 30d RMSE | Req 90d RMSE | Fail 1d RMSE | Fail 30d RMSE | Fail 90d RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **SDG-MH (D_JointPlanning)** | Infinite h >= 1 (evaluated [1, 420]) | 2314.5$\pm$32.2 | 11034.3$\pm$261.3 | 43866.8$\pm$1247.3 | 126664.8$\pm$4104.2 | 1159.4$\pm$28.9 | 7468.4$\pm$583.1 | 14003.3$\pm$991.9 |
| **Tree-2l (MH-RemEnv)** | Finite Dyadic T <= 512 (calibrated [1, 420]) | 4378.6$\pm$64.2 | 12716.7$\pm$176.4 | 44587.3$\pm$924.4 | 126186.5$\pm$3209.9 | 2724.1$\pm$51.1 | 9467.8$\pm$498.9 | 16154.0$\pm$1224.1 |
| **Tree-2l (Daily-d Parity)** | Finite Dyadic T <= 512 (calibrated [1, 420]) | 4407.0$\pm$64.8 | 12753.4$\pm$176.6 | 44612.6$\pm$927.6 | 126197.5$\pm$3223.4 | 2743.0$\pm$51.2 | 9530.3$\pm$504.1 | 16253.4$\pm$1233.7 |
| **Tree-U (MH-RemEnv)** | Finite Dyadic T <= 512 (calibrated [1, 420]) | 27907.0$\pm$528.7 | 49855.3$\pm$601.3 | 72369.7$\pm$1298.6 | 138543.3$\pm$3926.2 | 14142.6$\pm$284.3 | 40923.8$\pm$1438.6 | 45397.7$\pm$2852.4 |
| **W-BD-30 (MH-Calibrated)** | Rolling Adaptive Finite [1, 420] envelope | 3649.9$\pm$198.6 | 15780.3$\pm$1294.4 | 51601.9$\pm$4170.4 | 133714.2$\pm$4199.2 | 1278.4$\pm$86.4 | 14172.9$\pm$2672.6 | 24464.7$\pm$3159.7 |

## 4. Recomputed Competitive Baseline Diff: Frozen v2 (PR #6) vs Revised v3 (Unified Post-Processing + 128-bit Seeds)

| Method | Metric | Frozen v2 Mean | Revised v3 Mean | Diff (`v3 - v2`) | Rel Change |
|---|---|---:|---:|---:|---:|
| SDG-MH (D_JointPlanning) | `requests_1d_rmse` | 2310.52 | 2314.55 | **+4.03** | +0.17% |
| SDG-MH (D_JointPlanning) | `requests_7d_rmse` | 10919.23 | 11034.31 | **+115.08** | +1.05% |
| SDG-MH (D_JointPlanning) | `requests_30d_rmse` | 43235.69 | 43866.77 | **+631.08** | +1.46% |
| SDG-MH (D_JointPlanning) | `requests_90d_rmse` | 125238.08 | 126664.80 | **+1426.72** | +1.14% |
| SDG-MH (D_JointPlanning) | `failures_1d_rmse` | 1176.56 | 1159.37 | **-17.18** | -1.46% |
| SDG-MH (D_JointPlanning) | `failures_7d_rmse` | 3366.74 | 3244.51 | **-122.23** | -3.63% |
| SDG-MH (D_JointPlanning) | `failures_30d_rmse` | 7667.43 | 7468.41 | **-199.03** | -2.60% |
| SDG-MH (D_JointPlanning) | `failures_90d_rmse` | 14345.86 | 14003.27 | **-342.58** | -2.39% |
| Tree-2l (MH-RemEnv) | `requests_1d_rmse` | 4378.64 | 4378.56 | **-0.08** | -0.00% |
| Tree-2l (MH-RemEnv) | `requests_7d_rmse` | 12735.96 | 12716.70 | **-19.26** | -0.15% |
| Tree-2l (MH-RemEnv) | `requests_30d_rmse` | 45308.80 | 44587.25 | **-721.55** | -1.59% |
| Tree-2l (MH-RemEnv) | `requests_90d_rmse` | 129162.86 | 126186.47 | **-2976.39** | -2.30% |
| Tree-2l (MH-RemEnv) | `failures_1d_rmse` | 3009.76 | 2724.12 | **-285.65** | -9.49% |
| Tree-2l (MH-RemEnv) | `failures_7d_rmse` | 5507.10 | 5409.28 | **-97.83** | -1.78% |
| Tree-2l (MH-RemEnv) | `failures_30d_rmse` | 9682.61 | 9467.79 | **-214.82** | -2.22% |
| Tree-2l (MH-RemEnv) | `failures_90d_rmse` | 16704.47 | 16153.98 | **-550.49** | -3.29% |
| Tree-2l (Daily-d Parity) | `requests_1d_rmse` | 4406.89 | 4406.96 | **+0.07** | +0.00% |
| Tree-2l (Daily-d Parity) | `requests_7d_rmse` | 12771.78 | 12753.38 | **-18.41** | -0.14% |
| Tree-2l (Daily-d Parity) | `requests_30d_rmse` | 45338.49 | 44612.61 | **-725.87** | -1.60% |
| Tree-2l (Daily-d Parity) | `requests_90d_rmse` | 129195.64 | 126197.53 | **-2998.11** | -2.32% |
| Tree-2l (Daily-d Parity) | `failures_1d_rmse` | 3033.47 | 2742.97 | **-290.50** | -9.58% |
| Tree-2l (Daily-d Parity) | `failures_7d_rmse` | 5548.99 | 5450.35 | **-98.65** | -1.78% |
| Tree-2l (Daily-d Parity) | `failures_30d_rmse` | 9748.67 | 9530.30 | **-218.37** | -2.24% |
| Tree-2l (Daily-d Parity) | `failures_90d_rmse` | 16817.41 | 16253.40 | **-564.01** | -3.35% |
| Tree-U (MH-RemEnv) | `requests_1d_rmse` | 27742.39 | 27907.01 | **+164.62** | +0.59% |
| Tree-U (MH-RemEnv) | `requests_7d_rmse` | 50365.74 | 49855.26 | **-510.48** | -1.01% |
| Tree-U (MH-RemEnv) | `requests_30d_rmse` | 72781.34 | 72369.70 | **-411.65** | -0.57% |
| Tree-U (MH-RemEnv) | `requests_90d_rmse` | 143671.18 | 138543.26 | **-5127.93** | -3.57% |
| Tree-U (MH-RemEnv) | `failures_1d_rmse` | 26822.80 | 14142.56 | **-12680.24** | -47.27% |
| Tree-U (MH-RemEnv) | `failures_7d_rmse` | 36535.19 | 29784.86 | **-6750.32** | -18.48% |
| Tree-U (MH-RemEnv) | `failures_30d_rmse` | 42092.51 | 40923.85 | **-1168.66** | -2.78% |
| Tree-U (MH-RemEnv) | `failures_90d_rmse` | 46870.99 | 45397.68 | **-1473.31** | -3.14% |
| W-BD-30 (MH-Calibrated) | `requests_1d_rmse` | 3565.58 | 3649.86 | **+84.28** | +2.36% |
| W-BD-30 (MH-Calibrated) | `requests_7d_rmse` | 15476.96 | 15780.27 | **+303.31** | +1.96% |
| W-BD-30 (MH-Calibrated) | `requests_30d_rmse` | 49154.31 | 51601.92 | **+2447.61** | +4.98% |
| W-BD-30 (MH-Calibrated) | `requests_90d_rmse` | 125582.16 | 133714.24 | **+8132.09** | +6.48% |
| W-BD-30 (MH-Calibrated) | `failures_1d_rmse` | 1315.44 | 1278.37 | **-37.07** | -2.82% |
| W-BD-30 (MH-Calibrated) | `failures_7d_rmse` | 6445.68 | 5877.19 | **-568.50** | -8.82% |
| W-BD-30 (MH-Calibrated) | `failures_30d_rmse` | 20410.14 | 14172.86 | **-6237.27** | -30.56% |
| W-BD-30 (MH-Calibrated) | `failures_90d_rmse` | 53110.64 | 24464.65 | **-28645.98** | -53.94% |

## 5. Figures & Reproducible Commands
```powershell
# Smoke:
.\.venv\Scripts\python.exe experiments/sdg_mh/run_long_horizon_420d_audit_v3.py --smoke
# Full (K=8 seeds x 5 CRN MC = 40 runs/config):
.\.venv\Scripts\python.exe experiments/sdg_mh/run_long_horizon_420d_audit_v3.py --full
```
- [fig1_v3_same_query_date_d_vs_b.png](figures/fig1_v3_same_query_date_d_vs_b.png)
- [fig2_v3_uniform_postproc_baselines.png](figures/fig2_v3_uniform_postproc_baselines.png)