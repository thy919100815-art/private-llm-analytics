# Banded / Toeplitz Matrix Factorization Competitive Baseline Report (v4)

## 1. Mechanism Specification & Literature Grounding
We implement `BandedToeplitzBaseline` in `src/private_llm_analytics/sdg_mh/baselines.py` following:
- **Fichtenberger, Henzinger & Upadhyay (ICML 2023)**, *Constant Matters: Fine-grained Error Bound on Differentially Private Continual Observation*: canonical lower-triangular Toeplitz strategy matrix with generating function $(1 - z)^{-\alpha}$ ($\alpha = 0.50$), $f_0 = 1, f_k = f_{k-1}(2k-1)/(2k)$.
- **Choquette-Choo et al. (NeurIPS 2023)** & **Denisov et al. (NeurIPS 2022)**: $b$-banded lower-triangular Toeplitz strategy matrix $\mathbf{C}^{(b)} \in \mathbb{R}^{T \times T}$ where $C_{t,\tau} = f_{t-\tau}$ for $0 \le t - \tau < b$ and $0$ otherwise.

We evaluate two bandwidth/exponent configurations over the exact 420-day horizon ($T = 420$, $K = 8$ data seeds $\times 5$ CRN trials, $C_r = 50, C_f = 10$):
1. `Banded-MF (b=30, MH-RemEnv)`: bandwidth $b = 30$ (matching the 30-day rolling window scale), canonical exponent $\alpha = 0.50$.
2. `Banded-MF (b=8, MH-RemEnv)`: short-band Toeplitz factorization with $b = 8$, $\alpha = 0.25$.

## 2. Multi-Horizon User-Time DP Calibration vs. Naive Single-Day Calibration
Under $(u, I)$-User-Time Adjacency over any contiguous $h$-day interval $I = [s, s+h-1]$, a user active throughout $I$ perturbs all $h$ columns of $\mathbf{C}^{(b)}$ in $I$ coherently:
$$\Delta_2^2(h) = C_{\text{clip}}^2 \max_{0 \le s \le T-h} \mathbf{1}_I^\top (\mathbf{C}^{(b)\top} \mathbf{C}^{(b)}) \mathbf{1}_I.$$
- For single-day participation ($h=1$), the column norm squared is $\sum_{k=0}^{b-1} f_k^2 = O(\log b)$.
- For multi-day participation ($h \gg b$), coherent cross-day contributions sum before squaring: $\mathbf{1}_I^\top (\mathbf{C}^{(b)\top} \mathbf{C}^{(b)}) \mathbf{1}_I \approx h \cdot \bigl(\sum_{k=0}^{b-1} f_k\bigr)^2$.
- **Naive $h=1$ Calibration Violation**: Calibrating $\mathbf{C}^{(b)}$ solely at $h=1$ against $\bar{\rho}_{\text{add}}(1)$ violates the 365-day Multi-Horizon commitment $\bar{\rho}_{\text{add}}(365)$ by **`275.86x`** for `Banded-MF (b=30, alpha=0.50)` and **`47.74x`** for `Banded-MF (b=8, alpha=0.25)`.
- **Valid Multi-Horizon Calibration (`MH-RemEnv`)**: Enforcing $\rho_{\text{MF}}(h) \le \bar{\rho}_{\text{add}}(h)$ for all $h \in [1, 420]$ binds at `tight_h = 394` and scales $\sigma_{\text{mf}}$ by $\sum_{k=0}^{b-1} f_k$, satisfying Theorem 7(b) ($\text{Var}(\mathbf{1}_w^\top \hat{\mathbf{x}}) \ge w^2 C_{\text{clip}}^2 / (2\rho(w))$) everywhere.

## 3. Empirical Results on the Frozen v3 420-Day Benchmark ($K=8$ Seeds $\times 5$ CRN Trials)

Banded-MF rows come from `banded_mf_competitive_summary.csv` / `banded_mf_systems_overhead.csv` (this run). SDG-MH, Tree-2l, and W-BD-30 RMSE rows are copied from the frozen v3 `long_horizon_420d_v3_final/audit_v3_competitive_baselines_summary.csv` (same workload, envelope, and post-processing). Systems overhead was measured only for SDG-MH and Banded-MF; "n/m" = not measured in this run.

| Mechanism | Horizon Scope | Supports Distinct | Preproc (ms) | Online 420d (ms) | State (B) | 1d Req RMSE | 7d Req RMSE | 30d Req RMSE | 90d Req RMSE | 30d Fail RMSE |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SDG-MH (D_Joint)` | Infinite $h \ge 1$ | Yes (7/30/90d) | 66.20 | 0.420 | 16 | 2314.5 | 11034.3 | 43866.8 | 126664.8 | 7468.4 |
| `Banded-MF (b=8, MH)` | Finite $T=420$ | No | 214.42 | 0.152 | 128 | 3446.9 | 11589.1 | 43975.5 | 126617.7 | 7829.9 |
| `Banded-MF (b=30, MH)` | Finite $T=420$ | No | 173.01 | 0.182 | 480 | 9919.5 | 18846.8 | 46369.0 | 127926.1 | 13248.0 |
| `Tree-2l (MH-RemEnv)` | Finite dyadic $T \le 512$ | No | n/m | n/m | n/m | 4378.6 | 12716.7 | 44587.3 | 126186.5 | 9467.8 |
| `W-BD-30 (MH-Calib)` | Rolling finite $[1,420]$ | No | n/m | n/m | n/m | 3649.9 | 15780.3 | 51601.9 | 133714.2 | 14172.9 |

**Unfavorable results kept:** at 90d, Banded-MF (b=8) is 0.04% below SDG-MH on Requests (126617.7 vs 126664.8), and on 90d Failures it is lower (13880.6 vs 14003.3). Tree-2l is lowest on 90d Requests (126186.5). In all three cases the confidence intervals overlap. On online execution time, Banded-MF (0.152–0.182 ms) is faster than SDG-MH (0.420 ms).
