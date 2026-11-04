# SDG-MH (Sealed Discrete Gaussian - Multi-Horizon) Experiment Reproduction Guide

This directory contains the complete experimental suite for evaluating **SDG-MH** against state-of-the-art differential privacy baselines in enterprise LLM analytics.

---

## 1. Environment & Setup

- **Operating System**: Windows (PowerShell)
- **Python Version**: Python 3.14.0 (virtual environment at `./.venv`)
- **Key Dependencies**: `numpy>=1.26.0`, `scipy>=1.18.0`, `pandas>=2.2.0`, `matplotlib>=3.8.0`, `pytest>=8.0.0`
- **Timezone**: UTC strictly across all components (沿用现有系统时区)
- **Security Boundaries**: No access to real corporate telemetry; all synthetic generators parameterized according to enterprise specifications. Direct exact data reads (`include_exact=True`) are strictly disabled across all experiments.

Activate the environment or use the direct interpreter:
```powershell
# Virtual environment Python executable
.\.venv\Scripts\python.exe
```

---

## 2. Test Suite Execution (Unit Tests & Invariants)

Run core invariant tests covering discrete Gaussian sampling, ledger closed-form agreements, revisit gate, raw data isolation, and display suppression rules:

```powershell
.\.venv\Scripts\pytest.exe tests/sdg_mh/ -v
```

Expected output: `12 passed in ~0.8s`.

---

## 3. End-to-End Experiment Execution Order

To reproduce all tables, figures, and CSVs in `results/sdg_mh/` from scratch, execute the scripts in the following order:

### Step 1: Baseline Calibration (Calibers a & b)
Calculates exact noise scaling multipliers $\beta_a$ ($\epsilon_{30}$-matched) and $\beta_b$ ($\rho(h) \le \rho_{\text{sdg}}(h) \ \forall h$) for all 11 mechanisms:
```powershell
.\.venv\Scripts\python.exe experiments/sdg_mh/run_calibration.py
```
*Outputs*: `results/sdg_mh/baseline_calibration_table.csv`

---

### Step 2: RQ3 Binary Tree Crossover Analysis
Verifies reference table replication ($T=512, L=8$), computes analytical efficiency ratios across all $(d, w)$, audits 300 random layer noise tree configurations, and generates heatmaps:
```powershell
.\.venv\Scripts\python.exe experiments/sdg_mh/rq3_binary_tree_crossover.py
```
*Outputs*:
- `results/sdg_mh/rq3_reference_check.csv`
- `results/sdg_mh/rq3_variance_ratios.csv`
- `results/sdg_mh/rq3_linf_ratios.csv`
- `results/sdg_mh/rq3_efficiency_ratios.csv`
- `results/sdg_mh/rq3_random_300_audit.csv`
- `results/sdg_mh/fig_rq3_variance_ratio_heatmap.png`
- `results/sdg_mh/fig_rq3_linf_ratio_heatmap.png`

---

### Step 3: RQ4 Distinct Channel Calendar Sweep & Trade-Offs
Sweeps refresh steps $s \in \{1, 3, 7, 14, 30, 90\}$, evaluates Calendars A and B across 30 trials, and generates trade-off curves:
```powershell
.\.venv\Scripts\python.exe experiments/sdg_mh/rq4_distinct_refresh_sweep.py
```
*Outputs*:
- `results/sdg_mh/rq4_distinct_sweep_results.csv`
- `results/sdg_mh/rq4_calendar_comparison_trials.csv`
- `results/sdg_mh/fig_rq4_refresh_tradeoff.png`

---

### Step 4: RQ5 Query Reuse & Dashboard Traffic
Simulates 180 days of realistic dashboard clickstream traffic (89,497 query clicks) comparing zero-cost sealed post-processing against on-demand query budgeting:
```powershell
.\.venv\Scripts\python.exe experiments/sdg_mh/rq5_query_reuse.py
```
*Outputs*:
- `results/sdg_mh/rq5_query_reuse_cumulative.csv`
- `results/sdg_mh/fig_rq5_cumulative_privacy_loss.png`

---

### Step 5: RQ6 Workforce Turnover & Lifetime Exposure
Simulates 3-year employee lifecycle (15% annual turnover, 5% rejoin rate) evaluating per-individual cumulative privacy loss under SDG-MH vs Epoch90 global accumulation:
```powershell
.\.venv\Scripts\python.exe experiments/sdg_mh/rq6_workforce_turnover.py
```
*Outputs*:
- `results/sdg_mh/rq6_individual_turnover_exposure.csv`
- `results/sdg_mh/rq6_tenure_exposure_summary.csv`
- `results/sdg_mh/fig_rq6_turnover_exposure.png`

---

### Step 6: RQ2 10-Year Long-Term Accuracy Stability
Evaluates 10-year longitudinal interaction volume error across turnover rates {5%, 15%, 30%} (30 trials each):
```powershell
.\.venv\Scripts\python.exe experiments/sdg_mh/rq2_long_term_stability.py
```
*Outputs*:
- `results/sdg_mh/rq2_long_term_stability.csv`
- `results/sdg_mh/fig_rq2_long_term_stability.png`

---

### Step 7: RQ1 Comprehensive Accuracy Benchmark
Runs 30 Monte Carlo trials across 11 mechanisms $\times$ 2 calibers $\times$ 6 query windows (7d, 30d, 90d, 365d, Month 1, Q1), reporting MAPE, CI coverage, prevalence error, Top-K NDCG@10, and truncation bias:
```powershell
.\.venv\Scripts\python.exe experiments/sdg_mh/rq1_accuracy_benchmark.py
```
*Outputs*:
- `results/sdg_mh/rq1_accuracy_trials.csv` (3,960 trials)
- `results/sdg_mh/rq1_accuracy_summary.csv`
- `results/sdg_mh/rq1_clipping_statistics.csv`
- `results/sdg_mh/fig_rq1_requests_error_by_window.png`
- `results/sdg_mh/fig_rq1_ci_coverage.png`
- `results/sdg_mh/fig_rq1_clipping_comparison.png`
- `results/sdg_mh/fig_rq1_topk_ndcg.png`

---

### Step 8: Formal Correctness Checks (Checks 1 to 4)
Runs automated verification of the 4 required mathematical invariants and security gates:
```powershell
.\.venv\Scripts\python.exe experiments/sdg_mh/verify_formal_checks.py
```
*Outputs*: `results/sdg_mh/formal_checks.csv`

---

### Step 9: Engineering Metrics & Precomputed Display Rules
Measures SealJob daily duration, annual storage footprint, query latency over 10,000 requests, and evaluates confidence interval display suppression rules:
```powershell
.\.venv\Scripts\python.exe experiments/sdg_mh/measure_engineering_metrics.py
```
*Outputs*:
- `results/sdg_mh/engineering_metrics.csv`
- `results/sdg_mh/display_rules_table.csv`

---

## 4. One-Click Reproduction Script

To run the entire pipeline sequentially:
```powershell
.\.venv\Scripts\python.exe -c "
import subprocess, sys

scripts = [
    'experiments/sdg_mh/run_calibration.py',
    'experiments/sdg_mh/rq3_binary_tree_crossover.py',
    'experiments/sdg_mh/rq4_distinct_refresh_sweep.py',
    'experiments/sdg_mh/rq5_query_reuse.py',
    'experiments/sdg_mh/rq6_workforce_turnover.py',
    'experiments/sdg_mh/rq2_long_term_stability.py',
    'experiments/sdg_mh/rq1_accuracy_benchmark.py',
    'experiments/sdg_mh/verify_formal_checks.py',
    'experiments/sdg_mh/measure_engineering_metrics.py',
]

for s in scripts:
    print(f'=== Running {s} ===')
    subprocess.run([sys.executable, s], check=True)
print('ALL EXPERIMENTS REPRODUCED SUCCESSFULLY.')
"
```
