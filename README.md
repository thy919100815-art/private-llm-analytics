# Private LLM Analytics (v0.1.7 — Final Gate Patch / Freeze Candidate)

A **User-Level Central Differential Privacy Analytics Layer** prototype for enterprise multi-model AI telemetry.

> [!IMPORTANT]
> **v0.1.7 Final Gate Patch & Freeze Candidate**:
> Version 0.1.7 completes the final correctness and distribution hardening gates for the v0.1.x series:
> - **Gate Fix A**: Low-level `ReleaseOrchestrator` unconditionally blocks external recipients in `EXPERIMENT` mode prior to planning, data access, reservation, noise generation, or commit (`RecipientSecurityViolation`), ensuring zero privacy or accounting side-effects.
> - **Gate Fix B**: `run_all_experiments.py` reproduces the intended $N=50,000$ main epsilon sweep with fail-fast population alignment (`active_users <= public_population`), plus `--smoke` mode for rapid, non-destructive verification.
> - **Gate Fix C & D**: Duplicate dataset aliases (`interactions_n10k.parquet`, `interactions_n50k.parquet`) removed; standardized on canonical full numeric naming (`interactions_n10000.parquet`, `interactions_n50000.parquet`). $N=100,000$ dataset excluded from distribution archive to maintain compact size.
> - **Gate Fix E**: Standard machine-parseable `data/checksums.sha256` in `<sha256>  <filename>` format covering included files.
> - **Gate Fix F**: `verify_zip_distribution.py` dynamically discovers and validates all 15 verification scripts and 159 unit/regression tests in an isolated clean temporary directory.
>
> **Security Boundaries**:
> - Experiment mode is restricted to trusted internal evaluation at every API layer, including the low-level `ReleaseOrchestrator`.
> - No external recipient may receive Experiment-mode releases.
> - Dataset content binding uses a full 256-bit digest over canonicalized hashed-row representation (`data_sha256_<64 hex chars>`).
>
> Longitudinal RDP composition, continual observation, dynamic taxonomy, adaptive clipping, Top-K, and Web UI remain strictly deferred to v0.2.

---

## 1. Problem Statement

Modern enterprises widely adopt multiple frontier Large Language Model (LLM) providers simultaneously (e.g., `provider_a`, `provider_b`, `provider_c`). Enterprise IT and AI governance teams need to understand workforce adoption patterns:
- Which providers and task domains are gaining adoption?
- What is the workforce prevalence across AI task categories?
- How does an employee's workflow distribute across different providers and capabilities?
- What is the total enterprise interaction volume per domain?

While the enterprise itself acts as a **trusted curator** with access to raw, detailed interaction telemetry logs, releasing statistical summaries to external recipients (such as model providers or market analysts) poses severe privacy risks:
- Exposing exact usage can reveal an individual employee's workflow, sensitive projects, or anomalous behavior.
- In add/remove user adjacency, leaking exact active user counts directly breaks user-level Differential Privacy.
- If external recipients can control or supply the random seed, they can reconstruct exact private aggregates via $q(D) = \widetilde{q}(D) - Z$.
- Concurrent worker processes racing to query the analytics layer could cause uncoordinated budget overruns.
- Legacy caches decoupled from authoritative accounting could return differential privacy payloads with zero spent budget.
- Experiments sharing databases or cache keys with production could allow attackers to recover private aggregates from known-seed experiment payloads.

**Private LLM Analytics** resolves these tensions by inserting an immutable, mathematically verified, transactionally persistent Differential Privacy layer between internal telemetry and external releases.

---

## 2. Threat Model & Privacy Framework

- **Trusted Curator**: The enterprise internal telemetry service collects interaction logs, performs user-level grouping, and runs contribution transformations.
- **External Recipient**: Third-party model providers (`provider_a`, `provider_b`, `provider_c`) or external evaluators who receive released aggregated statistics. External recipients are untrusted and must never receive exact data, private active user counts, competitor market shares, or the ability to influence DP randomness.
- **Privacy Model**: **Central Differential Privacy** (not Local DP).
- **Privacy Unit**: **User-level**. The privacy guarantee covers **all interactions of an employee within the designated statistical window** (e.g., 30 days).
- **Adjacency**: **Add/Remove User Adjacency**. Two neighboring datasets $D$ and $D'$ differ by the presence or absence of one entire user's interaction history in the statistical window:
  $$D \sim D' \iff D' = D \cup \{u\} \quad \text{or} \quad D' = D \setminus \{u\}$$
- **Public Population Roster**: Under add/remove adjacency, the active user count is sensitive private data. The system strictly mandates a fixed, public eligible employee universe ($N_{\text{public}} = 50,000$). Prevalence denominators strictly use $N_{\text{public}}$, and active user counts are strictly stripped from all external responses.

---

## 3. The Core Invariants of v0.1.7

$$\boxed{1.\ \text{External / Production caller cannot control DP randomness}}$$
- `AnalyticsService.compute()` strictly forbids any caller-supplied RNG, seed, or noise vector for **all** recipients (raising `ProductionRNGControlError`).
- Production DP releases strictly use secure unseeded internal entropy (`np.random.default_rng()`).
- Deterministic seed evaluation is isolated exclusively to `compute_experiment()` / `ExperimentRunner` for trusted internal recipients.

$$\boxed{2.\ \text{Experiment mode can never serve an external recipient}}$$
- Experiment mode is restricted to trusted internal evaluation at every API layer, including low-level `ReleaseOrchestrator` (`RecipientSecurityViolation`).
- No external recipient may receive Experiment-mode releases. Block occurs before planning, data access, state reservation, noise generation, or release commit, with zero budget or state side-effects.

$$\boxed{3.\ \text{Same semantic query} \implies \text{Same persistent DP release}}$$
- In production mode, `reuse_existing_release=True` is strictly enforced (fail-closed; configuring `False` raises `ProductionSafetyConfigurationError`).
- Concurrent or repeated identical queries return bitwise identical memoized payloads with 0 incremental budget charge, eliminating repeated-query noise averaging attacks.

$$\boxed{4.\ \text{Committed privacy spend} \le \text{Global privacy budget (Exact Decimal Accounting)}}$$
- In production mode, `allow_budget_overrun=False` is strictly enforced (fail-closed; configuring `True` raises `ProductionSafetyConfigurationError`).
- Budget checks fetch individual reservation rows and calculate cumulative spend using exact Python `Decimal` arithmetic.
- Eliminates SQLite `SUM(REAL)` IEEE 754 floating-point undercounting (e.g., $0.01 + 0.06 = 0.06999999999999999 < 0.07$ drift is eliminated; $0.01 + 0.06 + 5 \times 10^{-18} > 0.07$ is strictly rejected).

$$\boxed{5.\ \text{Experiment mode cannot contaminate Production release / cache / accounting}}$$
- Experiment runs use a dedicated `ExperimentRunner` writing to an isolated state store (`results/experiment_state.db` or temporary SQLite).
- Canonical release keys are explicitly namespaced by execution mode (`"execution_mode": "experiment"` vs `"execution_mode": "production"`).
- In experiment mode, `noise_seed` or `trial_id` is embedded directly into the canonical semantics (`"noise_seed_identity": "seed_1"`), preventing distinct Monte Carlo trials from colliding or hitting each other's cache.
- Low-level `ReleaseOrchestrator(ExecutionMode.EXPERIMENT)` defaults to `results/experiment_state.db` and fails closed if pointed to production `results/privacy_state.db`.

$$\boxed{6.\ \text{Dataset identity change} \implies \text{Release key change}}$$
- `set_data(df)` computes a full 256-bit digest over canonicalized hashed-row representation (`data_sha256_<64 hex chars>`).
- If a human dataset version label is provided, it is combined with the content hash (`f"{label}@{content_fp}"`), preventing human label collisions across differing datasets.

---

## 4. The Three Analytics Metrics

### Metric 1: User Prevalence ($H^{prev}$ and $R^{prev}$)
- **Question**: *How many employees used a given Provider for a given Task in the window?*
- **User Transformation**: 9-dimensional binary vector $b_u \in \{0, 1\}^9$:
  $$b_{u, p, c} = \mathbf{1}[m_{u, p, c} > 0]$$
- **Contribution Bounding**: Bounded to at most $K$ cells (default $K=9$).
- **Sensitivities**:
  - $L_2$ Sensitivity: $\Delta_2 = \sqrt{K} = \sqrt{9} = 3.0$
  - $L_1$ Sensitivity: $\Delta_1 = K = 9.0$
- **Rate Calculation (Mandatory Public Denominator)**:
  $$R^{prev} = \frac{\max(0, H^{prev})}{N_{\text{public}}}$$
  Uses public eligible roster size $N_{\text{public}}$, never private `df["user_id"].nunique()`.
- **Evaluation Methodology**: Prevalence rates are 9 overlapping Bernoulli marginal indicators ($\sum p_i$ is not constrained to 1.0). Total Variation distance is invalid and omitted from prevalence evaluation. Reported metrics: MAE, MAE percentage points, RMSE, ranking accuracy.

### Metric 2: User-Normalized Share ($S$ and $\bar{S}$)
- **Question**: *How does a typical employee distribute their AI usage across Provider $\times$ Task?*
- **User Transformation**: Normalized simplex vector $x_u$:
  $$x_u = \frac{m_u}{\|m_u\|_1} \quad \text{if } \|m_u\|_1 > 0 \quad (\text{else } \mathbf{0})$$
  - $\|x_u\|_1 = 1.0$
  - $\|x_u\|_2 = \sqrt{\sum_j x_{u,j}^2} \le \sum_j x_{u,j} = 1.0$
- **Norm Bounding**: `norm_bound` is fixed to $1.0$ and validated at runtime.
- **Sensitivities**:
  - $L_2$ Sensitivity: $\Delta_2 = 1.0$
  - $L_1$ Sensitivity: $\Delta_1 = 1.0$
- **Evaluation**: User-Normalized Share forms a true probability simplex; Total Variation distance ($0.5 \sum |p_i - q_i|$) is valid and reported.

### Metric 3: Interaction Volume ($V$)
- **Question**: *Where is enterprise total interaction volume distributed?*
- **User Transformation**: User-level $L_2$ norm clipping with threshold $C$:
  $$\bar{m}_u = m_u \cdot \min\left(1, \frac{C}{\|m_u\|_2}\right)$$
  - $\|\bar{m}_u\|_2 \le C$
- **Clipping Sensitivity Alignment**: Query-level overrides for `clip_norm` are prohibited. Planner and execution share the exact immutable configuration object, guaranteeing declared sensitivity == actual clipping bound.
- **Sensitivities**:
  - $L_2$ Sensitivity: $\Delta_2 = C$ (default $C=20.0$)
  - $L_1$ Sensitivity: $\Delta_1 = \sqrt{d} \cdot C = \sqrt{9} \cdot C = 3 \cdot C$ (by Cauchy-Schwarz on $\mathbb{R}^9$)
- **Error Decomposition & Trade-off**:
  $$\text{Total Error} = \underbrace{(V^{clip} - V^{raw})}_{\text{Clipping Bias}} + \underbrace{(V^{DP} - V^{clip})}_{\text{DP Noise Error}}$$

---

## 5. Differential Privacy Mechanisms

### Gaussian Mechanism (Primary)
- Calibrated using the **Analytic Gaussian Mechanism (AGM)** (Balle & Wang, ICML 2018).
- Solves the exact root condition for the Gaussian privacy loss tail:
  $$\delta(\sigma) = \Phi\left(\frac{\Delta_2}{2\sigma} - \frac{\varepsilon\sigma}{\Delta_2}\right) - e^\varepsilon \Phi\left(-\frac{\Delta_2}{2\sigma} - \frac{\varepsilon\sigma}{\Delta_2}\right) = \delta$$

### Laplace Mechanism (Baseline)
- Calibrated using classical $L_1$ sensitivity:
  $$b = \frac{\Delta_1}{\varepsilon}$$

---

## 6. Architecture & Concurrency Control

```
                 +-----------------------------------------+
                 |       Enterprise Interaction Log       |
                 +-----------------------------------------+
                                      |
                        [User-Level Aggregation]
                                      v
                 +-----------------------------------------+
                 |        User Matrix (N x 9 Grid)         |
                 +-----------------------------------------+
                                      |
                +----------------------+----------------------+
                |                                             |
    [Production Path: compute()]              [Experiment Path: compute_experiment()]
                |                                             |
    Validate Fail-Closed Config                  ExperimentRunner / ReleaseOrchestrator
    Prohibit Caller RNG (Secure Entropy)         Role check: ONLY 'internal' allowed
    Key: "execution_mode": "production"          Key: "execution_mode": "experiment"
                |                                Seed Identity: "seed_{noise_seed}"
                v                                             v
    +-------------------------------------------------------------+
    |             PrivacyStateManager (SQLite + WAL)              |
    |  - Monotonic CAS lease fencing                              |
    |  - Single-flight in-flight wait                            |
    |  - Exact Decimal budget comparison                          |
    |  - Commit irreversibility                                   |
    +-------------------------------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 |       Recipient Security Filter         |
                 |  - external: stripped active_user_count |
                 |  - external: filtered competitors       |
                 +-----------------------------------------+
```

---

## 7. Verification Suite (16 Standalone Scripts)

All 16 standalone verification scripts pass from a clean database state:

| Script | Description | Status |
|---|---|---|
| `experiments/verify_smoke_non_destructive.py` | Smoke execution runs non-destructively; verifies official research artifacts unchanged | **PASS** |
| `experiments/verify_experiment_external_block.py` | Low-level ReleaseOrchestrator blocks external recipients in EXPERIMENT mode (zero side-effects) | **PASS** |
| `experiments/verify_exp02_population.py` | Validates N=50,000 public population consistency, fail-fast checks, and non-destructive reproduction | **PASS** |
| `experiments/verify_multiworker_budget.py` | 8 concurrent workers compete for budget cap; exactly 3 commit, 5 rejected | **PASS** |
| `experiments/verify_same_key_concurrency.py` | 6 concurrent identical queries receive bitwise identical release, 1 budget charge | **PASS** |
| `experiments/verify_atomic_restart.py` | Crash recovery before reserve, before commit, and commit irreversibility | **PASS** |
| `experiments/verify_slow_owner_singleflight.py` | Slow owner (11s) computes while fast owner waits; bitwise identical, 1 charge | **PASS** |
| `experiments/verify_fencing.py` | Stale worker lease takeover rejected via CAS generation fencing (`ReservationLostError`) | **PASS** |
| `experiments/verify_experiment_production_isolation.py` | Experiment seeded query cannot hit production; production unseeded only | **PASS** |
| `experiments/verify_experiment_seed_identity.py` | Distinct experiment noise seeds produce distinct release keys and no cache collisions | **PASS** |
| `experiments/verify_budget_precision.py` | Exact Decimal accounting rejects adversarial IEEE 754 float drift and zero-slack cap overspend | **PASS** |
| `experiments/verify_dataset_identity.py` | Changing data or human label override changes fingerprint and release key; preserves seed=0 | **PASS** |
| `experiments/verify_cache_restart.py` | Crash recovery and state consistency across persistent database restarts | **PASS** |
| `experiments/verify_500_noise_trials.py` | Validates 500 independent Monte Carlo evaluation trials under EVALUATION mode | **PASS** |
| `experiments/verify_custom_state_isolation.py` | Resolves canonical paths and prevents collision with custom production DB paths | **PASS** |
| `experiments/verify_decimal_storage_schema.py` | Verifies TEXT NOT NULL schema migration and exact Decimal row-by-row summation | **PASS** |

---

## 8. Quick Start

### Installation

```bash
pip install -e .
```

### Run Tests

```bash
pytest tests -q
# 161 passed in tests/
```

### Run All 16 Standalone Verifications

```bash
python -c "import subprocess, sys; from pathlib import Path; [subprocess.run([sys.executable, str(s)], check=True) for s in sorted(Path('experiments').glob('verify_*.py'))]"
```

### Production Query Example

```python
from private_llm_analytics.service.analytics import AnalyticsService

# Initialize production service
service = AnalyticsService.from_config_dir("configs")
service.load_data("data/interactions_n50000.parquet")

# Production query (secure unseeded DP randomness, persistent release memoization)
release = service.compute(
    metric="user_prevalence",
    recipient="provider_a",
    epsilon=1.0,
    delta=1e-6,
)
print(release["results"])
```

### Run Research Experiment Suite

```bash
# Full research suite (500 trials per configuration, output root: results/):
python run_all_experiments.py

# Rapid non-destructive smoke verification:
python run_all_experiments.py --smoke
```

> **Smoke Mode Isolation**:
> - Smoke outputs are written exclusively to `results/verification/` (`results/verification/tables/`, `results/verification/raw/`, `results/verification/figures/`).
> - Official research artifacts (`results/tables/*`, `results/raw/*`, `results/figures/*`) remain 100% byte-identical and unmodified.

> **Main Epsilon Sweep Configuration**:
> - $N = 50,000$ (dataset: `data/interactions_n50000.parquet`)
> - $\delta = 10^{-6}$
> - $\varepsilon \in \{0.25, 0.5, 1.0, 2.0, 4.0, 8.0\}$
