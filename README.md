# Private LLM Analytics (v0.4 — User-Level Continual Observation & Sliding-Window Differential Privacy)

A **User-Level Central Differential Privacy Analytics Layer** prototype for enterprise multi-model AI telemetry.

> [!IMPORTANT]
> **v0.2 Milestone: Longitudinal Multi-Recipient Privacy Accounting**:
> Version 0.2 extends the frozen v0.1.7 core with longitudinal differential privacy accounting across long time horizons, multiple external recipients, and recurring reporting cadences:
> - **Rényi Differential Privacy (RDP) Accountant**: Exact Gaussian RDP calibration ($\text{RDP}(\alpha) = \frac{\alpha \Delta_2^2}{2 \sigma^2}$) and conservative Laplace pure-DP bounds over candidate orders $\alpha \in \{1.25, 1.5, 2, 3, 4, 5, 8, 10, 16, 32, 64, 128\}$.
> - **Historical Release Composition**: Order-wise additive accumulation $\text{RDP}_{\text{total}}(\alpha) = \sum_t \text{RDP}_t(\alpha)$ before optimal $(\epsilon, \delta)$ conversion, achieving sublinear $O(\sqrt{T})$ scaling over longitudinal horizons.
> - **Lifetime Privacy Horizon**: External recipients retain historical releases indefinitely. Production strictly enforces `privacy_history_mode = "lifetime"`; rolling resets fail closed (`ProductionSafetyConfigurationError`).
> - **Multi-Recipient Composition Policies**: Full support for `GLOBAL_EXTERNAL` (unified external scope preventing provider switching evasion), `COLLUSION_GROUP` (shared scopes among colluding recipients), and `PER_RECIPIENT` (isolated accounting per provider).
> - **Atomic Concurrency & Recovery Hardening**: Integrated SQLite `BEGIN IMMEDIATE` compare-and-reserve transaction, CAS lease fencing, single-flight memoization with 0 duplicate charge, and crash-recovery budget reclamation.
> - **Longitudinal Experiments (Exp 05–08)**: Systematic evaluations of Basic vs RDP composition, 90-day release frequency planning, multi-recipient collusion impact, and sequential authorization trajectories.
>
> Top-K, heavy hitters, continual observation tree mechanisms, web UI, dynamic taxonomy, adaptive clipping planners, and ML training remain deferred to future versions.

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

## 5. Longitudinal Privacy Accounting (v0.2)

### 5.1 The Fundamental Challenge of Longitudinal Releases
In enterprise production deployments, analytical releases occur repeatedly across months or quarters to multiple external recipients. Under standard differential privacy, independent releases over time compound privacy loss. Simple basic composition ($\sum_{t=1}^T \varepsilon_t$) scales linearly ($O(T)$), which rapidly exhausts available privacy budgets.

To solve this, **Private LLM Analytics v0.2** introduces a mathematically rigorous **Rényi Differential Privacy (RDP) Accountant** combined with persistent historical release state.

### 5.2 Mathematical Formulation of Rényi Differential Privacy
For Rényi divergence order $\alpha > 1$, Gaussian mechanism noise scale $\sigma$, and $L_2$ query sensitivity $\Delta_2$, the exact RDP cost is:
$$\text{RDP}(\alpha) = \frac{\alpha \Delta_2^2}{2 \sigma^2}$$

For pure $\varepsilon$-DP Laplace releases, we apply the conservative pure-DP bound:
$$\text{RDP}(\alpha) \le \varepsilon \quad (\forall \alpha > 1)$$

#### Historical Additive Composition
Over a sequence of $T$ releases, privacy loss composes by pointwise additive summation at each candidate order $\alpha$:
$$\text{RDP}_{\text{total}}(\alpha) = \sum_{t=1}^T \text{RDP}_t(\alpha)$$

#### Conversion to Cumulative $(\varepsilon, \delta)$
Given an enterprise policy parameter $\delta_{\text{global}} \in (0, 1)$, the effective cumulative privacy loss is computed via exact Decimal logarithm without floating-point conversion drift:
$$\varepsilon_{\text{effective}} = \min_{\alpha > 1} \left[ \text{RDP}_{\text{total}}(\alpha) + \frac{\ln(1 / \delta_{\text{global}})}{\alpha - 1} \right]$$

Authoritative production authorization strictly enforces $\varepsilon_{\text{effective}} \le \varepsilon_{\text{cap}}$ against the target $\delta_{\text{global}}$, avoiding unproved mixed bounds across differing $\delta$ levels.

### 5.3 Multi-Recipient Composition Policies
External model providers (`provider_a`, `provider_b`, `provider_c`) interact under configurable collusion policies:
- **`GLOBAL_EXTERNAL` (Default & Production Recommendation)**:
  All external releases share a unified scope (`global_external`). An enterprise release to Provider A consumes global budget, preventing external recipients from evading privacy caps by rotating provider identities.
- **`COLLUSION_GROUP`**:
  Recipients configured within a known collusion group (e.g., `group_ab = [provider_a, provider_b]`) share a collective accounting scope (`collusion:group_ab`), while independent groups (`collusion:group_c`) maintain separate accounting.
- **`PER_RECIPIENT`**:
  Each external recipient possesses an independent accounting scope (`recipient:<name>`), formally valid only under the strict assumption of zero information sharing between providers.
- **`internal`**:
  Internal analytics remain completely excluded from external differential privacy accounting.

### 5.4 Absolute Prohibition of Rolling Privacy Reset
> [!CAUTION]
> **No Rolling Resets in Production (`privacy_history_mode = "lifetime"`)**:
> External recipients permanently store and analyze historical releases. Deleting historical rows from the database after 30, 60, or 90 days does not erase the information already disclosed. Consequently, production configuration strictly enforces `privacy_history_mode = "lifetime"`. Configuring `"rolling_reset"` fails closed with `ProductionSafetyConfigurationError`.

### 5.5 Replay Memoization with Zero Incremental Spend
Repeated queries with identical semantic parameters return memoized outputs with `cache_hit=True` and strictly `0.0` incremental budget charge. This eliminates noise averaging attacks while enabling arbitrary repeated reporting at zero marginal privacy cost.

### 5.6 Empirical Longitudinal Findings (Exp 05–08)
Empirical evaluations demonstrate:
1. **Exp 05 (Basic vs RDP Composition)**: Evaluates distinct guarantees honestly without artificial min: Basic $(\sum \varepsilon_i, \sum \delta_i)$ vs RDP $(\varepsilon_{\text{RDP}}, \delta_{\text{global}})$. At $T=100$, basic composition accumulates to $(100.0, 10^{-4})$, whereas RDP limits cumulative privacy loss to $(15.31, 10^{-6})$.
2. **Exp 06 (Release Frequency Planning)**: Under an $\varepsilon_{\text{cap}}=10.0$ ceiling over 90 days, daily releases exhaust the cap at release 48 (42 rejected), weekly releases execute all 13 releases with 48.9% budget utilization ($\varepsilon=4.89$), and monthly releases consume $\varepsilon=2.27$.
3. **Exp 07 (Collusion Overhead)**: Under `GLOBAL_EXTERNAL`, the budget cap is reached at week 8; under `COLLUSION_GROUP`, group A+B exhausts at week 10 while isolated group C executes unhindered.
4. **Exp 08 (Longitudinal Authorization Trajectory)**: Verifies step-by-step transaction safety and cap enforcement until exhaustion across 15 sequential queries.

---

## 6. Verification Suite (23 Standalone Scripts)

All 23 standalone verification scripts pass from a clean database state:

| Script | Description | Status |
|---|---|---|
| `experiments/verify_rdp_math.py` | RDP formulas, order linearity, monotonic scaling, and Laplace bounds | **PASS** |
| `experiments/verify_rdp_100_composition.py` | 100-step RDP historical composition scaling benefit ($O(\sqrt{T})$ vs $O(T)$) | **PASS** |
| `experiments/verify_recipient_collusion.py` | Multi-recipient collusion policies (`GLOBAL_EXTERNAL`, `COLLUSION_GROUP`, `PER_RECIPIENT`) | **PASS** |
| `experiments/verify_same_release_zero_cost.py` | Same-query replay memoization with 0 incremental privacy budget spend | **PASS** |
| `experiments/verify_no_privacy_reset.py` | Production fail-closed rejection of rolling reset & lifetime accumulation | **PASS** |
| `experiments/verify_rdp_multiworker_budget.py` | 8 concurrent worker processes competing under RDP cap; 3 commit, 5 rejected | **PASS** |
| `experiments/verify_rdp_history_restart.py` | RDP historical view reconstruction across process restarts & stale lease recovery | **PASS** |
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

## 7. Quick Start

### Installation

```bash
pip install -e .
```

### Run Tests

```bash
pytest tests -q
# 186 passed in tests/
```

### Run All 23 Standalone Verifications

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
print("Results:", release["results"])
print("Privacy Accounting:", release["privacy_accounting"])

# Retrieve authoritative longitudinal PrivacyView
view = service.get_privacy_view(recipient="provider_a")
print(f"Cumulative Releases: {view.release_count}")
print(f"Effective Epsilon:  {view.effective_epsilon:.4f}")
print(f"Remaining Budget:   {view.remaining_epsilon:.4f}")
```

### Run Research Experiment Suite

```bash
# Full research suite (Exp 01-04 + Exp 05-08 longitudinal, output root: results/):
python run_all_experiments.py --include-longitudinal

# Rapid non-destructive smoke verification:
python run_all_experiments.py --smoke --include-longitudinal
```

> **Smoke Mode Isolation**:
> - Smoke outputs are written exclusively to `results/verification/` (`results/verification/tables/`, `results/verification/raw/`, `results/verification/figures/`).
> - Official research artifacts (`results/tables/*`, `results/raw/*`, `results/figures/*`) remain 100% byte-identical and unmodified.


---

## 8. Small-n Utility-Aware Differential Privacy Release Planner (v0.3)

### 8.1 Motivation & Research Question
In mid-sized enterprise environments ($N \in [500, 5000]$ employees), raw telemetry volume is orders of magnitude smaller than national population datasets. Under user-level Differential Privacy:
1. Setting clipping bound $C$ too low introduces severe systematic clipping bias: $E_{\text{clip}}(C) = \mathbb{E}[(R_u - C)_+]$.
2. Setting $C$ too high inflates $L_2$ query sensitivity $\Delta_2 = C$, injecting overwhelming DP Gaussian noise: $\sigma = \text{calibrate\_analytic\_gaussian}(\epsilon, \delta, C)$.
3. Releasing fine-grained 9-cell statistics spreads signal thin, causing noise to dominate each cell. Aggregating to coarse marginals ($G \in \{\text{provider\_3\_cell}, \text{task\_3\_cell}, \text{overall\_1\_cell}\}$) trades dimensional granularity against variance reduction.
4. Releasing frequently ($F=1$ or $F=7$) rapidly consumes cumulative longitudinal privacy budget over a planning horizon $H$.

**The v0.3 Release Planner** automatically determines the optimal release policy tuple:
$$\pi^* = (C^*, G^*, W^*, F^*)$$
minimizing dimensionless normalized loss while strictly satisfying the lifetime privacy budget.

---

### 8.2 Direct Granularity User-Level DP Semantics

Unlike post-clipping linear aggregation (which inflates operator norm and sensitivity by $\sqrt{3}$ or $3.0$), **Direct Granularity Release** performs linear aggregation directly in user-space before target-dimensional clipping:

1. **Volume Metric**:
   $$m_u^G = A_G m_u \in \mathbb{R}^{d_G}$$
   $$\bar{m}_u^G = m_u^G \cdot \min\left(1, \frac{C}{\|m_u^G\|_2}\right)$$
   $$V_G = \sum_{u=1}^N \bar{m}_u^G \implies \Delta_2 = C$$
   Gaussian noise is calibrated with sensitivity $\Delta_2 = C$ in the target dimension $d_G \in \{9, 3, 3, 1\}$.

2. **Prevalence Metric**:
   Direct binary indicators are constructed per user at target granularity:
   - Provider: $b_{u,p} = \mathbf{1}[\text{user } u \text{ used provider } p \text{ at least once}] \implies \Delta_2 = \sqrt{3}$
   - Task: $b_{u,t} = \mathbf{1}[\text{user } u \text{ used task } t \text{ at least once}] \implies \Delta_2 = \sqrt{3}$
   - Overall: $b_u = \mathbf{1}[\text{user } u \text{ active}] \implies \Delta_2 = 1.0$
   - Full 9-cell: $b_{u,p,t} = \mathbf{1}[\text{user } u \text{ used cell } (p,t)] \implies \Delta_2 = 3.0$

3. **Baseline Comparison (`POST_PROCESS_9_CELL`)**:
   Full 9-cell DP release followed by deterministic linear aggregation $A_G \widetilde{V}_9$. Incurs **zero additional privacy cost**, but carries higher noise variance than direct coarse releases under small $N$.

---

### 8.3 Public Time Window Model & Dimensionless Loss

To evaluate arbitrary statistical windows $W$ without private data leakage, the planner employs a public first-two-moment scaling prior calibrated against reference window $W_{\text{ref}} = 30$ days:
$$\mu_W = \mu_{\text{ref}} \cdot \frac{W}{W_{\text{ref}}}, \quad \sigma_W = \sigma_{\text{ref}} \cdot \sqrt{\frac{W}{W_{\text{ref}}}}$$

#### Default Objective: `EXPECTED_RELEASE_L1`
Under the production default objective mode, the total loss directly optimizes expected workload release accuracy alongside dimensionless temporal freshness and relevance penalties:
$$L_{\text{total}}(\pi) = w_{\text{data}} L_{\text{expected\_release\_workload}}(\pi) + w_{\text{stale}} L_{\text{stale}}(\pi) + w_{\text{window}} L_{\text{window}}(\pi)$$
where:
- **Workload Expected Release Error ($L_{\text{expected\_release\_workload}}$)**: Evaluated across declared workload queries $Q \in \{Q_9, Q_p, Q_t, Q_o\}$ (full 9-cell, provider marginal, task marginal, and overall scalar total) weighted by $(w_9, w_p, w_t, w_o)$:
  $$L_{\text{expected\_release\_workload}} = \frac{1}{\sum_{Q} w_Q} \sum_{Q} w_Q \frac{\sum_k \mathbb{E}[|(Q \hat{V}_9 - Q V_{\text{raw}})_k|]}{\max(1, \mathbb{E}[\|V_{\text{raw}}\|_1])}$$
  where $\hat{V}_9 = R_G(V_{G, \text{DP}})$ is the deterministic reconstruction to 9-cell space and each coordinate error $\mathbb{E}[|(Q \hat{V}_9 - Q V_{\text{raw}})_k|]$ follows a folded-normal closed form under Gaussian DP noise.
- **Reporting Freshness / Staleness Penalty ($L_{\text{stale}}$)**:
  $$L_{\text{stale}} = \frac{F}{2 W}$$
- **Window Misalignment Penalty ($L_{\text{window}}$)**:
  $$L_{\text{window}} = \frac{|W - W_{\text{ref}}|}{W_{\text{ref}}}$$
All business trade-offs are governed strictly by the non-negative weights $(w_{\text{data}}, w_{\text{stale}}, w_{\text{window}})$ without auxiliary hidden multiplier distortions.

#### Diagnostic / Legacy Objective: `COMPONENT_SURROGATE`
For offline analysis and ablation studies, the planner retains the decoupled component surrogate loss:
$$L_{\text{total}}(\pi) = w_{\text{clip}} L_{\text{clip}} + w_{\text{noise}} L_{\text{noise}} + w_{\text{agg}} L_{\text{agg}} + w_{\text{stale}} L_{\text{stale}} + w_{\text{window}} L_{\text{window}}$$
where $L_{\text{clip}} = \frac{E_{\text{clip}}(C, W)}{N \cdot \mu_W}$, $L_{\text{noise}} = \frac{E_{\text{noise}}(C, G, \epsilon)}{N \cdot \mu_W}$ (with $\mathbb{E}[|Z|] = \sigma \sqrt{2/\pi}$ for Gaussian and $\mathbb{E}[|Z|] = b$ for Laplace), and $L_{\text{agg}}$ measures deterministic reconstruction error.

Subject to lifetime privacy budget constraint:
$$\text{RDP}_{\text{projected}}(\alpha) = \text{RDP}_{\text{historical}}(\alpha) + \left\lceil \frac{H}{F} \right\rceil \cdot \text{RDP}_{\text{single}}(\alpha; \pi) \implies \epsilon_{\text{projected}} \le \epsilon_{\text{cap}}$$

---

### 8.4 Strict Privacy Provenance & Production Isolation
$$\boxed{\text{Production Planner operates exclusively on Public or Committed DP Feedback}}$$
- **STATIC_PUBLIC Mode**: Relies exclusively on public configurations, public prior $\mathcal{TN}(\mu, \sigma)$, public population $N_{\text{public}}$, and candidate policy grids.
- **DP_FEEDBACK Mode**: May incorporate committed historical release IDs (`DP_RELEASE:<id>`) through authoritative `PrivacyStateManager`. Raw private context is rejected (`ProductionSafetyConfigurationError`).
- **ORACLE_EXACT_EVAL Mode**: Permitted exclusively in offline evaluation benchmarks. If `ORACLE_EXACT_EVAL` appears in production execution, the system fails closed immediately.

---

### 8.5 Empirical Research Findings (Formal Exp 09–14)

> [!NOTE]
> **Metric Scope**: v0.3 utility-aware parameter optimization is formally evaluated and production-enabled for Interaction Volume. Existing v0.1/v0.2 DP release support for prevalence and normalized share remains active and unchanged.

All values are automatically verified and synchronized from [`results/tables/v03_key_results.json`](file:///f:/Private%20LLM%20Analytics/results/tables/v03_key_results.json):

| Metric / Dimension | Formal Finding & Empirical Result |
| :--- | :--- |
| **Selected $C$ by $N$ (Exp 10)** | $N=500 \to C^*=40$ (Oracle $C^*=40$, relative regret $= 0.0\%$); $N=1000 \to C^*=80$ (Oracle $C^*=80$, regret $= 0.0\%$); $N=2500 \to C^*=80$ (regret $= 0.0\%$); $N=5000 \to C^*=80$ (regret $= 0.0\%$). Across all $N \in [500, 5000]$, static public planning with closed-form folded-normal expected $L_1$ release loss achieves **0.0% regret**, perfectly matching the empirical Oracle optimal clipping bound. |
| **Estimator Calibration (Exp 09)** | Evaluated under unified normalized total L1 loss with `PublicContributionPrior`: **Absolute Calibration MAE = 0.002203**, **Relative Calibration Error = 0.55%** (Global Relative MAE = 0.55%, Mean Condition Relative Error = 1.64%, Worst Condition Relative Error = 13.34%), **Calibration Bias = +0.001743**, **Spearman Rank Correlation $\rho = 0.9860$** ($p = 2.50 \times 10^{-23}$, 30 conditions, 500 noise trials per condition). Confirms exceptional magnitude calibration and ranking accuracy. |
| **Selected $G$ by $N$ (Exp 11)** | Direct target sensitivity release $\Delta_2 = C$ decomposes total loss into clipping loss, DP noise loss, and reconstruction loss. Noise variance reduction reaches **66.6%** for 3-cell granularities and **89.4%** for overall 1-cell, eliminating the $\sqrt{3}\times$ sensitivity penalty of post-clipping linear aggregation. |
| **Selected $W$ by $N$ & Oracle Regret (Exp 13)** | In joint $(C, G, W, F)$ policy optimization over 90-day horizon with first-two-moment temporal model alignment / scaling matched (daily Negative Binomial increments matching public scaling laws with mean error $< 0.55\%$ and std error $< 5.63\%$): <br>• Under $\epsilon_{\text{cap}}=2.0$: Correctly detects infeasibility across all $N \in \{500, 1000, 5000\}$ and reports `NO_FEASIBLE_POLICY` (minimum projected $\epsilon=2.2657 > 2.0$, 0 feasible candidates). <br>• Under $\epsilon_{\text{cap}}=4.0$: Selects $W=30$ days, $F=14$ days, $G=\text{full\_9\_cell}$. At $N=500$, selects $C=80$ with true empirical utility $0.1773$ (Oracle $C=40$, true utility $0.1735$, relative regret $= 2.2\%$, 95% CI: Planner $[0.1751, 0.1796]$, Oracle $[0.1714, 0.1756]$). At $N=1000$, selects $C=80$ with true utility $0.1467$ matching Oracle ($C=80$, true utility $0.1467$, **0.0% regret**, 95% CI: $[0.1456, 0.1478]$). At $N=5000$, selects $C=80$ with true utility $0.1227$ matching Oracle ($C=80$, true utility $0.1227$, **0.0% regret**, 95% CI: $[0.1225, 0.1229]$). |
| **Selected $F$ by Budget (Exp 12)** | Under $\epsilon_{\text{cap}}=4.0$ with horizon $H=90$, initial cadence selects $F=14$ days ($K=7$ releases, $\epsilon_{\text{proj}}=3.50$). When historical budget consumption reaches $73.4\%$ ($\epsilon_{\text{hist}}=2.9358$), the planner automatically throttles to $F=30$ days ($K=3$ releases, $\epsilon_{\text{proj}}=3.77$), strictly preserving budget feasibility. |
| **Prior Misspecification (Exp 14)** | Evaluated across three distinct ground-truth regimes (`light` $\mu=25$, `default` $\mu=60$, `heavy_tail` $\mu=100$) using independent public seeds: <br>• Under matched priors: **0.00% regret** across all three regimes. <br>• `light` regime: Mean misspecified regret $= 4.48\%$, max regret $= 13.44\%$ (vs fixed mid baseline $13.44\%$). <br>• `default` regime: Mean misspecified regret $= 29.96\%$, max regret $= 89.87\%$ (vs fixed mid baseline $89.87\%$). <br>• `heavy_tail` regime: Mean misspecified regret $= 33.74\%$, max regret $= 101.21\%$ (vs fixed mid baseline $101.21\%$). In the tested misspecification set, regret did not exceed the fixed-mid baseline. |
| **Budget Utilization** | 100% compliant with lifetime RDP budget composition. No over-allocation or budget violation occurred in any experimental condition. |


---

## Continual Observation and Sliding-Window DP

Version 0.4 introduces **User-Level Continual Observation** and **Sliding-Window Differential Privacy** for enterprise multi-model AI analytics, solving the problem of continuous temporal monitoring (e.g., daily 7-day, 30-day, 90-day sliding windows, quarter-to-date prefixes, and arbitrary fixed intervals) without incurring linear privacy budget degradation from overlapping releases.

### 1. The Overlapping Release Challenge
In traditional differential privacy architectures, releasing statistics over overlapping sliding windows (e.g. Day $t-29$ to Day $t$ published daily) creates a distinct release for every day. Under standard DP composition theorems, each fresh noisy release leaks additional privacy budget. Over a 365-day year, 336 daily 30-day sliding window releases accumulate linear privacy cost, causing lifetime $\epsilon$ to explode (e.g. accumulating $\epsilon > 900$ for standard release configurations) or requiring impractical noise scales that render analytics useless.

### 2. User-Level vs. Event-Level Continual Observation
In literature, continual observation is often presented under **event-level DP**, where each time step contains independent events from disjoint individuals. In enterprise LLM telemetry, this assumption is fundamentally false:
- The authoritative privacy unit is the **USER**.
- The DP adjacency relationship dictates that two databases differ by **one user's entire history of LLM interactions across the entire observation epoch**.
- A user may participate actively on every single day, or concentrate their activity in bursts.
- Treating daily events as independent would lead to a catastrophic privacy violation.

### 3. Group-$L_1$-of-$L_2$ Temporal Contribution Bounding
To provide rigorous user-level differential privacy across continuous temporal queries, v0.4 establishes a two-stage hierarchical contribution bounding operator:

1. **Target-Space Projection**: Raw daily 9-cell interaction vectors are first projected into the target granularity space $\mathbb{R}^d$ via linear operator $A_G$:
   $$x_{u, t} = A_G m_{u, t}$$
2. **Daily $L_2$ Clipping**: Daily contributions are clipped to bound $C_{\text{day}}$:
   $$y_{u, t} = x_{u, t} \cdot \min\left(1, \frac{C_{\text{day}}}{\max(\|x_{u, t}\|_2, 10^{-12})}\right)$$
3. **Epoch Group-$L_1$-of-$L_2$ Clipping**: The user's total temporal activity across the epoch $T$ is bounded by evaluating the sum of daily $L_2$ norms $S_u = \sum_{t=1}^T \|y_{u, t}\|_2$, and scaling all daily vectors by:
   $$z_{u, t} = y_{u, t} \cdot \min\left(1, \frac{C_{\text{epoch}}}{\max(S_u, 10^{-12})}\right)$$

This mathematical construction guarantees two simultaneous invariants for every user $u$ and day $t$:
$$\|z_{u, t}\|_2 \le C_{\text{day}} \quad \text{and} \quad \sum_{t=1}^T \|z_{u, t}\|_2 \le C_{\text{epoch}}$$

### 4. Per-Level Sensitivity Proof for Binary Time Trees
A binary time tree decomposes an epoch of $T$ days (padded to $\text{tree\_size} = 2^{\lceil \log_2(T) \rceil}$) into $L = \lceil \log_2(T) \rceil + 1$ dyadic levels:
- Level $0$: single-day leaf nodes (length $1$)
- Level $l$: dyadic nodes of duration $2^l$ days
- Root Level $L-1$: node covering the entire tree size

**Theorem (Per-Level $L_2$ Sensitivity Bound)**:
> On any tree level $l$, the temporal node intervals $I_{l, j}$ are pairwise mutually disjoint:
> $$I_{l, j} \cap I_{l, j'} = \emptyset \quad \text{for } j \ne j'$$
> Let $v_{u, l, j} = \sum_{t \in I_{l, j}} z_{u, t}$ denote user $u$'s contribution vector to node $j$ on level $l$.
> By the triangle inequality:
> $$\|v_{u, l, j}\|_2 = \left\| \sum_{t \in I_{l, j}} z_{u, t} \right\|_2 \le \sum_{t \in I_{l, j}} \|z_{u, t}\|_2$$
> Let $V_{u, l} = (v_{u, l, 0}, v_{u, l, 1}, \dots, v_{u, l, M-1})$ denote the concatenated joint vector of user $u$'s contributions across all nodes on level $l$. The Euclidean norm of $V_{u, l}$ satisfies:
> $$\|V_{u, l}\|_2 = \sqrt{\sum_{j=0}^{M-1} \|v_{u, l, j}\|_2^2} \le \sum_{j=0}^{M-1} \|v_{u, l, j}\|_2 \le \sum_{j=0}^{M-1} \sum_{t \in I_{l, j}} \|z_{u, t}\|_2 = \sum_{t=1}^T \|z_{u, t}\|_2 \le C_{\text{epoch}}$$
> Therefore, for ANY level $l$, the user-level $L_2$ sensitivity of the joint level vector query is bounded by:
> $$\Delta_l \le C_{\text{epoch}}$$

### 5. Level Gaussian Vector Mechanism & RDP Composition
Each tree level $l$ is materialized as a joint Gaussian vector mechanism:
$$\mathcal{M}_l(D) = \text{LevelAggregate}_l(D) + \mathcal{N}(0, \sigma^2 I)$$
- All levels share an equal noise scale $\sigma = \sigma_{\text{tree}}$ in v0.4.
- For each Rényi order $lpha$, the level mechanism incurs RDP cost:
  $$\rho_l(\alpha) = \frac{\alpha C_{\text{epoch}}^2}{2 \sigma^2}$$
- Across the $L$ levels of the tree, the full mechanism composes via RDP addition:
  $$\text{RDP}_{\text{tree}}(lpha) = L \cdot \frac{\alpha C_{\text{epoch}}^2}{2 \sigma^2}$$
- The number of charged tree level mechanisms is $O(\log T)$ per epoch. Actual effective $\epsilon$ is determined by RDP composition and calibrated $\sigma$.

### 6. Epoch Boundaries Do NOT Reset Lifetime Privacy
Epochs partition tree construction and coordinate indexing (e.g. 90-day quarters). However:
- **An epoch is NOT a privacy reset**.
- Users active across multiple epochs (Epoch 1, Epoch 2, ...) have their lifetime privacy budget composed additively across all epochs and prior v0.1/v0.2/v0.3 releases:
  $$\text{RDP}_{\text{lifetime}}(\alpha) = \sum_{\text{epochs}} \text{RDP}_{\text{epoch}}(\alpha) + \sum_{\text{releases}} \text{RDP}_{\text{release}}(\alpha)$$
- Production strictly enforces lifetime continuity; rolling resets fail closed.

### 7. Decoupling Node Noise Identity from Recipient Exposure Accounting
To guarantee both mathematical privacy and security against multi-recipient collusion:
1. **Canonical Node Key**:
   $$\text{node\_id} = \text{hash}(\text{stream\_fingerprint}, \text{epoch\_id}, \text{level}, \text{node\_index})$$
   A node's noisy payload depends strictly on its temporal identity and never on recipient or query scope.
   **All recipients querying the same node receive the bitwise identical noisy payload**. This completely prevents noise averaging attacks across colluding recipients.
2. **Recipient Exposure Accounting**:
   $$\text{exposure\_key} = (\text{budget\_scope\_id}, \text{stream\_fingerprint}, \text{epoch\_id}, \text{level})$$
   A privacy scope is charged for a level mechanism **once** when any node on that level is first revealed to that scope. Subsequent queries utilizing that level (or other nodes on the same level) incur **zero additional privacy charge**.
3. **Multi-Recipient Composition Policies**:
   - `GLOBAL_EXTERNAL`: All external recipients share a unified global external scope. The first external query pays for level $l$; all subsequent external queries reuse level $l$ at zero additional cost.
   - `PER_RECIPIENT`: Each external recipient pays for level $l$ once in their isolated scope, but both recipients receive bitwise identical noisy nodes.
   - `COLLUSION_GROUP`: Recipients in a collusion consortium share level exposures within the group scope.

### 8. Deterministic Query Post-Processing
All user-facing temporal queries (prefix $[1, t]$, fixed interval $[l, r]$, and sliding window $[t-W+1, t]$) are executed purely as **deterministic linear combinations of canonical committed noisy nodes**:
$$V_{\text{query}} = \sum_{\text{node} \in \text{canonical\_cover}} V_{\text{node}}$$
- **Query execution has ZERO RNG at query time**.
- Once nodes are materialized in the persistent SQLite store, query execution never accesses raw interactions telemetry (verified via raw loader fault injection).

### 9. Finalized Data Barrier and Immutability
To protect privacy invariants:
- Only days marked as finalized (outside `lateness_grace_days`) can be materialized into queryable tree nodes.
- Once a tree node is materialized and released, retroactively altering underlying telemetry is rejected (fail-closed immutability).

### 10. Required Limitations (Section 92)
The v0.4 Continual Observation implementation is subject to the following explicit operational and theoretical boundaries:
1. **Interaction Volume Metric Only**: Production continual observation strictly supports the `interaction_volume` metric. Calls with `user_prevalence` or `user_normalized_share` fail closed with `ProductionSafetyConfigurationError`.
2. **Gaussian Mechanism Only**: Production continual observation strictly requires the Gaussian mechanism; Laplace requests fail closed.
3. **Binary Tree Topology Only**: Topology is fixed to dyadic binary trees with next-power-of-two padding; arbitrary wavelet or non-dyadic splits are not supported.
4. **Fixed Public Topology**: Tree structure and interval partitions are public, fixed, and data-independent. No data-dependent adaptive tree topology is claimed or implemented.
5. **Fixed Epoch Boundaries**: Epoch lengths are predetermined by configuration; dynamic epoch partitioning is not supported.
6. **Finalized Data Immutability**: Underlying telemetry for released nodes cannot be silently updated or backfilled.
7. **Temporal Clipping Bias**: Group-$L_1$-of-$L_2$ clipping introduces deterministic temporal bias ($C_{\text{epoch}}$ truncation), which must be balanced against noise variance.
8. **Lifetime Composition**: Epoch transitions do not clear consumed privacy budget; lifetime RDP composes indefinitely.
9. **No Optimality Claims**: v0.4 does not claim theoretical optimality over all possible continual mechanisms; it provides a rigorous, verified user-level baseline.
10. **Prevalence / Share Continual Observation**: Continual tracking of user prevalence and normalized share metrics remains a subject for future releases.

---

### Key Empirical Results (Exp 15 – Exp 19)

| Experiment | Metric / Finding | Quantitative Result |
| :--- | :--- | :--- |
| **Exp 15: Tree Privacy Scaling** | Epsilon Reduction Factor (Fixed Noise) | Tree accumulates up to **1.79x lower lifetime $\epsilon$** over $T=365$ days compared to independent daily releases. |
| **Exp 15: Noise Reduction** | Noise Reduction Factor (Fixed $\epsilon=2.0$) | At fixed lifetime $\epsilon=2.0$, tree achieves **up to 4.14x lower query noise MAE** for 30-day queries over a 1-year horizon. |
| **Exp 16: Prefix Tracking** | Cumulative Noise Growth | Prefix query noise grows as $O(\log t)$ in binary tree vs. $O(t)$ linear variance growth in independent daily sums (**up to 3.98x noise reduction** at $t=90$). |
| **Exp 17: Sliding-Window DP** | Lifetime Privacy Savings | Over a 365-day horizon, daily 7-day sliding-window releases accumulate $\epsilon=952.76$ under independent releases vs. $\epsilon=53.82$ under binary tree (**17.7x lifetime privacy savings**). |
| **Exp 18: Temporal Clipping Tradeoff** | Bias-Variance Pareto Frontier | As $C_{\text{epoch}}$ increases from 10 to 160 (at $C_{\text{day}}=10$), relative clipping bias drops from $91.5\%$ to $2.7\%$, while relative DP noise increases from $0.6\%$ to $9.8\%$. Minimum total relative error is achieved at $C_{\text{epoch}}=160$. |
| **Exp 19: Long-Horizon System Scaling** | Query Latency & Storage Footprint | Cached interval query latency remains constant at **$\approx 13$ ms** from $T=30$ to $T=730$ days. SQLite storage footprint remains compact at **$\approx 104$ KB**. |
