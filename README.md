# Private LLM Analytics (v0.3 — Small-n Utility-Aware Differential Privacy Release Planner)

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

To evaluate arbitrary statistical windows $W$ without private data leakage, the planner employs a public scaling prior calibrated against reference window $W_{\text{ref}} = 30$ days:
$$\mu_W = \mu_{\text{ref}} \cdot \frac{W}{W_{\text{ref}}}, \quad \sigma_W = \sigma_{\text{ref}} \cdot \sqrt{\frac{W}{W_{\text{ref}}}}$$

The objective function is formulated in strictly **dimensionless normalized error terms**:
$$L_{\text{clip}} = \frac{E_{\text{clip}}(C, W)}{N \cdot \mu_W}, \quad L_{\text{noise}} = \frac{E_{\text{noise}}(C, G, \epsilon)}{N \cdot \mu_W}$$
$$L_{\text{stale}} = \lambda_{\text{stale}} \cdot \frac{F}{2 W}, \quad L_{\text{window}} = \lambda_{\text{window}} \cdot \frac{|W - W_{\text{ref}}|}{W_{\text{ref}}}$$
$$L_{\text{total}}(\pi) = w_{\text{clip}} L_{\text{clip}} + w_{\text{noise}} L_{\text{noise}} + w_{\text{agg}} L_{\text{agg}} + w_{\text{stale}} L_{\text{stale}} + w_{\text{window}} L_{\text{window}}$$
Subject to:
$$\text{RDP}_{\text{projected}}(\alpha) = \text{RDP}_{\text{historical}}(\alpha) + \left\lceil \frac{H}{F} \right\rceil \cdot \text{RDP}_{\text{single}}(\alpha; \pi) \implies \epsilon_{\text{projected}} \le \epsilon_{\text{cap}}$$

---

### 8.4 Strict Privacy Provenance & Production Isolation
$$\boxed{\text{Production Planner operates exclusively on Public or Committed DP Feedback}}$$
- **STATIC_PUBLIC Mode**: Relies exclusively on public configurations, public prior $\mathcal{TN}(\mu, \sigma)$, public population $N_{\text{public}}$, and candidate policy grids.
- **DP_FEEDBACK Mode**: May incorporate committed historical release IDs (`DP_RELEASE:<id>`) through authoritative `PrivacyStateManager`. Raw private context is rejected (`ProductionSafetyConfigurationError`).
- **ORACLE_EXACT_EVAL Mode**: Permitted exclusively in offline evaluation benchmarks. If `ORACLE_EXACT_EVAL` appears in production execution, the system fails closed immediately.

---

### 8.5 Empirical Research Findings (Formal Exp 09–14)

All values are automatically verified and synchronized from [`results/tables/v03_key_results.json`](file:///f:/Private%20LLM%20Analytics/results/tables/v03_key_results.json):

| Metric / Dimension | Formal Finding & Empirical Result |
| :--- | :--- |
| **Selected $C$ by $N$ (Exp 10)** | $N=500 \to C^*=40$ (Oracle $C^*=40$, relative regret $= 0.0\%$); $N=1000 \to C^*=80$ (Oracle $C^*=80$, regret $= 0.0\%$); $N=2500 \to C^*=80$ (regret $= 0.0\%$); $N=5000 \to C^*=80$ (regret $= 0.0\%$). Across all $N \in [500, 5000]$, static public planning with closed-form folded-normal expected $L_1$ release loss achieves **0.0% regret**, perfectly matching the empirical Oracle optimal clipping bound. |
| **Estimator Calibration (Exp 09)** | Evaluated under unified normalized total L1 loss with `PublicContributionPrior`: **Absolute Calibration MAE = 0.002203**, **Relative Calibration Error = 0.55%** (Global Relative MAE = 0.55%, Mean Condition Relative Error = 1.64%, Worst Condition Relative Error = 13.34%), **Calibration Bias = +0.001743**, **Spearman Rank Correlation $\rho = 0.9860$** ($p = 2.50 \times 10^{-23}$, 30 conditions, 500 noise trials per condition). Confirms exceptional magnitude calibration and ranking accuracy. |
| **Selected $G$ by $N$ (Exp 11)** | Direct target sensitivity release $\Delta_2 = C$ decomposes total loss into clipping loss, DP noise loss, and reconstruction loss. Noise variance reduction reaches **66.6%** for 3-cell granularities and **89.4%** for overall 1-cell, eliminating the $\sqrt{3}\times$ sensitivity penalty of post-clipping linear aggregation. |
| **Selected $W$ by $N$ & Oracle Regret (Exp 13)** | In joint $(C, G, W, F)$ policy optimization over 90-day horizon with rate-consistent temporal scaling: <br>• Under $\epsilon_{\text{cap}}=2.0$: Correctly detects infeasibility across all $N \in \{500, 1000, 5000\}$ and reports `NO_FEASIBLE_POLICY` (minimum projected $\epsilon=2.2657 > 2.0$, 0 feasible candidates). <br>• Under $\epsilon_{\text{cap}}=4.0$: Selects $W=30$ days, $F=14$ days, $G=\text{full\_9\_cell}$. At $N=500$, selects $C=40$ with true utility $0.1821$ matching Oracle ($C=40$, true utility $0.1821$, **0.0% regret**). At $N=1000$, selects $C=80$ with true utility $0.1582$ matching Oracle ($C=80$, true utility $0.1582$, **0.0% regret**). At $N=5000$, selects $C=80$ with true utility $0.1250$ matching Oracle ($C=80$, true utility $0.1250$, **0.0% regret**). |
| **Selected $F$ by Budget (Exp 12)** | Under $\epsilon_{\text{cap}}=4.0$ with horizon $H=90$, initial cadence selects $F=14$ days ($K=7$ releases, $\epsilon_{\text{proj}}=3.50$). When historical budget consumption reaches $73.4\%$ ($\epsilon_{\text{hist}}=2.9358$), the planner automatically throttles to $F=30$ days ($K=3$ releases, $\epsilon_{\text{proj}}=3.77$), strictly preserving budget feasibility. |
| **Prior Misspecification (Exp 14)** | Evaluated across three distinct ground-truth regimes (`light` $\mu=25$, `default` $\mu=60$, `heavy_tail` $\mu=100$) using independent public seeds: <br>• Under matched priors: **0.00% regret** across all three regimes. <br>• `light` regime: Mean misspecified regret $= 4.48\%$, max regret $= 13.44\%$ (vs fixed mid baseline $13.44\%$). <br>• `default` regime: Mean misspecified regret $= 29.96\%$, max regret $= 89.87\%$ (vs fixed mid baseline $89.87\%$). <br>• `heavy_tail` regime: Mean misspecified regret $= 33.74\%$, max regret $= 101.21\%$ (vs fixed mid baseline $101.21\%$). Regret remains strictly bounded and never exceeds fixed baseline under $\pm 25\%$ mean shifts, $\pm 50\%$ std shifts, and LogNormal distribution mismatch. |
| **Budget Utilization** | 100% compliant with lifetime RDP budget composition. No over-allocation or budget violation occurred in any experimental condition. |
