# Private LLM Analytics

**Protect enterprise LLM usage analytics at the user level—even across repeated reports over time.**

User-level DP · Lifetime RDP Accounting · Multi-Recipient Composition · Continual Observation · Sliding-Window Releases

[![CI](https://github.com/thy919100815-art/private-llm-analytics/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thy919100815-art/private-llm-analytics/actions/workflows/ci.yml)
[![Release Gate](https://github.com/thy919100815-art/private-llm-analytics/actions/workflows/release-gate.yml/badge.svg?branch=main)](https://github.com/thy919100815-art/private-llm-analytics/actions/workflows/release-gate.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Latest Release](https://img.shields.io/github/v/release/thy919100815-art/private-llm-analytics?color=orange)](https://github.com/thy919100815-art/private-llm-analytics/releases/tag/v0.4.0)

Private LLM Analytics is a privacy layer for organizations that need to analyze employee usage of multiple LLM providers without exposing individual user behavior to external vendors or analysts.

Current stable release: [`v0.4.0`](https://github.com/thy919100815-art/private-llm-analytics/releases/tag/v0.4.0)

---

## Why This Exists

**How can an organization share LLM usage statistics with external model vendors without exposing individual employee behavior—even after repeated reports over time?**

Enterprises increasingly deploy multiple frontier LLM providers across internal business units. IT, procurement, and AI governance teams need empirical telemetry to monitor adoption patterns, capability migration, and provider utilization.

However, sharing telemetry with external model vendors or analysts introduces critical privacy risks:
- In sparse task categories or specialized teams, exact interaction counts can expose individual employee workflows, sensitive internal initiatives, or anomalous behavioral patterns.
- Repeated longitudinal reports on recurring cadences accumulate privacy loss, enabling noise-averaging or reconstruction attacks.
- External vendors may retain released reports indefinitely, requiring non-replenishing lifetime privacy accounting.

Private LLM Analytics provides a security-hardened user-level differential privacy analytics layer with persistent privacy accounting, tested security invariants, and reproducible evaluation.

---

## Features

- **User-Level Central DP**: Privacy unit is the complete user trajectory within statistical windows under add/remove user adjacency.
- **Gaussian & Laplace Releases**: Exact calibrated noise addition tailored to bounded $L_1$ and $L_2$ user contribution spaces.
- **Lifetime RDP Accounting**: Exact Gaussian Rényi Differential Privacy accumulation across multi-release horizons with conversion to $(\epsilon, \delta)$.
- **Multi-Recipient Composition**: Flexible budget scoping supporting `GLOBAL_EXTERNAL`, `PER_RECIPIENT`, and `COLLUSION_GROUP` policies.
- **Utility-Aware Release Planning**: Small-$n$ adaptive release planning balancing clipping bias, noise variance, staleness, and temporal averaging.
- **Continual Observation**: Prefix-stable binary time-tree continual observation enabling repeated time-range releases via canonical tree node reuse.
- **Sliding-Window Analytics**: Low-noise rolling 7-day, 30-day, and 90-day analytics via canonical tree node reuse.
- **Persistent Privacy State**: ACID-compliant SQLite transaction store preventing state loss across restarts.
- **Concurrency-Safe Memoization**: Atomic compare-and-reserve transactions with lease fencing and single-flight memoization to eliminate duplicate noise generation and budget double-charging.

---

## Supported Analytics

Statistical aggregates are computed across a standard $3 \times 3$ Provider $\times$ Task Category grid (9 cells):

1. **User Prevalence**: The proportion of employees in the public eligible roster who utilized a specific model capability within the observation window.
2. **User-Normalized Share**: Distribution of an employee's workflow across providers and capabilities (simplex-normalized per user before aggregation).
3. **Interaction Volume**: Total volume of model interactions, bounded by daily $L_2$ sensitivity and epoch-wide group-$L_{1,2}$ limits.

---

## Architecture

```mermaid
flowchart LR
    A[Enterprise LLM Telemetry]
    B[User-Level Contribution Bounding]
    C[DP Release Engine]
    D[Lifetime RDP Accountant]
    E[Persistent Privacy State]
    F[Provider / External Recipient]
    G[Continual Observation Tree]
    H[Sliding / Prefix / Interval Queries]

    A --> B
    B --> C
    C --> D
    D --> E
    C --> F
    B --> G
    G --> H
    H --> F
```

---

## Privacy Model

- **Privacy Unit**: Individual user $u$. All interactions of an employee within the designated statistical window or epoch are protected jointly.
- **Adjacency**: Add/Remove User Adjacency. Two datasets $D$ and $D'$ are neighboring if they differ by the complete presence or absence of a single user:
  $$D \sim D' \iff D' = D \cup \{u\} \quad \text{or} \quad D' = D \setminus \{u\}$$
- **Trusted Curator**: The internal analytics service holds raw interaction records, enforces contribution bounding, and applies differential privacy mechanisms before releasing results.
- **External Recipients**: Untrusted third parties receiving statistical summaries. Recipients cannot supply random seeds or access unaggregated counts.
- **Lifetime RDP Accounting**: External recipients can retain historical releases indefinitely. Privacy loss is tracked as non-replenishing cumulative spend via Rényi Differential Privacy:
  $$\text{RDP}_{\text{total}}(\alpha) = \sum_{t=1}^M \text{RDP}_t(\alpha)$$
  For Gaussian mechanisms with $L_2$ sensitivity $\Delta_2$ and noise scale $\sigma$:
  $$\text{RDP}(\alpha) = \frac{\alpha \Delta_2^2}{2 \sigma^2}$$
  Converted to canonical $(\epsilon, \delta)$-DP at query time via:
  $$\epsilon(\delta) = \min_{\alpha > 1} \left\{ \text{RDP}_{\text{total}}(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1} \right\}$$

---

## Quick Start

### Installation

```bash
git clone https://github.com/thy919100815-art/private-llm-analytics.git
cd private-llm-analytics
pip install -e ".[dev,web]"
python -m pytest tests/ -q
```

### Minimal Usage Example

```python
import pandas as pd
from private_llm_analytics.config import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService

# 1. Load authoritative configuration
config = load_app_config("configs")

# 2. Initialize trusted analytics service
service = AnalyticsService(config=config)

# 3. Ingest enterprise telemetry logs
telemetry_df = pd.DataFrame([
    {"user_id": "emp_001", "provider": "provider_a", "task": "technical"},
    {"user_id": "emp_002", "provider": "provider_b", "task": "document"},
])
service.set_data(telemetry_df, dataset_version="telemetry_v1")

# 4. Compute DP release for an external recipient
result = service.compute(
    metric="interaction_volume",
    recipient="provider_a",
    epsilon=1.0,
    delta=1e-6,
)

print(result["results"]["cell_volumes"])
```

---

## Local Web Console

Private LLM Analytics includes a lightweight, secure local web console for trusted internal administrators to configure queries, inspect lifetime privacy consumption, review small-$n$ planner recommendations, and execute differential privacy releases.

![Private LLM Analytics Local Web Console](docs/images/web-console.png)

### Security & Privacy Invariants

- **Localhost Bound**: The server binds strictly to `127.0.0.1` by default (never `0.0.0.0`) and is designed exclusively for local trusted curators.
- **Zero External Egress**: Uses standard server-rendered HTML, vanilla CSS, and vanilla JavaScript with zero external CDN dependencies, remote fonts, or network telemetry.
- **Authoritative Accounting**: The Web Console does not implement standalone DP logic; every release and budget inquiry delegates strictly to the authoritative `AnalyticsService` and `PrivacyStateManager`.
- **RNG Fencing**: External request payloads cannot supply random seeds or tamper with cryptographic noise generators.

### Starting the Console

```bash
# Install with web dependencies
pip install -e ".[web]"

# Launch the local console (defaults to http://127.0.0.1:8000)
python -m private_llm_analytics.web

# Or specify custom host and port:
python -m private_llm_analytics.web --host 127.0.0.1 --port 8080
```

Open your browser at `http://127.0.0.1:8000`.

### Key Console Capabilities

1. **Dataset Management**: Load bundled enterprise synthetic demo telemetry (50,000 users, 3M+ rows) or upload local CSV/Parquet telemetry datasets.
2. **Authoritative Privacy Monitor**: Real-time inspection of cumulative lifetime RDP budget spend, current effective $(\epsilon, \delta)$ consumption, and per-recipient scope breakdown.
3. **Query Configuration**: Interactive selection across configured providers, task categories, analytics metrics, and target privacy parameters $(\epsilon, \delta)$.
4. **Utility-Aware Planner Preview**: Preview recommended clipping bounds and utility loss decompositions (noise variance, bias, total loss) before spending privacy budget (zero budget spend for preview).
5. **Execution & Concurrency Confirmation**: Explicit two-step confirmation modal before budget expenditure; automatic detection and notification for memoized releases (cache hit with 0 incremental spend).
6. **Release History**: Audit trail of completed releases with sanitization of internal user identities.
7. **Continual Observation**: Status inspection of binary time-tree continual observation and interval queries for Interaction Volume.

---

## Continual Observation

To support recurring sliding-window and longitudinal reporting without linear budget depletion, Private LLM Analytics implements user-level continual observation over binary time trees:

- **Prefix-Stable Temporal Contribution Bounding**:
  Raw daily user vectors $x_{u,t}$ are first clipped to daily bound $C_{\text{day}}$ ($y_{u,t} = x_{u,t} \cdot \min(1, C_{\text{day}} / \|x_{u,t}\|_2)$). To guarantee prefix stability as new days arrive progressively, an online chronological remaining-budget filter is enforced:
  $$R_{u,0} = C_{\text{epoch}}$$
  $$a_{u,t} = \min\left(1, \frac{R_{u,t-1}}{\|y_{u,t}\|_2}\right), \quad z_{u,t} = a_{u,t} y_{u,t}, \quad R_{u,t} = \max\left(0, R_{u,t-1} - \|z_{u,t}\|_2\right)$$
  This guarantees that past daily contributions $z_{u,1:t}$ depend strictly on past data $y_{u,1:t}$ and never change retrospectively, while bounding cumulative sensitivity across any tree level by $\sum_{t} \|z_{u,t}\|_2 \le C_{\text{epoch}}$.

- **Binary Time Tree**:
  The timeline is partitioned into a dyadic tree of canonical intervals. Any arbitrary interval query $[s, e]$ decomposes into at most $O(\log T)$ disjoint canonical tree nodes.

- **Sliding-Window DP**:
  Binary-tree continual observation reuses canonical noisy nodes across overlapping temporal queries and uses $O(\log T)$ tree levels per epoch. Effective privacy loss is determined by the configured Gaussian noise scale and lifetime RDP composition.

---

## Configuration

Production and experimental policies are specified in [`configs/privacy.yaml`](configs/privacy.yaml):

- `global_budget`: Enterprise-wide $(\epsilon, \delta)$ privacy ceiling.
- `accountant`: RDP composition mode, evaluation orders $\alpha \in [1.25, 128]$, and strict fail-closed enforcement.
- `recipient_composition`: Composition scope policy (`global_external`, `per_recipient`, `collusion_group`).
- `release`: Replay memoization (`reuse_existing_release: true`) and overrun prevention (`allow_budget_overrun: false`).
- `planner`: Static public prior distribution and expected release utility objective weights.
- `continual`: Time-tree parameters, $C_{\text{day}}$ and $C_{\text{epoch}}$ clipping bounds, and target epoch epsilon.

---

## Experiments

The repository includes reproducible research experiments covering release planning and continual observation:

- **Exp 09**: Small-$n$ Adaptive Release Planning vs Baseline Regret
- **Exp 10**: Staleness vs DP Noise Pareto Frontier
- **Exp 11**: Budget Depletion Allocation Trajectories
- **Exp 12**: Prior Robustness under Extreme Parameter Misspecification
- **Exp 13**: Release Granularity vs Direct Sensitivity Amplification
- **Exp 14**: End-to-End Planning Verification
- **Exp 15**: Continual Binary Tree vs Independent DP Scaling
- **Exp 16**: Prefix Query Noise Accumulation
- **Exp 17**: Sliding-Window Continual Differential Privacy
- **Exp 18**: Prefix-Stable Temporal Contribution Clipping Tradeoff
- **Exp 19**: Long-Horizon Continual System Scaling & Latency Footprint

### Reproducing Experiments

```bash
# Fast smoke verification of all experiment pipelines
python experiments/run_all_experiments.py --smoke

# Full formal experiment suite
python experiments/run_all_experiments.py

# Run all 30 formal mathematical and architectural verifications
python -m pytest tests/ -q
python experiments/verify_smoke_non_destructive.py
```

Generated tables are stored in `results/tables/`, figures in `results/figures/`, and key metrics in `results/tables/v04_key_results.json`.

---

## Repository Structure

```text
private-llm-analytics/
├── configs/            # System and privacy configuration YAMLs
├── experiments/        # Formal research experiments (Exp01–Exp19) & 30 verification scripts
├── results/            # Research tables (CSV/JSON), publication figures (PDF/PNG), state DB
├── src/                # Core library
│   └── private_llm_analytics/
│       ├── continual/  # Binary time tree, continual query engine, and temporal mechanisms
│       ├── evaluation/ # Analytical utility estimators and accuracy metrics
│       ├── privacy/    # Differential privacy mechanisms, state manager, and RDP accounting
│       ├── service/    # High-level analytics service API
│       └── web/        # Local Web Analytics Console (FastAPI, zero external egress)
└── tests/              # Comprehensive pytest verification suite (439 tests)
```

---

## Security

Please see [`SECURITY.md`](SECURITY.md) for security policies, vulnerability reporting guidelines, and architectural safety invariants.

---

## Limitations

- **Central Differential Privacy**: Assumes a trusted curator operating within the enterprise security boundary; does not support local DP (LDP) or shuffle/federated models.
- **Continual Metric Scope**: The continual observation tree mechanism is verified for Interaction Volume with the Gaussian mechanism. Continual prevalence and normalized share mechanisms remain subjects for future releases.
- **Fixed Public Topology**: Tree structure and interval partitions are public, fixed, and data-independent. Dynamic or data-dependent tree topologies are not supported.
- **Temporal Clipping Bias**: Prefix-stable online clipping introduces deterministic clipping bias for high-frequency users across long horizons, which must be balanced against additive DP noise variance.

---

## Citation

```bibtex
@misc{private_llm_analytics,
  title={Private LLM Analytics: User-Level Differential Privacy for Enterprise LLM Telemetry},
  author={Private LLM Analytics Research Team},
  year={2026},
  url={https://github.com/thy919100815-art/private-llm-analytics}
}
```

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
