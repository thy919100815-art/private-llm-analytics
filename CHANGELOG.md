# Changelog

All notable changes to **Private LLM Analytics** are documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0]

### Added
- **Local Web Console**: Localhost FastAPI analytics console with zero external network egress and no frontend build system.
- **Authoritative Trailing-Window Release Selection**: Trailing 7/14/30/60/90-day release windows derived strictly from dataset maximum timestamps.
- **Privacy-Budget and Release-History Inspection**: Real-time monitoring of cumulative lifetime RDP budget spend and sanitized persistent audit ledger.
- **Utility-Planner Preview**: Zero-spend preview of recommended clipping bounds and utility loss decompositions with explicit recommendation-only labeling.
- **Local Telemetry Ingestion**: Support for local CSV and Parquet telemetry uploads with schema and taxonomy validation.
- **Canonical DP Release Replay UX**: Replay memoization indicators with zero incremental privacy cost on repeated queries.
- **Continual-Observation Status**: Inspection of binary time-tree continual observation with fail-closed stream initialization semantics.
- **Test Suite Expansion**: Comprehensive regression suite increased from 421 to 448 tests.

## [0.4.0]

### Added
- **User-Level Continual Observation**: Prefix-stable online chronological contribution bounding ($C_{\text{day}}$ and $C_{\text{epoch}}$) over temporal telemetry streams.
- **Binary Time Tree Query Engine**: Canonical dyadic interval decomposition supporting arbitrary $[s, e]$ time ranges and sliding-window releases with node reuse.
- **Cross-Epoch Materialization**: Fail-closed cross-epoch query planner with strict finalized frontier verification.
- **Formal Continual DP Experiments**: Research suites and empirical validations for temporal privacy scaling, prefix queries, sliding windows, and latency footprints (Exp 15–19).

## [0.3.0]

### Added
- **Adaptive Release Planning**: Small-$n$ joint optimization across clipping bounds, release granularities, and observation frequencies.
- **Utility and Loss Modeling**: Analytical estimation of DP noise variance, empirical clipping bias, staleness, and temporal averaging error.
- **Empirical Experiments**: Empirical verification of adaptive planning regret, staleness Pareto frontiers, and prior misspecification robustness (Exp 09–14).

## [0.2.0]

### Added
- **Longitudinal RDP Accounting**: Exact Gaussian Rényi Differential Privacy accumulation with conversion to $(\epsilon, \delta)$-DP across long time horizons.
- **Lifetime Privacy Horizon**: Non-replenishing cumulative privacy expenditure tracking for external recipients.
- **Multi-Recipient Composition Policies**: Fine-grained budget scoping for `GLOBAL_EXTERNAL`, `COLLUSION_GROUP`, and `PER_RECIPIENT` recipients.
- **Atomic Concurrency Hardening**: SQLite `BEGIN IMMEDIATE` compare-and-reserve transaction management, lease fencing, and single-flight memoization to eliminate duplicate charges.

## [0.1.0]

### Added
- **User-Level Central DP**: Privacy framework protecting individual user trajectories under add/remove user adjacency.
- **Core DP Mechanisms**: Analytically calibrated Gaussian and Laplace noise addition.
- **Supported Enterprise Metrics**: User prevalence, user-normalized share, and interaction volume across a $3 \times 3$ provider-task category grid.
- **Recipient Isolation**: Post-processing filtering restricting cross-provider comparison and preventing active user count leakage.
