# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| v0.4.x  | :white_check_mark: |
| v0.3.x  | :white_check_mark: |
| < v0.3  | :x:                |

## Reporting a Vulnerability

If you discover a potential privacy or security vulnerability in **Private LLM Analytics**, please do not report it through public GitHub issues or discussions.

Instead, please send a private vulnerability report detailing:
1. Description of the vulnerability and attack scenario (e.g. privacy budget evasion, RNG control, memoization desynchronization).
2. Minimal reproduction script or configuration.
3. Observed privacy loss or invariant breach.
4. Proposed remediation if available.

Reports will be acknowledged within 48 hours, followed by coordinated verification and patch development.

---

## Core Security & Privacy Invariants

The architecture enforces the following mandatory invariants across all execution paths:

1. **No External RNG / Seed Control**: External callers cannot supply or influence random number generators, seeds, or noise vectors (`ProductionRNGControlError`). Production releases use unseeded entropy exclusively.
2. **Experiment Isolation**: Experiment-mode executions (`compute_experiment` / `ExperimentRunner`) write to isolated databases and are barred from serving external recipients.
3. **Fail-Closed Budget Fencing**: Budget checks execute in SQLite atomic transactions (`BEGIN IMMEDIATE`) with exact Python `Decimal` arithmetic. No budget overruns are permitted.
4. **Strict Add/Remove User Adjacency**: Active user counts and private denominators are never released externally. All prevalence rates strictly use the immutable public roster size ($N_{\text{public}}$).
5. **Single-Flight Memoization**: Identical semantic queries share in-flight generation and retrieve bitwise-identical memoized releases with zero duplicate privacy charges, preventing noise averaging attacks.
6. **Authoritative Finalized Frontier**: Continuous observation and cross-epoch queries fail closed if data within the requested interval is unfinalized or manifests are unsealed.
