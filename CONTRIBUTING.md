# Contributing to Private LLM Analytics

Thank you for your interest in contributing to **Private LLM Analytics**.

Because this project enforces strict differential privacy guarantees and security invariants for enterprise environments, all code changes are subject to rigorous verification.

---

## 1. Development Setup

Clone the repository and install the editable package with development dependencies:

```bash
git clone https://github.com/thy919100815-art/private-llm-analytics.git
cd private-llm-analytics
pip install -e ".[dev]"
```

Supported Python versions: **3.10**, **3.11**, and **3.12**.

---

## 2. Running Tests & Verifications

Before opening a pull request, ensure the entire test and verification suite passes locally:

```bash
# 1. Run full unit and integration test suite (420+ tests)
python -m pytest tests/ -q

# 2. Run fast smoke verification
python experiments/run_all_experiments.py --smoke

# 3. Run non-destructive artifact safety check
python experiments/verify_smoke_non_destructive.py
```

All tests and formal verification scripts must pass without failures.

---

## 3. Pull Request Requirements

- **Branching**: Branch off `main` for bug fixes and documentation; create feature branches for larger proposals.
- **Documentation**: Code changes must update relevant docstrings, configuration docs, or README sections.
- **Code Style**: Maintain clean, idiomatic Python with complete type hints (`mypy`/PEP 484 compliant).
- **Clean Git History**: Avoid merge commits with conflicts; write concise, descriptive commit messages.

---

## 4. Privacy & Security-Sensitive Expectations

Any pull request touching differential privacy mechanisms, state tracking, or recipient filtering must strictly preserve the following architectural invariants:

1. **No External RNG / Seed Control**: External callers must never control or influence random number generators, noise seeds, or perturbation vectors (`ProductionRNGControlError`).
2. **Experiment Isolation**: Research and experiment code must never share production database paths or serve external recipients.
3. **Fail-Closed Budget Fencing**: All budget accounting and reservation operations must execute in atomic transactions (`BEGIN IMMEDIATE`) with exact Python `Decimal` arithmetic.
4. **Strict Add/Remove User Adjacency**: Active user counts and private denominators must never be released externally.
5. **Deterministic Post-Processing**: Continual observation queries must reuse canonical materialized tree nodes; query execution must perform zero noise addition or raw data access at query time.
6. **Finalized Frontier Barrier**: Unfinalized temporal data must never be materialized into queryable canonical tree nodes.

Security vulnerabilities should be reported privately as outlined in [`SECURITY.md`](SECURITY.md).
