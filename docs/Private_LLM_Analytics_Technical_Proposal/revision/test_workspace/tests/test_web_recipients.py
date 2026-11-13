"""Unit and security tests for Recipient Display Names and Scope Isolation.

Verifies Section 35:
1. recipient config loaded with display names;
2. changing display name in memory/config does NOT change recipient_id;
3. changing display name does NOT reset lifetime budget;
4. changing display name does NOT split privacy ledger;
5. changing display name does NOT change canonical release identity;
6. changing display name does NOT create a different privacy scope;
7. authorization continues to strictly enforce stable recipient_id;
8. external recipient cannot see other providers' scoped data even if human label is changed.
"""

import copy
from pathlib import Path
import numpy as np
import pytest

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.models import RecipientPolicy
from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.planner.planner import AnalyticsPlanner
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.composition import resolve_budget_scope_id
from private_llm_analytics.privacy.exceptions import RecipientSecurityViolation
from private_llm_analytics.release.recipients import filter_release_for_recipient

CONFIGS_DIR = Path("configs")


def test_01_recipient_config_loaded_with_display_names():
    """Verify default recipient configurations load with expected human-readable display names."""
    config = load_app_config(CONFIGS_DIR)
    recipients = config.recipients.recipients

    assert "internal" in recipients
    assert "provider_a" in recipients
    assert "provider_b" in recipients
    assert "provider_c" in recipients

    assert recipients["internal"].get_display_name("internal") == "内部 AI 治理团队"
    assert recipients["provider_a"].get_display_name("provider_a") == "OpenAI"
    assert recipients["provider_b"].get_display_name("provider_b") == "Kimi"
    assert recipients["provider_c"].get_display_name("provider_c") == "DeepSeek"


def test_02_changing_display_name_does_not_change_recipient_id():
    """Verify changing display_name does not alter machine recipient_id key."""
    config = load_app_config(CONFIGS_DIR)
    recipients = config.recipients.recipients

    policy_a = recipients["provider_a"]
    original_id = "provider_a"
    policy_a.display_name = "Modified Human Display Name"

    # recipient_id is the dictionary key and machine identifier, unchanged
    assert original_id in recipients
    assert recipients[original_id].display_name == "Modified Human Display Name"
    assert original_id == "provider_a"


def test_03_04_changing_display_name_does_not_reset_budget_or_split_ledger():
    """Verify budget scope resolution and ledger binding depend on stable recipient_id and role, not display name."""
    config = load_app_config(CONFIGS_DIR)
    policy = config.recipients.recipients["provider_a"]

    # Global external scope mode
    scope_default = resolve_budget_scope_id(
        recipient="provider_a",
        recipient_role=policy.role,
        mode="global_external",
    )

    policy.display_name = "Completely New Label"
    scope_after_label_change = resolve_budget_scope_id(
        recipient="provider_a",
        recipient_role=policy.role,
        mode="global_external",
    )
    assert scope_default == scope_after_label_change == "global_external"

    # Per-recipient scope mode
    scope_per_rec_1 = resolve_budget_scope_id(
        recipient="provider_a",
        recipient_role=policy.role,
        mode="per_recipient",
    )
    policy.display_name = "Yet Another Name"
    scope_per_rec_2 = resolve_budget_scope_id(
        recipient="provider_a",
        recipient_role=policy.role,
        mode="per_recipient",
    )
    assert scope_per_rec_1 == scope_per_rec_2 == "recipient:provider_a"


def test_05_06_changing_display_name_does_not_change_canonical_release_identity_or_privacy_scope():
    """Verify that canonical semantics and release fingerprint are invariant to display name changes."""
    config_1 = load_app_config(CONFIGS_DIR)
    config_2 = copy.deepcopy(config_1)

    # Change display name in config_2
    config_2.recipients.recipients["provider_a"].display_name = "Super Altered Name"

    planner_1 = AnalyticsPlanner(config_1)
    planner_2 = AnalyticsPlanner(config_2)

    query = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        window_start="2026-01-01",
        window_end="2026-01-30",
        epsilon=1.0,
        delta=1e-6,
    )

    plan_1 = planner_1.create_plan(query)
    plan_2 = planner_2.create_plan(query)

    assert plan_1.release_key == plan_2.release_key
    assert plan_1.canonical_semantics == plan_2.canonical_semantics
    assert plan_1.privacy_scope_id == plan_2.privacy_scope_id


def test_07_08_authorization_and_isolation_strictly_enforce_stable_recipient_id():
    """Verify security isolation: external recipients cannot get exact data or competitor data even if display name mimics trusted roles."""
    config = load_app_config(CONFIGS_DIR)
    policy_a = copy.deepcopy(config.recipients.recipients["provider_a"])

    # Attempt to spoof display name as trusted admin
    policy_a.display_name = "内部超级管理员 (Trusted Admin)"

    dummy_release = DPReleaseResult(
        metric_name="user_prevalence",
        dp_vector=np.array([105.2, 198.7]),
        post_processed={
            "prevalence_rates": {
                "provider_a:technical": 0.01052,
                "provider_b:technical": 0.01987,
            },
            "cell_volumes": {
                "provider_a:technical": 105.2,
                "provider_b:technical": 198.7,
            },
        },
        epsilon=1.0,
        delta=1e-6,
        mechanism="gaussian",
        noise_scale=1.5,
        sensitivity=1.0,
        cell_names=["provider_a:technical", "provider_b:technical"],
        n_users=1000,
        public_population_size=10000,
    )

    # 1. Exact results attempt must fail with RecipientSecurityViolation because role remains 'external'
    with pytest.raises(RecipientSecurityViolation):
        filter_release_for_recipient(
            release_result=dummy_release,
            recipient_name="provider_a",
            policy=policy_a,
            include_exact=True,
            exact_data={"exact": True},
        )

    # 2. Filtering DP results must isolate provider_a only
    filtered = filter_release_for_recipient(
        release_result=dummy_release,
        recipient_name="provider_a",
        policy=policy_a,
        include_exact=False,
    )

    # Must contain provider_a only
    rates = filtered["results"]["prevalence_rates"]
    assert "provider_a:technical" in rates
    assert "provider_b:technical" not in rates

    volumes = filtered["results"]["cell_volumes"]
    assert "provider_a:technical" in volumes
    assert "provider_b:technical" not in volumes

    # Active user count must NOT be exposed to external recipients
    assert "active_user_count" not in filtered
