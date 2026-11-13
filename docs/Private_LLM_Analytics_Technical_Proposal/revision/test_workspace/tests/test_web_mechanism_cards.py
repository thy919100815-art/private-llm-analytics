"""Tests for Differential Privacy Mechanism Cards and Explainability (v1.0.x / v1.1).

Verifies Section 32 requirements:
- Every supported statistic has complete metadata and mechanism card
- Authoritative card sections: title, target, privacy unit, adjacency, contribution,
  sensitivity, mechanism, post-processing, result meaning, accounting
- Specific mathematical invariants for all 5 statistics
"""

import pytest

from private_llm_analytics.config import load_app_config
from private_llm_analytics.statistics.cards import get_all_mechanism_cards, get_mechanism_card
from private_llm_analytics.statistics.registry import StatisticProfileRegistry


class TestMechanismCards:
    @pytest.fixture
    def app_config(self):
        return load_app_config("configs")

    def test_all_five_statistics_registered(self):
        registered = StatisticProfileRegistry.list_statistics()
        expected = {
            "user_prevalence",
            "user_normalized_share",
            "interaction_volume",
            "success_rate",
            "top_k_tasks",
        }
        assert expected.issubset(set(registered))

    def test_all_mechanism_cards_have_required_sections(self, app_config):
        all_cards = get_all_mechanism_cards(app_config)
        assert len(all_cards) == 5

        for stat_id, card in all_cards.items():
            assert card["statistic_id"] == stat_id
            assert card["display_title"]
            assert card["display_name_zh"]
            assert card["profile_name_zh"]
            assert card["privacy_unit"]
            assert card["adjacency"]
            assert card["target"]
            assert "raw_example" in card and card["raw_example"]["note"] == "示例数据，仅用于解释机制"
            assert "columns" in card["raw_example"] and len(card["raw_example"]["rows"]) > 0
            assert "user_contribution" in card and card["user_contribution"]["formula"]
            assert "contribution_bound" in card and card["contribution_bound"]["formula"]
            assert "sensitivity" in card and card["sensitivity"]["formula"]
            assert "dp_mechanism" in card and card["dp_mechanism"]["equation"]
            assert "post_processing" in card and len(card["post_processing"]["steps"]) > 0
            assert "result_meaning" in card and card["result_meaning"]["interpretation"]
            assert "privacy_accounting" in card and card["privacy_accounting"]["accounting"]

    def test_user_prevalence_card_invariants(self, app_config):
        card = get_mechanism_card("user_prevalence", app_config)
        assert "sqrt(k)" in card.sensitivity["conservative_bound"] or "Delta_2 <= sqrt(k)" in card.sensitivity["conservative_bound"]
        assert "sqrt(min(" in card.sensitivity["tight_bound"]
        assert "N_{\\text{public}}" in card.sensitivity["rate_sensitivity"] or "N_public" in card.sensitivity["rate_sensitivity"]
        assert "严禁使用私有活跃用户数" in card.post_processing["explanation"]
        assert "公开员工名册中有多少比例" in card.target
        assert "1 / N_{\\text{public}}" in card.sensitivity["coordinate_sensitivity"]

    def test_user_normalized_share_card_invariants(self, app_config):
        card = get_mechanism_card("user_normalized_share", app_config)
        assert card.sensitivity["configured_value"] == 1.0
        assert "1.0" in card.sensitivity["formula"]
        # Must explicitly describe nonnegative clamping + simplex renormalization, NOT Euclidean projection
        assert "非负截断 + 单纯形重新归一化" in card.post_processing["explanation"] or "nonnegative clamping + simplex renormalization" in card.post_processing["explanation"]
        assert "并非欧氏单纯形投影" in card.post_processing["explanation"] or "not Euclidean" in card.post_processing["explanation"].lower()
        assert "1/D" in card.post_processing["explanation"] or "1/D" in str(card.post_processing["steps"])

    def test_interaction_volume_card_invariants(self, app_config):
        card = get_mechanism_card("interaction_volume", app_config)
        assert card.sensitivity["configured_value"] == 20.0
        assert "Delta_2 = C" in card.sensitivity["formula"]
        assert "max(0" in card.post_processing["steps"][0]
        assert card.utility_decomposition is not None
        terms = [d["term"] for d in card.utility_decomposition["definitions"]]
        assert any("Raw Exact" in t for t in terms)
        assert any("Clipped Exact" in t for t in terms)
        assert any("DP Output" in t for t in terms)
        assert any("Clipping Bias" in t for t in terms)
        assert any("DP Noise Error" in t for t in terms)
        assert any("Total Error" in t for t in terms)

    def test_success_rate_card_invariants(self, app_config):
        card = get_mechanism_card("success_rate", app_config)
        # Shared user-level clipping
        assert "a_u" in card.user_contribution["formula"]
        assert "C_{\\text{rate}}" in card.user_contribution["formula"]
        # Joint sensitivity
        assert "\\sqrt{2}" in card.sensitivity["formula"]
        # Single Joint Gaussian Release
        assert "单次联合高斯机制" in card.dp_mechanism["explanation"] or "Single Joint Gaussian" in card.dp_mechanism["explanation"]
        assert "单次 RDP" in card.dp_mechanism["explanation"] or "单次 RDP" in card.privacy_accounting["accounting"]
        # Post-processing clipping to [0, 1] and denominator thresholding
        assert "None" in card.post_processing["steps"][2] or "抑制" in card.post_processing["steps"][2]

    def test_top_k_tasks_card_invariants(self, app_config):
        card = get_mechanism_card("top_k_tasks", app_config)
        # Configured finite taxonomy universe, not unbounded
        assert "有限任务" in card.target or "finite task taxonomy" in card.target.lower() or "有限任务分类体系" in card.target
        assert "并非开放/无界" in card.user_contribution["explanation"] or "not open/unbounded" in card.user_contribution["explanation"].lower()
        # Conservative and tight bounds
        assert "sqrt(k_tasks)" in card.sensitivity["conservative_bound"]
        assert "sqrt(min(" in card.sensitivity["tight_bound"]
        # DP ranking is post-processing on noisy values
        assert "后处理" in card.post_processing["explanation"]
        assert "完全在加噪后的 DP 估计值上进行" in card.post_processing["explanation"]
