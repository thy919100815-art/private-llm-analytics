"""Statistic-Specific DP Mechanism Framework.

Each supported statistic owns a complete privacy-preserving release profile:
statistical target, contribution definition, contribution bounding, sensitivity,
approved DP mechanism, privacy cost descriptor, post-processing, utility model, and validation.
"""

from private_llm_analytics.statistics.base import (
    DPStatisticProfile,
    ProfileMetadata,
    MechanismPlan,
    PrivacyCostDescriptor,
)
from private_llm_analytics.statistics.prevalence import UserPrevalenceProfile
from private_llm_analytics.statistics.normalized_share import UserNormalizedShareProfile
from private_llm_analytics.statistics.interaction_volume import InteractionVolumeProfile
from private_llm_analytics.statistics.success_rate import SuccessRateProfile
from private_llm_analytics.statistics.registry import StatisticProfileRegistry

__all__ = [
    "DPStatisticProfile",
    "ProfileMetadata",
    "MechanismPlan",
    "PrivacyCostDescriptor",
    "UserPrevalenceProfile",
    "UserNormalizedShareProfile",
    "InteractionVolumeProfile",
    "SuccessRateProfile",
    "StatisticProfileRegistry",
]
