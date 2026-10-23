"""authoritative registry for statistic-specific dp profiles."""

from __future__ import annotations
from typing import Dict, List, Optional
import threading

from private_llm_analytics.privacy.exceptions import UnknownStatisticProfile
from private_llm_analytics.statistics.base import DPStatisticProfile, ProfileMetadata
from private_llm_analytics.statistics.prevalence import UserPrevalenceProfile
from private_llm_analytics.statistics.normalized_share import UserNormalizedShareProfile
from private_llm_analytics.statistics.interaction_volume import InteractionVolumeProfile
from private_llm_analytics.statistics.success_rate import SuccessRateProfile


class StatisticProfileRegistry:
    """Thread-safe registry for approved statistic-specific DP profiles."""

    _lock = threading.Lock()
    _profiles: Dict[str, DPStatisticProfile] = {}
    _initialized = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        with cls._lock:
            if not cls._initialized:
                cls._profiles = {
                    "user_prevalence": UserPrevalenceProfile(),
                    "user_normalized_share": UserNormalizedShareProfile(),
                    "interaction_volume": InteractionVolumeProfile(),
                    "success_rate": SuccessRateProfile(),
                }
                cls._initialized = True

    @classmethod
    def register(cls, profile: DPStatisticProfile) -> None:
        cls._ensure_initialized()
        with cls._lock:
            cls._profiles[profile.statistic_id] = profile

    @classmethod
    def get(cls, statistic_id: str) -> DPStatisticProfile:
        cls._ensure_initialized()
        with cls._lock:
            profile = cls._profiles.get(statistic_id)
            if profile is None:
                raise UnknownStatisticProfile(
                    f"Statistic profile '{statistic_id}' is not registered. "
                    f"Available statistics: {list(cls._profiles.keys())}"
                )
            return profile

    @classmethod
    def has(cls, statistic_id: str) -> bool:
        cls._ensure_initialized()
        with cls._lock:
            return statistic_id in cls._profiles

    @classmethod
    def list_statistic_ids(cls) -> List[str]:
        cls._ensure_initialized()
        with cls._lock:
            return list(cls._profiles.keys())

    list_statistics = list_statistic_ids

    @classmethod
    def get_metadata(cls, statistic_id: str) -> ProfileMetadata:
        return cls.get(statistic_id).metadata()

    @classmethod
    def get_all(cls) -> Dict[str, DPStatisticProfile]:
        cls._ensure_initialized()
        with cls._lock:
            return dict(cls._profiles)

    @classmethod
    def get_metadata_list(cls) -> List[ProfileMetadata]:
        cls._ensure_initialized()
        with cls._lock:
            return [p.metadata() for p in cls._profiles.values()]

    @classmethod
    def reset(cls) -> None:
        """Reset the registry to initial approved default profiles."""
        with cls._lock:
            cls._profiles = {
                "user_prevalence": UserPrevalenceProfile(),
                "user_normalized_share": UserNormalizedShareProfile(),
                "interaction_volume": InteractionVolumeProfile(),
                "success_rate": SuccessRateProfile(),
            }
            cls._initialized = True
