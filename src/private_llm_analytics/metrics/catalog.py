"""Metric catalog and factory."""

from typing import Dict, Optional
from private_llm_analytics.config.models import MetricsConfig
from private_llm_analytics.metrics.base import BaseMetric
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.metrics.normalized_share import UserNormalizedShareMetric
from private_llm_analytics.metrics.prevalence import UserPrevalenceMetric
from private_llm_analytics.metrics.success_rate import SuccessRateMetric
from private_llm_analytics.metrics.top_k_tasks import TopKTasksMetric


class MetricCatalog:
    """Registry and factory for configured metrics."""

    def __init__(self, metrics_config: Optional[MetricsConfig] = None):
        self._metrics: Dict[str, BaseMetric] = {}
        if metrics_config:
            self._init_from_config(metrics_config)
        else:
            self._init_defaults()

    def _init_defaults(self):
        self.register(UserPrevalenceMetric())
        self.register(UserNormalizedShareMetric())
        self.register(InteractionVolumeMetric())
        self.register(SuccessRateMetric())
        self.register(TopKTasksMetric())

    def _init_from_config(self, cfg: MetricsConfig):
        for name, m_def in cfg.metrics.items():
            if not m_def.enabled:
                continue
            if m_def.type == "prevalence":
                k = m_def.contribution.max_cells_per_user or 9
                self.register(UserPrevalenceMetric(name=name, max_cells_per_user=k))
            elif m_def.type == "normalized_share":
                bound = m_def.contribution.norm_bound or 1.0
                self.register(UserNormalizedShareMetric(name=name, norm_bound=bound))
            elif m_def.type == "volume":
                c = m_def.contribution.clip_norm or 20.0
                self.register(InteractionVolumeMetric(name=name, clip_norm=c))
            elif m_def.type == "success_rate":
                c_rate = getattr(m_def.contribution, "clip_norm_rate", 10.0) or 10.0
                min_den = getattr(m_def.contribution, "min_noisy_denominator", 5.0) or 5.0
                self.register(SuccessRateMetric(name=name, clip_norm_rate=c_rate, min_noisy_denominator=min_den))
            elif m_def.type == "top_k_tasks":
                top_k = getattr(m_def.contribution, "top_k", 5) or 5
                k_bound = getattr(m_def.contribution, "max_tasks_per_user", 5) or 5
                thresh = getattr(m_def.contribution, "threshold", None)
                self.register(TopKTasksMetric(name=name, k=top_k, max_tasks_per_user=k_bound, threshold=thresh))
            else:
                raise ValueError(f"Unknown metric type: {m_def.type}")

    def register(self, metric: BaseMetric) -> None:
        self._metrics[metric.name] = metric

    def get(self, name: str) -> BaseMetric:
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found. Available: {list(self._metrics.keys())}")
        return self._metrics[name]

    def list_metrics(self) -> Dict[str, BaseMetric]:
        return dict(self._metrics)
