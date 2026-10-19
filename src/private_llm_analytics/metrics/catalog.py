"""Metric catalog and factory."""

from typing import Dict, Optional
from private_llm_analytics.config.models import MetricsConfig
from private_llm_analytics.metrics.base import BaseMetric
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.metrics.normalized_share import UserNormalizedShareMetric
from private_llm_analytics.metrics.prevalence import UserPrevalenceMetric


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
