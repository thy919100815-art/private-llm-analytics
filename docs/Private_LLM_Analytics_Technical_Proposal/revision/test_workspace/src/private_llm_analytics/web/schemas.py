"""Pydantic schemas for the Local Web Console API in v0.5."""

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


VALID_METRICS = {
    "user_prevalence",
    "user_normalized_share",
    "interaction_volume",
    "success_rate",
    "top_k_tasks",
}
FORBIDDEN_RNG_KEYS = {"rng", "seed", "noise_seed", "noise", "random_state"}


class ReleaseRequest(BaseModel):
    """Request payload for differential privacy release."""

    metric: str = Field(..., description="Analytics metric to compute")
    recipient: str = Field(..., description="Authorized recipient name")
    epsilon: float = Field(..., gt=0.0, description="Differential privacy epsilon")
    delta: float = Field(..., gt=0.0, lt=1.0, description="Differential privacy delta")
    window_days: Optional[int] = Field(None, gt=0, description="Observation window in days")
    window_start: Optional[str] = Field(None, description="ISO window start timestamp")
    window_end: Optional[str] = Field(None, description="ISO window end timestamp")

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        norm = v.lower().strip()
        if norm not in VALID_METRICS:
            raise ValueError(
                f"Invalid metric '{v}'. Allowed metrics: {sorted(list(VALID_METRICS))}"
            )
        return norm

    @field_validator("epsilon")
    @classmethod
    def validate_epsilon(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v) or v <= 0:
            raise ValueError("Epsilon must be a finite positive number.")
        return v

    @field_validator("delta")
    @classmethod
    def validate_delta(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v) or not (0 < v < 1):
            raise ValueError("Delta must be a finite number strictly between 0 and 1.")
        return v

    @model_validator(mode="before")
    @classmethod
    def reject_rng_control(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k in FORBIDDEN_RNG_KEYS:
                if k in data:
                    raise ValueError(
                        f"External caller cannot specify '{k}'. Production release strictly forbids RNG/seed control."
                    )
        return data

    @model_validator(mode="after")
    def validate_window_precedence(self) -> "ReleaseRequest":
        has_days = self.window_days is not None
        has_start = self.window_start is not None
        has_end = self.window_end is not None

        if has_days and (has_start or has_end):
            raise ValueError(
                "Cannot specify both 'window_days' and explicit 'window_start'/'window_end'."
            )

        if (has_start and not has_end) or (has_end and not has_start):
            raise ValueError(
                "Both 'window_start' and 'window_end' must be provided for explicit windowing."
            )

        return self


class PlannerRecommendRequest(BaseModel):
    """Request payload for utility planner recommendations."""

    metric: str = Field("interaction_volume", description="Metric to plan for")
    recipient: str = Field("provider_a", description="Target recipient")
    planning_horizon_days: Optional[int] = Field(90, gt=0, description="Horizon in days")

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        norm = v.lower().strip()
        if norm not in VALID_METRICS:
            raise ValueError(f"Invalid metric '{v}'. Allowed metrics: {sorted(list(VALID_METRICS))}")
        return norm


class ContinualQueryRequest(BaseModel):
    """Request payload for dyadic continual interval queries."""

    start_day: int = Field(..., ge=1, description="Start day (1-indexed)")
    end_day: int = Field(..., ge=1, description="End day (1-indexed)")
    metric: str = Field("interaction_volume", description="Continual metric")
    recipient: str = Field("provider_a", description="Target recipient")

    @model_validator(mode="after")
    def validate_interval(self) -> "ContinualQueryRequest":
        if self.start_day > self.end_day:
            raise ValueError(
                f"start_day ({self.start_day}) cannot be greater than end_day ({self.end_day})."
            )
        if self.metric.lower().strip() != "interaction_volume":
            raise ValueError(
                "Continual observation is currently supported for Interaction Volume only."
            )
        return self


class ConfigMetricItem(BaseModel):
    id: str
    name: str
    description: str
    name_zh: Optional[str] = None
    description_zh: Optional[str] = None


class ConfigResponse(BaseModel):
    providers: List[str]
    tasks: List[str]
    metrics: List[ConfigMetricItem]
    default_epsilon: float
    default_delta: float
    candidate_windows: List[int]
    recipients: List[str]
    continual_enabled: bool
    granularities: List[str]
    statistics: Optional[List[Dict[str, Any]]] = None
    recipient_details: Optional[List[Dict[str, Any]]] = None
    mechanism_cards: Optional[Dict[str, Any]] = None
    privacy_policy: Optional[Dict[str, Any]] = None


class DatasetStatusResponse(BaseModel):
    loaded: bool
    rows: int
    active_users: int
    dataset_version: Optional[str]
    source: str
    user_count: Optional[int] = None
    provider_count: Optional[int] = None
    task_count: Optional[int] = None
    time_range: Optional[Dict[str, Optional[str]]] = None
    has_success_field: Optional[bool] = None
    quality_checks: Optional[List[Dict[str, Any]]] = None


class InputPreviewResponse(BaseModel):
    total_rows: int
    preview_limit: int
    columns: List[str]
    rows: List[Dict[str, Any]]


class AccuracyEvaluationResponse(BaseModel):
    metric: str
    primary_metric_name: str
    primary_metric_display: str
    primary_metric_value: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mean_relative_error_pct: Optional[float] = None
    max_relative_error_pct: Optional[float] = None
    mean_pp_error: Optional[float] = None
    max_pp_error: Optional[float] = None
    l1_distance: Optional[float] = None
    precision_at_k: Optional[float] = None
    jaccard_similarity: Optional[float] = None
    k: Optional[int] = None
    overlap_count: Optional[int] = None
    explanation: Optional[str] = None
    rows: List[Dict[str, Any]] = Field(default_factory=list)


class EpsilonSweepRequest(BaseModel):
    metric: str = Field("interaction_volume", description="Target statistic for epsilon sweep")
    recipient: str = Field("internal", description="Requesting recipient ID (must be trusted)")
    candidate_epsilons: Optional[List[float]] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    n_trials: Optional[int] = Field(10, ge=1, le=50)

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        norm = v.lower().strip()
        if norm not in VALID_METRICS:
            raise ValueError(f"Invalid metric '{v}'. Allowed metrics: {sorted(list(VALID_METRICS))}")
        return norm


class SweepPoint(BaseModel):
    epsilon: float
    mean_error: float
    std_error: float
    median_error: float
    p90_error: float
    trials_count: int


class EpsilonSweepResponse(BaseModel):
    metric: str
    y_axis_label: str
    candidate_epsilons: List[float]
    n_trials: int
    points: List[SweepPoint]
    explanation: str


class PrivacyStatusResponse(BaseModel):
    budget_scope_id: str
    accountant_mode: str
    history_mode: str
    recipient_composition_mode: str
    effective_epsilon: float
    epsilon_cap: float
    remaining_headroom: float
    delta_global: float
    release_count: int
    best_alpha: Optional[float]
    spend_percentage: float


class PlannerRecommendResponse(BaseModel):
    recommended_clipping: Optional[float]
    recommended_granularity: str
    recommended_window_days: int
    recommended_frequency_days: int
    estimated_clipping_loss: Optional[float]
    estimated_dp_noise: Optional[float]
    estimated_utility: Optional[float]
    budget_feasible: bool
    projected_epsilon: float
    decision_reason: str
    is_recommendation_only: bool = True


class ReleaseResponse(BaseModel):
    release_id: str
    release_key: str
    metric: str
    recipient: str
    recipient_name: Optional[str] = None
    epsilon: float
    delta: float
    noise_scale: float
    mechanism: str
    cache_hit: bool
    incremental_privacy_cost: float
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    results: Dict[str, Any]
    privacy_summary: Dict[str, Any]


class ReleaseHistoryItem(BaseModel):
    release_id: str
    timestamp: str
    metric: str
    recipient: str
    recipient_name: Optional[str] = None
    budget_scope_id: str
    window_start: Optional[str]
    window_end: Optional[str]
    mechanism: str
    epsilon: float
    delta: float
    cache_hit: bool
    status: str = "COMMITTED"
    download_available: bool = True


class AdminPrivacyPolicyResponse(BaseModel):
    lifetime_epsilon_cap: float
    global_delta: float
    default_query_epsilon: float
    default_query_delta: float
    current_effective_epsilon: float
    audit_history: List[Dict[str, Any]] = []


class AdminPrivacyPolicyUpdateRequest(BaseModel):
    recipient: str = Field("internal", description="Requesting recipient (must be trusted admin)")
    lifetime_epsilon_cap: float = Field(..., gt=0.0, description="New lifetime epsilon cap")
    global_delta: float = Field(..., gt=0.0, lt=1.0, description="New global delta")
    default_query_epsilon: float = Field(..., gt=0.0, description="Default query epsilon")
    default_query_delta: float = Field(..., gt=0.0, lt=1.0, description="Default query delta")
    reason: Optional[str] = Field("", description="Governance reason for policy modification")

    @field_validator("lifetime_epsilon_cap")
    @classmethod
    def validate_cap(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v) or v <= 0:
            raise ValueError("Lifetime epsilon cap must be a finite positive number.")
        return v

    @field_validator("global_delta")
    @classmethod
    def validate_gdelta(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v) or not (0 < v < 1):
            raise ValueError("Global delta must be a finite number strictly between 0 and 1.")
        return v

    @field_validator("default_query_epsilon")
    @classmethod
    def validate_qeps(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v) or v <= 0:
            raise ValueError("Default query epsilon must be a finite positive number.")
        return v

    @field_validator("default_query_delta")
    @classmethod
    def validate_qdelta(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v) or not (0 < v < 1):
            raise ValueError("Default query delta must be a finite number strictly between 0 and 1.")
        return v


class ContinualStatusResponse(BaseModel):
    enabled: bool
    mode: str
    epoch_days: int
    daily_l2_bound: float
    epoch_group_l12_bound: float
    tree_topology: str
    supported_metric: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    type: str
    details: Optional[Any] = None
