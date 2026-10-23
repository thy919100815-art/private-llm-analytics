"""Pydantic configuration models for Private LLM Analytics v0.1.1."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    source: Literal["synthetic", "csv", "parquet"]
    path: Optional[str] = None


class PopulationConfig(BaseModel):
    mode: Literal["fixed_public_population"] = "fixed_public_population"
    n_users: int = Field(default=10000, gt=0)
    n_days: int = Field(default=30, gt=0)


class UsageFrequencyConfig(BaseModel):
    distribution: Literal["normal", "truncated_normal"] = "normal"
    mean: float = Field(default=60.0, gt=0)
    std: float = Field(default=20.0, gt=0)
    minimum: int = Field(default=5, ge=1)
    maximum: int = Field(default=150, ge=5)

    @model_validator(mode="after")
    def check_min_max(self) -> "UsageFrequencyConfig":
        if self.minimum >= self.maximum:
            raise ValueError(f"minimum ({self.minimum}) must be less than maximum ({self.maximum})")
        return self


class ScenarioConfig(BaseModel):
    dataset: DatasetConfig
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    providers: List[str] = Field(default_factory=lambda: ["provider_a", "provider_b", "provider_c"])
    tasks: List[str] = Field(default_factory=lambda: ["technical", "document", "analysis"])
    usage_frequency: UsageFrequencyConfig = Field(default_factory=UsageFrequencyConfig)
    random_seed: int = 42

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ScenarioConfig":
        if len(self.providers) != 3:
            raise ValueError(f"v0.1.1 strictly requires exactly 3 providers, got {len(self.providers)}")
        if len(self.tasks) != 3:
            raise ValueError(f"v0.1.1 strictly requires exactly 3 tasks, got {len(self.tasks)}")
        return self


class GlobalBudgetConfig(BaseModel):
    epsilon: float = Field(gt=0)
    delta: float = Field(default=1e-6, ge=0.0, lt=1.0)


class PrivacyAccountantConfig(BaseModel):
    mode: Literal["basic", "rdp"] = "rdp"
    type: str = "basic"
    epsilon_cap: float = Field(default=4.0, gt=0)
    delta_global: float = Field(default=1e-6, gt=0.0, lt=1.0)
    orders: List[float] = Field(
        default_factory=lambda: [1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 16.0, 32.0, 64.0, 128.0]
    )

    @model_validator(mode="after")
    def validate_accountant_config(self) -> "PrivacyAccountantConfig":
        if self.type != "basic":
            raise ValueError(
                f"Unsupported accountant type '{self.type}'. v0.1.1 strictly supports only 'basic' "
                f"persistent privacy accounting (RDP is scheduled for v0.2)."
            )
        if self.mode not in ("basic", "rdp"):
            raise ValueError(f"Unsupported accountant mode '{self.mode}'. Choose 'basic' or 'rdp'.")
        if self.mode == "rdp":
            if not self.orders:
                raise ValueError("RDP orders cannot be empty when accountant mode is 'rdp'.")
            for alpha in self.orders:
                if alpha <= 1.0:
                    raise ValueError(f"RDP order alpha must be strictly greater than 1.0, got {alpha}")
            if self.delta_global <= 0.0 or self.delta_global >= 1.0:
                raise ValueError(f"delta_global must be in (0, 1), got {self.delta_global}")
            if self.epsilon_cap <= 0.0:
                raise ValueError(f"epsilon_cap must be > 0, got {self.epsilon_cap}")
        return self


class PrivacyHistoryConfig(BaseModel):
    mode: Literal["lifetime", "rolling_reset"] = "lifetime"
    planning_horizon_days: Optional[int] = None


class RecipientCompositionConfig(BaseModel):
    mode: Literal["per_recipient", "collusion_group", "global_external"] = "global_external"
    groups: Dict[str, List[str]] = Field(default_factory=dict)
    recipient_roles: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_no_overlapping_groups(self) -> "RecipientCompositionConfig":
        seen_recipients: Dict[str, str] = {}
        for group_name, members in self.groups.items():
            for member in members:
                if member in seen_recipients:
                    raise ValueError(
                        f"Collusion group overlap detected: recipient '{member}' appears in both "
                        f"'{seen_recipients[member]}' and '{group_name}'. Overlapping collusion groups are strictly forbidden."
                    )
                seen_recipients[member] = group_name
        return self


class DefaultMechanismConfig(BaseModel):
    type: Literal["gaussian", "laplace"] = "gaussian"


class PublicPriorConfig(BaseModel):
    distribution: Literal["truncated_normal", "lognormal", "contribution_model"] = "contribution_model"
    mean: float = Field(default=60.0, gt=0)
    std: float = Field(default=20.0, gt=0)
    lower: float = Field(default=5.0, ge=0)
    upper: float = Field(default=150.0, gt=0)
    public_seed: int = 900001
    num_reference_users: int = 2000

    @model_validator(mode="after")
    def validate_bounds(self) -> "PublicPriorConfig":
        if self.lower >= self.upper:
            raise ValueError(f"lower ({self.lower}) must be strictly less than upper ({self.upper})")
        return self


class UtilityWeightsConfig(BaseModel):
    objective_mode: Literal["EXPECTED_RELEASE_L1", "COMPONENT_SURROGATE"] = "EXPECTED_RELEASE_L1"
    data: Optional[float] = Field(default=None, ge=0)
    clipping: float = Field(default=1.0, ge=0)
    noise: float = Field(default=1.0, ge=0)
    aggregation: float = Field(default=1.0, ge=0)
    staleness: float = Field(default=0.5, ge=0)
    window: float = Field(default=0.2, ge=0)

    @model_validator(mode="after")
    def validate_non_zero_sum(self) -> "UtilityWeightsConfig":
        if self.data is None:
            if (
                self.clipping == 0
                and self.noise == 0
                and self.aggregation == 0
                and self.staleness == 0
                and self.window == 0
            ):
                self.data = 0.0
            else:
                self.data = 1.0

        if self.objective_mode == "EXPECTED_RELEASE_L1":
            total = self.data + self.staleness + self.window
        else:
            total = self.clipping + self.noise + self.aggregation + self.staleness + self.window

        if total <= 0:
            raise ValueError("Sum of utility weights must be strictly positive.")
        return self


class PlannerWorkloadConfig(BaseModel):
    default_arrival_rate: float = Field(default=0.1, gt=0)
    full_9_cell_weight: float = Field(default=0.4, ge=0)
    provider_weight: float = Field(default=0.3, ge=0)
    task_weight: float = Field(default=0.2, ge=0)
    overall_weight: float = Field(default=0.1, ge=0)
    provider_shares: Dict[str, float] = Field(
        default_factory=lambda: {"provider_a": 0.5, "provider_b": 0.3, "provider_c": 0.2}
    )
    task_shares: Dict[str, float] = Field(
        default_factory=lambda: {"technical": 0.4, "document": 0.35, "analysis": 0.25}
    )

    @model_validator(mode="after")
    def validate_workload_weights(self) -> "PlannerWorkloadConfig":
        total = self.full_9_cell_weight + self.provider_weight + self.task_weight + self.overall_weight
        if total <= 0:
            raise ValueError("Sum of workload weights must be strictly positive.")
        return self


class WindowModelConfig(BaseModel):
    reference_days: int = Field(default=30, gt=0)
    mean_scaling: Literal["linear", "sublinear"] = "linear"
    std_scaling: Literal["sqrt_time", "linear"] = "sqrt_time"
    temporal_averaging_penalty: float = Field(default=0.2, ge=0)


class PlannerConfig(BaseModel):
    enabled: bool = False
    mode: Literal["static_public", "dp_feedback", "oracle_eval_only"] = "static_public"
    prior: PublicPriorConfig = Field(default_factory=PublicPriorConfig)
    weights: UtilityWeightsConfig = Field(default_factory=UtilityWeightsConfig)
    workloads: PlannerWorkloadConfig = Field(default_factory=PlannerWorkloadConfig)
    window_model: WindowModelConfig = Field(default_factory=WindowModelConfig)
    candidate_clipping: List[float] = Field(
        default_factory=lambda: [5.0, 10.0, 20.0, 40.0, 80.0]
    )
    candidate_granularities: List[str] = Field(
        default_factory=lambda: ["full_9_cell", "provider_3_cell", "task_3_cell", "overall_1_cell"]
    )
    candidate_windows: List[int] = Field(
        default_factory=lambda: [7, 14, 30, 60, 90]
    )
    candidate_frequencies: List[int] = Field(
        default_factory=lambda: [1, 7, 14, 30]
    )
    planning_horizon_days: int = Field(default=90, gt=0)


class PrivacyReleaseConfig(BaseModel):
    reuse_existing_release: bool = True
    allow_budget_overrun: bool = False
    state_db_path: str = "results/privacy_state.db"
    single_flight_timeout: float = 30.0
    lease_timeout: float = 300.0


class ContinualTreeConfigModel(BaseModel):
    topology: Literal["binary"] = "binary"
    padding: Literal["next_power_of_two"] = "next_power_of_two"


class ContinualNoiseConfigModel(BaseModel):
    mechanism: Literal["gaussian"] = "gaussian"
    target_epoch_epsilon: float = 2.0
    calibration_delta: float = 1e-6
    equal_sigma_per_level: bool = True


class ContinualConfigModel(BaseModel):
    enabled: bool = False
    mode: Literal["binary_tree"] = "binary_tree"
    epoch_days: int = 90
    timezone: str = "UTC"
    daily_l2_bound: float = 10.0
    epoch_l12_bound: float = 40.0
    lateness_grace_days: int = 1
    tree: ContinualTreeConfigModel = Field(default_factory=ContinualTreeConfigModel)
    noise: ContinualNoiseConfigModel = Field(default_factory=ContinualNoiseConfigModel)


class PrivacyConfig(BaseModel):
    privacy_unit: Literal["user"] = "user"
    adjacency: Literal["add_remove"] = "add_remove"
    horizon: Dict[str, Any] = Field(default_factory=lambda: {"days": 90})
    global_budget: GlobalBudgetConfig
    default_query_budget: GlobalBudgetConfig = Field(
        default_factory=lambda: GlobalBudgetConfig(epsilon=1.0, delta=1e-6)
    )
    accountant: PrivacyAccountantConfig = Field(default_factory=PrivacyAccountantConfig)
    history: PrivacyHistoryConfig = Field(default_factory=PrivacyHistoryConfig)
    recipient_composition: RecipientCompositionConfig = Field(default_factory=RecipientCompositionConfig)
    default_mechanism: DefaultMechanismConfig = Field(default_factory=DefaultMechanismConfig)
    release: PrivacyReleaseConfig = Field(default_factory=PrivacyReleaseConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    continual: ContinualConfigModel = Field(default_factory=ContinualConfigModel)


class ContributionConfig(BaseModel):
    mode: Literal["binary", "l1_normalized", "l2_clip", "joint_l2_clip"]
    max_cells_per_user: Optional[int] = 9
    norm_bound: Optional[float] = 1.0
    clip_norm: Optional[float] = 20.0
    clip_norm_rate: Optional[float] = 10.0
    min_noisy_denominator: Optional[float] = 5.0

    @model_validator(mode="after")
    def validate_bounds(self) -> "ContributionConfig":
        if self.mode == "l1_normalized":
            if self.norm_bound is None or abs(self.norm_bound - 1.0) > 1e-12:
                raise ValueError(
                    f"User-Normalized Share strictly requires norm_bound == 1.0 in v0.1.2, got {self.norm_bound}."
                )
        elif self.mode == "binary":
            if self.max_cells_per_user is None or self.max_cells_per_user != 9:
                raise ValueError(
                    f"User Prevalence strictly requires max_cells_per_user == 9 in v0.1.2, got {self.max_cells_per_user}."
                )
        elif self.mode == "l2_clip":
            if self.clip_norm is None or self.clip_norm <= 0:
                raise ValueError(
                    f"Interaction Volume requires clip_norm > 0, got {self.clip_norm}."
                )
        elif self.mode == "joint_l2_clip":
            if self.clip_norm_rate is None or self.clip_norm_rate <= 0:
                raise ValueError(
                    f"Success Rate requires clip_norm_rate > 0, got {self.clip_norm_rate}."
                )
        return self


class MechanismItemConfig(BaseModel):
    type: Literal["gaussian", "laplace"] = "gaussian"


class MetricDefinition(BaseModel):
    enabled: bool = True
    type: Literal["prevalence", "normalized_share", "volume", "success_rate"]
    version: int = 1
    dimensions: List[str] = Field(default_factory=lambda: ["provider", "task"])
    contribution: ContributionConfig
    mechanism: MechanismItemConfig = Field(default_factory=MechanismItemConfig)


class MetricsConfig(BaseModel):
    metrics: Dict[str, MetricDefinition]


class RecipientPolicy(BaseModel):
    role: Literal["trusted", "external"]
    providers: List[str]
    metrics: List[str]
    exact_results: bool = False
    dp_results: bool = True
    cross_provider_comparison: bool = False

    @model_validator(mode="after")
    def check_external_exact(self) -> "RecipientPolicy":
        if self.role == "external" and self.exact_results:
            raise ValueError("Security violation: External recipients cannot be configured with exact_results=True.")
        return self


class RecipientsConfig(BaseModel):
    recipients: Dict[str, RecipientPolicy]


class AppConfig(BaseModel):
    scenario: ScenarioConfig
    privacy: PrivacyConfig
    metrics: MetricsConfig
    recipients: RecipientsConfig
