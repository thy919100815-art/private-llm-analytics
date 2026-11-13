import numpy as np
import pandas as pd
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.evaluation.engine import evaluate_prevalence_accuracy

service = AnalyticsService.from_config_dir("configs")
service.load_data("data/interactions_n50000.parquet")
user_matrix = service.get_user_matrix()

exp_res = service.compute_experiment(
    metric="user_prevalence",
    recipient="internal",
    epsilon=1.0,
    delta=1e-6,
    include_exact=True,
    noise_seed=42,
)

eval_res = evaluate_prevalence_accuracy(user_matrix, exp_res["results"])
print("Summary:", eval_res["primary_metric_name"], eval_res["primary_metric_display"])
for row in eval_res["rows"]:
    print(row)
