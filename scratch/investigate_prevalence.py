import numpy as np
import pandas as pd
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.statistics.registry import StatisticProfileRegistry

# Initialize service
service = AnalyticsService.from_config_dir("configs")
df_demo = service.generate_synthetic_data(n_users=50000, seed=42)

print("=== DATASET STATS ===")
print(f"Total rows: {len(df_demo)}")
print(f"Total unique users in telemetry: {df_demo['user_id'].nunique()}")
print(f"Configured scenario population: {service.config.scenario.population.n_users}")

user_matrix = service.get_user_matrix()
print(f"UserMatrix n_users: {user_matrix.n_users}")
print(f"UserMatrix public_population_size: {user_matrix.public_population_size}")

# Check cell: provider_a:technical
cell_idx = user_matrix.cell_map[("provider_a", "technical")]
cell_name = user_matrix.cell_names[cell_idx]
print(f"\nTarget cell: {cell_name} (index {cell_idx})")

# Raw telemetry count of unique users who used provider_a for technical:
df_cell = df_demo[(df_demo["provider"] == "provider_a") & (df_demo["task"] == "technical")]
raw_unique_users_cell = df_cell["user_id"].nunique()
print(f"Raw unique users who used {cell_name}: {raw_unique_users_cell}")
print(f"Raw unique users / total active users ({user_matrix.n_users}): {raw_unique_users_cell / user_matrix.n_users:.6f}")
print(f"Raw unique users / public_population_size ({user_matrix.public_population_size}): {raw_unique_users_cell / user_matrix.public_population_size:.6f}")

# Profile
profile = StatisticProfileRegistry.get("user_prevalence")
contributions = profile.build_user_contributions(user_matrix)
bounded_exact_count = np.sum(contributions, axis=0)[cell_idx]
print(f"Bounded exact count for {cell_name}: {bounded_exact_count}")
print(f"Bounded count / public_population_size: {bounded_exact_count / user_matrix.public_population_size:.6f}")

# Check all cells
raw_counts = np.sum(user_matrix.matrix > 0, axis=0)
for i, name in enumerate(user_matrix.cell_names):
    print(f"Cell {name}: raw_unique={raw_counts[i]}, bounded={np.sum(contributions, axis=0)[i]}, rate_pub={raw_counts[i]/user_matrix.public_population_size:.4f}, rate_act={raw_counts[i]/user_matrix.n_users:.4f}")
