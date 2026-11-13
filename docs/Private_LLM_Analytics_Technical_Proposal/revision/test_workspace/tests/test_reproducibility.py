"""Unit tests for reproducibility under fixed random seeds."""

import pandas as pd
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def test_synthetic_data_reproducibility():
    scenario = SimulationScenario(
        n_users=50,
        n_days=7,
        random_seed=42,
    )
    gen = SyntheticDataGenerator(scenario)

    df1 = gen.generate(seed=42)
    df2 = gen.generate(seed=42)

    pd.testing.assert_frame_equal(df1, df2)

    stats1 = gen.compute_summary_statistics(df1)
    stats2 = gen.compute_summary_statistics(df2)

    assert stats1["n_users"] == stats2["n_users"]
    assert stats1["total_interactions"] == stats2["total_interactions"]
    assert stats1["task_distribution"] == stats2["task_distribution"]
