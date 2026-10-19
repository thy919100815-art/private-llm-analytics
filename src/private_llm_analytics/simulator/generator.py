"""End-to-end synthetic dataset generator."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from private_llm_analytics.ingestion.validation import validate_interaction_dataframe
from private_llm_analytics.simulator.interactions import generate_interaction_dataframe
from private_llm_analytics.simulator.scenario import SimulationScenario
from private_llm_analytics.simulator.users import sample_user_profiles


class SyntheticDataGenerator:
    """Simulator that generates realistic enterprise AI interaction logs with user heterogeneity."""

    def __init__(self, scenario: Optional[SimulationScenario] = None):
        self.scenario = scenario or SimulationScenario()

    def sample_interaction_counts(self, rng: np.random.Generator) -> np.ndarray:
        """Sample M_u ~ TruncatedNormal(mean, std^2, [min, max]) for each user."""
        sc = self.scenario
        a = (sc.freq_min - sc.freq_mean) / sc.freq_std
        b = (sc.freq_max - sc.freq_mean) / sc.freq_std

        # Draw from truncated normal
        raw_samples = truncnorm.rvs(
            a, b,
            loc=sc.freq_mean,
            scale=sc.freq_std,
            size=sc.n_users,
            random_state=rng,
        )
        counts = np.rint(raw_samples).astype(np.int64)
        # Ensure clipping to bounds
        counts = np.clip(counts, sc.freq_min, sc.freq_max)
        return counts

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        """Generate long-format interaction DataFrame for the entire population."""
        sc = self.scenario
        current_seed = seed if seed is not None else sc.random_seed
        rng = np.random.default_rng(current_seed)

        # 1. Sample interaction counts per user
        interaction_counts = self.sample_interaction_counts(rng)

        # 2. Sample user profiles with diverse preferences
        user_ids, joint_probs = sample_user_profiles(
            n_users=sc.n_users,
            interaction_counts=interaction_counts,
            providers=sc.providers,
            tasks=sc.tasks,
            rng=rng,
        )

        # 3. Sample multinomial interaction counts across 9 cells for each user
        # Vectorized batch processing
        user_counts = np.zeros((sc.n_users, 9), dtype=np.int64)
        for u in range(sc.n_users):
            user_counts[u] = rng.multinomial(interaction_counts[u], joint_probs[u])

        # 4. Expand into DataFrame
        df = generate_interaction_dataframe(
            user_ids=user_ids,
            user_counts=user_counts,
            providers=sc.providers,
            tasks=sc.tasks,
            start_date=sc.start_date,
            n_days=sc.n_days,
            rng=rng,
        )

        # 5. Validate against ingestion schema
        validated_df = validate_interaction_dataframe(
            df,
            allowed_providers=sc.providers,
            allowed_tasks=sc.tasks,
        )
        return validated_df

    def generate_and_save(
        self,
        output_path: Union[str, Path],
        file_format: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate and save to CSV or Parquet file."""
        df = self.generate(seed=seed)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        fmt = file_format.lower() if file_format else out_p.suffix.lstrip(".").lower()
        if fmt == "parquet":
            df.to_parquet(out_p, index=False, engine="pyarrow")
        elif fmt == "csv":
            df.to_csv(out_p, index=False)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Choose 'parquet' or 'csv'.")
        return df

    def compute_summary_statistics(self, df: pd.DataFrame) -> Dict[str, object]:
        """Compute statistical validation metrics on generated dataset."""
        user_counts = df.groupby("user_id").size()

        # Task and Provider distributions
        task_dist = df["task"].value_counts(normalize=True).to_dict()
        provider_dist = df["provider"].value_counts(normalize=True).to_dict()
        joint_dist = df.groupby(["provider", "task"]).size().unstack(fill_value=0)

        # User heterogeneity: compute user task entropy
        user_task_counts = df.groupby(["user_id", "task"]).size().unstack(fill_value=0)
        user_task_probs = user_task_counts.div(user_task_counts.sum(axis=1), axis=0) + 1e-12
        user_entropies = - (user_task_probs * np.log2(user_task_probs)).sum(axis=1)

        return {
            "n_users": int(df["user_id"].nunique()),
            "total_interactions": int(len(df)),
            "user_frequency": {
                "mean": float(user_counts.mean()),
                "std": float(user_counts.std()),
                "median": float(user_counts.median()),
                "min": int(user_counts.min()),
                "max": int(user_counts.max()),
                "p25": float(user_counts.quantile(0.25)),
                "p75": float(user_counts.quantile(0.75)),
            },
            "task_distribution": task_dist,
            "provider_distribution": provider_dist,
            "mean_user_entropy": float(user_entropies.mean()),
            "std_user_entropy": float(user_entropies.std()),
        }
