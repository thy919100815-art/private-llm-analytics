"""User preference modeling and archetype generation."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class UserProfile:
    user_id: str
    persona: str
    n_interactions: int
    task_probs: np.ndarray          # shape (3,)
    provider_task_probs: np.ndarray # shape (9,) joint distribution P(provider, task)


# Base Dirichlet hyper-parameters for 4 employee personas:
# Tasks order: [technical, document, analysis]
PERSONA_TASK_ALPHA = {
    "technical_specialist": np.array([7.0, 1.5, 1.5]),
    "document_specialist": np.array([1.5, 7.0, 1.5]),
    "analyst_specialist": np.array([1.5, 1.5, 7.0]),
    "generalist": np.array([3.0, 3.0, 3.0]),
}

# Base provider preferences conditional on task:
# Providers order: [provider_a, provider_b, provider_c]
# Provider A is stronger at technical, Provider B at document, Provider C at analysis.
BASE_TASK_PROVIDER_WEIGHTS = {
    "technical": np.array([0.65, 0.20, 0.15]),
    "document": np.array([0.15, 0.65, 0.20]),
    "analysis": np.array([0.20, 0.20, 0.60]),
}


def sample_user_profiles(
    n_users: int,
    interaction_counts: np.ndarray,
    providers: List[str],
    tasks: List[str],
    rng: np.random.Generator,
) -> Tuple[List[str], np.ndarray]:
    """Sample user profiles in vectorized batch mode.

    Returns:
        user_ids: List of user IDs.
        joint_probs: shape (n_users, 9) where columns correspond to the 9 (provider, task) pairs.
    """
    assert len(interaction_counts) == n_users
    n_p = len(providers)
    n_t = len(tasks)
    n_cells = n_p * n_t

    if n_p == 3 and n_t == 3 and tasks == ["technical", "document", "analysis"]:
        persona_names = list(PERSONA_TASK_ALPHA.keys())
        persona_weights = [0.30, 0.30, 0.25, 0.15]
        assigned_personas_idx = rng.choice(len(persona_names), size=n_users, p=persona_weights)
        task_probs = np.zeros((n_users, 3), dtype=np.float64)
        for p_idx, p_name in enumerate(persona_names):
            mask = (assigned_personas_idx == p_idx)
            count = int(np.sum(mask))
            if count > 0:
                alpha = PERSONA_TASK_ALPHA[p_name]
                task_probs[mask] = rng.dirichlet(alpha, size=count)
        user_provider_affinity = rng.dirichlet(np.array([2.0, 2.0, 2.0]), size=n_users)
        joint_probs = np.zeros((n_users, 9), dtype=np.float64)
        base_weights_matrix = np.vstack([
            BASE_TASK_PROVIDER_WEIGHTS[tasks[0]],
            BASE_TASK_PROVIDER_WEIGHTS[tasks[1]],
            BASE_TASK_PROVIDER_WEIGHTS[tasks[2]],
        ])
        for t_idx in range(3):
            bw = base_weights_matrix[t_idx]
            cond_p = user_provider_affinity * bw
            cond_p = cond_p / cond_p.sum(axis=1, keepdims=True)
            p_task = task_probs[:, t_idx:t_idx+1]
            for p_idx in range(3):
                cell_idx = p_idx * 3 + t_idx
                joint_probs[:, cell_idx] = (p_task * cond_p[:, p_idx:p_idx+1]).ravel()
        joint_probs = joint_probs / joint_probs.sum(axis=1, keepdims=True)
    else:
        # Dynamic long-tail Zipfian task distribution across 10-15+ tasks
        zipf_weights = np.array([1.0 / ((i + 1) ** 0.85) for i in range(n_t)])
        base_alpha = 15.0 * (zipf_weights / zipf_weights.sum())
        base_alpha = np.maximum(base_alpha, 0.2)
        task_probs = rng.dirichlet(base_alpha, size=n_users)

        prov_alpha = np.ones(n_p) * 2.0
        user_provider_affinity = rng.dirichlet(prov_alpha, size=n_users)

        joint_probs = np.zeros((n_users, n_cells), dtype=np.float64)
        for t_idx in range(n_t):
            # Base provider preference with slight variation across tasks
            t_hash = hash(tasks[t_idx]) % 1000 / 1000.0
            p_weights = np.array([1.0 + 0.3 * np.sin(t_hash * (i + 1)) for i in range(n_p)])
            p_weights = p_weights / p_weights.sum()

            cond_p = user_provider_affinity * p_weights
            cond_p = cond_p / cond_p.sum(axis=1, keepdims=True)

            p_task = task_probs[:, t_idx:t_idx+1]
            for p_idx in range(n_p):
                cell_idx = p_idx * n_t + t_idx
                joint_probs[:, cell_idx] = (p_task * cond_p[:, p_idx:p_idx+1]).ravel()

        joint_probs = joint_probs / joint_probs.sum(axis=1, keepdims=True)

    pad_width = max(5, len(str(n_users)))
    user_ids = [f"U{i+1:0{pad_width}d}" for i in range(n_users)]
    return user_ids, joint_probs
