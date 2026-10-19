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
    assert len(providers) == 3
    assert len(tasks) == 3
    assert len(interaction_counts) == n_users

    persona_names = list(PERSONA_TASK_ALPHA.keys())
    persona_weights = [0.30, 0.30, 0.25, 0.15]

    # Assign personas
    assigned_personas_idx = rng.choice(len(persona_names), size=n_users, p=persona_weights)

    # Sample task probabilities for each user using their persona's Dirichlet distribution
    task_probs = np.zeros((n_users, 3), dtype=np.float64)
    for p_idx, p_name in enumerate(persona_names):
        mask = (assigned_personas_idx == p_idx)
        count = int(np.sum(mask))
        if count > 0:
            alpha = PERSONA_TASK_ALPHA[p_name]
            task_probs[mask] = rng.dirichlet(alpha, size=count)

    # Sample user-specific provider affinities (variation across employees)
    user_provider_affinity = rng.dirichlet(np.array([2.0, 2.0, 2.0]), size=n_users) # shape (n_users, 3)

    # Compute P(provider | task, user)
    # Combine user provider affinity with task-specific provider strengths
    # Joint probabilities: shape (n_users, 9)
    # Cell indexing: 3 providers x 3 tasks
    # Ordered as: (p_0, t_0), (p_0, t_1), (p_0, t_2), (p_1, t_0), ..., (p_2, t_2)
    # where p_i in providers, t_j in tasks
    joint_probs = np.zeros((n_users, 9), dtype=np.float64)

    base_weights_matrix = np.vstack([
        BASE_TASK_PROVIDER_WEIGHTS[tasks[0]], # task 0
        BASE_TASK_PROVIDER_WEIGHTS[tasks[1]], # task 1
        BASE_TASK_PROVIDER_WEIGHTS[tasks[2]], # task 2
    ]) # shape (3, 3) -> row=task, col=provider

    for t_idx in range(3):
        # Base weights for this task: shape (3,)
        bw = base_weights_matrix[t_idx]
        # Element-wise product of user affinity and base task weights
        cond_p = user_provider_affinity * bw # shape (n_users, 3)
        # Normalize over providers
        cond_p = cond_p / cond_p.sum(axis=1, keepdims=True)

        p_task = task_probs[:, t_idx:t_idx+1] # shape (n_users, 1)

        # For each provider p_idx:
        for p_idx in range(3):
            cell_idx = p_idx * 3 + t_idx
            joint_probs[:, cell_idx] = (p_task * cond_p[:, p_idx:p_idx+1]).ravel()

    # Normalize joint_probs to sum to 1.0 per user
    joint_probs = joint_probs / joint_probs.sum(axis=1, keepdims=True)

    pad_width = max(5, len(str(n_users)))
    user_ids = [f"U{i+1:0{pad_width}d}" for i in range(n_users)]

    return user_ids, joint_probs
