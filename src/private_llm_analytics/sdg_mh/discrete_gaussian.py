"""Exact Discrete Gaussian mechanism sampler implementing Canonne-Kamath-Steinke (2020).

Reference:
  Canonne, Kamath, Steinke (2020): "The Discrete Gaussian for Differential Privacy", NeurIPS 2020.
  Distribution:
    For parameter sigma^2 > 0 (or precision t = floor(sigma) + 1),
    P(X = x) proportional to exp(-x^2 / (2 * sigma^2)) for all integer x in Z.
"""

from __future__ import annotations

import math
from typing import Optional, Union
import numpy as np


def sample_discrete_gaussian(
    sigma: float,
    size: Optional[Union[int, tuple[int, ...]]] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Union[int, np.ndarray]:
    """Sample from the discrete Gaussian distribution N_Z(0, sigma^2).

    Uses exact rejection sampling from a Discrete Laplace proposal:
      Proposal: Y ~ DiscreteLaplace(scale=t) where t = floor(sigma) + 1.
      Sampled via Y = U_1 - U_2, with U_1, U_2 ~ Geometric(1 - exp(-1/t)).
      Acceptance probability: exp(- (|y| - sigma^2 / t)^2 / (2 * sigma^2)).
      Acceptance rate is >= sqrt(2*pi) * sigma / (2*(floor(sigma)+1)) ~ 0.798 for large sigma.

    Args:
        sigma: Standard deviation parameter sigma > 0.
        size: Output shape. If None, returns a scalar integer.
        rng: Optional numpy Generator for reproducibility.
        seed: Optional integer seed (used if rng is None).

    Returns:
        Integer or numpy ndarray of integers sampled from N_Z(0, sigma^2).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}")

    if rng is None:
        rng = np.random.default_rng(seed)

    is_scalar = size is None
    target_count = 1 if is_scalar else int(np.prod(size))

    # Proposal parameter t = floor(sigma) + 1
    t = int(math.floor(sigma)) + 1
    p_geom = 1.0 - math.exp(-1.0 / t)
    inv_two_sig_sq = 1.0 / (2.0 * sigma * sigma)
    bias_term = (sigma * sigma) / t

    accepted: list[int] = []
    # Vectorized rejection loop
    while len(accepted) < target_count:
        needed = target_count - len(accepted)
        batch_size = max(needed * 2, 64)
        u1 = rng.geometric(p_geom, size=batch_size) - 1
        u2 = rng.geometric(p_geom, size=batch_size) - 1
        y = u1 - u2
        # Acceptance ratio
        diff = np.abs(y) - bias_term
        acc_prob = np.exp(-(diff * diff) * inv_two_sig_sq)
        u_rand = rng.uniform(0.0, 1.0, size=batch_size)
        mask = u_rand <= acc_prob
        accepted.extend(y[mask].tolist())

    res = np.array(accepted[:target_count], dtype=np.int64)
    if is_scalar:
        return int(res[0])
    return res.reshape(size)
