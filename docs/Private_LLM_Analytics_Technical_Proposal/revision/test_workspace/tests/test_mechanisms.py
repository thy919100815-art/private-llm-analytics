"""Unit tests for DP mechanisms calibration and execution."""

import numpy as np
from private_llm_analytics.mechanisms.gaussian import (
    GaussianMechanism,
    calibrate_analytic_gaussian,
    gaussian_privacy_loss_tail,
)
from private_llm_analytics.mechanisms.laplace import (
    LaplaceMechanism,
    calibrate_laplace,
)


def test_analytic_gaussian_calibration():
    # Verify exact root condition across epsilons
    for eps in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        delta = 1e-6
        sens = 1.0
        sigma = calibrate_analytic_gaussian(eps, delta, sens)
        tail = gaussian_privacy_loss_tail(sigma, eps, sens)
        assert abs(tail - delta) <= 1e-10, f"AGM tail {tail} deviates from delta {delta}"


def test_laplace_calibration():
    eps = 1.0
    sens = 9.0
    scale = calibrate_laplace(eps, sens)
    assert scale == 9.0

    mech = LaplaceMechanism(epsilon=0.5, sensitivity=3.0)
    assert mech.noise_scale == 6.0


def test_gaussian_noise_vector():
    mech = GaussianMechanism(epsilon=1.0, delta=1e-6, sensitivity=1.0)
    clean = np.zeros(9)
    rng = np.random.default_rng(42)
    noisy = mech.add_noise(clean, rng=rng)
    assert noisy.shape == (9,)
    assert not np.all(noisy == 0.0)
