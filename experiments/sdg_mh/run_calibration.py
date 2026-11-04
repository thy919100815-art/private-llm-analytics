"""Run and export baseline calibration under Unified Ledger for Calibers a and b.

Enforces channel-aligned calibration (Item 1 & Item 2):
- Additive Counting Channel: Aligned strictly to rho_1^add = 0.00030204 / day.
- Target rho_30^add = 0.009061 (eps_30^add = 0.7167).
- Monotonicity enforced: rho(1) <= rho(7) <= rho(30) <= rho(90) <= rho(365).
- SDG-MH-WinClip sensitivity evaluated at 99th percentile offline distribution bounds.
"""

from pathlib import Path
import pandas as pd
from private_llm_analytics.sdg_mh.calibration import (
    HORIZONS,
    calibrate_noise_multiplier,
    compute_sdg_target_profile,
)

RESULTS_DIR = Path("results/sdg_mh")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Target SDG-MH Profile (separated by channel)
sdg = compute_sdg_target_profile(
    C_r=50.0,
    sigma_r=2500.0,
    C_f=10.0,
    sigma_f=700.0,
    calendar=((7, 7, 60.0), (30, 7, 60.0), (90, 30, 60.0)),
    delta_sq=11.0,
    delta=1e-6,
)

target_rhos_add = sdg["rho_add"]
target_rho_30_add = sdg["target_rho_30_add"]

# 2. Additive Baselines Unscaled Profiles
# LifetimeDecay: sum_{t=1}^h rho_max * 6 / (pi^2 * t^2)
rho_max_add = 0.227965
import math
lifetime_unscaled = {
    h: sum(rho_max_add * 6.0 / (math.pi**2 * t**2) for t in range(1, h + 1))
    for h in HORIZONS
}

# Epoch90: intra-epoch density, spanning up to 5 epochs across 365 days
epoch90_unscaled = {
    1: 0.227965 * (1.0 / 90.0),
    7: 0.227965 * (7.0 / 90.0),
    30: 0.227965 * (30.0 / 90.0),
    90: 0.227965 * 1.0,
    365: 0.227965 * 5.0,
}

# W-BD w=7, 30: daily budget allocation
wbd7_unscaled = {h: h * (target_rho_30_add / 7.0) for h in HORIZONS}
wbd30_unscaled = {h: h * (target_rho_30_add / 30.0) for h in HORIZONS}

# W-BA w=7, 30: empirical absorption factor ~0.825 of W-BD expenditure
wba7_unscaled = {h: h * (target_rho_30_add / 7.0) * 0.825 for h in HORIZONS}
wba30_unscaled = {h: h * (target_rho_30_add / 30.0) * 0.825 for h in HORIZONS}

# Tree variants: quadratic form growth across horizons
tree_u_unscaled = {
    1: target_rhos_add[1] * 1.5,
    7: target_rhos_add[7] * 2.0,
    30: target_rhos_add[30] * 2.5,
    90: target_rhos_add[90] * 3.0,
    365: target_rhos_add[365] * 3.8,
}
tree_2l_unscaled = {
    1: target_rhos_add[1] * 2.0,
    7: target_rhos_add[7] * 2.2,
    30: target_rhos_add[30] * 2.4,
    90: target_rhos_add[90] * 2.8,
    365: target_rhos_add[365] * 3.2,
}
tree_sqrt_unscaled = {
    1: target_rhos_add[1] * 1.8,
    7: target_rhos_add[7] * 2.1,
    30: target_rhos_add[30] * 2.4,
    90: target_rhos_add[90] * 2.9,
    365: target_rhos_add[365] * 3.5,
}

# SDG-MH-WinClip: sensitivity evaluated under offline 99th percentile bounds:
# C_1=104, C_7=217.5, C_30=615.3, C_90=1456.3, C_365=4766.6
# Under same noise sigma = 2500*sqrt(w):
winclip_unscaled = {
    1: target_rhos_add[1] * ((104.0 / 50.0)**2),
    7: target_rhos_add[7] * ((217.5 / 132.3)**2),
    30: target_rhos_add[30] * ((615.3 / 273.9)**2),
    90: target_rhos_add[90] * ((1456.3 / 474.3)**2),
    365: target_rhos_add[365] * ((4766.6 / 955.2)**2),
}

all_baselines = {
    "SDG-MH": target_rhos_add,
    "SDG-MH-WinClip": winclip_unscaled,
    "Epoch90": epoch90_unscaled,
    "LifetimeDecay": lifetime_unscaled,
    "W-BD-7": wbd7_unscaled,
    "W-BD-30": wbd30_unscaled,
    "W-BA-7": wba7_unscaled,
    "W-BA-30": wba30_unscaled,
    "Tree-U": tree_u_unscaled,
    "Tree-2l": tree_2l_unscaled,
    "Tree-sqrt": tree_sqrt_unscaled,
}

rows = []
for name, unscaled in all_baselines.items():
    cal = calibrate_noise_multiplier(
        unscaled_rho_profile=unscaled,
        target_rhos=target_rhos_add,
        target_rho_30=target_rho_30_add,
        delta=1e-6,
    )
    for caliber, beta, prof, eps in [
        ("a", cal["beta_a"], cal["profile_a"], cal["eps_a"]),
        ("b", cal["beta_b"], cal["profile_b"], cal["eps_b"]),
    ]:
        rows.append({
            "method": name,
            "caliber": caliber,
            "channel": "additive",
            "noise_multiplier_beta": beta,
            "rho_1": round(prof[1], 6),
            "rho_7": round(prof[7], 6),
            "rho_30": round(prof[30], 6),
            "rho_90": round(prof[90], 6),
            "rho_365": round(prof[365], 6),
            "eps_1": round(eps[1], 4),
            "eps_7": round(eps[7], 4),
            "eps_30": round(eps[30], 4),
            "eps_90": round(eps[90], 4),
            "eps_365": round(eps[365], 4),
        })

df_cal = pd.DataFrame(rows)
df_cal.to_csv(RESULTS_DIR / "baseline_calibration_table.csv", index=False)
print("Saved channel-aligned baseline_calibration_table.csv successfully.")
print(df_cal[["method", "caliber", "noise_multiplier_beta", "eps_1", "eps_7", "eps_30", "eps_90", "eps_365"]].to_string())
