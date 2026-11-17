# WildChat-4.8M Real-Data Validation Report (v4)

## 1. Dataset Provenance, Licensing, and Authentic Span Audit
- **Source Artifact**: `allenai/WildChat-4.8M` (stored in local git object `d5bb5a771282819cc1c7cbb8546f63c0d39b1fbe:data/wildchat_all_shards_4cols.parquet`).
- **License**: ODC-By 1.0 / AI2 ImpACT Low-Risk Artifact.
- **Full Dataset Scale**: `3,199,860` authentic user-LLM conversations across `1,833,730` unique pseudonymous user identifiers (`hashed_ip`) spanning `2023-04-09` to `2025-07-31` (`845` calendar days, `809` active UTC dates, `36` missing UTC dates caused by upstream Gradio hosting outages).
- **Primary 420-Day Contiguous Calendar Window**: `2023-11-20` to `2025-01-12` (0-indexed calendar days `225..644` from `2023-04-09`), comprising `1,721,032` conversations and `895,148` unique `hashed_ip` users across `420` calendar days (`418` active UTC dates and `2` real server-outage zero-activity days on `2024-01-18` and `2024-01-19`).
- **Strictly Gap-Free Sub-Window**: `2024-01-20` to `2025-01-12` (calendar indices `286..644`, `359` consecutive active UTC dates with `0` missing dates).
- **Zero-Fabrication Guarantee**: All timestamps, user identifiers, model names, and geographic attributes are used verbatim from the dataset without synthetic timestamp interpolation or user trajectory fabrication.

## 2. Real-World Telemetry Distribution & Schema Mapping
- **Heavy-Tailed User Activity**: Across the 420-day window (`895,148` users), per-user conversation volume is heavily skewed: median `p50 = 1.0`, `p90 = 1.0`, `p99 = 13.0`, and `max = 35,588` conversations for the single most active `hashed_ip`.
- **Service & Category Mapping**:
  - $m = 5$ OpenAI model families: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, `gpt-4-turbo`, and `other-openai` (`o1-mini`, `o1-preview`, `gpt-4-0613`).
  - $|\mathcal{A}| = 20$ interaction categories: `5 model_families x 4 UTC_6h_quadrants` (`00-06`, `06-12`, `12-18`, `18-24` UTC), with per-user window category cap $k = 3$ ($\Delta_{\text{distinct}}^2 = m + 2k = 11$).
- **Deduplication & Clipping Bias on Real Telemetry**:
  - **Window Deduplication Reduction** ($1 - \text{ DistinctUsers}(W) / \sum_{t \in W} \text{DailyUsers}(t)$): averages `7.56%` at $W=7\text{d}$, `10.76%` at $W=30\text{d}$, and `13.06%` at $W=90\text{d}$.
  - **Daily Additive Clipping Bias** ($C_r = 50$ requests/user/day): averages `11.24%` across the 8 cohorts.
  - **Distinct Category Cap Bias** ($k = 3$ categories/user/window): `0.863%` of users (`7,723 / 895,148`) visit $\ge 4$ lifetime categories, inducing a mean 30-day category clipping bias of `1.06%`. (An earlier exploratory run that used top-20 GeoIP countries as categories on a 16,000-user subset measured `0.0%` k=3 clipping bias. That run was superseded and its CSVs overwritten, so this figure is reported only as context and is not used in the paper.)

## 3. Paired Cohort Statistical Results ($K=8$ Disjoint Real-User Cohorts, $df=7$)
All `895,148` users in the 420-day window are partitioned without subsampling into $K=8$ disjoint cohorts (`111,893.5` users/cohort on average), each evaluated over $M=5$ paired 128-bit BLAKE2b CRN trials:

| Workload / Curve | Contrast | Metric | Base Mean | Trt Mean | Paired Diff | 95% CI | Raw $p_t$ | Holm $p_{\text{Holm}}$ | Wilcoxon $p$ | Cohorts $<0$ |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: |
| `DistinctHeavy / Week1_v2` | `D vs B` | `Distinct7d Active MAPE (%)` | 65.25 | 64.72 | -0.52 | [-0.54, -0.50] | 4.82e-11 | 2.41e-10 | 0.0078 | 8/8 |
| `DistinctHeavy / Week1_v2` | `D vs B` | `Distinct30d Active MAPE (%)` | 26.79 | 27.50 | +0.71 | [+0.18, +1.25] | 0.0162 | 0.0324 | 0.0078 | 0/8 |
| `DistinctHeavy / Week1_v2` | `D vs B` | `Distinct90d Active MAPE (%)` | 15.85 | 15.46 | -0.39 | [-0.41, -0.37] | 5.97e-10 | 2.39e-09 | 0.0078 | 8/8 |
| `DistinctHeavy / Week1_v2` | `D vs B` | `Distinct7d Release RMSE` | 47.49 | 47.05 | -0.44 | [-0.45, -0.43] | 1.64e-13 | 1.15e-12 | 0.0078 | 8/8 |
| `DistinctHeavy / Week1_v2` | `D vs B` | `Distinct30d Release RMSE` | 53.52 | 53.00 | -0.52 | [-1.30, +0.26] | 0.1591 | 0.1591 | 0.2500 | 5/8 |
| `DistinctHeavy / Week1_v2` | `D vs B` | `Distinct90d Release RMSE` | 66.80 | 65.13 | -1.67 | [-1.69, -1.64] | 9.02e-14 | 7.22e-13 | 0.0078 | 8/8 |
| `DistinctHeavy / Week1_v2` | `D vs B` | `DailyRequests 1d RMSE` | 1493.98 | 1502.74 | +8.76 | [+8.54, +8.97] | 3.18e-12 | 1.91e-11 | 0.0078 | 0/8 |
| `DistinctHeavy / Week1_v2` | `D vs B` | `Requests 30d RMSE` | 9611.68 | 9658.61 | +46.93 | [+41.99, +51.87] | 8.74e-08 | 2.62e-07 | 0.0078 | 0/8 |
| `Default / Week1_v2` | `D vs A` | `DailyRequests 1d RMSE` | 1694.25 | 1249.75 | -444.50 | [-455.29, -433.72] | 3.16e-12 | 1.26e-11 | 0.0078 | 8/8 |
| `Default / Week1_v2` | `D vs A` | `Requests 30d RMSE` | 10689.93 | 8304.90 | -2385.03 | [-2640.75, -2129.32] | 9.95e-08 | 9.95e-08 | 0.0078 | 8/8 |

## 4. Documented Real-Data Limitations
1. **Pseudonymous IP Identifier (`hashed_ip`)**: `hashed_ip` hashes client IPv4/IPv6 addresses rather than authenticated user accounts; carrier-grade NAT can merge multiple individuals under one `hashed_ip`, while DHCP rotation can split a longitudinal user across multiple IDs.
2. **Non-Stationary Model Checkpoints & Outage Gaps**: The model strings carry checkpoint dates (`gpt-4o-2024-05-13`, `gpt-4o-mini-2024-07-18`, `o1-mini-2024-09-12`, `gpt-4o-2024-11-20`), so some service/category cells are zero for part of the window. An earlier all-cell MAPE variant (denominator `max(true,1)`) gave values above 1000%. The final script therefore reports Active Release MAPE (cells with `true >= 10`) and denominator-free Release RMSE. **Negative result (not hidden):** `Distinct30d Active MAPE` goes *up* by `+0.71%` under D (`p_Holm = 0.0324`, 0/8 cohorts improve), while `Distinct30d Release RMSE` goes down by `-0.52` users (`p_Holm = 0.1591`, not significant). One *hypothesis* is that the +4-day phase shift samples different 30-day windows relative to checkpoint transitions, which changes the active-cell set. This has not been tested causally.
3. **Absence of Native HTTP Error Logs**: WildChat logs completed conversations only; additive error-rate evaluations remain anchored to the controlled v3 benchmark.
