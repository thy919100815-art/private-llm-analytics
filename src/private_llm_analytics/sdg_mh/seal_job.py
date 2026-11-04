"""Daily Sealing Job for SDG Additive Channels.

Executes SEAL(t) at day t + L (default L = 2 days lateness grace period).
Enforces:
1. Data-independent per-user daily clipping.
2. Exact Discrete Gaussian perturbation N_Z(0, sigma^2).
3. Immutable append to SealedStore.
4. Automatic registration in UnifiedPrivacyLedger.
5. Performance latency benchmarking (execution duration in ms).
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from private_llm_analytics.sdg_mh.clipping import clip_daily_additive
from private_llm_analytics.sdg_mh.discrete_gaussian import sample_discrete_gaussian
from private_llm_analytics.sdg_mh.ledger import UnifiedPrivacyLedger
from private_llm_analytics.sdg_mh.sealed_store import DailySealedRecord, SealedStore


class SealJob:
    """Orchestrates daily SEAL(t) executions."""

    def __init__(
        self,
        sealed_store: SealedStore,
        ledger: UnifiedPrivacyLedger,
        C_r: float = 50.0,
        C_f: float = 10.0,
        sigma_r: float = 2500.0,
        sigma_f: float = 700.0,
        C_p: Optional[float] = None,
        sigma_p: Optional[float] = None,
        providers_list: Optional[Sequence[str]] = None,
        lateness_grace_days: int = 2,
    ):
        self.sealed_store = sealed_store
        self.ledger = ledger
        self.C_r = float(C_r)
        self.C_f = float(C_f)
        self.sigma_r = float(sigma_r)
        self.sigma_f = float(sigma_f)
        self.C_p = float(C_p) if C_p is not None else None
        self.sigma_p = float(sigma_p) if sigma_p is not None else None
        self.providers_list = list(providers_list) if providers_list is not None else None
        self.lateness_grace_days = int(lateness_grace_days)
        self.execution_times_ms: List[float] = []

    def seal_day(
        self,
        day_index: int,
        date_str: str,
        df_day: pd.DataFrame,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> DailySealedRecord:
        """Execute SEAL(t) for day_index with timestamped telemetry records."""
        t0 = time.perf_counter()

        # 1. User daily contribution clipping
        c_req, c_fail, c_prov, diags = clip_daily_additive(
            df_day=df_day,
            C_r=self.C_r,
            C_f=self.C_f,
            providers_list=self.providers_list,
            C_p=self.C_p,
        )

        # 2. Add Discrete Gaussian noise
        noise_r = sample_discrete_gaussian(sigma=self.sigma_r, rng=rng, seed=seed)
        noise_f = sample_discrete_gaussian(
            sigma=self.sigma_f,
            rng=rng,
            seed=None if seed is None else seed + 1000003,
        )

        y_r = int(round(c_req)) + int(noise_r)
        y_f = int(round(c_fail)) + int(noise_f)

        y_prov_list = None
        if c_prov is not None and self.sigma_p is not None:
            noise_p = sample_discrete_gaussian(
                sigma=self.sigma_p,
                size=len(c_prov),
                rng=rng,
                seed=None if seed is None else seed + 2000003,
            )
            y_prov_list = (c_prov + noise_p).tolist()

        # 3. Create immutable record
        now_iso = datetime.now(timezone.utc).isoformat()
        record = DailySealedRecord(
            day_index=day_index,
            date_str=date_str,
            y_requests=y_r,
            y_failures=y_f,
            y_providers=y_prov_list,
            sigma_r=self.sigma_r,
            sigma_f=self.sigma_f,
            sigma_p=self.sigma_p,
            C_r=self.C_r,
            C_f=self.C_f,
            sealed_at=now_iso,
        )

        # 4. Append to SealedStore
        self.sealed_store.append(record)

        # 5. Register in Unified Ledger
        self.ledger.record_gaussian_release(
            release_id=f"sdg_additive_req_{day_index}",
            days_accessed=[day_index],
            sensitivity=self.C_r,
            sigma=self.sigma_r,
            tag="sdg_requests",
        )
        self.ledger.record_gaussian_release(
            release_id=f"sdg_additive_fail_{day_index}",
            days_accessed=[day_index],
            sensitivity=self.C_f,
            sigma=self.sigma_f,
            tag="sdg_failures",
        )
        if self.C_p is not None and self.sigma_p is not None:
            self.ledger.record_gaussian_release(
                release_id=f"sdg_additive_prov_{day_index}",
                days_accessed=[day_index],
                sensitivity=self.C_p,
                sigma=self.sigma_p,
                tag="sdg_providers",
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.execution_times_ms.append(elapsed_ms)
        return record

    def get_benchmarks(self) -> Dict[str, float]:
        """Compute latency statistics in milliseconds."""
        if not self.execution_times_ms:
            return {"count": 0, "p50_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
        arr = np.array(self.execution_times_ms)
        return {
            "count": len(arr),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
        }
