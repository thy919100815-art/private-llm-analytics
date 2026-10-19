"""Unit tests for user matrix aggregation."""

import pandas as pd
import numpy as np
from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.ingestion.schema import (
    COL_INTERACTION_ID,
    COL_USER_ID,
    COL_TIMESTAMP,
    COL_PROVIDER,
    COL_TASK,
)


def test_user_aggregation():
    # 2 users, multiple interactions
    df = pd.DataFrame({
        COL_INTERACTION_ID: ["I01", "I02", "I03", "I04", "I05"],
        COL_USER_ID: ["U1", "U1", "U1", "U2", "U2"],
        COL_TIMESTAMP: pd.to_datetime(["2026-10-01"] * 5),
        COL_PROVIDER: ["provider_a", "provider_a", "provider_b", "provider_c", "provider_c"],
        COL_TASK: ["technical", "technical", "document", "analysis", "analysis"],
    })
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]

    umatrix = build_user_matrix(df, providers=provs, tasks=tsks)

    assert umatrix.n_users == 2
    assert umatrix.n_cells == 9
    assert umatrix.total_interactions == 5.0

    # U1 has: 2 in (provider_a, technical), 1 in (provider_b, document)
    u1_idx = umatrix.user_ids.index("U1")
    assert umatrix.matrix[u1_idx, umatrix.cell_map[("provider_a", "technical")]] == 2.0
    assert umatrix.matrix[u1_idx, umatrix.cell_map[("provider_b", "document")]] == 1.0
    assert umatrix.matrix[u1_idx, umatrix.cell_map[("provider_c", "analysis")]] == 0.0

    # U2 has: 2 in (provider_c, analysis)
    u2_idx = umatrix.user_ids.index("U2")
    assert umatrix.matrix[u2_idx, umatrix.cell_map[("provider_c", "analysis")]] == 2.0
