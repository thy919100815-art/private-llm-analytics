"""Candidate definitions and enumeration for Utility-Aware Release Planning (v0.3)."""

from dataclasses import dataclass
from decimal import Decimal
import itertools
from typing import Any, Dict, List, Optional, Sequence, Union


VALID_GRANULARITIES = (
    "full_9_cell",
    "provider_3_cell",
    "task_3_cell",
    "overall_1_cell",
)

VALID_MECHANISMS = (
    "gaussian",
    "laplace",
)

DEFAULT_CANDIDATE_C: List[float] = [5.0, 10.0, 20.0, 40.0, 80.0]
DEFAULT_CANDIDATE_G: List[str] = [
    "full_9_cell",
    "provider_3_cell",
    "task_3_cell",
    "overall_1_cell",
]
DEFAULT_CANDIDATE_W: List[int] = [7, 14, 30, 60, 90]
DEFAULT_CANDIDATE_F: List[int] = [1, 7, 14, 30]


@dataclass(frozen=True)
class ReleasePolicyCandidate:
    """A release policy candidate specifying (C, G, W, F) and mechanism parameters."""

    contribution_bound: Optional[float] = None
    granularity: str = "full_9_cell"
    window_days: int = 30
    frequency_days: int = 7
    mechanism: str = "gaussian"
    epsilon_per_release: float = 1.0
    delta_per_release: float = 1e-6

    def __post_init__(self):
        if self.contribution_bound is not None and self.contribution_bound <= 0:
            raise ValueError(f"contribution_bound must be > 0, got {self.contribution_bound}")
        if self.granularity not in VALID_GRANULARITIES:
            raise ValueError(
                f"Invalid granularity '{self.granularity}'. Must be one of {VALID_GRANULARITIES}"
            )
        if self.window_days <= 0:
            raise ValueError(f"window_days must be > 0, got {self.window_days}")
        if self.frequency_days <= 0:
            raise ValueError(f"frequency_days must be > 0, got {self.frequency_days}")
        mech_norm = str(self.mechanism).lower().strip()
        if mech_norm not in VALID_MECHANISMS:
            raise ValueError(
                f"Invalid mechanism '{self.mechanism}'. Must be one of {VALID_MECHANISMS}"
            )
        object.__setattr__(self, "mechanism", mech_norm)
        if self.epsilon_per_release <= 0:
            raise ValueError(f"epsilon_per_release must be > 0, got {self.epsilon_per_release}")
        if mech_norm == "gaussian":
            if self.delta_per_release <= 0 or self.delta_per_release >= 1:
                raise ValueError(
                    f"delta_per_release must be in (0, 1) for Gaussian mechanism, got {self.delta_per_release}"
                )

    @property
    def candidate_id(self) -> str:
        c_str = f"C{int(self.contribution_bound)}" if self.contribution_bound is not None else "CNone"
        return f"{c_str}_G{self.granularity}_W{self.window_days}_F{self.frequency_days}_M{self.mechanism}_eps{self.epsilon_per_release:.2f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "C": self.contribution_bound,
            "G": self.granularity,
            "W": self.window_days,
            "F": self.frequency_days,
            "mechanism": self.mechanism,
            "epsilon_per_release": self.epsilon_per_release,
            "delta_per_release": self.delta_per_release,
            "candidate_id": self.candidate_id,
        }


def generate_candidate_set(
    contribution_bounds: Optional[Sequence[Optional[float]]] = None,
    granularities: Optional[Sequence[str]] = None,
    window_days: Optional[Sequence[int]] = None,
    frequency_days: Optional[Sequence[int]] = None,
    mechanisms: Optional[Sequence[str]] = None,
    epsilon_per_release: float = 1.0,
    delta_per_release: float = 1e-6,
) -> List[ReleasePolicyCandidate]:
    """Deterministically enumerate the finite candidate space for release planning."""
    c_list = list(contribution_bounds) if contribution_bounds is not None else [None]
    g_list = list(granularities) if granularities is not None else list(DEFAULT_CANDIDATE_G)
    w_list = list(window_days) if window_days is not None else list(DEFAULT_CANDIDATE_W)
    f_list = list(frequency_days) if frequency_days is not None else list(DEFAULT_CANDIDATE_F)
    m_list = list(mechanisms) if mechanisms is not None else ["gaussian"]

    candidates: List[ReleasePolicyCandidate] = []
    # Deterministic cartesian product order: C -> G -> W -> F -> M
    for c, g, w, f, m in itertools.product(c_list, g_list, w_list, f_list, m_list):
        candidates.append(
            ReleasePolicyCandidate(
                contribution_bound=c,
                granularity=g,
                window_days=w,
                frequency_days=f,
                mechanism=m,
                epsilon_per_release=epsilon_per_release,
                delta_per_release=delta_per_release,
            )
        )
    return candidates
