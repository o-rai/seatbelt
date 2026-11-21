"""
seatbelt/judging/config.py

Verification settings and the call/cost estimate shown before a run.

Consistency: each judge grades the same response N times. N defaults to 5
and falls back to 3 when a call budget is set and N=5 would exceed it.
With early_stop, every judge first runs fallback_n times and continues to N
only if those runs disagree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Optional


class BudgetError(ValueError):
    """Even the fallback N exceeds the call budget."""


@dataclass
class VerificationConfig:
    consistency_n: int = 5
    fallback_n: int = 3
    early_stop: bool = True
    # Share of a judge's runs that must match its majority label.
    # 0.8 means 4/5 at N=5, and 3/3 (unanimous) at N=3.
    min_agreement: float = 0.8
    # 1 attempt + 1 retry when a reply doesn't match the schema (spec S4).
    max_schema_attempts: int = 2
    # Used only for the pre-run estimate.
    deliberation_rate_estimate: float = 0.15
    # Max judge calls for a run. None = no cap (always N = consistency_n).
    max_judge_calls: Optional[int] = None

    def validate(self) -> "VerificationConfig":
        if self.fallback_n < 1 or self.consistency_n < self.fallback_n:
            raise ValueError("Need 1 <= fallback_n <= consistency_n.")
        if not 0.5 < self.min_agreement <= 1.0:
            raise ValueError("min_agreement must be in (0.5, 1].")
        if self.max_schema_attempts < 1:
            raise ValueError("max_schema_attempts must be >= 1.")
        if not 0.0 <= self.deliberation_rate_estimate <= 1.0:
            raise ValueError("deliberation_rate_estimate must be in [0, 1].")
        return self


def estimate_judge_calls(
    n_rows: int,
    n: int,
    judges_per_row: int = 2,
    deliberation_rate: float = 0.15,
    tiebreak: bool = False,
) -> dict[str, float]:
    """
    expected: every judge runs n times; deliberating rows add 1 call per judge,
              plus n tiebreak runs if a tiebreak judge is configured.
    worst:    every row deliberates (and tiebreaks). Schema retries not included.
    """
    per_row = judges_per_row * n
    extra = judges_per_row + (n if tiebreak else 0)
    return {
        "n": n,
        "rows": n_rows,
        "expected": round(n_rows * (per_row + deliberation_rate * extra)),
        "worst": n_rows * (per_row + extra),
    }


def choose_n(n_rows: int, config: VerificationConfig, tiebreak: bool = False) -> int:
    """Return consistency_n if it fits the budget, else fallback_n, else raise."""
    config.validate()
    if config.max_judge_calls is None:
        return config.consistency_n
    for n in (config.consistency_n, config.fallback_n):
        est = estimate_judge_calls(
            n_rows, n, 2, config.deliberation_rate_estimate, tiebreak
        )
        if est["expected"] <= config.max_judge_calls:
            return n
    est = estimate_judge_calls(
        n_rows, config.fallback_n, 2, config.deliberation_rate_estimate, tiebreak
    )
    raise BudgetError(
        f"{n_rows} rows need about {est['expected']} judge calls even at N={config.fallback_n}, "
        f"over the budget of {config.max_judge_calls}. Raise the budget or run fewer rows."
    )


def compute_config_hash(**parts: Any) -> str:
    """Stable hash of everything that shaped a result (config, rubric versions, judge models)."""

    def _norm(v: Any) -> Any:
        if hasattr(v, "__dataclass_fields__"):
            return asdict(v)
        return v

    blob = json.dumps(
        {k: _norm(v) for k, v in parts.items()}, sort_keys=True, default=str
    )
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()
