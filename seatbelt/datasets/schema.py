"""
seatbelt/datasets/schema.py

The canonical dataset row. Every source format (JSONL, legacy probe JSON,
BBQ-style CSV) is converted into a DatasetRow before anything else sees it.

Each row carries exactly one risk dimension. That dimension decides which
judge panel scores the row; there is no router.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# The six dimension judges. Names match the existing council agents.
DIMENSIONS: tuple[str, ...] = (
    "deception",
    "fairness",
    "sociotech",
    "regulatory",
    "transparency",
    "privacy",
)

# Inspect AI stages from the design doc.
STAGES: tuple[str, ...] = ("mcq", "open_ended", "safety")


class DatasetError(ValueError):
    """Raised when a dataset file or row is invalid. Never swallowed."""


class UnknownDimensionError(DatasetError):
    """Raised when a row names a dimension that has no judge."""


def require_dimension(dimension: Any, where: str = "") -> str:
    """Return the dimension if it is one of DIMENSIONS, else raise."""
    if dimension not in DIMENSIONS:
        loc = f" ({where})" if where else ""
        raise UnknownDimensionError(
            f"Unknown dimension {dimension!r}{loc}. Expected one of: {', '.join(DIMENSIONS)}."
        )
    return dimension


_FIELDS = (
    "row_id",
    "dimension",
    "stage",
    "turns",
    "protected_class",
    "template_id",
    "expected_answer",
    "target_answer",
    "rubric_hints",
    "weight",
    "source",
    "metadata",
)


@dataclass
class DatasetRow:
    """
    One evaluation item.

    turns:            turns[0] is the first prompt sent to the tested model.
                      Later entries are scripted follow-ups (e.g. pushbacks),
                      sent in the same conversation.
    expected_answer:  the correct answer when one exists (e.g. MCQ letter).
    target_answer:    the answer a pressure follow-up pushes toward
                      (stereotyped answer for ambiguous items, a wrong answer
                      for disambiguated items).
    rubric_hints:     reference material for the judge, e.g. red_flags /
                      green_flags from older probe files. Hints, not scores.
    """

    row_id: str
    dimension: str
    stage: str
    turns: list[str]
    protected_class: Optional[str] = None
    template_id: Optional[str] = None
    expected_answer: Optional[str] = None
    target_answer: Optional[str] = None
    rubric_hints: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "DatasetRow":
        where = f"row {self.row_id!r}" + (f" in {self.source}" if self.source else "")
        if not isinstance(self.row_id, str) or not self.row_id.strip():
            raise DatasetError(f"Missing row_id ({self.source or 'unknown source'}).")
        require_dimension(self.dimension, where)
        if self.stage not in STAGES:
            raise DatasetError(
                f"Unknown stage {self.stage!r} ({where}). Expected one of: {', '.join(STAGES)}."
            )
        if (
            not isinstance(self.turns, list)
            or not self.turns
            or not all(isinstance(t, str) and t.strip() for t in self.turns)
        ):
            raise DatasetError(
                f"turns must be a non-empty list of non-empty strings ({where})."
            )
        if not isinstance(self.weight, (int, float)) or self.weight <= 0:
            raise DatasetError(f"weight must be a positive number ({where}).")
        if not isinstance(self.rubric_hints, dict) or not isinstance(
            self.metadata, dict
        ):
            raise DatasetError(f"rubric_hints and metadata must be objects ({where}).")
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "") -> "DatasetRow":
        """Build a row from the canonical JSON shape. Unknown keys are an error."""
        if not isinstance(data, dict):
            raise DatasetError(f"Row must be a JSON object ({source}).")
        unknown = set(data) - set(_FIELDS)
        if unknown:
            raise DatasetError(
                f"Unknown field(s) {sorted(unknown)} in row {data.get('row_id')!r} ({source}). "
                "Put extra information under 'metadata'."
            )
        missing = [
            k for k in ("row_id", "dimension", "stage", "turns") if k not in data
        ]
        if missing:
            raise DatasetError(
                f"Missing required field(s) {missing} in row {data.get('row_id')!r} ({source})."
            )
        kwargs = dict(data)
        kwargs.setdefault("source", source)
        return cls(**kwargs).validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
