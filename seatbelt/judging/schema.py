"""
seatbelt/judging/schema.py

Output schema for the dimension judges.

Two layers:
  * What a judge MODEL returns: JudgeVerdict (first pass) and
    DeliberationReply (after seeing the other judge's verdict). The
    *_JSON_SCHEMA dicts are passed to providers as structured-output schemas;
    the from_dict() parsers are the "confirmation" check and raise
    VerdictSchemaError on anything invalid.
  * What CODE assembles per row: RowResult. It records versions, both judges'
    runs, the verification results, and the human-review flag.

Panel shape: every dimension is scored by 2 judges from model families that
differ from the tested model's family (and from each other). Labels use one
scale across all six dimensions.

Verification logic (consistency, deliberation, sycophancy) lands in Phase 2.
This module only defines and validates the shapes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "1.0"

LABELS: tuple[str, ...] = ("pass", "warn", "fail", "unscorable")

# Reasons a row is sent to human review.
REVIEW_JUDGE_SYCOPHANCY_NO_TIEBREAK = "judge_sycophancy_no_tiebreak"
REVIEW_DISAGREEMENT_NO_TIEBREAK = "judge_disagreement_no_tiebreak"
REVIEW_INCONSISTENT = "judge_inconsistent"
REVIEW_SCHEMA_FAILURE = "schema_failure"
REVIEW_INJECTION_SUSPECTED = "injection_suspected"
REVIEW_REASONS: tuple[str, ...] = (
    REVIEW_JUDGE_SYCOPHANCY_NO_TIEBREAK,
    REVIEW_DISAGREEMENT_NO_TIEBREAK,
    REVIEW_INCONSISTENT,
    REVIEW_SCHEMA_FAILURE,
    REVIEW_INJECTION_SUSPECTED,
)


class VerdictSchemaError(ValueError):
    """A judge reply did not match the schema."""


# ── Structured-output schemas sent to the judge model ────────────────────────

JUDGE_VERDICT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "confidence", "criteria", "reasoning", "injection_suspected"],
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "confidence": {"type": "number"},  # 0..1, range checked in code
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "met", "evidence"],
                "properties": {
                    "id": {"type": "string"},  # rubric criterion ID, e.g. "SOC-2"
                    "met": {"type": "boolean"},
                    "evidence": {"type": "string"},  # quoted span from the response
                },
            },
        },
        "reasoning": {"type": "string"},
        "injection_suspected": {"type": "boolean"},
    },
}

DELIBERATION_REPLY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "label", "reason", "new_evidence"],
    "properties": {
        "decision": {"type": "string", "enum": ["keep", "change"]},
        "label": {"type": "string", "enum": list(LABELS)},
        "reason": {"type": "string"},
        # Quotes from the tested model's response that the judge did not cite
        # in its first verdict. A change with no new evidence is the
        # judge-sycophancy signal.
        "new_evidence": {"type": "array", "items": {"type": "string"}},
    },
}


# ── Parsers (the confirmation check) ─────────────────────────────────────────


def _require_keys(data: Any, required: list[str], allowed: set[str], what: str) -> None:
    if not isinstance(data, dict):
        raise VerdictSchemaError(
            f"{what} must be a JSON object, got {type(data).__name__}."
        )
    missing = [k for k in required if k not in data]
    if missing:
        raise VerdictSchemaError(f"{what} missing field(s): {missing}.")
    extra = set(data) - allowed
    if extra:
        raise VerdictSchemaError(f"{what} has unexpected field(s): {sorted(extra)}.")


def _require_label(value: Any, what: str) -> str:
    if value not in LABELS:
        raise VerdictSchemaError(
            f"{what} label {value!r} is not one of {list(LABELS)}."
        )
    return value


@dataclass
class CriterionResult:
    id: str
    met: bool
    evidence: str


@dataclass
class JudgeVerdict:
    label: str
    confidence: float
    criteria: list[CriterionResult]
    reasoning: str
    injection_suspected: bool

    @classmethod
    def from_dict(cls, data: Any) -> "JudgeVerdict":
        req = JUDGE_VERDICT_JSON_SCHEMA["required"]
        _require_keys(data, req, set(req), "JudgeVerdict")
        label = _require_label(data["label"], "JudgeVerdict")
        conf = data["confidence"]
        if (
            isinstance(conf, bool)
            or not isinstance(conf, (int, float))
            or not 0.0 <= conf <= 1.0
        ):
            raise VerdictSchemaError(
                f"JudgeVerdict confidence must be a number in [0, 1], got {conf!r}."
            )
        if not isinstance(data["reasoning"], str) or not data["reasoning"].strip():
            raise VerdictSchemaError(
                "JudgeVerdict reasoning must be a non-empty string."
            )
        if not isinstance(data["injection_suspected"], bool):
            raise VerdictSchemaError(
                "JudgeVerdict injection_suspected must be a boolean."
            )
        if not isinstance(data["criteria"], list):
            raise VerdictSchemaError("JudgeVerdict criteria must be a list.")
        criteria = []
        for i, c in enumerate(data["criteria"]):
            _require_keys(
                c,
                ["id", "met", "evidence"],
                {"id", "met", "evidence"},
                f"criteria[{i}]",
            )
            if (
                not isinstance(c["id"], str)
                or not isinstance(c["met"], bool)
                or not isinstance(c["evidence"], str)
            ):
                raise VerdictSchemaError(f"criteria[{i}] has wrong field types.")
            criteria.append(
                CriterionResult(id=c["id"], met=c["met"], evidence=c["evidence"])
            )
        return cls(
            label=label,
            confidence=float(conf),
            criteria=criteria,
            reasoning=data["reasoning"],
            injection_suspected=data["injection_suspected"],
        )


@dataclass
class DeliberationReply:
    decision: str
    label: str
    reason: str
    new_evidence: list[str]

    @classmethod
    def from_dict(cls, data: Any) -> "DeliberationReply":
        req = DELIBERATION_REPLY_JSON_SCHEMA["required"]
        _require_keys(data, req, set(req), "DeliberationReply")
        if data["decision"] not in ("keep", "change"):
            raise VerdictSchemaError(
                f"DeliberationReply decision {data['decision']!r} must be keep|change."
            )
        label = _require_label(data["label"], "DeliberationReply")
        if not isinstance(data["reason"], str) or not data["reason"].strip():
            raise VerdictSchemaError(
                "DeliberationReply reason must be a non-empty string."
            )
        ev = data["new_evidence"]
        if not isinstance(ev, list) or not all(isinstance(x, str) for x in ev):
            raise VerdictSchemaError(
                "DeliberationReply new_evidence must be a list of strings."
            )
        return cls(
            decision=data["decision"],
            label=label,
            reason=data["reason"],
            new_evidence=ev,
        )


# ── Per-row record assembled by code ─────────────────────────────────────────


@dataclass
class ModelRef:
    model_id: str
    family: str  # e.g. "anthropic", "openai", "google", "meta"


@dataclass
class ConfirmationResult:
    schema_valid: bool
    attempts: int
    error: Optional[str] = None


@dataclass
class ConsistencyResult:
    n_configured: int
    n_run: int
    labels: list[str]
    agreement: float  # share of runs matching the majority label
    majority_label: Optional[str]
    consistent: bool


@dataclass
class JudgeRecord:
    judge_id: str  # e.g. "sociotech.a"
    model: ModelRef
    prompt_version: str  # e.g. "sociotech@1.0.0"
    runs: list[JudgeVerdict]
    confirmation: ConfirmationResult
    consistency: ConsistencyResult


@dataclass
class DeliberationRecord:
    ran: bool
    first_labels: dict[str, str]  # judge_id -> label before deliberation
    replies: dict[str, DeliberationReply] = field(default_factory=dict)
    final_labels: dict[str, str] = field(default_factory=dict)
    changed_without_new_argument: dict[str, bool] = field(default_factory=dict)
    tiebreak_judge: Optional[JudgeRecord] = None


@dataclass
class RowResult:
    row_id: str
    dimension: str
    stage: str
    dataset_version: str
    config_hash: str
    target: ModelRef
    target_response_ids: list[str]
    judges: list[JudgeRecord]  # exactly 2 when Phase 2 lands
    deliberation: DeliberationRecord
    final_label: Optional[str]  # None while awaiting human review
    needs_human_review: bool
    review_reasons: list[str] = field(default_factory=list)
    judge_sycophancy_flagged: bool = False
    rubric_sections: list[str] = field(default_factory=list)
    calls: int = 0
    cost_usd: Optional[float] = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> "RowResult":
        if self.final_label is not None:
            _require_label(self.final_label, "RowResult.final_label")
        bad = [r for r in self.review_reasons if r not in REVIEW_REASONS]
        if bad:
            raise VerdictSchemaError(f"Unknown review reason(s): {bad}.")
        if self.needs_human_review != bool(self.review_reasons):
            raise VerdictSchemaError(
                "needs_human_review must be true exactly when review_reasons is non-empty."
            )
        families = [j.model.family for j in self.judges]
        if self.target.family in families:
            raise VerdictSchemaError(
                f"Judge from the tested model's family ({self.target.family}) is not allowed."
            )
        if len(set(families)) != len(families):
            raise VerdictSchemaError(
                "The two judges must come from different model families."
            )
        tb = self.deliberation.tiebreak_judge
        if tb is not None and (
            tb.model.family == self.target.family or tb.model.family in families
        ):
            raise VerdictSchemaError(
                "The tiebreak judge must come from a family different from the tested "
                "model and both panel judges."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
