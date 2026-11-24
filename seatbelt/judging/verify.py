"""
seatbelt/judging/verify.py

Verification layer for one row: two judges, each checked three ways.

  1. Confirmation  every judge reply must parse into the schema with a valid
                   label. One retry with the parse error, then the run counts
                   as a schema failure.
  2. Consistency   each judge grades the same response N times (N from
                   choose_n: 5, or 3 under a tight budget). With early stop,
                   N=5 runs 3 first and continues only if they disagree.
  3. Sycophancy    if the two judges' majority labels differ, one
                   deliberation round runs. A judge that changes its label
                   without citing new evidence (a verbatim quote from the
                   tested model's replies that it hadn't cited) is flagged.

Resolution: if the judges agree (and nobody flipped without evidence), that
label is final. Otherwise a tiebreak judge decides if one is configured; if
not, the row is NOT auto-resolved: needs_human_review=True with a reason and
final_label=None.

Judge models are plain callables here: complete(system, user, json_schema) -> str.
Provider adapters (Anthropic, OpenAI, Google, Inspect) plug in at that seam.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

from seatbelt.datasets.schema import DatasetRow
from seatbelt.judging.config import VerificationConfig
from seatbelt.judging.prompts import (
    Turn,
    build_deliberation_prompts,
    build_system_prompt,
    build_user_prompt,
    model_text,
    parse_json_reply,
    unescape_data,
)
from seatbelt.judging.rubrics import Rubric, get_rubric, select_sections
from seatbelt.judging.schema import (
    DELIBERATION_REPLY_JSON_SCHEMA,
    JUDGE_VERDICT_JSON_SCHEMA,
    REVIEW_DISAGREEMENT_NO_TIEBREAK,
    REVIEW_INCONSISTENT,
    REVIEW_INJECTION_SUSPECTED,
    REVIEW_JUDGE_SYCOPHANCY_NO_TIEBREAK,
    REVIEW_SCHEMA_FAILURE,
    ConfirmationResult,
    ConsistencyResult,
    DeliberationRecord,
    DeliberationReply,
    JudgeRecord,
    JudgeVerdict,
    ModelRef,
    RowResult,
    VerdictSchemaError,
)

T = TypeVar("T")
CompleteFn = Callable[[str, str, dict], str]


@dataclass
class Judge:
    judge_id: str
    model_id: str
    family: str
    complete: CompleteFn

    @property
    def ref(self) -> ModelRef:
        return ModelRef(model_id=self.model_id, family=self.family)


# ── Confirmation ─────────────────────────────────────────────────────────────


def call_with_schema(
    judge: Judge,
    system: str,
    user: str,
    schema: dict,
    parser: Callable[[object], T],
    max_attempts: int,
) -> tuple[Optional[T], int, Optional[str]]:
    """Return (parsed or None, attempts used, last error)."""
    error: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        prompt = user
        if error is not None:
            prompt += (
                f"\n\nYour previous reply was invalid ({error}). "
                "Respond only with JSON matching the required schema."
            )
        try:
            raw = judge.complete(system, prompt, schema)
        except Exception as e:  # provider failure counts as a failed attempt
            error = f"judge call failed: {e}"
            continue
        try:
            return parser(parse_json_reply(raw)), attempt, None
        except (
            json.JSONDecodeError,
            VerdictSchemaError,
            TypeError,
            AttributeError,
        ) as e:
            error = str(e)
    return None, max_attempts, error


# ── Consistency ──────────────────────────────────────────────────────────────


def run_judge(
    judge: Judge,
    rubric: Rubric,
    system: str,
    user: str,
    n: int,
    config: VerificationConfig,
) -> tuple[JudgeRecord, int]:
    """Run one judge up to n times. Returns its record and the calls used."""
    first_batch = (
        config.fallback_n if (config.early_stop and n > config.fallback_n) else n
    )
    verdicts: list[JudgeVerdict] = []
    errors: list[str] = []
    calls = 0

    def run_once() -> None:
        nonlocal calls
        v, attempts, err = call_with_schema(
            judge,
            system,
            user,
            JUDGE_VERDICT_JSON_SCHEMA,
            JudgeVerdict.from_dict,
            config.max_schema_attempts,
        )
        calls += attempts
        if v is None:
            errors.append(err or "invalid reply")
        else:
            verdicts.append(v)

    for _ in range(first_batch):
        run_once()
    unanimous = len(verdicts) == first_batch and len({v.label for v in verdicts}) == 1
    if n > first_batch and not unanimous:
        for _ in range(n - first_batch):
            run_once()

    labels = [v.label for v in verdicts]
    majority: Optional[str] = None
    agreement = 0.0
    if labels:
        ranked = Counter(labels).most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            majority = ranked[0][0]
        agreement = ranked[0][1] / len(labels)
    consistent = majority is not None and agreement >= config.min_agreement

    record = JudgeRecord(
        judge_id=judge.judge_id,
        model=judge.ref,
        prompt_version=rubric.prompt_version,
        runs=verdicts,
        confirmation=ConfirmationResult(
            schema_valid=not errors,
            attempts=calls,
            error=errors[-1] if errors else None,
        ),
        consistency=ConsistencyResult(
            n_configured=n,
            n_run=len(verdicts) + len(errors),
            labels=labels,
            agreement=round(agreement, 3),
            majority_label=majority,
            consistent=consistent,
        ),
    )
    return record, calls


# ── Sycophancy: evidence check ───────────────────────────────────────────────


def _norm(text: str) -> str:
    return " ".join(unescape_data(text).lower().split())


def valid_new_evidence(
    quotes: list[str], transcript: list[Turn], own_runs: list[JudgeVerdict]
) -> list[str]:
    """
    Keep quotes that (a) appear verbatim in the tested model's replies and
    (b) were not already cited by this judge in its first-pass runs.
    """
    replies = _norm(model_text(transcript))
    cited = [
        _norm(c.evidence) for v in own_runs for c in v.criteria if c.evidence.strip()
    ]
    out = []
    for q in quotes:
        nq = _norm(q)
        if not nq or nq not in replies:
            continue  # fabricated or paraphrased "evidence" doesn't count
        if any(nq in c for c in cited):
            continue  # already cited before deliberation
        out.append(q)
    return out


# ── Row ──────────────────────────────────────────────────────────────────────


def verify_row(
    row: DatasetRow,
    transcript: list[Turn],
    judges: list[Judge],
    config: VerificationConfig,
    n: int,
    target: ModelRef,
    dataset_version: str,
    config_hash: str,
    target_response_ids: Optional[list[str]] = None,
    tiebreak: Optional[Judge] = None,
) -> RowResult:
    config.validate()
    if len(judges) != 2:
        raise ValueError(f"Each row needs exactly 2 judges, got {len(judges)}.")
    families = [j.family for j in judges]
    if target.family in families or len(set(families)) != 2:
        raise ValueError(
            f"Judges must come from 2 different families, neither equal to the tested "
            f"model's family ({target.family}); got {families}."
        )
    if tiebreak is not None and (
        tiebreak.family == target.family or tiebreak.family in families
    ):
        raise ValueError(
            "Tiebreak judge must come from a third family, not the tested model's."
        )
    if not any(t.role == "assistant" for t in transcript):
        raise ValueError(
            f"Row {row.row_id!r}: transcript has no tested-model reply to judge."
        )

    rubric = get_rubric(row.dimension)
    sections = select_sections(rubric, row)
    system = build_system_prompt(rubric, sections)
    user = build_user_prompt(row, transcript)

    reasons: list[str] = []

    def flag(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    calls = 0
    records: list[JudgeRecord] = []
    for j in judges:
        rec, c = run_judge(j, rubric, system, user, n, config)
        records.append(rec)
        calls += c

    for rec in records:
        if not rec.confirmation.schema_valid:
            flag(REVIEW_SCHEMA_FAILURE)
        if rec.runs and not rec.consistency.consistent:
            flag(REVIEW_INCONSISTENT)
        if rec.runs and sum(v.injection_suspected for v in rec.runs) * 2 > len(
            rec.runs
        ):
            flag(REVIEW_INJECTION_SUSPECTED)

    first = {rec.judge_id: rec.consistency.majority_label for rec in records}
    delib = DeliberationRecord(ran=False, first_labels=dict(first))
    final: Optional[str] = None
    sycophancy = False

    if all(label is not None for label in first.values()):
        if len(set(first.values())) == 1:
            final = next(iter(first.values()))
        else:
            delib.ran = True
            reps = {
                rec.judge_id: next(
                    v for v in rec.runs if v.label == rec.consistency.majority_label
                )
                for rec in records
            }
            replies_ok = True
            for i, (judge, rec) in enumerate(zip(judges, records)):
                other = records[1 - i]
                d_system, d_user = build_deliberation_prompts(
                    rubric,
                    sections,
                    row,
                    transcript,
                    reps[rec.judge_id],
                    reps[other.judge_id],
                )
                reply, attempts, _err = call_with_schema(
                    judge,
                    d_system,
                    d_user,
                    DELIBERATION_REPLY_JSON_SCHEMA,
                    DeliberationReply.from_dict,
                    config.max_schema_attempts,
                )
                calls += attempts
                if reply is None:
                    replies_ok = False
                    flag(REVIEW_SCHEMA_FAILURE)
                    delib.final_labels[rec.judge_id] = first[rec.judge_id]
                    continue
                delib.replies[rec.judge_id] = reply
                delib.final_labels[rec.judge_id] = reply.label
                changed = reply.label != first[rec.judge_id]
                evidence = valid_new_evidence(reply.new_evidence, transcript, rec.runs)
                flipped_without_argument = changed and not evidence
                delib.changed_without_new_argument[rec.judge_id] = (
                    flipped_without_argument
                )
                sycophancy = sycophancy or flipped_without_argument

            agreed = len(set(delib.final_labels.values())) == 1
            if agreed and replies_ok and not sycophancy:
                final = next(iter(delib.final_labels.values()))
            elif tiebreak is not None:
                tb, c = run_judge(tiebreak, rubric, system, user, n, config)
                calls += c
                delib.tiebreak_judge = tb
                if tb.consistency.consistent and tb.confirmation.schema_valid:
                    final = tb.consistency.majority_label
                else:
                    flag(REVIEW_INCONSISTENT if tb.runs else REVIEW_SCHEMA_FAILURE)
            elif sycophancy:
                flag(REVIEW_JUDGE_SYCOPHANCY_NO_TIEBREAK)
            else:
                flag(REVIEW_DISAGREEMENT_NO_TIEBREAK)

    # Anything except a suspected injection means the label can't be trusted.
    if any(r != REVIEW_INJECTION_SUSPECTED for r in reasons):
        final = None

    return RowResult(
        row_id=row.row_id,
        dimension=row.dimension,
        stage=row.stage,
        dataset_version=dataset_version,
        config_hash=config_hash,
        target=target,
        target_response_ids=list(target_response_ids or []),
        judges=records,
        deliberation=delib,
        final_label=final,
        needs_human_review=bool(reasons),
        review_reasons=reasons,
        judge_sycophancy_flagged=sycophancy,
        rubric_sections=[s.id for s in sections],
        calls=calls,
    ).validate()
