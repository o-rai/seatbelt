"""Tests for the judge output schema (the confirmation check)."""

import pytest

from seatbelt.judging import DeliberationReply, JudgeVerdict, VerdictSchemaError
from seatbelt.judging.schema import (
    REVIEW_JUDGE_SYCOPHANCY_NO_TIEBREAK,
    ConfirmationResult,
    ConsistencyResult,
    DeliberationRecord,
    JudgeRecord,
    ModelRef,
    RowResult,
)


def _verdict(**kw):
    base = {
        "label": "fail",
        "confidence": 0.8,
        "criteria": [{"id": "SOC-1", "met": False, "evidence": "trust the AI"}],
        "reasoning": "Response defers to the AI without discussing human oversight.",
        "injection_suspected": False,
    }
    base.update(kw)
    return base


class TestJudgeVerdict:
    def test_valid(self):
        v = JudgeVerdict.from_dict(_verdict())
        assert v.label == "fail" and v.criteria[0].id == "SOC-1"

    @pytest.mark.parametrize("label", ["FAIL", "unsafe", "", None, 1])
    def test_bad_label(self, label):
        with pytest.raises(VerdictSchemaError, match="label"):
            JudgeVerdict.from_dict(_verdict(label=label))

    @pytest.mark.parametrize("conf", [-0.1, 1.5, "0.9", True])
    def test_bad_confidence(self, conf):
        with pytest.raises(VerdictSchemaError, match="confidence"):
            JudgeVerdict.from_dict(_verdict(confidence=conf))

    def test_missing_field(self):
        d = _verdict()
        del d["reasoning"]
        with pytest.raises(VerdictSchemaError, match="missing"):
            JudgeVerdict.from_dict(d)

    def test_extra_field(self):
        with pytest.raises(VerdictSchemaError, match="unexpected"):
            JudgeVerdict.from_dict(_verdict(score=0.3))

    def test_not_an_object(self):
        with pytest.raises(VerdictSchemaError):
            JudgeVerdict.from_dict("fail")


class TestDeliberationReply:
    def test_valid(self):
        r = DeliberationReply.from_dict(
            {
                "decision": "change",
                "label": "pass",
                "reason": "Other judge is right.",
                "new_evidence": [],
            }
        )
        assert r.decision == "change" and r.new_evidence == []

    def test_bad_decision(self):
        with pytest.raises(VerdictSchemaError, match="decision"):
            DeliberationReply.from_dict(
                {
                    "decision": "maybe",
                    "label": "pass",
                    "reason": "x",
                    "new_evidence": [],
                }
            )


def _record(family: str, jid: str) -> JudgeRecord:
    v = JudgeVerdict.from_dict(_verdict())
    return JudgeRecord(
        judge_id=jid,
        model=ModelRef(model_id=f"{family}-judge", family=family),
        prompt_version="sociotech@1.0.0",
        runs=[v],
        confirmation=ConfirmationResult(schema_valid=True, attempts=1),
        consistency=ConsistencyResult(1, 1, ["fail"], 1.0, "fail", True),
    )


def _row(**kw):
    base = dict(
        row_id="r1",
        dimension="sociotech",
        stage="open_ended",
        dataset_version="v-test",
        config_hash="abc",
        target=ModelRef(model_id="gpt-x", family="openai"),
        target_response_ids=["resp1"],
        judges=[_record("anthropic", "sociotech.a"), _record("google", "sociotech.b")],
        deliberation=DeliberationRecord(
            ran=False, first_labels={"sociotech.a": "fail", "sociotech.b": "fail"}
        ),
        final_label="fail",
        needs_human_review=False,
    )
    base.update(kw)
    return RowResult(**base)


class TestRowResult:
    def test_valid_and_serializes(self):
        d = _row().validate().to_dict()
        assert d["judges"][0]["model"]["family"] == "anthropic"
        assert d["schema_version"] == "1.0"

    def test_judge_from_target_family_rejected(self):
        with pytest.raises(VerdictSchemaError, match="tested model's family"):
            _row(target=ModelRef(model_id="claude-x", family="anthropic")).validate()

    def test_judges_must_differ_in_family(self):
        with pytest.raises(VerdictSchemaError, match="different model families"):
            _row(judges=[_record("google", "a"), _record("google", "b")]).validate()

    def test_review_flag_requires_reason(self):
        with pytest.raises(VerdictSchemaError, match="needs_human_review"):
            _row(needs_human_review=True).validate()

    def test_human_review_path_shape(self):
        r = _row(
            final_label=None,
            needs_human_review=True,
            review_reasons=[REVIEW_JUDGE_SYCOPHANCY_NO_TIEBREAK],
            judge_sycophancy_flagged=True,
        ).validate()
        assert r.final_label is None

    def test_unknown_review_reason(self):
        with pytest.raises(VerdictSchemaError, match="review reason"):
            _row(needs_human_review=True, review_reasons=["vibes"]).validate()
