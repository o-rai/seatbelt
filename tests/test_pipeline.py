"""
Phase 3 tests: dimension -> judge mapping, unknown dimension, multi-turn
target calls, N=3/N=5 through the pipeline, and the human-review path.

Tested models and judges are fakes. No API calls.
"""

import json
import re
from pathlib import Path

import pytest

from seatbelt.datasets import (
    DIMENSIONS,
    DatasetConfig,
    DatasetRow,
    UnknownDimensionError,
    load_dataset,
)
from seatbelt.judging import (
    BudgetError,
    JudgeModel,
    JudgePanelError,
    JudgeRegistry,
    RubricError,
    VerificationConfig,
)
from seatbelt.judging.schema import REVIEW_JUDGE_SYCOPHANCY_NO_TIEBREAK
from seatbelt.pipeline import DirectStageRunner, TargetModel, TargetModelError, evaluate

FIXTURES = Path(__file__).parent / "fixtures" / "dataset"


# ── Fakes ─────────────────────────────────────────────────────────────────────


def verdict(label="pass", injection=False):
    return json.dumps(
        {
            "label": label,
            "confidence": 0.9,
            "criteria": [{"id": "X-1", "met": label == "pass", "evidence": ""}],
            "reasoning": f"Judged {label}.",
            "injection_suspected": injection,
        }
    )


class FakeJudgeModel:
    """Returns a fixed first-pass label; in deliberation, keeps or flips (without evidence)."""

    def __init__(self, family, label="pass", flip_to=None):
        self.family, self.label, self.flip_to = family, label, flip_to
        self.systems = []

    def __call__(self, system, user, schema):
        self.systems.append(system)
        if "Deliberation round" in system:
            label = self.flip_to or self.label
            decision = "change" if label != self.label else "keep"
            return json.dumps(
                {
                    "decision": decision,
                    "label": label,
                    "reason": "r",
                    "new_evidence": [],
                }
            )
        return verdict(self.label)

    def model(self):
        return JudgeModel(f"{self.family}-judge", self.family, self)

    def dimensions_seen(self):
        return [re.search(r"You are the (\w+) judge", s).group(1) for s in self.systems]


class FakeChat:
    def __init__(self):
        self.calls = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        return f"<think>hidden</think>Reply {len(self.calls)}"


def target(family="openai", chat=None):
    return TargetModel("target-x", family, chat or FakeChat())


def registry(*fakes, use_tiebreak=False):
    return JudgeRegistry.uniform([f.model() for f in fakes], use_tiebreak=use_tiebreak)


def row(dimension="sociotech", row_id="r1", turns=("Question?",), **meta):
    meta.setdefault(
        "probe_type", "automation_bias" if dimension == "sociotech" else None
    )
    meta = {k: v for k, v in meta.items() if v is not None}
    return DatasetRow(
        row_id=row_id,
        dimension=dimension,
        stage="open_ended",
        turns=list(turns),
        metadata=meta,
    )


# ── Dimension -> judge mapping ───────────────────────────────────────────────


class TestJudgeMapping:
    def setup_method(self):
        self.reg = registry(
            FakeJudgeModel("openai"),
            FakeJudgeModel("anthropic"),
            FakeJudgeModel("google"),
            FakeJudgeModel("meta"),
        )

    @pytest.mark.parametrize("dimension", DIMENSIONS)
    def test_each_dimension_maps_to_its_own_panel(self, dimension):
        panel = self.reg.panel_for(dimension, target_family="openai")
        assert panel.dimension == dimension
        assert [j.judge_id for j in panel.judges] == [
            f"{dimension}.anthropic",
            f"{dimension}.google",
        ]

    def test_tested_model_family_is_excluded(self):
        for fam in ("openai", "anthropic", "google", "meta"):
            families = [j.family for j in self.reg.panel_for("privacy", fam).judges]
            assert fam not in families and len(set(families)) == 2

    def test_tiebreak_is_a_third_family(self):
        reg = JudgeRegistry(self.reg.candidates, use_tiebreak=True)
        panel = reg.panel_for("fairness", "openai")
        assert panel.tiebreak.family == "meta"

    def test_tiebreak_absent_when_no_third_family(self):
        reg = registry(
            FakeJudgeModel("anthropic"), FakeJudgeModel("google"), use_tiebreak=True
        )
        assert reg.panel_for("fairness", "openai").tiebreak is None

    def test_not_enough_families_fails_loudly(self):
        reg = registry(FakeJudgeModel("openai"), FakeJudgeModel("anthropic"))
        with pytest.raises(JudgePanelError, match="needs 2 judge families"):
            reg.panel_for("deception", "openai")

    def test_per_dimension_candidates(self):
        a, g, m = (
            FakeJudgeModel("anthropic").model(),
            FakeJudgeModel("google").model(),
            FakeJudgeModel("meta").model(),
        )
        reg = JudgeRegistry({"sociotech": [a, g], "privacy": [g, m]})
        assert [j.family for j in reg.panel_for("privacy", "openai").judges] == [
            "google",
            "meta",
        ]
        with pytest.raises(JudgePanelError, match="No judges registered"):
            reg.panel_for("fairness", "openai")

    @pytest.mark.parametrize("bad", ["socio", "Fairness", "", "router"])
    def test_unknown_dimension_error(self, bad):
        with pytest.raises(UnknownDimensionError):
            self.reg.panel_for(bad, "openai")

    def test_unknown_dimension_in_registry_rejected(self):
        with pytest.raises(UnknownDimensionError):
            JudgeRegistry({"socio": [FakeJudgeModel("google").model()]})


# ── Tested model calls ───────────────────────────────────────────────────────


class TestTargetModel:
    def test_turns_sent_as_one_conversation(self):
        chat = FakeChat()
        transcript = DirectStageRunner().run(
            row(turns=("Q1", "Pushback")), target(chat=chat)
        )
        assert chat.calls[0] == [{"role": "user", "content": "Q1"}]
        assert chat.calls[1] == [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Pushback"},
        ]
        assert [t.role for t in transcript] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert transcript[1].content == "Reply 1"  # <think> stripped

    def test_legacy_prompt_fn_replays_history(self):
        prompts = []
        t = TargetModel.from_prompt_fn(
            lambda p: prompts.append(p) or "ok", "legacy", "meta"
        )
        DirectStageRunner().run(row(turns=("Q1", "Are you sure?")), t)
        assert prompts[0] == "Q1"
        assert "User: Q1" in prompts[1] and "Assistant: ok" in prompts[1]
        assert prompts[1].endswith("Now the user says: Are you sure?")

    def test_target_failure_names_the_row(self):
        def broken(messages):
            raise ConnectionError("down")

        with pytest.raises(TargetModelError, match="row 'r1'"):
            DirectStageRunner().run(row(), target(chat=broken))


# ── evaluate(): end to end ───────────────────────────────────────────────────


class TestEvaluate:
    def test_rows_go_to_their_dimension_judges(self):
        a, g = FakeJudgeModel("anthropic"), FakeJudgeModel("google")
        rows = [
            row("sociotech", "s1"),
            row("privacy", "p1", probe_type="pii_leakage"),
            row("transparency", "t1", probe_type="self_disclosure"),
        ]
        result = evaluate(rows, target(), registry(a, g))
        assert [r.dimension for r in result.rows] == [
            "sociotech",
            "privacy",
            "transparency",
        ]
        assert [r.judges[0].judge_id for r in result.rows] == [
            "sociotech.anthropic",
            "privacy.anthropic",
            "transparency.anthropic",
        ]
        # 3 early-stop runs per row, each in the row's own dimension
        assert (
            a.dimensions_seen()
            == ["sociotech"] * 3 + ["privacy"] * 3 + ["transparency"] * 3
        )
        assert all(
            r.final_label == "pass" and not r.needs_human_review for r in result.rows
        )
        assert result.rows[0].target_response_ids == ["s1#turn2"]

    def test_unknown_dimension_fails_before_any_call(self):
        chat, a = FakeChat(), FakeJudgeModel("anthropic")
        bad = DatasetRow(
            row_id="x", dimension="socio", stage="open_ended", turns=["q"]
        )  # not validated
        with pytest.raises(UnknownDimensionError, match="socio"):
            evaluate(
                [row(), bad], target(chat=chat), registry(a, FakeJudgeModel("google"))
            )
        assert chat.calls == [] and a.systems == []

    def test_unmatched_rubric_section_fails_before_any_call(self):
        chat = FakeChat()
        with pytest.raises(RubricError):
            evaluate(
                [row("fairness", probe_type="mystery")],
                target(chat=chat),
                registry(FakeJudgeModel("anthropic"), FakeJudgeModel("google")),
            )
        assert chat.calls == []

    def test_missing_panel_fails_before_any_call(self):
        chat = FakeChat()
        with pytest.raises(JudgePanelError):
            evaluate(
                [row()],
                target("anthropic", chat),
                registry(FakeJudgeModel("anthropic"), FakeJudgeModel("google")),
            )
        assert chat.calls == []

    def test_default_n_is_5(self):
        a = FakeJudgeModel("anthropic")
        result = evaluate([row()], target(), registry(a, FakeJudgeModel("google")))
        assert result.n == 5
        assert result.rows[0].judges[0].consistency.n_configured == 5
        assert result.estimate["n"] == 5

    def test_budget_falls_back_to_n3(self):
        rows = [row(row_id=f"r{i}") for i in range(10)]
        cfg = VerificationConfig(max_judge_calls=80)  # N=5 needs ~103, N=3 ~63
        result = evaluate(
            rows,
            target(),
            registry(FakeJudgeModel("anthropic"), FakeJudgeModel("google")),
            cfg,
        )
        assert result.n == 3
        assert all(r.judges[0].consistency.n_configured == 3 for r in result.rows)

    def test_budget_too_small_fails_before_any_call(self):
        chat = FakeChat()
        with pytest.raises(BudgetError):
            evaluate(
                [row()],
                target(chat=chat),
                registry(FakeJudgeModel("anthropic"), FakeJudgeModel("google")),
                VerificationConfig(max_judge_calls=2),
            )
        assert chat.calls == []

    def test_human_review_flag_path(self):
        a = FakeJudgeModel("anthropic", label="pass")
        g = FakeJudgeModel(
            "google", label="fail", flip_to="pass"
        )  # flips with no new evidence
        result = evaluate([row()], target(), registry(a, g))
        r = result.rows[0]
        assert r.judge_sycophancy_flagged
        assert r.needs_human_review is True
        assert r.review_reasons == [REVIEW_JUDGE_SYCOPHANCY_NO_TIEBREAK]
        assert r.final_label is None
        assert result.needs_review() == [r]

    def test_tiebreak_resolves_instead_of_review(self):
        a = FakeJudgeModel("anthropic", label="pass")
        g = FakeJudgeModel("google", label="fail", flip_to="pass")
        m = FakeJudgeModel("meta", label="pass")
        result = evaluate([row()], target(), registry(a, g, m, use_tiebreak=True))
        assert result.rows[0].final_label == "pass" and not result.needs_review()

    def test_loaded_dataset_end_to_end(self):
        ds = load_dataset(DatasetConfig(local_dir=str(FIXTURES)))
        result = evaluate(
            ds,
            target(),
            registry(FakeJudgeModel("anthropic"), FakeJudgeModel("google")),
        )
        assert len(result.rows) == len(ds.rows) == 6
        assert result.dataset_version.startswith("local-unversioned+")
        sections = {r.row_id: r.rubric_sections for r in result.rows}
        assert sections["FX-01-orig-amb-nonneg"] == ["stereotype_reliance", "pressure"]
        assert sections["fair_fx_001"] == ["decision_basis"]
        # BBQ pushback row: 3 user turns -> 3 tested-model replies
        assert len(result.transcripts["FX-01-orig-amb-nonneg"]) == 6
        json.dumps(result.to_dict())  # serializable for the results database

    def test_config_hash_changes_with_n(self):
        reg = registry(FakeJudgeModel("anthropic"), FakeJudgeModel("google"))
        h5 = evaluate([row()], target(), reg).config_hash
        h3 = evaluate(
            [row()], target(), reg, VerificationConfig(consistency_n=3)
        ).config_hash
        assert h5 != h3

    def test_on_row_callback(self):
        seen = []
        evaluate(
            [row(row_id="a"), row(row_id="b")],
            target(),
            registry(FakeJudgeModel("anthropic"), FakeJudgeModel("google")),
            on_row=seen.append,
        )
        assert [r.row_id for r in seen] == ["a", "b"]
