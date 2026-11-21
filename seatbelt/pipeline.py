"""
seatbelt/pipeline.py

The per-dimension evaluation pipeline:

    dataset rows ──► stage runner ──► tested model (multi-turn)
                                          │ transcript
                                          ▼
                   row.dimension ──► judge panel (2 judges) ──► verification ──► RowResult

There is no router: each row's `dimension` picks its panel directly.

Seams for the Inspect AI integration:
  * StageRunner: DirectStageRunner sends a row's turns to the tested model in
    one conversation. An Inspect-backed runner (MCQ / open ended / safety
    stages as Inspect tasks and solvers) can replace it as long as it returns
    the same transcript (list[Turn]).
  * Judge models are plain callables (see JudgeModel); Inspect model
    providers can supply them.

The legacy rule-based `audit()` in seatbelt/core.py is not used here.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Protocol, Union

from seatbelt.datasets.loader import LoadedDataset
from seatbelt.datasets.schema import DatasetRow, require_dimension
from seatbelt.judging.config import (
    VerificationConfig,
    choose_n,
    compute_config_hash,
    estimate_judge_calls,
)
from seatbelt.judging.prompts import Turn
from seatbelt.judging.registry import JudgeRegistry
from seatbelt.judging.rubrics import RUBRICS, get_rubric, select_sections
from seatbelt.judging.schema import ModelRef, RowResult
from seatbelt.judging.verify import verify_row

Message = dict  # {"role": "user" | "assistant", "content": str}
ChatFn = Callable[[list[Message]], str]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class TargetModelError(RuntimeError):
    """The tested model failed on a row."""


# ── Tested model ─────────────────────────────────────────────────────────────


@dataclass
class TargetModel:
    """
    The model under evaluation. `chat` receives the whole conversation so far
    (OpenAI/Anthropic-style messages) and returns the next reply.
    """

    model_id: str
    family: str
    chat: ChatFn

    @classmethod
    def from_prompt_fn(
        cls, fn: Callable[[str], str], model_id: str, family: str
    ) -> "TargetModel":
        """
        Wrap a legacy `prompt -> reply` function. Earlier turns are replayed
        as text inside a single prompt, since the function keeps no history.
        Prefer a real `chat` function where the API supports it.
        """

        def chat(messages: list[Message]) -> str:
            if len(messages) == 1:
                return fn(messages[0]["content"])
            history = "\n\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in messages[:-1]
            )
            return fn(
                "Earlier in this conversation:\n\n"
                f"{history}\n\nNow the user says: {messages[-1]['content']}"
            )

        return cls(model_id, family, chat)

    @property
    def ref(self) -> ModelRef:
        return ModelRef(model_id=self.model_id, family=self.family)


# ── Stage runner ─────────────────────────────────────────────────────────────


class StageRunner(Protocol):
    def run(self, row: DatasetRow, target: TargetModel) -> list[Turn]: ...


class DirectStageRunner:
    """Send each of the row's turns to the tested model, in one conversation."""

    def run(self, row: DatasetRow, target: TargetModel) -> list[Turn]:
        messages: list[Message] = []
        transcript: list[Turn] = []
        for prompt in row.turns:
            messages.append({"role": "user", "content": prompt})
            transcript.append(Turn("user", prompt))
            try:
                reply = target.chat(list(messages))
            except Exception as e:
                raise TargetModelError(
                    f"Tested model {target.model_id} failed on row {row.row_id!r}: {e}"
                ) from e
            reply = _THINK_RE.sub("", str(reply)).strip()
            messages.append({"role": "assistant", "content": reply})
            transcript.append(Turn("assistant", reply))
        return transcript


# ── Pipeline ─────────────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    rows: list[RowResult]
    dataset_version: str
    config_hash: str
    n: int
    estimate: dict
    target: ModelRef
    transcripts: dict[str, list[Turn]] = field(default_factory=dict)

    def needs_review(self) -> list[RowResult]:
        return [r for r in self.rows if r.needs_human_review]

    def by_dimension(self) -> dict[str, list[RowResult]]:
        out: dict[str, list[RowResult]] = {}
        for r in self.rows:
            out.setdefault(r.dimension, []).append(r)
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def preflight(
    rows: list[DatasetRow], target: TargetModel, registry: JudgeRegistry
) -> None:
    """Check every row can be judged before any model is called."""
    for row in rows:
        require_dimension(row.dimension, f"row {row.row_id!r}")
        registry.panel_for(row.dimension, target.family)
        select_sections(get_rubric(row.dimension), row)


def evaluate(
    dataset: Union[LoadedDataset, list[DatasetRow]],
    target: TargetModel,
    registry: JudgeRegistry,
    config: Optional[VerificationConfig] = None,
    stage_runner: Optional[StageRunner] = None,
    dataset_version: Optional[str] = None,
    on_row: Optional[Callable[[RowResult], None]] = None,
) -> PipelineResult:
    """
    Run every row: tested model → the row's dimension panel → verification.

    Raises before any model call if a row has an unknown dimension, no
    matching rubric section, or no valid judge panel for this target, or if
    the call budget can't fit even N=fallback_n.
    """
    config = (config or VerificationConfig()).validate()
    stage_runner = stage_runner or DirectStageRunner()
    if isinstance(dataset, LoadedDataset):
        rows = dataset.rows
        dataset_version = (
            dataset_version or f"{dataset.version}+{dataset.content_hash[:12]}"
        )
    else:
        rows = list(dataset)
        dataset_version = dataset_version or "unversioned"

    preflight(rows, target, registry)
    n = choose_n(len(rows), config, tiebreak=registry.use_tiebreak)
    estimate = estimate_judge_calls(
        len(rows), n, 2, config.deliberation_rate_estimate, registry.use_tiebreak
    )
    config_hash = compute_config_hash(
        verification=config,
        n=n,
        rubrics=sorted(r.prompt_version for r in RUBRICS.values()),
        judges=registry.describe(),
        use_tiebreak=registry.use_tiebreak,
        target=asdict(target.ref),
        dataset_version=dataset_version,
    )

    results: list[RowResult] = []
    transcripts: dict[str, list[Turn]] = {}
    for row in rows:
        transcript = stage_runner.run(row, target)
        transcripts[row.row_id] = transcript
        panel = registry.panel_for(row.dimension, target.family)
        result = verify_row(
            row,
            transcript,
            panel.judges,
            config,
            n,
            target.ref,
            dataset_version=dataset_version,
            config_hash=config_hash,
            target_response_ids=[
                f"{row.row_id}#turn{i}"
                for i, t in enumerate(transcript, start=1)
                if t.role == "assistant"
            ],
            tiebreak=panel.tiebreak,
        )
        results.append(result)
        if on_row:
            on_row(result)

    return PipelineResult(
        rows=results,
        dataset_version=dataset_version,
        config_hash=config_hash,
        n=n,
        estimate=estimate,
        target=target.ref,
        transcripts=transcripts,
    )
