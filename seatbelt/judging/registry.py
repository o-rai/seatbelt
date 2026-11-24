"""
seatbelt/judging/registry.py

Direct mapping from a row's dimension to its judge panel. No router.

For each dimension the registry holds an ordered list of candidate judge
models. For a given tested model, the panel is the first two candidates
whose families differ from the tested model's family and from each other.
If tiebreaks are enabled, the next candidate from a third family is the
tiebreak judge.

Everything that can go wrong fails loudly, before any model is called:
unknown dimension, or not enough judge families left for this target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from seatbelt.datasets.schema import DIMENSIONS, require_dimension
from seatbelt.judging.verify import CompleteFn, Judge


class JudgePanelError(ValueError):
    """Not enough judge families for this dimension and tested model."""


@dataclass
class JudgeModel:
    """A judge model before it is bound to a dimension."""

    model_id: str
    family: str
    complete: CompleteFn


@dataclass
class Panel:
    dimension: str
    judges: list[Judge]
    tiebreak: Optional[Judge] = None


@dataclass
class JudgeRegistry:
    candidates: dict[str, list[JudgeModel]] = field(default_factory=dict)
    use_tiebreak: bool = False

    def __post_init__(self) -> None:
        for dim, models in self.candidates.items():
            require_dimension(dim, "JudgeRegistry.candidates")
            if not models:
                raise JudgePanelError(f"No judge candidates for dimension {dim!r}.")

    @classmethod
    def uniform(
        cls, models: list[JudgeModel], use_tiebreak: bool = False
    ) -> "JudgeRegistry":
        """Same candidate list for all six dimensions."""
        return cls({d: list(models) for d in DIMENSIONS}, use_tiebreak=use_tiebreak)

    def panel_for(self, dimension: str, target_family: str) -> Panel:
        require_dimension(dimension, "panel_for")
        if dimension not in self.candidates:
            raise JudgePanelError(f"No judges registered for dimension {dimension!r}.")

        picked: list[JudgeModel] = []
        families: set[str] = {target_family}
        for m in self.candidates[dimension]:
            if m.family in families:
                continue
            picked.append(m)
            families.add(m.family)
            if len(picked) == (3 if self.use_tiebreak else 2):
                break

        if len(picked) < 2:
            available = sorted({m.family for m in self.candidates[dimension]})
            raise JudgePanelError(
                f"Dimension {dimension!r} needs 2 judge families other than the tested "
                f"model's ({target_family}); registry has {available}."
            )

        def bind(m: JudgeModel) -> Judge:
            return Judge(f"{dimension}.{m.family}", m.model_id, m.family, m.complete)

        tiebreak = bind(picked[2]) if len(picked) == 3 else None
        return Panel(dimension, [bind(m) for m in picked[:2]], tiebreak)

    def describe(self) -> dict[str, list[str]]:
        """Model IDs per dimension, for the config hash."""
        return {
            d: [f"{m.family}:{m.model_id}" for m in ms]
            for d, ms in sorted(self.candidates.items())
        }
