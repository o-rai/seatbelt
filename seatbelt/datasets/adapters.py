"""
seatbelt/datasets/adapters.py

Convert the formats the dataset currently exists in into DatasetRow objects.

  - canonical JSONL / JSON (the target format, see README "Dataset contract")
  - legacy probe JSON (e.g. sociotech/private_probes.json)
  - BBQ-style CSV exported from the per-class spreadsheet tabs, plus the
    "Templates" tab that maps Template ID -> protected class

Adapters fail loudly. A row we can't convert is an error, not a skip.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Optional

from seatbelt.datasets.schema import DatasetError, DatasetRow

# ── Canonical ────────────────────────────────────────────────────────────────


def parse_canonical_jsonl(text: str, source: str) -> list[DatasetRow]:
    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON on line {lineno} of {source}: {e}") from e
        rows.append(DatasetRow.from_dict(data, source=f"{source}:{lineno}"))
    return rows


def parse_canonical_json(items: list[dict], source: str) -> list[DatasetRow]:
    return [DatasetRow.from_dict(item, source=source) for item in items]


# ── Legacy probe JSON ────────────────────────────────────────────────────────

_PAIRED_KEYS = ("prompt_a", "prompt_en", "monitored_prompt")


def parse_legacy_probes(
    items: list[dict], dimension: str, source: str, default_stage: str = "open_ended"
) -> list[DatasetRow]:
    """
    Old probe dicts: {"id", "type", "weight", "prompt" | "setup_prompt"+"followup_prompt"
    | "initial_prompt"+"pressure_prompt", "red_flags", "green_flags", "notes", ...}.
    """
    rows = []
    for item in items:
        if not isinstance(item, dict):
            raise DatasetError(f"Probe entries must be objects ({source}).")
        pid = item.get("id")
        if any(k in item for k in _PAIRED_KEYS):
            raise DatasetError(
                f"Probe {pid!r} in {source} is a paired probe (A/B variants). Paired probes "
                "are not supported by the loader yet: split them into two rows that share "
                "metadata.pair_id."
            )
        if "prompt" in item:
            turns = [item["prompt"]]
        elif "setup_prompt" in item and "followup_prompt" in item:
            turns = [item["setup_prompt"], item["followup_prompt"]]
        elif "initial_prompt" in item and "pressure_prompt" in item:
            turns = [item["initial_prompt"], item["pressure_prompt"]]
        else:
            raise DatasetError(f"Probe {pid!r} in {source} has no prompt field.")

        hints = {k: item[k] for k in ("red_flags", "green_flags") if k in item}
        known = {
            "id",
            "type",
            "weight",
            "prompt",
            "setup_prompt",
            "followup_prompt",
            "initial_prompt",
            "pressure_prompt",
            "red_flags",
            "green_flags",
            "stage",
        }
        metadata: dict[str, Any] = {
            "probe_type": item.get("type"),
            "source_format": "legacy_probe",
        }
        metadata.update({k: v for k, v in item.items() if k not in known})

        rows.append(
            DatasetRow(
                row_id=pid,
                dimension=dimension,
                stage=item.get("stage", default_stage),
                turns=turns,
                rubric_hints=hints,
                weight=float(item.get("weight", 1.0)),
                source=source,
                metadata=metadata,
            ).validate()
        )
    return rows


# ── BBQ-style CSV ────────────────────────────────────────────────────────────

_FORMAT_TO_STAGE = {"multiple choice": "mcq", "story": "open_ended"}
_PUSHBACK_HEADER = re.compile(r"^pushback\s+(\d+)\b", re.IGNORECASE)
# Optional label columns. README asks the dataset owner to add these.
_EXPECTED_COLS = ("expected answer", "correct answer", "answer")
_TARGET_COLS = ("target answer", "stereotyped answer")


def _read_csv_with_header(
    text: str, required: str, source: str
) -> list[dict[str, str]]:
    """
    Spreadsheet exports sometimes have a grouping row above the real header.
    Find the first row containing `required` and use it as the header.
    """
    reader = list(csv.reader(io.StringIO(text)))
    for idx, row in enumerate(reader):
        cells = [c.strip() for c in row]
        if required in cells:
            header = cells
            out = []
            for raw in reader[idx + 1 :]:
                if not any(c.strip() for c in raw):
                    continue
                padded = raw + [""] * (len(header) - len(raw))
                out.append({h: v for h, v in zip(header, padded) if h})
            return out
    raise DatasetError(f"{source}: header row with column {required!r} not found.")


def parse_templates_csv(text: str, source: str) -> dict[str, str]:
    """Templates tab: Template ID -> Protected class."""
    records = _read_csv_with_header(text, "Template ID", source)
    mapping = {}
    for r in records:
        tid = r.get("Template ID", "").strip()
        cls = r.get("Protected class", "").strip()
        if tid:
            if not cls:
                raise DatasetError(
                    f"{source}: template {tid!r} has no protected class."
                )
            mapping[tid] = cls
    return mapping


def _first(record: dict[str, str], names: tuple[str, ...]) -> Optional[str]:
    lowered = {k.strip().lower(): v for k, v in record.items()}
    for n in names:
        v = lowered.get(n)
        if v and v.strip():
            return v.strip()
    return None


def parse_bbq_csv(
    text: str,
    dimension: str,
    source: str,
    templates: Optional[dict[str, str]] = None,
) -> list[DatasetRow]:
    """
    One CSV per protected-class tab. Columns used:
      Prompt ID, Template ID, Format, Context, Question, Name order,
      Original prompt & stereotype from, Prompt, Follow-up, Pushback N (...),
      optional: Protected class, Expected answer, Target answer.
    Pre-collected model replies (Claude/GPT/Gemini columns) are ignored here.
    """
    records = _read_csv_with_header(text, "Prompt ID", source)
    if not records:
        return []
    headers = list(records[0].keys())
    pushback_cols = sorted(
        (int(m.group(1)), h) for h in headers if (m := _PUSHBACK_HEADER.match(h))
    )
    templates = templates or {}
    rows = []
    for r in records:
        pid = r.get("Prompt ID", "").strip()
        if not pid:
            continue
        fmt = r.get("Format", "").strip()
        stage = _FORMAT_TO_STAGE.get(fmt.lower())
        if stage is None:
            raise DatasetError(f"{source}: row {pid!r} has unknown Format {fmt!r}.")
        prompt = r.get("Prompt", "").strip()
        follow_ups = [r[h].strip() for _, h in pushback_cols if r.get(h, "").strip()]
        tid = r.get("Template ID", "").strip() or None
        protected_class = _first(r, ("protected class",)) or (
            templates.get(tid) if tid else None
        )
        follow_up_type = r.get("Follow-up", "").strip()

        rows.append(
            DatasetRow(
                row_id=pid,
                dimension=dimension,
                stage=stage,
                turns=[prompt] + follow_ups,
                protected_class=protected_class,
                template_id=tid,
                expected_answer=_first(r, _EXPECTED_COLS),
                target_answer=_first(r, _TARGET_COLS),
                source=source,
                metadata={
                    "source_format": "bbq_csv",
                    "format": fmt,
                    "context": r.get("Context", "").strip() or None,
                    "question_polarity": r.get("Question", "").strip() or None,
                    "name_order": r.get("Name order", "").strip() or None,
                    "stereotype_source": r.get(
                        "Original prompt & stereotype from", ""
                    ).strip()
                    or None,
                    "follow_up_type": (
                        None if follow_up_type in ("", "None") else follow_up_type
                    ),
                },
            ).validate()
        )
    return rows
