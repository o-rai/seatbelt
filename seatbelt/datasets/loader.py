"""
seatbelt/datasets/loader.py

Load a versioned dataset into DatasetRow objects.

Two sources:
  1. Local directory (SEATBELT_DATASET_DIR or DatasetConfig.local_dir).
     Used for development and for teammates preparing the dataset.
  2. The private GitHub repo at a pinned ref (tag or commit SHA), fetched
     file by file through a manifest.json with sha256 checksums.

Layout (both sources):
    <root>/manifest.json                (required for remote, optional locally)
    <root>/<dimension>/<files>          one folder per dimension
    <root>/<files>                      root-level canonical files must carry
                                        "dimension" in every row

Everything fails loudly: a missing file, a bad checksum, an invalid row, an
unknown dimension, or a duplicate row_id raises DatasetError. There is no
silent fallback to "fewer rows".

See README "Dataset contract" for the manifest format.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from seatbelt.datasets.adapters import (
    parse_bbq_csv,
    parse_canonical_json,
    parse_canonical_jsonl,
    parse_legacy_probes,
    parse_templates_csv,
)
from seatbelt.datasets.schema import (
    DIMENSIONS,
    DatasetError,
    DatasetRow,
    require_dimension,
)

DEFAULT_REPO = "mishi93999/seatbelt-probes-private"
FORMATS = ("jsonl", "json", "legacy_probes", "bbq_csv", "bbq_templates")

Fetcher = Callable[[str, Optional[str]], bytes]


@dataclass
class DatasetConfig:
    """Where to load the dataset from. Pinned versions only for remote loads."""

    local_dir: Optional[str] = None
    repo: str = DEFAULT_REPO
    ref: Optional[str] = None  # tag (preferred) or full commit SHA
    token_env: str = "SEATBELT_DATASET_TOKEN"
    cache_dir: Optional[str] = None
    dimensions: tuple[str, ...] = DIMENSIONS  # subset to load
    limit_per_dimension: Optional[int] = None  # for quick test runs


@dataclass
class LoadedDataset:
    rows: list[DatasetRow]
    version: str
    content_hash: str
    source: str
    files: list[str] = field(default_factory=list)

    def by_dimension(self) -> dict[str, list[DatasetRow]]:
        out: dict[str, list[DatasetRow]] = defaultdict(list)
        for row in self.rows:
            out[row.dimension].append(row)
        return dict(out)


@dataclass
class _FileEntry:
    path: str
    format: str
    dimension: Optional[str]
    sha256: Optional[str] = None


# ── Public API ───────────────────────────────────────────────────────────────


def load_dataset(
    config: Optional[DatasetConfig] = None, fetch: Optional[Fetcher] = None
) -> LoadedDataset:
    config = config or DatasetConfig()
    for d in config.dimensions:
        require_dimension(d, "DatasetConfig.dimensions")

    local_dir = config.local_dir or os.environ.get("SEATBELT_DATASET_DIR")
    if local_dir:
        return _load_local(Path(local_dir), config)
    return _load_remote(config, fetch or _github_fetch)


# ── Local ────────────────────────────────────────────────────────────────────


def _load_local(root: Path, config: DatasetConfig) -> LoadedDataset:
    if not root.is_dir():
        raise DatasetError(f"Dataset directory not found: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        version, entries = _parse_manifest(
            manifest_path.read_bytes(), str(manifest_path)
        )
    else:
        version, entries = "local-unversioned", _discover(root)

    blobs = {}
    for e in entries:
        p = root / e.path
        if not p.is_file():
            raise DatasetError(f"File listed in dataset not found: {p}")
        data = p.read_bytes()
        if e.sha256:
            _verify(e, data)
        blobs[e.path] = data
    return _build(entries, blobs, version, source=str(root), config=config)


def _discover(root: Path) -> list[_FileEntry]:
    """Infer files and formats when there is no manifest (dev mode only)."""
    entries = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith(".") or p.name == "manifest.json":
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        dimension = None
        if len(parts) > 1:
            dimension = parts[0]
            if dimension not in DIMENSIONS:
                raise DatasetError(
                    f"Folder {dimension!r} in {root} is not a dimension. "
                    f"Expected one of: {', '.join(DIMENSIONS)}."
                )
        fmt = _infer_format(p)
        if fmt is None:
            continue  # README, notes, etc.
        entries.append(_FileEntry(path=rel.as_posix(), format=fmt, dimension=dimension))
    return entries


def _infer_format(p: Path) -> Optional[str]:
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "bbq_templates" if "template" in p.stem.lower() else "bbq_csv"
    if suffix == ".json":
        try:
            items = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise DatasetError(f"Invalid JSON in {p}: {e}") from e
        if not isinstance(items, list) or not items:
            raise DatasetError(f"{p} must contain a non-empty JSON list.")
        return "json" if "row_id" in items[0] else "legacy_probes"
    return None


# ── Remote (GitHub, pinned ref) ──────────────────────────────────────────────


def _load_remote(config: DatasetConfig, fetch: Fetcher) -> LoadedDataset:
    if not config.ref:
        raise DatasetError(
            "No dataset source. Set SEATBELT_DATASET_DIR to a local copy, or set "
            "DatasetConfig.ref to a pinned tag/commit of the dataset repo."
        )
    token = os.environ.get(config.token_env)
    base = f"https://raw.githubusercontent.com/{config.repo}/{config.ref}"
    cache_root = Path(config.cache_dir or Path.home() / ".seatbelt" / "dataset_cache")
    cache_root = cache_root / config.repo.replace("/", "__") / config.ref

    manifest_bytes = fetch(f"{base}/manifest.json", token)
    version, entries = _parse_manifest(
        manifest_bytes, f"{config.repo}@{config.ref}/manifest.json"
    )
    missing_sha = [e.path for e in entries if not e.sha256]
    if missing_sha:
        raise DatasetError(f"Remote manifest entries must have sha256: {missing_sha}")

    blobs = {}
    for e in entries:
        cached = cache_root / e.path
        if cached.is_file() and _sha256(cached.read_bytes()) == e.sha256:
            blobs[e.path] = cached.read_bytes()
            continue
        data = fetch(f"{base}/{e.path}", token)
        _verify(e, data)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
        blobs[e.path] = data
    return _build(
        entries, blobs, version, source=f"{config.repo}@{config.ref}", config=config
    )


def _github_fetch(url: str, token: Optional[str]) -> bytes:
    headers = {"Authorization": f"token {token}"} if token else {}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=30
        ) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        hint = (
            " (private repo: is the token env var set?)"
            if e.code in (401, 403, 404)
            else ""
        )
        raise DatasetError(f"Failed to fetch dataset file: HTTP {e.code}{hint}") from e
    except urllib.error.URLError as e:
        raise DatasetError(f"Failed to fetch dataset file: {e.reason}") from e


# ── Shared ───────────────────────────────────────────────────────────────────


def _parse_manifest(raw: bytes, where: str) -> tuple[str, list[_FileEntry]]:
    try:
        m = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise DatasetError(f"Invalid manifest {where}: {e}") from e
    if (
        not isinstance(m, dict)
        or not m.get("version")
        or not isinstance(m.get("files"), list)
    ):
        raise DatasetError(f"Manifest {where} needs 'version' and a 'files' list.")
    entries = []
    for f in m["files"]:
        if not isinstance(f, dict) or "path" not in f or "format" not in f:
            raise DatasetError(
                f"Manifest {where}: each file needs 'path' and 'format'."
            )
        if f["format"] not in FORMATS:
            raise DatasetError(
                f"Manifest {where}: unknown format {f['format']!r} for {f['path']}."
            )
        dim = f.get("dimension")
        if dim is not None:
            require_dimension(dim, f"manifest entry {f['path']}")
        entries.append(
            _FileEntry(
                path=f["path"],
                format=f["format"],
                dimension=dim,
                sha256=f.get("sha256"),
            )
        )
    return str(m["version"]), entries


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify(entry: _FileEntry, data: bytes) -> None:
    actual = _sha256(data)
    if actual != entry.sha256:
        raise DatasetError(
            f"Checksum mismatch for {entry.path}: manifest {entry.sha256}, actual {actual}."
        )


def _build(
    entries: list[_FileEntry],
    blobs: dict[str, bytes],
    version: str,
    source: str,
    config: DatasetConfig,
) -> LoadedDataset:
    # Template lookups first, so class CSVs can resolve Template ID -> class.
    templates: dict[str, str] = {}
    for e in entries:
        if e.format == "bbq_templates":
            templates.update(parse_templates_csv(_text(blobs[e.path], e.path), e.path))

    rows: list[DatasetRow] = []
    for e in entries:
        if e.format == "bbq_templates":
            continue
        rows.extend(_parse_entry(e, blobs[e.path], templates))

    seen: dict[str, str] = {}
    for r in rows:
        if r.row_id in seen:
            raise DatasetError(
                f"Duplicate row_id {r.row_id!r} in {seen[r.row_id]} and {r.source}."
            )
        seen[r.row_id] = r.source

    wanted = set(config.dimensions)
    rows = [r for r in rows if r.dimension in wanted]
    if config.limit_per_dimension is not None:
        counts: dict[str, int] = defaultdict(int)
        limited = []
        for r in rows:
            if counts[r.dimension] < config.limit_per_dimension:
                limited.append(r)
                counts[r.dimension] += 1
        rows = limited

    content_hash = _sha256(
        "".join(f"{p}:{_sha256(blobs[p])}\n" for p in sorted(blobs)).encode("utf-8")
    )
    return LoadedDataset(
        rows=rows,
        version=version,
        content_hash=content_hash,
        source=source,
        files=sorted(blobs),
    )


def _text(data: bytes, path: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise DatasetError(f"{path} is not UTF-8: {e}") from e


def _parse_entry(
    e: _FileEntry, data: bytes, templates: dict[str, str]
) -> list[DatasetRow]:
    text = _text(data, e.path)
    if e.format == "jsonl":
        rows = parse_canonical_jsonl(text, e.path)
    elif e.format == "json":
        rows = parse_canonical_json(_json_list(text, e.path), e.path)
    elif e.format == "legacy_probes":
        rows = parse_legacy_probes(_json_list(text, e.path), _need_dim(e), e.path)
    elif e.format == "bbq_csv":
        rows = parse_bbq_csv(text, _need_dim(e), e.path, templates)
    else:  # pragma: no cover - guarded by FORMATS
        raise DatasetError(f"Unsupported format {e.format!r}")

    if e.dimension is not None:
        for r in rows:
            if r.dimension != e.dimension:
                raise DatasetError(
                    f"Row {r.row_id!r} says dimension {r.dimension!r} but lives under "
                    f"{e.dimension!r} ({e.path})."
                )
    return rows


def _need_dim(e: _FileEntry) -> str:
    if e.dimension is None:
        raise DatasetError(
            f"{e.path}: format {e.format!r} has no dimension column, so the file must sit in "
            "a dimension folder or have 'dimension' set in the manifest."
        )
    return e.dimension


def _json_list(text: str, path: str) -> list:
    try:
        items = json.loads(text)
    except json.JSONDecodeError as e:
        raise DatasetError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(items, list):
        raise DatasetError(f"{path} must contain a JSON list.")
    return items
