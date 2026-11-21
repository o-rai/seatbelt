"""
Tests for the dataset loader. Uses synthetic fixtures in tests/fixtures/dataset
and a fake fetcher for the remote path. No network, no private data.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from seatbelt.datasets import (
    DIMENSIONS,
    DatasetConfig,
    DatasetError,
    DatasetRow,
    UnknownDimensionError,
    load_dataset,
    require_dimension,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dataset"


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("SEATBELT_DATASET_DIR", raising=False)
    monkeypatch.delenv("SEATBELT_DATASET_TOKEN", raising=False)


def _copy_fixtures(tmp_path: Path) -> Path:
    dst = tmp_path / "dataset"
    shutil.copytree(FIXTURES, dst)
    return dst


def _write_manifest(root: Path, version: str = "v-test.1") -> dict:
    files = []
    fmt = {
        "sociotech/private_probes.json": "legacy_probes",
        "sociotech/templates.csv": "bbq_templates",
        "sociotech/fx_sex.csv": "bbq_csv",
        "fairness/rows.jsonl": "jsonl",
    }
    for path, f in fmt.items():
        sha = hashlib.sha256((root / path).read_bytes()).hexdigest()
        files.append(
            {"path": path, "format": f, "dimension": path.split("/")[0], "sha256": sha}
        )
    manifest = {"version": version, "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


# ── Local discovery (no manifest) ────────────────────────────────────────────


class TestLocalDiscovery:
    def test_loads_all_formats(self):
        ds = load_dataset(DatasetConfig(local_dir=str(FIXTURES)))
        by_dim = ds.by_dimension()
        assert set(by_dim) == {"sociotech", "fairness"}
        assert len(by_dim["sociotech"]) == 2 + 3  # 2 legacy probes + 3 CSV rows
        assert ds.version == "local-unversioned"
        assert len(ds.content_hash) == 64

    def test_env_var_selects_local_dir(self, monkeypatch):
        monkeypatch.setenv("SEATBELT_DATASET_DIR", str(FIXTURES))
        ds = load_dataset(DatasetConfig())
        assert ds.source == str(FIXTURES)

    def test_bbq_csv_row_mapping(self):
        rows = {
            r.row_id: r
            for r in load_dataset(DatasetConfig(local_dir=str(FIXTURES))).rows
        }
        mcq = rows["FX-01-orig-amb-nonneg"]
        assert mcq.stage == "mcq"
        assert mcq.protected_class == "Sex"  # resolved through templates.csv
        assert mcq.template_id == "FX-01"
        assert mcq.turns[0].startswith("A man and a female pilot")
        assert mcq.turns[1:] == [
            "I think it's A (the man).",
            "I've seen this many times. It's A.",
        ]
        assert mcq.metadata["context"] == "Ambiguous"
        assert mcq.metadata["follow_up_type"] == 'Pushback: "I think"'

        no_followup = rows["FX-01-orig-amb-neg"]
        assert len(no_followup.turns) == 1
        assert no_followup.metadata["follow_up_type"] is None

        story = rows["FX-01-orig-story"]
        assert story.stage == "open_ended"

    def test_legacy_probe_mapping(self):
        rows = {
            r.row_id: r
            for r in load_dataset(DatasetConfig(local_dir=str(FIXTURES))).rows
        }
        p = rows["ab_fx_001"]
        assert p.dimension == "sociotech"
        assert p.stage == "open_ended"
        assert p.weight == 1.2
        assert p.rubric_hints["red_flags"] == ["follow the ai", "the ai knows best"]
        assert p.metadata["probe_type"] == "automation_bias"

    def test_dimension_subset_and_limit(self):
        ds = load_dataset(
            DatasetConfig(
                local_dir=str(FIXTURES),
                dimensions=("sociotech",),
                limit_per_dimension=2,
            )
        )
        assert [r.dimension for r in ds.rows] == ["sociotech", "sociotech"]

    def test_non_dimension_folder_fails(self, tmp_path):
        root = _copy_fixtures(tmp_path)
        (root / "socio").mkdir()
        (root / "socio" / "x.jsonl").write_text("")
        with pytest.raises(DatasetError, match="not a dimension"):
            load_dataset(DatasetConfig(local_dir=str(root)))

    def test_duplicate_row_id_fails(self, tmp_path):
        root = _copy_fixtures(tmp_path)
        line = (root / "fairness" / "rows.jsonl").read_text()
        (root / "fairness" / "more.jsonl").write_text(line)
        with pytest.raises(DatasetError, match="Duplicate row_id"):
            load_dataset(DatasetConfig(local_dir=str(root)))

    def test_row_in_wrong_folder_fails(self, tmp_path):
        root = _copy_fixtures(tmp_path)
        row = json.loads((root / "fairness" / "rows.jsonl").read_text())
        row["row_id"] = "moved"
        (root / "sociotech" / "moved.jsonl").write_text(json.dumps(row))
        with pytest.raises(DatasetError, match="lives under 'sociotech'"):
            load_dataset(DatasetConfig(local_dir=str(root)))

    def test_paired_legacy_probe_fails_loudly(self, tmp_path):
        root = _copy_fixtures(tmp_path)
        (root / "fairness" / "pairs.json").write_text(
            json.dumps([{"id": "cf_x", "prompt_a": "a", "prompt_b": "b"}])
        )
        with pytest.raises(DatasetError, match="paired probe"):
            load_dataset(DatasetConfig(local_dir=str(root)))

    def test_missing_dir_fails(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(DatasetConfig(local_dir=str(tmp_path / "nope")))


# ── Manifest ─────────────────────────────────────────────────────────────────


class TestManifest:
    def test_local_manifest_sets_version(self, tmp_path):
        root = _copy_fixtures(tmp_path)
        _write_manifest(root, version="v2026.11.0")
        ds = load_dataset(DatasetConfig(local_dir=str(root)))
        assert ds.version == "v2026.11.0"
        assert len(ds.rows) == 6

    def test_checksum_mismatch_fails(self, tmp_path):
        root = _copy_fixtures(tmp_path)
        _write_manifest(root)
        (root / "fairness" / "rows.jsonl").write_text(
            (root / "fairness" / "rows.jsonl").read_text().replace("Sam", "Alex")
        )
        with pytest.raises(DatasetError, match="Checksum mismatch"):
            load_dataset(DatasetConfig(local_dir=str(root)))

    def test_unknown_dimension_in_manifest_fails(self, tmp_path):
        root = _copy_fixtures(tmp_path)
        m = _write_manifest(root)
        m["files"][0]["dimension"] = "socio"
        (root / "manifest.json").write_text(json.dumps(m))
        with pytest.raises(UnknownDimensionError):
            load_dataset(DatasetConfig(local_dir=str(root)))


# ── Remote (fake fetcher, no network) ────────────────────────────────────────


class TestRemote:
    def test_requires_pinned_ref(self):
        with pytest.raises(DatasetError, match="pinned tag"):
            load_dataset(DatasetConfig())

    def test_fetches_by_manifest_and_caches(self, tmp_path, monkeypatch):
        root = _copy_fixtures(tmp_path)
        _write_manifest(root)
        monkeypatch.setenv("SEATBELT_DATASET_TOKEN", "fake-token")
        calls = []

        def fake_fetch(url, token):
            calls.append(url)
            assert token == "fake-token"
            prefix = "https://raw.githubusercontent.com/org/data/v-test.1/"
            assert url.startswith(prefix)
            return (root / url[len(prefix) :]).read_bytes()

        cfg = DatasetConfig(
            repo="org/data", ref="v-test.1", cache_dir=str(tmp_path / "cache")
        )
        ds = load_dataset(cfg, fetch=fake_fetch)
        assert ds.version == "v-test.1"
        assert ds.source == "org/data@v-test.1"
        assert len(ds.rows) == 6
        first_round = len(calls)

        load_dataset(cfg, fetch=fake_fetch)
        assert len(calls) - first_round == 1  # only the manifest is re-fetched

    def test_remote_manifest_requires_sha(self, tmp_path):
        manifest = {
            "version": "v1",
            "files": [{"path": "fairness/rows.jsonl", "format": "jsonl"}],
        }

        def fake_fetch(url, token):
            return json.dumps(manifest).encode()

        with pytest.raises(DatasetError, match="sha256"):
            load_dataset(
                DatasetConfig(ref="v1", cache_dir=str(tmp_path)), fetch=fake_fetch
            )


# ── Row schema ───────────────────────────────────────────────────────────────


class TestRowSchema:
    def _row(self, **kw):
        base = {
            "row_id": "r1",
            "dimension": "privacy",
            "stage": "open_ended",
            "turns": ["hi"],
        }
        base.update(kw)
        return base

    def test_valid_row(self):
        r = DatasetRow.from_dict(self._row())
        assert r.dimension == "privacy"

    @pytest.mark.parametrize("dim", ["socio", "Fairness", "", None])
    def test_unknown_dimension(self, dim):
        with pytest.raises(UnknownDimensionError):
            DatasetRow.from_dict(self._row(dimension=dim))

    def test_unknown_stage(self):
        with pytest.raises(DatasetError, match="stage"):
            DatasetRow.from_dict(self._row(stage="essay"))

    def test_unknown_field_rejected(self):
        with pytest.raises(DatasetError, match="Unknown field"):
            DatasetRow.from_dict(self._row(dimention="privacy"))

    def test_empty_turns_rejected(self):
        with pytest.raises(DatasetError, match="turns"):
            DatasetRow.from_dict(self._row(turns=[]))

    def test_require_dimension_accepts_all_six(self):
        assert [require_dimension(d) for d in DIMENSIONS] == list(DIMENSIONS)


# ── Opt-in check against a real local copy of the dataset ────────────────────


@pytest.mark.skipif(
    not os.environ.get("SEATBELT_REAL_DATASET_DIR"),
    reason="set SEATBELT_REAL_DATASET_DIR to validate a real dataset checkout",
)
def test_real_dataset_validates():
    ds = load_dataset(DatasetConfig(local_dir=os.environ["SEATBELT_REAL_DATASET_DIR"]))
    assert ds.rows, "dataset loaded but contained no rows"
