"""Tests for 0.6-M16b JSONL output format.

When ``output.format == "jsonl"`` is set on a config, every table is
written as ``<table_name>.jsonl`` with one JSON object per line via
``DataFrame.to_json(orient='records', lines=True, date_format='iso',
force_ascii=False)``. Nested struct / array cells serialise as native
JSON; NaN / None becomes ``null``; date columns emit as ISO-8601 strings
rather than pandas' default epoch-ms milliseconds. Denormalized wide
sidecars, holdout splits, and the entity-features file all follow the
same encoding so a run never produces mixed-encoding output.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotsim.builder import create, create_from_yaml
from plotsim.config import OutputConfig
from plotsim.output import (
    _resolve_output_format,
    _write_jsonl,
    write_single_table,
    write_tables,
)
from plotsim.tables import generate_tables


ROOT = Path(__file__).resolve().parent.parent


# --- Helpers ---------------------------------------------------------------


def _saas_jsonl_config(tmp_path: Path):
    """Load the saas template and switch it to jsonl output."""
    cfg = create_from_yaml(ROOT / "tests" / "configs" / "saas_template.yaml")
    return cfg.model_copy(
        update={
            "output": cfg.output.model_copy(update={"format": "jsonl", "directory": str(tmp_path)}),
        }
    )


def _tables_for(cfg) -> dict[str, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_tables(cfg, np.random.default_rng(cfg.seed))


def _read_jsonl(path: Path) -> list[dict]:
    """Return parsed rows from a JSONL file (one dict per line)."""
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _builder_kwargs(**overrides):
    """Minimal ``create()`` kwargs for builder-passthrough tests."""
    base = {
        "about": "jsonl test domain",
        "unit": "company",
        "window": {"start": "2023-01", "end": "2023-06", "every": "monthly"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
        ],
        "segments": [
            {"name": "alpha", "count": 4, "archetype": "growth"},
        ],
    }
    base.update(overrides)
    return base


# --- Directory structure ---------------------------------------------------


class TestDirectoryStructure:
    """JSONL output emits one ``<table>.jsonl`` file per generated table
    under the output directory."""

    def test_every_table_produces_jsonl_file(self, tmp_path):
        cfg = _saas_jsonl_config(tmp_path)
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        for tbl in cfg.tables:
            path = tmp_path / f"{tbl.name}.jsonl"
            assert path.is_file(), f"{tbl.name}: expected .jsonl file"
            # The CSV / Parquet variants should NOT also be written.
            assert not (tmp_path / f"{tbl.name}.csv").exists()
            assert not (tmp_path / f"{tbl.name}.parquet").exists()


# --- Per-line validity + row count -----------------------------------------


class TestPerLineValidity:
    """Each line in a generated .jsonl file is valid standalone JSON and
    the row count matches the source DataFrame."""

    def test_every_line_parses_as_json(self, tmp_path):
        cfg = _saas_jsonl_config(tmp_path)
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        for tbl in cfg.tables:
            path = tmp_path / f"{tbl.name}.jsonl"
            rows = _read_jsonl(path)
            # Every non-empty line parsed successfully (else _read_jsonl
            # would have raised JSONDecodeError); each record is a JSON
            # object.
            assert len(rows) > 0, f"{tbl.name}: expected at least one row"
            assert all(
                isinstance(r, dict) for r in rows
            ), f"{tbl.name}: every record should parse as a JSON object"

    def test_row_count_matches_csv_baseline(self, tmp_path):
        """JSONL row count per table must equal the CSV row count for
        the same config — both formats consume the same post-CDC /
        post-quality ``tables_to_write`` dict, so any drift is a writer
        bug. Compares CSV minus header against JSONL line count."""
        cfg = create_from_yaml(ROOT / "tests" / "configs" / "saas_template.yaml")
        csv_dir = tmp_path / "csv"
        jsonl_dir = tmp_path / "jsonl"
        csv_dir.mkdir()
        jsonl_dir.mkdir()
        cfg_csv = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(
                    update={"format": "csv", "directory": str(csv_dir)}
                ),
            }
        )
        cfg_jsonl = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(
                    update={"format": "jsonl", "directory": str(jsonl_dir)}
                ),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg_csv, np.random.default_rng(cfg_csv.seed))
            write_tables(tables, cfg_csv, output_dir=csv_dir)
            tables_j = generate_tables(cfg_jsonl, np.random.default_rng(cfg_jsonl.seed))
            write_tables(tables_j, cfg_jsonl, output_dir=jsonl_dir)

        for tbl in cfg.tables:
            csv_lines = (csv_dir / f"{tbl.name}.csv").read_text(encoding="utf-8").splitlines()
            jsonl_lines = [
                line
                for line in (jsonl_dir / f"{tbl.name}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            # CSV's first line is the header; subtract it.
            assert (
                len(jsonl_lines) == len(csv_lines) - 1
            ), f"{tbl.name}: JSONL rows {len(jsonl_lines)} != CSV rows {len(csv_lines) - 1}"

    def test_column_key_order_matches_config(self, tmp_path):
        """``write_single_table`` reorders DataFrame columns by config
        order (PK → FK → others). JSONL preserves DataFrame column order
        in each record, so the JSON key sequence per line should match
        the config order."""
        cfg = _saas_jsonl_config(tmp_path)
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        fact_name = "fct_revenue"
        path = tmp_path / f"{fact_name}.jsonl"
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        # Re-parse preserving key order — python 3.7+ dicts already do.
        parsed = json.loads(first_line)
        observed_keys = list(parsed.keys())
        # PK / FK columns from the config should appear before
        # non-key columns. ``date_key`` (FK to dim_date) and
        # ``company_id`` (FK to dim_company) come first on
        # ``fct_revenue`` in the saas template.
        assert observed_keys[0] == "date_key"
        assert "company_id" in observed_keys[:3]


# --- NaN / null handling ---------------------------------------------------


class TestNullHandling:
    """NaN / pd.NA / None values serialise as JSON ``null`` rather than
    NaN / 'NaN' / empty string."""

    def test_nan_serialised_as_null(self, tmp_path):
        cfg = _saas_jsonl_config(tmp_path)
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [1.0, float("nan"), 3.5],
                "label": ["a", None, "c"],
            }
        )
        path = write_single_table("ad_hoc", df, tmp_path, config=cfg)
        assert path == tmp_path / "ad_hoc.jsonl"
        rows = _read_jsonl(path)
        assert rows[0]["value"] == 1.0
        assert rows[1]["value"] is None, "NaN float should serialise as null"
        assert rows[1]["label"] is None, "Python None should serialise as null"
        assert rows[2]["label"] == "c"

    def test_pd_na_serialised_as_null(self, tmp_path):
        cfg = _saas_jsonl_config(tmp_path)
        df = pd.DataFrame(
            {
                "id": pd.array([1, pd.NA, 3], dtype="Int64"),
                "label": ["a", "b", "c"],
            }
        )
        path = write_single_table("with_na", df, tmp_path, config=cfg)
        rows = _read_jsonl(path)
        assert rows[0]["id"] == 1
        assert rows[1]["id"] is None, "pd.NA in Int64 should serialise as null"
        assert rows[2]["id"] == 3


# --- Date format -----------------------------------------------------------


class TestDateFormat:
    """``date_format='iso'`` is pinned in ``_write_jsonl`` so date /
    datetime columns emit as ISO-8601 strings rather than pandas'
    default epoch-ms for ``orient='records'``."""

    def test_datetime_emits_iso_string(self, tmp_path):
        cfg = _saas_jsonl_config(tmp_path)
        df = pd.DataFrame(
            {
                "event_ts": pd.to_datetime(["2024-01-15", "2024-06-30"]),
                "value": [10, 20],
            }
        )
        path = write_single_table("with_dates", df, tmp_path, config=cfg)
        rows = _read_jsonl(path)
        ts0 = rows[0]["event_ts"]
        assert isinstance(ts0, str), f"expected ISO string, got {type(ts0)}"
        assert ts0.startswith("2024-01-15"), ts0
        # No epoch milliseconds leak through (they would render as a
        # large integer like 1705276800000).
        assert not isinstance(ts0, int)


# --- Unicode preservation --------------------------------------------------


class TestUnicode:
    """``force_ascii=False`` is pinned so non-ASCII characters land
    verbatim in the file rather than as ``\\uXXXX`` escapes."""

    def test_non_ascii_strings_round_trip(self, tmp_path):
        df = pd.DataFrame({"name": ["Iñárritu", "東京", "São Paulo"]})
        _write_jsonl(df, tmp_path / "unicode.jsonl")
        raw = (tmp_path / "unicode.jsonl").read_text(encoding="utf-8")
        # No escape sequences in the on-disk bytes.
        assert "\\u" not in raw
        assert "Iñárritu" in raw
        assert "東京" in raw
        # Round-trip via json.loads still recovers the strings.
        rows = _read_jsonl(tmp_path / "unicode.jsonl")
        assert [r["name"] for r in rows] == ["Iñárritu", "東京", "São Paulo"]


# --- Nested struct / array columns -----------------------------------------


class TestNestedColumns:
    """0.6-M14c nested ``struct`` / ``array`` columns serialise as
    native JSON objects / arrays in JSONL — no JSON-string wrapping
    (which is what the CSV writer has to do because flat-string cells
    can't carry nested types natively)."""

    def test_struct_column_serialises_as_object(self, tmp_path):
        cfg = create_from_yaml(ROOT / "tests" / "configs" / "retail_template.yaml")
        cfg_j = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(
                    update={"format": "jsonl", "directory": str(tmp_path)}
                ),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg_j, np.random.default_rng(cfg_j.seed))
            write_tables(tables, cfg_j, output_dir=tmp_path)

        path = tmp_path / "dim_product_category.jsonl"
        assert path.is_file()
        rows = _read_jsonl(path)
        # The retail template's catalog_metadata struct field carries
        # aisle / seasonality / avg_basket_position.
        sample_struct = None
        for r in rows:
            cell = r.get("catalog_metadata")
            if isinstance(cell, dict):
                sample_struct = cell
                break
        assert sample_struct is not None, (
            "expected at least one row with a dict catalog_metadata cell — "
            "JSONL must preserve struct cells as native JSON objects"
        )
        assert {"aisle", "seasonality", "avg_basket_position"} <= set(
            sample_struct.keys()
        ), sample_struct


# --- Sidecar behaviour -----------------------------------------------------


class TestSidecars:
    """Denormalized wide tables, holdout splits, and the entity-features
    file all emit as ``.jsonl`` when ``output.format == "jsonl"`` so the
    run produces no mixed-encoding files. Companions (``config.yaml``,
    ``validation_report.txt``) stay in their canonical text form."""

    def test_denormalized_wide_emits_jsonl(self, tmp_path):
        cfg = _saas_jsonl_config(tmp_path)
        cfg_d = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"denormalized": True}),
            }
        )
        tables = _tables_for(cfg_d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg_d, output_dir=tmp_path)

        for fact in ("fct_engagement", "fct_revenue", "fct_support_tickets"):
            wide_path = tmp_path / f"{fact}_wide.jsonl"
            assert wide_path.is_file(), f"{fact}_wide.jsonl should exist"
            assert not (tmp_path / f"{fact}_wide.csv").exists()
            assert not (tmp_path / f"{fact}_wide.parquet").exists()
            # Wide row count should match the underlying fact (left join).
            wide_rows = _read_jsonl(wide_path)
            assert len(wide_rows) == len(tables[fact])

    def test_companions_stay_in_canonical_format(self, tmp_path):
        """``config.yaml`` and ``validation_report.txt`` are not table
        data; their format is fixed regardless of ``output.format``."""
        cfg = _saas_jsonl_config(tmp_path)
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        assert (tmp_path / "config.yaml").is_file()
        assert (tmp_path / "validation_report.txt").is_file()
        assert not (tmp_path / "config.jsonl").exists()
        assert not (tmp_path / "validation_report.jsonl").exists()


# --- Format Literal validation --------------------------------------------


class TestFormatLiteral:
    """The ``OutputConfig.format`` Literal accepts jsonl alongside csv
    and parquet, and rejects unknown words."""

    def test_jsonl_accepted_by_output_config(self):
        oc = OutputConfig(format="jsonl", directory="out")
        assert oc.format == "jsonl"

    def test_unknown_format_rejected(self):
        with pytest.raises(Exception, match="format"):
            OutputConfig(format="ndjson", directory="out")

    def test_resolve_output_format_returns_jsonl(self):
        cfg = create_from_yaml(ROOT / "tests" / "configs" / "saas_template.yaml")
        cfg_j = cfg.model_copy(
            update={"output": cfg.output.model_copy(update={"format": "jsonl"})},
        )
        assert _resolve_output_format(cfg_j) == "jsonl"


# --- Builder passthrough ---------------------------------------------------


class TestBuilderPassthrough:
    """``create(output="jsonl")`` shorthand and the dict form both
    resolve to ``PlotsimConfig.output.format == "jsonl"``."""

    def test_builder_shorthand_jsonl(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = create(**_builder_kwargs(output="jsonl"))
        assert cfg.output.format == "jsonl"
        assert cfg.output.directory == "output"

    def test_builder_dict_form_jsonl(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = create(
                **_builder_kwargs(output={"format": "jsonl", "directory": "out_dir"}),
            )
        assert cfg.output.format == "jsonl"
        assert cfg.output.directory == "out_dir"

    def test_builder_rejects_unknown_shorthand(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(Exception, match="unknown output format"):
                create(**_builder_kwargs(output="ndjson"))


# --- CSV regression guard --------------------------------------------------


class TestCsvUnchanged:
    """Adding the JSONL branch must not perturb CSV output — a config
    with ``format: csv`` produces byte-identical files to a baseline
    config with the field omitted, run after run."""

    def test_csv_output_byte_identical(self, tmp_path):
        cfg = create_from_yaml(ROOT / "tests" / "configs" / "saas_template.yaml")
        a = tmp_path / "run_a"
        b = tmp_path / "run_b"
        a.mkdir()
        b.mkdir()
        cfg_a = cfg.model_copy(
            update={"output": cfg.output.model_copy(update={"directory": str(a)})},
        )
        cfg_b = cfg.model_copy(
            update={
                "output": OutputConfig(format="csv", directory=str(b)),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables_a = _tables_for(cfg_a)
            tables_b = _tables_for(cfg_b)
            write_tables(tables_a, cfg_a, output_dir=a)
            write_tables(tables_b, cfg_b, output_dir=b)

        for tbl in cfg.tables:
            ba = (a / f"{tbl.name}.csv").read_bytes()
            bb = (b / f"{tbl.name}.csv").read_bytes()
            assert ba == bb, f"{tbl.name}.csv differs between runs"
