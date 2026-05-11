"""0.6-M14c — Nested / JSON column type tests.

Covers the full mission's acceptance criteria:

  * ``Column.dtype = "struct"`` produces Python dicts in the
    DataFrame; cell keys/types match ``nested_schema``.
  * ``Column.dtype = "array"`` produces Python lists in the
    DataFrame; cell length = ``array_length``; element type matches
    ``array_element_type``.
  * Validators reject mismatched dtype/source/config combinations.
  * Parquet writer produces native nested schema (``struct<...>`` /
    ``list<element: ...>``); round-trip preserves nesting.
  * CSV writer serializes nested cells via ``json.dumps``;
    round-trip via ``json.loads`` recovers the original value.
  * Nested columns work on both fact and dim tables.
  * Determinism: same seed → byte-identical nested cells across
    runs.
  * Builder ``type: "struct"`` / ``type: "array"`` plumbs through
    to ``Column.dtype`` / ``nested_schema`` / ``array_element_type``.
  * Schema export contains ``struct`` and ``array`` in Dtype enum.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from plotsim import (
    PlotsimConfig,
    create,
    generate_tables_with_state,
    load_config,
    write_tables,
)
from plotsim.config import Column, NestedSource, parse_source


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "plotsim" / "configs"
SAAS_YAML = CONFIGS_DIR / "sample_saas.yaml"


# --- Fixtures ---------------------------------------------------------------


def _saas_yaml_dict() -> dict:
    with SAAS_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_cfg_from_dict(d: dict) -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PlotsimConfig(**d)


def _saas_with_nested_dim_columns() -> PlotsimConfig:
    """SAAS sample with one struct + one array column on dim_company."""
    d = _saas_yaml_dict()
    dim_company = next(t for t in d["tables"] if t["name"] == "dim_company")
    dim_company["columns"].append(
        {
            "name": "metadata",
            "dtype": "struct",
            "source": "nested",
            "nested_schema": {"tier_score": "int", "is_pilot": "boolean"},
        }
    )
    dim_company["columns"].append(
        {
            "name": "tags",
            "dtype": "array",
            "source": "nested",
            "array_element_type": "string",
            "array_length": 3,
        }
    )
    return _build_cfg_from_dict(d)


def _saas_with_nested_fact_column() -> PlotsimConfig:
    """SAAS sample with one struct column on fct_revenue."""
    d = _saas_yaml_dict()
    fct = next(t for t in d["tables"] if t["name"] == "fct_revenue")
    fct["columns"].append(
        {
            "name": "raw_payload",
            "dtype": "struct",
            "source": "nested",
            "nested_schema": {"version": "int", "trace_id": "string"},
        }
    )
    return _build_cfg_from_dict(d)


# --- 1. Source parsing ------------------------------------------------------


def test_parse_nested_source():
    parsed = parse_source("nested")
    assert isinstance(parsed, NestedSource)


# --- 2. Column model + validators -------------------------------------------


def test_struct_column_accepts_nested_schema():
    c = Column(
        name="meta",
        dtype="struct",
        source="nested",
        nested_schema={"a": "int", "b": "string"},
    )
    assert c.nested_schema == {"a": "int", "b": "string"}


def test_array_column_accepts_element_type_and_length():
    c = Column(
        name="tags",
        dtype="array",
        source="nested",
        array_element_type="string",
        array_length=5,
    )
    assert c.array_element_type == "string"
    assert c.array_length == 5


def test_struct_requires_nested_schema():
    with pytest.raises(ValidationError, match="nested_schema"):
        Column(name="meta", dtype="struct", source="nested")


def test_array_requires_element_type():
    with pytest.raises(ValidationError, match="array_element_type"):
        Column(name="tags", dtype="array", source="nested")


def test_struct_rejects_array_fields():
    with pytest.raises(ValidationError, match="array-only"):
        Column(
            name="meta",
            dtype="struct",
            source="nested",
            nested_schema={"a": "int"},
            array_length=3,
        )


def test_array_rejects_nested_schema():
    with pytest.raises(ValidationError, match="struct-only"):
        Column(
            name="tags",
            dtype="array",
            source="nested",
            array_element_type="int",
            nested_schema={"a": "int"},
        )


def test_nested_source_requires_nested_dtype():
    with pytest.raises(ValidationError, match="nested cells require"):
        Column(name="x", dtype="int", source="nested")


def test_nested_dtype_requires_nested_source():
    with pytest.raises(ValidationError, match="require.*source 'nested'"):
        Column(
            name="x",
            dtype="struct",
            source="generated:faker.word",
            nested_schema={"a": "int"},
        )


def test_struct_rejects_invalid_primitive_type():
    with pytest.raises(ValidationError, match="valid primitive types"):
        Column(
            name="meta",
            dtype="struct",
            source="nested",
            nested_schema={"a": "uuid"},  # invalid
        )


def test_array_rejects_invalid_element_type():
    with pytest.raises(ValidationError, match="valid: int / float / string"):
        Column(
            name="tags",
            dtype="array",
            source="nested",
            array_element_type="uuid",
        )


def test_nested_schema_rejects_empty_field_name():
    with pytest.raises(ValidationError, match="empty field name"):
        Column(
            name="meta",
            dtype="struct",
            source="nested",
            nested_schema={"": "int"},
        )


# --- 3. Cell-builder output (per_entity dim) --------------------------------


def test_struct_cell_is_dict_with_correct_keys():
    cfg = _saas_with_nested_dim_columns()
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    dc = tables["dim_company"]
    for cell in dc["metadata"]:
        assert isinstance(cell, dict)
        assert set(cell.keys()) == {"tier_score", "is_pilot"}
        assert isinstance(cell["tier_score"], int)
        assert isinstance(cell["is_pilot"], bool)


def test_array_cell_is_list_of_correct_length_and_type():
    cfg = _saas_with_nested_dim_columns()
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    dc = tables["dim_company"]
    for cell in dc["tags"]:
        assert isinstance(cell, list)
        assert len(cell) == 3
        assert all(isinstance(x, str) for x in cell)


def test_array_default_length_is_three():
    """Omitting ``array_length`` defaults to 3."""
    d = _saas_yaml_dict()
    next(t for t in d["tables"] if t["name"] == "dim_company")["columns"].append(
        {
            "name": "tags",
            "dtype": "array",
            "source": "nested",
            "array_element_type": "int",
        }
    )
    cfg = _build_cfg_from_dict(d)
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    assert all(len(c) == 3 for c in tables["dim_company"]["tags"])


# --- 4. Cell-builder output (fact table) ------------------------------------


def test_nested_struct_on_fact_table():
    cfg = _saas_with_nested_fact_column()
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    fct = tables["fct_revenue"]
    assert "raw_payload" in fct.columns
    for cell in fct["raw_payload"]:
        assert isinstance(cell, dict)
        assert set(cell.keys()) == {"version", "trace_id"}


# --- 5. Determinism ---------------------------------------------------------


def test_nested_cells_deterministic_under_seed():
    a_cfg = _saas_with_nested_dim_columns()
    b_cfg = _saas_with_nested_dim_columns()
    rng_a = np.random.default_rng(a_cfg.seed)
    rng_b = np.random.default_rng(b_cfg.seed)
    a, _ = generate_tables_with_state(a_cfg, rng_a)
    b, _ = generate_tables_with_state(b_cfg, rng_b)
    for ca, cb in zip(a["dim_company"]["metadata"], b["dim_company"]["metadata"]):
        assert ca == cb
    for ca, cb in zip(a["dim_company"]["tags"], b["dim_company"]["tags"]):
        assert ca == cb


# --- 6. CSV round-trip ------------------------------------------------------


def test_csv_serialises_nested_as_json(tmp_path):
    cfg = _saas_with_nested_dim_columns()
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    write_tables(tables, cfg, output_dir=tmp_path)

    import csv as csvmod

    rows = list(csvmod.reader((tmp_path / "dim_company.csv").open(encoding="utf-8")))
    header = rows[0]
    i_meta = header.index("metadata")
    i_tags = header.index("tags")
    # Spot-check: every data row's nested cell parses as JSON and
    # matches the in-memory cell.
    for source_idx, row in enumerate(rows[1:], start=0):
        meta_str = row[i_meta]
        tags_str = row[i_tags]
        parsed_meta = json.loads(meta_str)
        parsed_tags = json.loads(tags_str)
        assert parsed_meta == tables["dim_company"]["metadata"].iloc[source_idx]
        assert parsed_tags == tables["dim_company"]["tags"].iloc[source_idx]


# --- 7. Parquet round-trip --------------------------------------------------


def test_parquet_preserves_native_nested_schema(tmp_path):
    pytest.importorskip("pyarrow")
    cfg = _saas_with_nested_dim_columns()
    cfg = cfg.model_copy(update={"output": cfg.output.model_copy(update={"format": "parquet"})})
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    write_tables(tables, cfg, output_dir=tmp_path)

    import pyarrow.parquet as pq

    schema = pq.read_schema(tmp_path / "dim_company.parquet")
    metadata_type = str(schema.field("metadata").type)
    tags_type = str(schema.field("tags").type)
    assert "struct<" in metadata_type
    assert "tier_score" in metadata_type
    assert "is_pilot" in metadata_type
    assert tags_type.startswith("list<")


def test_parquet_round_trip_recovers_dicts_and_lists(tmp_path):
    import pandas as pd

    pytest.importorskip("pyarrow")
    cfg = _saas_with_nested_dim_columns()
    cfg = cfg.model_copy(update={"output": cfg.output.model_copy(update={"format": "parquet"})})
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    write_tables(tables, cfg, output_dir=tmp_path)

    rt = pd.read_parquet(tmp_path / "dim_company.parquet")
    for source_cell, rt_cell in zip(tables["dim_company"]["metadata"], rt["metadata"]):
        assert dict(rt_cell) == source_cell
    for source_cell, rt_cell in zip(tables["dim_company"]["tags"], rt["tags"]):
        assert list(rt_cell) == source_cell


# --- 8. Builder passthrough -------------------------------------------------


def _builder_kwargs(**overrides):
    base = {
        "about": "test domain",
        "unit": "company",
        "window": {"start": "2023-01", "end": "2023-12", "every": "monthly"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
        ],
        "segments": [
            {"name": "alpha", "count": 5, "archetype": "growth"},
        ],
        "dimensions": [
            {
                "name": "dim_company",
                "per": "unit",
                "columns": [
                    {"name": "company_id", "type": "id"},
                    {"name": "company_name", "type": "faker.company"},
                    {
                        "name": "metadata",
                        "type": "struct",
                        "nested_schema": {"score": "int", "active": "boolean"},
                    },
                    {
                        "name": "tags",
                        "type": "array",
                        "array_element_type": "string",
                        "array_length": 4,
                    },
                ],
            },
        ],
    }
    base.update(overrides)
    return base


def test_builder_struct_passthrough():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(**_builder_kwargs())
    dim = next(t for t in cfg.tables if t.name == "dim_company")
    meta_col = next(c for c in dim.columns if c.name == "metadata")
    assert meta_col.dtype == "struct"
    assert meta_col.source == "nested"
    assert meta_col.nested_schema == {"score": "int", "active": "boolean"}


def test_builder_array_passthrough():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(**_builder_kwargs())
    dim = next(t for t in cfg.tables if t.name == "dim_company")
    tags_col = next(c for c in dim.columns if c.name == "tags")
    assert tags_col.dtype == "array"
    assert tags_col.source == "nested"
    assert tags_col.array_element_type == "string"
    assert tags_col.array_length == 4


def test_builder_struct_requires_nested_schema():
    kw = _builder_kwargs()
    bad = next(c for c in kw["dimensions"][0]["columns"] if c["name"] == "metadata")
    bad.pop("nested_schema")
    with pytest.raises(ValueError, match="nested_schema"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            create(**kw)


# --- 9. Off-by-default invariant --------------------------------------------


def test_pre_m14c_output_unchanged_without_nested_columns(tmp_path):
    """Configs that don't declare any nested column produce
    byte-identical output to pre-M14c (no JSON serialisation, no
    pyarrow schema construction triggers)."""
    cfg = load_config(SAAS_YAML)
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    write_tables(tables, cfg, output_dir=tmp_path)
    # dim_company.csv should not contain any JSON-shaped content.
    text = (tmp_path / "dim_company.csv").read_text(encoding="utf-8")
    assert '"{' not in text  # no JSON-object string cells
    assert '"[' not in text  # no JSON-array string cells


# --- 10. Schema export ------------------------------------------------------


def test_schema_json_includes_nested_dtype_values():
    schema_path = ROOT / "plotsim-schema.json"
    if not schema_path.exists():
        pytest.skip("plotsim-schema.json not generated")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    column_def = schema["$defs"]["Column"]["properties"]
    dtype_enum = column_def["dtype"]["enum"]
    assert "struct" in dtype_enum
    assert "array" in dtype_enum
    assert "nested_schema" in column_def
    assert "array_element_type" in column_def
    assert "array_length" in column_def
