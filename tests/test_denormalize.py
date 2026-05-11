"""0.6-M14a — Denormalization mode tests.

Covers the full mission's acceptance criteria:

  * ``OutputConfig.denormalized: false`` (default) — output byte-
    identical to pre-M14a (no ``*_wide`` files written).
  * ``OutputConfig.denormalized: true`` — every fact table emits
    ``<fct_name>_wide.{csv|parquet}`` alongside the normalized
    output. Normalized files still emitted untouched.
  * Wide table contains all dim columns joined; column-name
    prefixes (``<dim>__<col>``) prevent collisions when two dims
    share a column name.
  * SCD2 dims contribute current-state rows only (``is_current``
    filter); SCD2 audit columns excluded from the wide output.
  * dim_date columns are joined like any other dim.
  * Multi-fact: one wide frame per fact, no cross-fact joins.
  * Builder ``output={'denormalized': True}`` plumbs through.
  * Determinism: same config + same seed → byte-identical wide
    files across runs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from plotsim import (
    PlotsimConfig,
    SurrogateKeyWarning,
    create,
    generate_tables_with_state,
    load_config,
    write_tables,
)
from plotsim.config import OutputConfig
from plotsim.denormalize import denormalize_fact_tables


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "plotsim" / "configs"
SAAS_YAML = CONFIGS_DIR / "sample_saas.yaml"


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def saas_cfg() -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return load_config(SAAS_YAML)


@pytest.fixture
def saas_run(saas_cfg):
    rng = np.random.default_rng(saas_cfg.seed)
    tables, _state = generate_tables_with_state(saas_cfg, rng)
    return saas_cfg, tables


def _saas_yaml_dict() -> dict:
    with SAAS_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_cfg_from_dict(d: dict) -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(**d)


# --- 1. OutputConfig field --------------------------------------------------


def test_output_config_denormalized_default_false():
    cfg = OutputConfig(format="csv", directory="out")
    assert cfg.denormalized is False


def test_output_config_denormalized_round_trip_yaml():
    d = _saas_yaml_dict()
    d["output"] = {**d["output"], "denormalized": True}
    cfg = _build_cfg_from_dict(d)
    assert cfg.output.denormalized is True


# --- 2. Pure denormalize function -------------------------------------------


def test_denormalize_returns_one_wide_per_fact(saas_run):
    cfg, tables = saas_run
    wides = denormalize_fact_tables(tables, cfg)
    fact_names = {t.name for t in cfg.tables if t.type == "fact"}
    assert set(wides.keys()) == {f"{n}_wide" for n in fact_names}


def test_denormalize_skips_dim_event_bridge(saas_run):
    cfg, tables = saas_run
    wides = denormalize_fact_tables(tables, cfg)
    for name in wides:
        # Only fact tables get a wide companion.
        base = name.removesuffix("_wide")
        tbl = next(t for t in cfg.tables if t.name == base)
        assert tbl.type == "fact"


def test_denormalize_empty_when_no_facts():
    # PlotsimConfig requires ≥1 fact, so build a tables dict that
    # excludes the facts and assert the function no-ops on the
    # missing entries (defensive: partial dicts).
    cfg = load_config(SAAS_YAML)
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    dim_only = {n: df for n, df in tables.items() if not n.startswith("fct_")}
    wides = denormalize_fact_tables(dim_only, cfg)
    assert wides == {}


# --- 3. Wide-frame shape ----------------------------------------------------


def test_wide_preserves_fact_row_count(saas_run):
    cfg, tables = saas_run
    wides = denormalize_fact_tables(tables, cfg)
    for fct_name in (t.name for t in cfg.tables if t.type == "fact"):
        wide = wides[f"{fct_name}_wide"]
        assert len(wide) == len(tables[fct_name]), (
            f"{fct_name}_wide row count drift: left-join must not "
            f"add or drop rows; got {len(wide)} vs fact {len(tables[fct_name])}"
        )


def test_wide_includes_all_fact_columns(saas_run):
    cfg, tables = saas_run
    wides = denormalize_fact_tables(tables, cfg)
    for fct_name in (t.name for t in cfg.tables if t.type == "fact"):
        wide = wides[f"{fct_name}_wide"]
        for col in tables[fct_name].columns:
            assert col in wide.columns, f"{fct_name}_wide missing fact col {col}"


def test_wide_includes_dim_columns_with_prefix(saas_run):
    cfg, tables = saas_run
    wides = denormalize_fact_tables(tables, cfg)
    # fct_revenue FKs to dim_company and dim_plan and dim_date.
    wide = wides["fct_revenue_wide"]
    assert "dim_company__company_name" in wide.columns
    assert "dim_plan__plan_name" in wide.columns
    # dim_date join brings period_label / year / quarter / month.
    assert "dim_date__period_label" in wide.columns


def test_wide_has_no_unprefixed_dim_columns(saas_run):
    cfg, tables = saas_run
    wides = denormalize_fact_tables(tables, cfg)
    wide = wides["fct_revenue_wide"]
    fact_cols = set(tables["fct_revenue"].columns)
    # Every column not in the original fact must carry a dim prefix.
    for col in wide.columns:
        if col in fact_cols:
            continue
        assert "__" in col, (
            f"unprefixed non-fact column {col!r} in fct_revenue_wide; "
            f"every dim-side column must be prefixed <dim>__<col>"
        )


def test_wide_drops_dim_join_key_no_duplication(saas_run):
    """The dim's join key is the same column as the fact's FK — must
    not appear twice (once unprefixed, once prefixed) in the wide
    output."""
    cfg, tables = saas_run
    wide = denormalize_fact_tables(tables, cfg)["fct_revenue_wide"]
    # company_id is the fact FK; dim_company__company_id should NOT exist.
    assert "company_id" in wide.columns
    assert "dim_company__company_id" not in wide.columns


# --- 4. SCD2 current-state filter -------------------------------------------


def test_scd2_dim_current_state_only(saas_run):
    """The saas template has SCD2 plan_tier on dim_company. Without
    the current-state filter, joining would multiply fact rows by
    the number of versions per company. Assert the wide table has
    one row per fact row (no duplication)."""
    cfg, tables = saas_run
    # Sanity: dim_company is SCD2 (multiple rows per company_id).
    company_versions = tables["dim_company"].groupby("company_id").size().max()
    assert company_versions > 1, (
        "test fixture broken: saas template should have multi-version "
        "SCD2 dim_company; no SCD2 → no test signal"
    )
    wides = denormalize_fact_tables(tables, cfg)
    wide = wides["fct_revenue_wide"]
    # If the join had multiplied rows, wide would have ≥
    # len(fact) * company_versions rows.
    assert len(wide) == len(tables["fct_revenue"])


def test_scd2_audit_columns_excluded_from_wide(saas_run):
    cfg, tables = saas_run
    wide = denormalize_fact_tables(tables, cfg)["fct_revenue_wide"]
    for audit in ("dim_row_id", "valid_from", "valid_to", "is_current"):
        prefixed = f"dim_company__{audit}"
        assert prefixed not in wide.columns, (
            f"SCD2 audit column {prefixed!r} leaked into the wide "
            f"output; current-state-only view should drop these"
        )
        # Also confirm unprefixed forms aren't present (the fact may
        # already carry an additive ``dim_row_id`` from
        # ``attach_dim_row_id_to_facts`` — that's a fact column, not
        # a dim leak — so we only assert the dim-prefixed form here.)


def test_scd2_current_label_carried_into_wide(saas_run):
    """The current SCD2 label (`plan_tier`) must surface as a
    `dim_company__plan_tier` column in the wide output, with the
    value matching the dim's current row for each company."""
    cfg, tables = saas_run
    wide = denormalize_fact_tables(tables, cfg)["fct_revenue_wide"]
    assert "dim_company__plan_tier" in wide.columns

    current_dim = tables["dim_company"][tables["dim_company"]["is_current"]]
    expected_by_company = dict(zip(current_dim["company_id"], current_dim["plan_tier"]))
    # Sample a few rows: each fact row's plan_tier must match the
    # company's current label (regardless of the row's date_key).
    sample = wide.head(50)
    for _, row in sample.iterrows():
        assert row["dim_company__plan_tier"] == expected_by_company[row["company_id"]]


# --- 5. write_tables wiring -------------------------------------------------


def test_write_tables_off_produces_no_wide_files(tmp_path, saas_cfg):
    rng = np.random.default_rng(saas_cfg.seed)
    tables, _ = generate_tables_with_state(saas_cfg, rng)
    # default: denormalized = False
    write_tables(tables, saas_cfg, output_dir=tmp_path)
    wide_files = list(tmp_path.glob("*_wide.csv"))
    assert wide_files == [], f"denormalized=False must emit zero *_wide files; found {wide_files}"


def test_write_tables_on_produces_wide_csv_per_fact(tmp_path, saas_cfg):
    cfg = saas_cfg.model_copy(
        update={
            "output": OutputConfig(
                format="csv",
                directory=str(tmp_path),
                denormalized=True,
            )
        }
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    write_tables(tables, cfg, output_dir=tmp_path)
    fact_names = [t.name for t in cfg.tables if t.type == "fact"]
    for fct in fact_names:
        wide_path = tmp_path / f"{fct}_wide.csv"
        assert wide_path.exists(), f"{wide_path} not written"
        normalized_path = tmp_path / f"{fct}.csv"
        assert normalized_path.exists(), (
            f"{normalized_path} normalized companion missing — "
            f"denormalized mode must emit alongside, not in place of"
        )


def test_write_tables_normalized_byte_identical_when_flag_off(tmp_path, saas_cfg):
    """Off-by-default invariant: writing with denormalized=False
    must produce the same normalized files as a pre-M14a baseline.
    We assert by comparing a write with explicit False to a write
    with the field omitted (default)."""
    rng = np.random.default_rng(saas_cfg.seed)
    tables, _ = generate_tables_with_state(saas_cfg, rng)

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    cfg_default = saas_cfg
    cfg_explicit = saas_cfg.model_copy(
        update={
            "output": OutputConfig(
                format="csv",
                directory=str(b),
                denormalized=False,
            )
        }
    )

    write_tables(tables, cfg_default, output_dir=a)
    write_tables(tables, cfg_explicit, output_dir=b)

    for fct in (t.name for t in saas_cfg.tables if t.type == "fact"):
        assert (a / f"{fct}.csv").read_bytes() == (b / f"{fct}.csv").read_bytes()


def test_write_tables_wide_parquet_when_format_parquet(tmp_path, saas_cfg):
    pytest.importorskip("pyarrow")
    cfg = saas_cfg.model_copy(
        update={
            "output": OutputConfig(
                format="parquet",
                directory=str(tmp_path),
                denormalized=True,
            )
        }
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    write_tables(tables, cfg, output_dir=tmp_path)
    fact_names = [t.name for t in cfg.tables if t.type == "fact"]
    for fct in fact_names:
        assert (tmp_path / f"{fct}_wide.parquet").exists()
        assert (tmp_path / f"{fct}.parquet").exists()


# --- 6. Determinism ---------------------------------------------------------


def test_wide_is_deterministic(tmp_path, saas_cfg):
    cfg = saas_cfg.model_copy(
        update={
            "output": OutputConfig(
                format="csv",
                directory=str(tmp_path),
                denormalized=True,
            )
        }
    )
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for target in (a, b):
        rng = np.random.default_rng(cfg.seed)
        tables, _ = generate_tables_with_state(cfg, rng)
        write_tables(tables, cfg, output_dir=target)
    for fct in (t.name for t in saas_cfg.tables if t.type == "fact"):
        assert (a / f"{fct}_wide.csv").read_bytes() == (
            b / f"{fct}_wide.csv"
        ).read_bytes(), f"{fct}_wide.csv is non-deterministic"


# --- 7. Builder passthrough -------------------------------------------------


def _builder_kwargs(**overrides):
    """Bare-minimum ``create()`` kwargs mirroring `_minimal_input` in
    `tests/test_builder_interpreter.py`. Use ``overrides`` to set the
    `output=` field under test."""
    base = {
        "about": "test domain",
        "unit": "company",
        "window": {"start": "2023-01", "end": "2024-12", "every": "monthly"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {"name": "mrr", "type": "amount", "polarity": "positive", "range": [100, 50000]},
        ],
        "segments": [
            {"name": "alpha", "count": 10, "archetype": "growth"},
            {"name": "beta", "count": 10, "archetype": "decline"},
        ],
    }
    base.update(overrides)
    return base


def test_builder_output_denormalized_passthrough():
    """``create(output={'denormalized': True})`` plumbs through to
    ``PlotsimConfig.output.denormalized``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(
            **_builder_kwargs(output={"directory": "output", "denormalized": True}),
        )
    assert cfg.output.denormalized is True


def test_builder_output_denormalized_default_false():
    """Without ``denormalized`` in the builder output dict, default
    is False — preserves pre-M14a builder behaviour."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(**_builder_kwargs(output="csv"))
    assert cfg.output.denormalized is False


# --- 8. FK-integrity sanity in wide output ----------------------------------


def test_wide_no_unmatched_dim_attrs(saas_run):
    """Every fact row should successfully join to every FK'd dim
    (FK integrity is already validated upstream). Therefore no
    dim-prefixed column should be entirely null for any row in the
    wide output."""
    cfg, tables = saas_run
    wide = denormalize_fact_tables(tables, cfg)["fct_revenue_wide"]
    # Pick a stable dim-prefixed string column to validate.
    if "dim_company__company_name" in wide.columns:
        nulls = int(wide["dim_company__company_name"].isna().sum())
        assert nulls == 0, (
            f"dim_company__company_name has {nulls} nulls in the wide "
            f"output; every fact row must resolve to a dim row"
        )


# --- 9. Schema export -------------------------------------------------------


def test_schema_json_includes_denormalized_field():
    schema_path = ROOT / "plotsim-schema.json"
    if not schema_path.exists():
        pytest.skip("plotsim-schema.json not generated")
    import json

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    output_schema = schema["$defs"]["OutputConfig"]
    assert "denormalized" in output_schema["properties"]
    assert output_schema["properties"]["denormalized"]["default"] is False
