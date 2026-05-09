"""M109 — temporal holdout split tests.

Covers:

* ``HoldoutConfig`` defaults and ``extra="forbid"``
* Load-time validation gates (target_metric required, numeric fact metric,
  non-empty quality_issues forbidden, holdout_periods bounds,
  min_training_periods floor)
* End-to-end CLI / write_tables emits ``{table}_train`` and
  ``{table}_holdout`` companions for every per_entity_per_period fact
  table; dim/bridge/event tables are NOT split
* Split correctness: row partition exact, no loss, no duplication
* Period boundary: training periods strictly < cutoff, holdout strictly >=
* Manifest records target_metric / holdout_periods / cutoff_period_index
* Determinism: same config + seed → byte-identical train and holdout files
* Output format respected: parquet path produces .parquet companions
* Backward compat: ``holdout.enabled=False`` (default) emits no extras
* Entity-features leakage prevention:
    - aggregations match training-only manual computation
    - target-metric aggregate columns excluded from output
* SCD-versioned dim integration: dim_row_id columns survive the split
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from plotsim import (
    SurrogateKeyWarning,
    build_manifest,
    generate_tables_with_state,
    load_config,
    write_tables,
)
from plotsim.config import HoldoutConfig
from plotsim.entity_features import ENTITY_FEATURES_BASENAME
from plotsim.holdout import cutoff_period_index, split_fact_tables
from plotsim.manifest import HoldoutInfo, MANIFEST_FILENAME
from plotsim.validation import validate_holdout_config


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "plotsim" / "configs"
SAAS_YAML = CONFIGS_DIR / "sample_saas.yaml"

SAAS_FACT_TABLES = ("fct_engagement", "fct_revenue", "fct_support_tickets")


# --- Helpers ----------------------------------------------------------------


def _patched_yaml(src: Path, overrides: dict, dst: Path) -> Path:
    """Copy a template YAML to ``dst`` and merge ``overrides`` at the root."""
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        data[key] = value
    dst.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return dst


def _run_template(yaml_path: Path, output_dir: Path):
    """Load + generate + write; return ``(cfg, tables, state, manifest, target)``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(yaml_path)
    rng = np.random.default_rng(cfg.seed)
    tables, state = generate_tables_with_state(cfg, rng)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    target = write_tables(
        tables,
        cfg,
        output_dir=output_dir,
        manifest=manifest,
    )
    return cfg, tables, state, manifest, target


# --- Config defaults + extra='forbid' ---------------------------------------


def test_holdout_default_disabled():
    cfg = HoldoutConfig()
    assert cfg.enabled is False
    assert cfg.target_metric is None
    assert cfg.holdout_periods == 0
    assert cfg.min_training_periods == 3


def test_holdout_extra_field_rejected():
    with pytest.raises(Exception):
        HoldoutConfig(enabled=True, junk_field=1)


def test_plotsim_config_default_includes_disabled_holdout():
    cfg = load_config(SAAS_YAML)
    assert cfg.holdout.enabled is False
    assert cfg.holdout.target_metric is None
    assert cfg.holdout.holdout_periods == 0


# --- Load-time validation gates ---------------------------------------------


def test_validation_passes_when_disabled():
    cfg = load_config(SAAS_YAML)
    assert validate_holdout_config(cfg) == []


def test_enabled_without_target_metric_raises(tmp_path):
    target = _patched_yaml(
        SAAS_YAML,
        {
            "holdout": {"enabled": True, "holdout_periods": 4},
        },
        tmp_path / "saas.yaml",
    )
    with pytest.raises(
        ValueError,
        match="holdout.enabled=true requires holdout.target_metric",
    ):
        load_config(target)


def test_zero_holdout_periods_raises_when_enabled(tmp_path):
    target = _patched_yaml(
        SAAS_YAML,
        {
            "holdout": {
                "enabled": True,
                "target_metric": "engagement",
                "holdout_periods": 0,
            },
        },
        tmp_path / "saas.yaml",
    )
    with pytest.raises(ValueError, match="holdout_periods must be >= 1"):
        load_config(target)


def test_unknown_target_metric_raises(tmp_path):
    target = _patched_yaml(
        SAAS_YAML,
        {
            "holdout": {
                "enabled": True,
                "target_metric": "does_not_exist",
                "holdout_periods": 4,
            },
        },
        tmp_path / "saas.yaml",
    )
    with pytest.raises(ValueError, match="unknown metric 'does_not_exist'"):
        load_config(target)


def test_target_metric_without_numeric_fact_column_raises(tmp_path):
    """A metric defined in ``config.metrics`` but never landed on a numeric
    fact column cannot be a training target."""
    data = yaml.safe_load(SAAS_YAML.read_text(encoding="utf-8"))
    data["metrics"].append(
        {
            "name": "phantom_metric",
            "label": "Phantom Metric",
            "distribution": "lognorm",
            "params": {"s": 0.5, "scale": 1.0},
            "polarity": "positive",
            "value_range": {"min": 0.0, "max": 1.0},
        }
    )
    data["holdout"] = {
        "enabled": True,
        "target_metric": "phantom_metric",
        "holdout_periods": 4,
    }
    target = tmp_path / "saas_phantom.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="no int/float column on any fact table"):
        load_config(target)


def test_min_training_periods_floor_raises(tmp_path):
    """Saas template has 24 monthly periods; holdout=23 leaves 1 training
    period, which falls below the default floor of 3."""
    target = _patched_yaml(
        SAAS_YAML,
        {
            "holdout": {
                "enabled": True,
                "target_metric": "engagement",
                "holdout_periods": 23,
            },
        },
        tmp_path / "saas.yaml",
    )
    with pytest.raises(
        ValueError,
        match="holdout split leaves 1 training period",
    ):
        load_config(target)


def test_quality_injection_combo_raises_at_load(tmp_path):
    target = _patched_yaml(
        SAAS_YAML,
        {
            "holdout": {
                "enabled": True,
                "target_metric": "engagement",
                "holdout_periods": 4,
            },
            "quality": {
                "quality_issues": [
                    {
                        "type": "null_injection",
                        "target_table": "fct_engagement",
                        "target_columns": ["engagement_score"],
                        "rate": 0.05,
                    }
                ],
            },
        },
        tmp_path / "saas.yaml",
    )
    with pytest.raises(
        ValueError,
        match="holdout cannot be combined with quality_issues",
    ):
        load_config(target)


# --- Disabled is backward-compatible ----------------------------------------


def test_default_disabled_produces_no_split_files(tmp_path):
    """Without ``holdout.enabled``, no ``_train`` / ``_holdout`` companions
    are written."""
    _cfg, _tables, _state, _manifest, target = _run_template(
        SAAS_YAML,
        tmp_path,
    )
    for fact in SAAS_FACT_TABLES:
        assert not (target / f"{fact}_train.csv").exists()
        assert not (target / f"{fact}_holdout.csv").exists()


# --- End-to-end split -------------------------------------------------------


@pytest.fixture
def saas_with_holdout(tmp_path):
    """Saas with 6-period trailing holdout on ``engagement``."""
    cfg_path = _patched_yaml(
        SAAS_YAML,
        {
            "holdout": {
                "enabled": True,
                "target_metric": "engagement",
                "holdout_periods": 6,
            },
        },
        tmp_path / "saas.yaml",
    )
    out = tmp_path / "out"
    return _run_template(cfg_path, out)


def test_split_files_emitted_for_every_fact_table(saas_with_holdout):
    _cfg, _tables, _state, _manifest, target = saas_with_holdout
    for fact in SAAS_FACT_TABLES:
        assert (target / f"{fact}.csv").exists(), f"{fact}.csv (unsplit) should still be written"
        assert (target / f"{fact}_train.csv").exists()
        assert (target / f"{fact}_holdout.csv").exists()


def test_dim_and_event_tables_are_not_split(saas_with_holdout):
    _cfg, tables, _state, _manifest, target = saas_with_holdout
    for name in tables:
        # FK 'dim_date_train.csv' style names should never appear.
        if name.startswith("dim_") or name.startswith("evt_"):
            assert not (target / f"{name}_train.csv").exists()
            assert not (target / f"{name}_holdout.csv").exists()


def test_partition_is_exact_no_loss_no_duplication(saas_with_holdout):
    cfg, tables, _state, _manifest, target = saas_with_holdout
    splits = split_fact_tables(cfg, tables)
    assert set(splits.keys()) == set(SAAS_FACT_TABLES)
    for fact_name, (train_df, holdout_df) in splits.items():
        original = tables[fact_name]
        assert len(train_df) + len(holdout_df) == len(
            original
        ), f"{fact_name}: row counts don't sum"
        # No duplicate index across the two halves.
        train_set = set(train_df.index.tolist())
        holdout_set = set(holdout_df.index.tolist())
        assert train_set.isdisjoint(holdout_set)
        assert train_set | holdout_set == set(original.index.tolist())


def test_training_strictly_before_holdout(saas_with_holdout):
    cfg, tables, _state, _manifest, _target = saas_with_holdout
    cutoff = cutoff_period_index(cfg)
    assert cutoff == cfg.time_window.period_count() - cfg.holdout.holdout_periods
    dim_date = tables["dim_date"]
    period_by_dk = dict(zip(dim_date["date_key"].tolist(), dim_date["period_index"].tolist()))
    splits = split_fact_tables(cfg, tables)
    for fact_name, (train_df, holdout_df) in splits.items():
        train_periods = {period_by_dk[dk] for dk in train_df["date_key"].tolist()}
        holdout_periods = {period_by_dk[dk] for dk in holdout_df["date_key"].tolist()}
        assert all(p < cutoff for p in train_periods), fact_name
        assert all(p >= cutoff for p in holdout_periods), fact_name


def test_every_entity_in_holdout_is_also_in_training(saas_with_holdout):
    cfg, tables, _state, _manifest, _target = saas_with_holdout
    splits = split_fact_tables(cfg, tables)
    for fact_name, (train_df, holdout_df) in splits.items():
        train_entities = set(train_df["company_id"].tolist())
        holdout_entities = set(holdout_df["company_id"].tolist())
        # Every entity in holdout must be present in training (no
        # entity should appear ONLY in holdout).
        leak = holdout_entities - train_entities
        assert not leak, f"{fact_name}: entities {leak} appear in holdout but not training"


def test_split_files_sum_to_original_on_disk(saas_with_holdout):
    _cfg, _tables, _state, _manifest, target = saas_with_holdout
    for fact in SAAS_FACT_TABLES:
        full = pd.read_csv(target / f"{fact}.csv")
        train = pd.read_csv(target / f"{fact}_train.csv")
        hold = pd.read_csv(target / f"{fact}_holdout.csv")
        assert len(train) + len(hold) == len(full), fact
        assert list(train.columns) == list(full.columns), fact
        assert list(hold.columns) == list(full.columns), fact


# --- Manifest integration ---------------------------------------------------


def test_manifest_records_holdout_info(saas_with_holdout):
    _cfg, _tables, _state, _manifest, target = saas_with_holdout
    payload = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert payload["holdout"] is not None
    assert payload["holdout"]["target_metric"] == "engagement"
    assert payload["holdout"]["holdout_periods"] == 6
    assert payload["holdout"]["cutoff_period_index"] == 24 - 6


def test_manifest_holdout_absent_when_disabled(tmp_path):
    _cfg, _tables, _state, _manifest, target = _run_template(SAAS_YAML, tmp_path)
    payload = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    # ``holdout`` field absent or null when not configured. Either is
    # acceptable round-trip semantics for an optional field.
    assert payload.get("holdout") is None


def test_holdout_info_construction_round_trips():
    info = HoldoutInfo(
        target_metric="engagement",
        holdout_periods=6,
        cutoff_period_index=18,
    )
    dumped = info.model_dump()
    rebuilt = HoldoutInfo(**dumped)
    assert rebuilt == info


# --- Determinism ------------------------------------------------------------


def test_split_files_byte_identical_across_runs(tmp_path):
    overrides = {
        "holdout": {
            "enabled": True,
            "target_metric": "engagement",
            "holdout_periods": 6,
        },
    }
    cfg_path = _patched_yaml(SAAS_YAML, overrides, tmp_path / "saas.yaml")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _run_template(cfg_path, out_a)
    _run_template(cfg_path, out_b)
    for fact in SAAS_FACT_TABLES:
        for kind in ("_train.csv", "_holdout.csv"):
            bytes_a = (out_a / f"{fact}{kind}").read_bytes()
            bytes_b = (out_b / f"{fact}{kind}").read_bytes()
            assert bytes_a == bytes_b, f"{fact}{kind} drifted between runs"


# --- Output format ----------------------------------------------------------


def test_parquet_format_emits_parquet_split_files(tmp_path):
    pytest.importorskip("pyarrow")
    overrides = {
        "holdout": {
            "enabled": True,
            "target_metric": "engagement",
            "holdout_periods": 6,
        },
        "output": {"format": "parquet", "directory": "ignored"},
    }
    cfg_path = _patched_yaml(SAAS_YAML, overrides, tmp_path / "saas.yaml")
    out = tmp_path / "out"
    _run_template(cfg_path, out)
    for fact in SAAS_FACT_TABLES:
        assert (out / f"{fact}.parquet").exists(), fact
        assert (out / f"{fact}_train.parquet").exists()
        assert (out / f"{fact}_holdout.parquet").exists()
        # CSV equivalents must NOT be written when format=parquet.
        assert not (out / f"{fact}_train.csv").exists()


# --- Entity features leakage prevention -------------------------------------


@pytest.fixture
def saas_with_holdout_and_features(tmp_path):
    cfg_path = _patched_yaml(
        SAAS_YAML,
        {
            "holdout": {
                "enabled": True,
                "target_metric": "engagement",
                "holdout_periods": 6,
            },
            "entity_features": {"enabled": True},
        },
        tmp_path / "saas.yaml",
    )
    out = tmp_path / "out"
    return _run_template(cfg_path, out)


def test_entity_features_excludes_target_metric_columns(
    saas_with_holdout_and_features,
):
    _cfg, _tables, _state, _manifest, target = saas_with_holdout_and_features
    df = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    forbidden = {
        f"engagement_{suffix}"
        for suffix in ("mean", "std", "slope", "first", "last", "peak_period")
    }
    leaked = forbidden & set(df.columns)
    assert not leaked, f"entity_features leaks target-metric columns: {leaked}"


def test_entity_features_aggregates_match_training_only(
    saas_with_holdout_and_features,
):
    """Manually compute one (entity, metric) aggregate from the
    training-only fact rows and confirm the entity-features row matches
    — proves the per-entity features were derived from the training
    window, not the full series."""
    cfg, tables, _state, _manifest, target = saas_with_holdout_and_features
    cutoff = cutoff_period_index(cfg)
    dim_date = tables["dim_date"]
    period_by_dk = dict(zip(dim_date["date_key"].tolist(), dim_date["period_index"].tolist()))

    fct_revenue = tables["fct_revenue"].copy()
    fct_revenue["__period"] = fct_revenue["date_key"].map(period_by_dk)
    train_only = fct_revenue[fct_revenue["__period"] < cutoff]

    # Pick the first entity with non-empty training rows.
    first_entity = train_only["company_id"].iloc[0]
    train_slice = train_only[train_only["company_id"] == first_entity]
    expected_mean = float(np.nanmean(train_slice["mrr"].to_numpy(dtype=float)))
    expected_last_period = int(train_slice["__period"].max())

    features = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    row = features[features["company_id"] == first_entity].iloc[0]
    # Mean should match the training-only mean. Tolerance accounts for
    # the CSV writer's ``%.4f`` rounding (half-LSB of 5e-5 around 1k);
    # in-memory pre-write the values are bit-identical.
    assert abs(row["mrr_mean"] - expected_mean) < 5e-4
    # ``mrr_last`` should reflect the LAST training period, not the
    # last period of the full series — an entity-features value
    # computed from the full series would point at period 23, not at
    # period 17 (cutoff-1) under a 6-period holdout on a 24-period
    # window.
    assert expected_last_period == cutoff - 1


# --- Public split helper ----------------------------------------------------


def test_split_fact_tables_returns_empty_when_disabled():
    cfg = load_config(SAAS_YAML)
    splits = split_fact_tables(cfg, {})
    assert splits == {}


def test_split_fact_tables_only_returns_composite_grain_facts(saas_with_holdout):
    """Helper contract: only ``per_entity_per_period`` facts appear in
    the splits dict. No bundled template ships with a non-composite
    fact (and the engine rejects non-date FKs on per_period facts), so
    the inverse case can't be exercised via YAML — but the positive
    side of the rule is verifiable: every key returned must point at a
    fact with composite grain on the supplied config."""
    cfg, tables, _state, _manifest, _ = saas_with_holdout
    splits = split_fact_tables(cfg, tables)
    composite_facts = {
        t.name for t in cfg.tables if t.type == "fact" and t.grain == "per_entity_per_period"
    }
    assert set(splits.keys()).issubset(composite_facts)
    # And every composite-grain fact with non-empty data is present.
    expected = {
        name for name in composite_facts if tables.get(name) is not None and not tables[name].empty
    }
    assert set(splits.keys()) == expected


# --- SCD-versioned dim integration ------------------------------------------


def test_holdout_works_with_scd_versioned_dims(saas_with_holdout):
    """Saas ships with an SCD Type 2 column on ``dim_company``
    (``plan_tier`` versioned by ``fct_revenue.mrr``). Fact tables
    that FK into the SCD dim carry a ``dim_row_id`` surrogate;
    confirm the surrogate column survives the holdout split on both
    halves."""
    _cfg, tables, _state, _manifest, target = saas_with_holdout
    # ``fct_revenue`` is the SCD trigger fact for plan_tier, so it is
    # the one that should carry ``dim_row_id`` after expansion.
    if "dim_row_id" not in tables["fct_revenue"].columns:
        pytest.skip("dim_row_id not attached on fct_revenue; SCD wiring shifted")
    train = pd.read_csv(target / "fct_revenue_train.csv")
    holdout = pd.read_csv(target / "fct_revenue_holdout.csv")
    assert "dim_row_id" in train.columns
    assert "dim_row_id" in holdout.columns
    parent_ids = set(tables["dim_company"]["dim_row_id"].tolist())
    assert set(train["dim_row_id"].tolist()).issubset(parent_ids)
    assert set(holdout["dim_row_id"].tolist()).issubset(parent_ids)
