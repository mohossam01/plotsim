"""M108 — entity feature table tests.

Covers:

* ``EntityFeaturesConfig`` defaults and ``extra="forbid"``
* Load-time validation gates (manifest, quality, metric names)
* End-to-end CLI / write_tables emits ``_entity_features.<csv|parquet>``
* Output shape: one row per entity, six aggregates per metric
* Manual verification of ``_slope`` and ``_peak_period`` against fixed inputs
* ``include_labels`` toggles archetype + final_trajectory_position
* ``metrics`` filter respects configured subset
* Bridge metrics are excluded regardless of config
* NaN handling under MCAR noise
* Byte-deterministic across runs
* Works alongside SCD-versioned dims
* ``plotsim-schema.json`` exposes ``EntityFeaturesConfig``
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
from plotsim.config import EntityFeaturesConfig
from plotsim.entity_features import (
    ENTITY_FEATURES_BASENAME,
    _aggregate_series,
    _peak_period,
    _slope,
    build_entity_features,
)
from plotsim.validation import validate_entity_features_config


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "plotsim" / "configs"
SAAS_YAML = CONFIGS_DIR / "sample_saas.yaml"
HR_YAML = CONFIGS_DIR / "sample_hr.yaml"
EDUCATION_YAML = CONFIGS_DIR / "sample_education.yaml"
RETAIL_YAML = CONFIGS_DIR / "sample_retail.yaml"
MARKETING_YAML = CONFIGS_DIR / "sample_marketing.yaml"


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
        cfg, state.trajectories, tables,
        scd_state=state.scd, bridge_state=state.bridges,
    )
    target = write_tables(
        tables, cfg,
        output_dir=output_dir,
        manifest=manifest,
    )
    return cfg, tables, state, manifest, target


# --- Config defaults + extra='forbid' ---------------------------------------


def test_entity_features_default_disabled():
    cfg = EntityFeaturesConfig()
    assert cfg.enabled is False
    assert cfg.metrics == []
    assert cfg.include_labels is True


def test_entity_features_extra_field_rejected():
    with pytest.raises(Exception):
        EntityFeaturesConfig(enabled=True, junk_field=1)


def test_plotsim_config_default_includes_disabled_entity_features():
    cfg = load_config(SAAS_YAML)
    assert cfg.entity_features.enabled is False
    assert cfg.entity_features.metrics == []
    assert cfg.entity_features.include_labels is True


# --- Load-time validation gates ---------------------------------------------


def test_validation_passes_when_disabled():
    cfg = load_config(SAAS_YAML)
    assert validate_entity_features_config(cfg) == []


def test_no_manifest_raises_at_load(tmp_path):
    target = _patched_yaml(SAAS_YAML, {
        "entity_features": {"enabled": True},
        "manifest": {"include": False},
    }, tmp_path / "saas.yaml")
    with pytest.raises(ValueError, match="manifest.include=true"):
        load_config(target)


def test_quality_injection_combo_raises_at_load(tmp_path):
    target = _patched_yaml(SAAS_YAML, {
        "entity_features": {"enabled": True},
        "quality": {
            "quality_issues": [{
                "type": "null_injection",
                "target_table": "fct_engagement",
                "target_columns": ["engagement_score"],
                "rate": 0.05,
            }],
        },
    }, tmp_path / "saas.yaml")
    with pytest.raises(
        ValueError,
        match="entity_features cannot be combined with quality_issues",
    ):
        load_config(target)


def test_unknown_metric_in_metrics_raises(tmp_path):
    target = _patched_yaml(SAAS_YAML, {
        "entity_features": {
            "enabled": True,
            "metrics": ["does_not_exist"],
        },
    }, tmp_path / "saas.yaml")
    with pytest.raises(ValueError, match="unknown metric 'does_not_exist'"):
        load_config(target)


def test_metric_without_numeric_fact_column_raises(tmp_path):
    """A metric defined in ``config.metrics`` but never landed on a numeric
    fact column cannot be aggregated. Construct one by adding a dummy
    metric (no fact column references it) and listing it in
    ``entity_features.metrics``.
    """
    data = yaml.safe_load(SAAS_YAML.read_text(encoding="utf-8"))
    data["metrics"].append({
        "name": "phantom_metric",
        "label": "Phantom Metric",
        "distribution": "lognorm",
        "params": {"s": 0.5, "scale": 1.0},
        "polarity": "positive",
        "value_range": {"min": 0.0, "max": 1.0},
    })
    data["entity_features"] = {
        "enabled": True,
        "metrics": ["phantom_metric"],
    }
    target = tmp_path / "saas_phantom.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="no int/float column on any fact table",
    ):
        load_config(target)


# --- Disabled is backward-compatible ----------------------------------------


def test_default_disabled_produces_no_file(tmp_path):
    _cfg, _tables, _state, _manifest, target = _run_template(SAAS_YAML, tmp_path)
    assert not (target / f"{ENTITY_FEATURES_BASENAME}.csv").exists()
    assert not (target / f"{ENTITY_FEATURES_BASENAME}.parquet").exists()


# --- End-to-end output ------------------------------------------------------


@pytest.fixture
def saas_with_features(tmp_path):
    """Run the saas template with ``entity_features.enabled=true``."""
    cfg_path = _patched_yaml(SAAS_YAML, {
        "entity_features": {"enabled": True},
    }, tmp_path / "saas.yaml")
    return _run_template(cfg_path, tmp_path / "out")


def test_csv_file_present(saas_with_features):
    _cfg, _tables, _state, _manifest, target = saas_with_features
    assert (target / f"{ENTITY_FEATURES_BASENAME}.csv").is_file()


def test_one_row_per_entity(saas_with_features):
    cfg, _tables, _state, _manifest, target = saas_with_features
    df = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    assert len(df) == len(cfg.entities)


def test_six_aggregate_columns_per_metric(saas_with_features):
    cfg, _tables, _state, _manifest, target = saas_with_features
    df = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    suffixes = ("mean", "std", "slope", "first", "last", "peak_period")
    for metric in cfg.metrics:
        # Some metrics may not be exposed on a fact column; skip those.
        if not any(
            f"{metric.name}_{s}" in df.columns for s in suffixes
        ):
            continue
        for s in suffixes:
            assert f"{metric.name}_{s}" in df.columns, (
                f"missing {metric.name}_{s}"
            )


def test_label_columns_present_when_include_labels_true(saas_with_features):
    cfg, _tables, _state, _manifest, target = saas_with_features
    df = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    assert "archetype" in df.columns
    assert "final_trajectory_position" in df.columns
    expected_archetypes = {e.archetype for e in cfg.entities}
    assert set(df["archetype"]).issubset(expected_archetypes)


def test_label_columns_omitted_when_include_labels_false(tmp_path):
    cfg_path = _patched_yaml(SAAS_YAML, {
        "entity_features": {"enabled": True, "include_labels": False},
    }, tmp_path / "saas.yaml")
    _cfg, _tables, _state, _manifest, target = _run_template(
        cfg_path, tmp_path / "out",
    )
    df = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    assert "archetype" not in df.columns
    assert "final_trajectory_position" not in df.columns


def test_metrics_filter_emits_subset_only(tmp_path):
    cfg_path = _patched_yaml(SAAS_YAML, {
        "entity_features": {
            "enabled": True,
            "metrics": ["mrr"],
        },
    }, tmp_path / "saas.yaml")
    _cfg, _tables, _state, _manifest, target = _run_template(
        cfg_path, tmp_path / "out",
    )
    df = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    assert "mrr_mean" in df.columns
    # No other metric should have aggregates.
    assert not any(
        c.startswith("engagement_") or c.startswith("churn_risk_")
        for c in df.columns
    )


def test_default_metrics_aggregates_every_numeric_fact_metric(saas_with_features):
    cfg, _tables, _state, _manifest, target = saas_with_features
    df = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    # Every metric whose name appears as a metric:column on some fact
    # table must have ``{name}_mean`` in the output.
    from plotsim.config import MetricSource, parse_source
    expected: set[str] = set()
    for tbl in cfg.tables:
        if tbl.type != "fact":
            continue
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if (
                isinstance(parsed, MetricSource)
                and col.dtype in ("int", "float")
            ):
                expected.add(parsed.metric)
    for name in expected:
        assert f"{name}_mean" in df.columns, f"missing {name}_mean"


def test_bridge_metrics_excluded(tmp_path):
    """Education template ships a bridge table; its bridge-only metrics
    must not appear as aggregate columns in the entity features file.
    """
    cfg_path = _patched_yaml(EDUCATION_YAML, {
        "entity_features": {"enabled": True},
    }, tmp_path / "education.yaml")
    cfg, _tables, _state, _manifest, target = _run_template(
        cfg_path, tmp_path / "out",
    )
    if not cfg.bridges:
        pytest.skip("template has no bridges")
    df = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    for bridge in cfg.bridges:
        for bm in bridge.metrics:
            for suffix in ("mean", "std", "slope", "first", "last", "peak_period"):
                col = f"{bm.name}_{suffix}"
                assert col not in df.columns, (
                    f"bridge metric {bm.name!r} leaked into entity_features"
                )


def test_parquet_format_follows_output_format(tmp_path):
    pytest.importorskip("pyarrow")
    cfg_path = _patched_yaml(SAAS_YAML, {
        "entity_features": {"enabled": True},
        "output": {
            "format": "parquet",
            "directory": str(tmp_path / "out"),
        },
    }, tmp_path / "saas.yaml")
    _cfg, _tables, _state, _manifest, target = _run_template(
        cfg_path, tmp_path / "out",
    )
    assert (target / f"{ENTITY_FEATURES_BASENAME}.parquet").is_file()
    assert not (target / f"{ENTITY_FEATURES_BASENAME}.csv").exists()


# --- Determinism + SCD coexistence ------------------------------------------


def test_byte_identical_across_runs(tmp_path):
    cfg_path_a = _patched_yaml(SAAS_YAML, {
        "entity_features": {"enabled": True},
    }, tmp_path / "saas_a.yaml")
    cfg_path_b = _patched_yaml(SAAS_YAML, {
        "entity_features": {"enabled": True},
    }, tmp_path / "saas_b.yaml")
    _, _, _, _, ta = _run_template(cfg_path_a, tmp_path / "out_a")
    _, _, _, _, tb = _run_template(cfg_path_b, tmp_path / "out_b")
    bytes_a = (ta / f"{ENTITY_FEATURES_BASENAME}.csv").read_bytes()
    bytes_b = (tb / f"{ENTITY_FEATURES_BASENAME}.csv").read_bytes()
    assert bytes_a == bytes_b


def test_works_with_scd_dim(saas_with_features):
    """Saas template ships SCD on dim_company. The entity features file
    must still have one row per *entity* (the natural key), regardless
    of how many SCD versions live on the dim.
    """
    cfg, tables, _state, _manifest, target = saas_with_features
    dim_company = tables["dim_company"]
    # SCD expansion produces multiple rows per entity:
    assert len(dim_company) > len(cfg.entities)
    df = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    assert len(df) == len(cfg.entities)
    # First column is the entity PK and is unique:
    pk_col = df.columns[0]
    assert df[pk_col].is_unique


# --- Manual aggregate calculations ------------------------------------------


def test_slope_matches_polyfit_on_known_input():
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    # Perfect line, slope == 2.0
    assert _slope(x, y) == pytest.approx(2.0)


def test_slope_with_nan_drops_missing():
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = np.array([1.0, np.nan, 5.0, 7.0, 9.0])
    # Same line minus one point → slope still 2.0 over [0,2,3,4].
    assert _slope(x, y) == pytest.approx(2.0)


def test_slope_returns_nan_on_too_few_points():
    x = np.array([0.0])
    y = np.array([1.0])
    assert np.isnan(_slope(x, y))


def test_peak_period_matches_argmax():
    periods = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    values = np.array([0.1, 0.5, 0.9, 0.3, 0.2])
    assert _peak_period(periods, values) == 2.0


def test_peak_period_skips_nan():
    periods = np.array([0, 1, 2, 3], dtype=np.int64)
    values = np.array([0.1, np.nan, 0.9, 0.3])
    assert _peak_period(periods, values) == 2.0


def test_peak_period_all_nan_returns_nan():
    periods = np.array([0, 1, 2], dtype=np.int64)
    values = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(_peak_period(periods, values))


def test_aggregate_series_full_shape():
    periods = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    out = _aggregate_series(periods, values)
    assert out["mean"] == pytest.approx(5.0)
    assert out["std"] == pytest.approx(np.std(values))
    assert out["slope"] == pytest.approx(2.0)
    assert out["first"] == 1.0
    assert out["last"] == 9.0
    assert out["peak_period"] == 4.0


def test_aggregate_series_all_nan_yields_nan():
    periods = np.array([0, 1, 2], dtype=np.int64)
    values = np.array([np.nan, np.nan, np.nan])
    out = _aggregate_series(periods, values)
    for key in ("mean", "std", "slope", "first", "last", "peak_period"):
        assert np.isnan(out[key]), f"{key} should be NaN"


# --- NaN-bearing fact tables (MCAR noise) -----------------------------------


def test_high_mcar_noise_does_not_crash(tmp_path):
    """Cranking ``noise.mcar_rate`` introduces ``pd.NA`` cells in fact
    metric columns; entity features must still produce valid (possibly
    NaN) aggregates instead of raising.
    """
    cfg_path = _patched_yaml(SAAS_YAML, {
        "entity_features": {"enabled": True},
        "noise": {"gaussian_sigma": 0.05, "outlier_rate": 0.0, "mcar_rate": 0.1},
    }, tmp_path / "saas.yaml")
    _cfg, _tables, _state, _manifest, target = _run_template(
        cfg_path, tmp_path / "out",
    )
    df = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    # File must exist and have the right row count; some cells may be NaN.
    assert len(df) > 0
    # mean column for a known metric is finite for at least one entity:
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    assert any(df[c].notna().any() for c in mean_cols)


# --- Direct module call -----------------------------------------------------


def test_build_entity_features_returns_dataframe(saas_with_features):
    """Sanity: the module's pure entry point produces the same row count
    as the on-disk CSV (the writer is just a serializer).
    """
    cfg, tables, _state, manifest, target = saas_with_features
    df_built = build_entity_features(cfg, tables, manifest)
    df_disk = pd.read_csv(target / f"{ENTITY_FEATURES_BASENAME}.csv")
    assert len(df_built) == len(df_disk)
    # Columns must be identical and in the same order.
    assert list(df_built.columns) == list(df_disk.columns)


def test_build_entity_features_pure_function(saas_with_features):
    """Two calls with identical inputs return DataFrames with equal
    values. ``equals`` ignores index alignment and dtype-promotion
    quirks that arise on round-trip through CSV.
    """
    cfg, tables, _state, manifest, _target = saas_with_features
    df_a = build_entity_features(cfg, tables, manifest)
    df_b = build_entity_features(cfg, tables, manifest)
    assert df_a.equals(df_b)


# --- Schema export ----------------------------------------------------------


def test_schema_json_includes_entity_features_config():
    schema_path = ROOT / "plotsim-schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "EntityFeaturesConfig" in schema.get("$defs", {})
    plotsim_props = schema.get("properties", {})
    assert "entity_features" in plotsim_props


def test_schema_entity_features_field_constraints():
    schema_path = ROOT / "plotsim-schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    efc = schema["$defs"]["EntityFeaturesConfig"]
    assert efc.get("additionalProperties") is False
    props = efc["properties"]
    assert set(props.keys()) == {"enabled", "metrics", "include_labels"}
    assert props["enabled"]["default"] is False
    assert props["include_labels"]["default"] is True
    # ``metrics`` uses ``default_factory=list``; pydantic v2 omits the
    # ``default`` key for factory-defaulted fields, so just check the
    # type contract.
    assert props["metrics"]["type"] == "array"
