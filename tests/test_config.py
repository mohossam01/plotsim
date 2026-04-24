"""Tests for plotsim.config — Mission 001 acceptance criteria."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from plotsim.config import (
    DIRTY,
    NOISE_PRESETS,
    DerivedSource,
    FKSource,
    GeneratedSource,
    LagSource,
    MetricSource,
    PlotsimConfig,
    PERFECTLY_CLEAN,
    PKSource,
    ProportionalSource,
    REALISTIC,
    SLIGHTLY_MESSY,
    StaticSource,
    SurrogateKeyWarning,
    Table,
    ThresholdSource,
    dump_config,
    load_config,
    parse_source,
)

ROOT = Path(__file__).resolve().parent.parent
SAAS_YAML = ROOT / "plotsim" / "configs" / "sample_saas.yaml"
HR_YAML = ROOT / "plotsim" / "configs" / "sample_hr.yaml"


# --- Acceptance: valid load for both sample configs ---

def test_load_saas_config():
    c = load_config(SAAS_YAML)
    assert isinstance(c, PlotsimConfig)
    assert c.domain.entity_type == "customer_account"
    assert c.domain.entity_label == "Customer accounts"
    metric_names = {m.name for m in c.metrics}
    assert metric_names == {
        "engagement", "mrr", "support_tickets",
        "feature_adoption", "churn_risk", "nps",
    }
    archetype_names = {a.name for a in c.archetypes}
    assert archetype_names == {
        "rocket_then_cliff", "steady_grower", "slow_death",
        "seasonal_spiker", "zombie_account", "expansion_champion",
    }
    table_names = {t.name for t in c.tables}
    assert table_names == {
        "dim_date", "dim_company", "dim_user", "dim_plan",
        "fct_engagement", "fct_revenue", "fct_support_tickets",
        "evt_login", "evt_churn",
    }
    assert len(c.entities) == 3


def test_load_hr_config():
    c = load_config(HR_YAML)
    assert isinstance(c, PlotsimConfig)
    assert c.domain.entity_type == "employee"
    metric_names = {m.name for m in c.metrics}
    assert metric_names == {
        "performance_score", "engagement_index", "training_hours",
        "absence_rate", "attrition_risk",
    }
    archetype_names = {a.name for a in c.archetypes}
    assert archetype_names == {
        "fast_riser", "steady_performer", "quiet_quitter",
        "burnout_risk", "new_hire_ramp",
    }
    assert len(c.tables) == 7
    assert len(c.entities) == 4


# --- Acceptance: PlotsimConfig is frozen ---

def test_config_is_frozen():
    c = load_config(SAAS_YAML)
    with pytest.raises(ValidationError):
        c.seed = 9999  # type: ignore[misc]
    with pytest.raises(ValidationError):
        c.domain.name = "mutated"  # type: ignore[misc]


# --- Acceptance: noise presets are named constants ---

def test_noise_presets_named_constants():
    assert NOISE_PRESETS["Perfectly clean"] is PERFECTLY_CLEAN
    assert NOISE_PRESETS["Slightly messy"] is SLIGHTLY_MESSY
    assert NOISE_PRESETS["Realistic"] is REALISTIC
    assert NOISE_PRESETS["Dirty"] is DIRTY
    assert PERFECTLY_CLEAN.gaussian_sigma == 0.0
    assert SLIGHTLY_MESSY.gaussian_sigma == 0.03
    assert REALISTIC.gaussian_sigma == 0.05
    assert DIRTY.duplicate_rate == 0.01


# --- Acceptance: round-trip load → dump → load → equal ---

def test_roundtrip_saas():
    c1 = load_config(SAAS_YAML)
    dumped = dump_config(c1)
    c2 = PlotsimConfig(**yaml.safe_load(dumped))
    assert c1 == c2


def test_roundtrip_hr():
    c1 = load_config(HR_YAML)
    dumped = dump_config(c1)
    c2 = PlotsimConfig(**yaml.safe_load(dumped))
    assert c1 == c2


# --- Helpers for negative tests: a minimal valid raw dict to mutate ---

def _minimal_valid() -> dict:
    return {
        "domain": {
            "name": "test",
            "description": "minimal test fixture",
            "entity_type": "widget",
            "entity_label": "Widgets",
        },
        "time_window": {
            "start": "2024-01", "end": "2024-12", "granularity": "monthly",
        },
        "seed": 1,
        "metrics": [{
            "name": "m1",
            "label": "M1",
            "distribution": "lognorm",
            "params": {"s": 0.5, "scale": 1.0},
            "polarity": "positive",
            "default_curve": "sigmoid",
        }],
        "archetypes": [{
            "name": "a1",
            "label": "A1",
            "description": "single segment covering full range",
            "curve_segments": [{
                "curve": "sigmoid",
                "params": {"midpoint": 0.5, "steepness": 1.0},
                "start_pct": 0.0,
                "end_pct": 1.0,
            }],
        }],
        "entities": [{"name": "e1", "archetype": "a1", "size": 5}],
        "tables": [
            {
                "name": "dim_widget",
                "type": "dim",
                "grain": "per_entity",
                "columns": [
                    {"name": "widget_id", "dtype": "id", "source": "pk"},
                ],
                "primary_key": "widget_id",
            },
            {
                "name": "fct_m1",
                "type": "fact",
                "grain": "per_entity_per_period",
                "columns": [
                    {"name": "date_key", "dtype": "id", "source": "fk:dim_date.date_key"},
                    {"name": "widget_id", "dtype": "id", "source": "fk:dim_widget.widget_id"},
                    {"name": "m1", "dtype": "float", "source": "metric:m1"},
                ],
                "primary_key": ["date_key", "widget_id"],
                "foreign_keys": ["dim_date.date_key", "dim_widget.widget_id"],
            },
            {
                "name": "dim_date",
                "type": "dim",
                "grain": "per_period",
                "columns": [
                    {"name": "date_key", "dtype": "id", "source": "pk"},
                ],
                "primary_key": "date_key",
            },
        ],
        "output": {"format": "csv", "directory": "out"},
    }


def test_minimal_valid_loads():
    c = PlotsimConfig(**_minimal_valid())
    assert c.seed == 1
    assert c.entities[0].name == "e1"


# --- Acceptance: every validation error case ---

def test_missing_required_field_seed():
    raw = _minimal_valid()
    del raw["seed"]
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "seed" in str(exc.value)


def test_missing_required_field_domain():
    raw = _minimal_valid()
    del raw["domain"]
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_unknown_curve_type_in_segment():
    raw = _minimal_valid()
    raw["archetypes"][0]["curve_segments"][0]["curve"] = "bogus_curve"
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_metric_override_references_unknown_metric():
    raw = _minimal_valid()
    raw["archetypes"][0]["metric_overrides"] = {
        "ghost_metric": {"curve": "sigmoid"},
    }
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "ghost_metric" in str(exc.value)


def test_fk_source_to_unknown_table():
    raw = _minimal_valid()
    raw["tables"][1]["columns"][1]["source"] = "fk:dim_nowhere.widget_id"
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "dim_nowhere" in str(exc.value)


def test_foreign_keys_list_references_unknown_table():
    raw = _minimal_valid()
    raw["tables"][1]["foreign_keys"] = ["dim_nowhere.widget_id"]
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_start_pct_not_less_than_end_pct():
    raw = _minimal_valid()
    raw["archetypes"][0]["curve_segments"] = [
        {"curve": "sigmoid", "params": {}, "start_pct": 0.6, "end_pct": 0.4},
    ]
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "start_pct" in str(exc.value)


def test_start_pct_equal_to_end_pct():
    raw = _minimal_valid()
    raw["archetypes"][0]["curve_segments"] = [
        {"curve": "sigmoid", "params": {}, "start_pct": 0.5, "end_pct": 0.5},
    ]
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_segments_do_not_cover_full_range_start():
    raw = _minimal_valid()
    raw["archetypes"][0]["curve_segments"] = [
        {"curve": "sigmoid", "params": {}, "start_pct": 0.1, "end_pct": 1.0},
    ]
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_segments_do_not_cover_full_range_end():
    raw = _minimal_valid()
    raw["archetypes"][0]["curve_segments"] = [
        {"curve": "sigmoid", "params": {}, "start_pct": 0.0, "end_pct": 0.8},
    ]
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_segments_with_gap():
    raw = _minimal_valid()
    raw["archetypes"][0]["curve_segments"] = [
        {"curve": "sigmoid", "params": {}, "start_pct": 0.0, "end_pct": 0.4},
        {"curve": "plateau", "params": {}, "start_pct": 0.5, "end_pct": 1.0},
    ]
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_bad_polarity():
    raw = _minimal_valid()
    raw["metrics"][0]["polarity"] = "neutral"
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_column_source_metric_not_in_metrics_list():
    raw = _minimal_valid()
    raw["tables"][1]["columns"][2]["source"] = "metric:ghost"
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "ghost" in str(exc.value)


def test_entity_references_unknown_archetype():
    raw = _minimal_valid()
    raw["entities"][0]["archetype"] = "nonexistent_archetype"
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "nonexistent_archetype" in str(exc.value)


def test_bad_source_prefix():
    raw = _minimal_valid()
    raw["tables"][1]["columns"][2]["source"] = "wrong:format"
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_fk_source_missing_dot():
    raw = _minimal_valid()
    raw["tables"][1]["columns"][1]["source"] = "fk:dim_widget_no_dot"
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_duplicate_metric_names():
    raw = _minimal_valid()
    raw["metrics"].append(copy.deepcopy(raw["metrics"][0]))
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_correlation_references_unknown_metric():
    raw = _minimal_valid()
    raw["correlations"] = [
        {"metric_a": "m1", "metric_b": "ghost", "coefficient": 0.5},
    ]
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_bad_time_window_format():
    raw = _minimal_valid()
    raw["time_window"]["start"] = "2024/01"
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_time_window_start_not_before_end():
    raw = _minimal_valid()
    raw["time_window"]["start"] = "2024-12"
    raw["time_window"]["end"] = "2024-01"
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_primary_key_not_in_columns():
    raw = _minimal_valid()
    raw["tables"][0]["primary_key"] = "nonexistent_col"
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


# --- Flag 1: new grain enum ---

def test_new_grain_values_all_accepted():
    raw = _minimal_valid()
    # Add a table per new grain value not already present in the minimal config
    raw["tables"].append({
        "name": "dim_plan",
        "type": "dim",
        "grain": "per_reference",
        "columns": [{"name": "plan_id", "dtype": "id", "source": "pk"}],
        "primary_key": "plan_id",
    })
    raw["tables"].append({
        "name": "evt_thing",
        "type": "event",
        "grain": "variable",
        "columns": [{"name": "event_id", "dtype": "id", "source": "pk"}],
        "primary_key": "event_id",
    })
    c = PlotsimConfig(**raw)
    grains = {t.grain for t in c.tables}
    assert {"per_entity", "per_period", "per_entity_per_period",
            "per_reference", "variable"} <= grains


def test_legacy_grain_values_rejected():
    raw = _minimal_valid()
    raw["tables"][0]["grain"] = "one_per_entity"  # old name
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


# --- Flag 2: composite primary keys ---

def test_composite_primary_key_accepted():
    raw = _minimal_valid()
    c = PlotsimConfig(**raw)
    fct = next(t for t in c.tables if t.name == "fct_m1")
    assert isinstance(fct.primary_key, list)
    assert fct.primary_key == ["date_key", "widget_id"]
    assert fct.primary_key_cols == ["date_key", "widget_id"]


def test_composite_pk_with_missing_column_rejected():
    raw = _minimal_valid()
    raw["tables"][1]["primary_key"] = ["date_key", "nonexistent_col"]
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "nonexistent_col" in str(exc.value)


def test_composite_pk_with_duplicate_columns_rejected():
    raw = _minimal_valid()
    raw["tables"][1]["primary_key"] = ["date_key", "date_key"]
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_empty_pk_list_rejected():
    raw = _minimal_valid()
    raw["tables"][1]["primary_key"] = []
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_surrogate_pk_on_composite_grain_emits_warning():
    raw = _minimal_valid()
    # Replace the composite PK with a single surrogate column on a
    # per_entity_per_period fact table — should warn but still load.
    raw["tables"][1]["columns"].insert(0, {
        "name": "row_id", "dtype": "id", "source": "pk",
    })
    raw["tables"][1]["primary_key"] = "row_id"
    with pytest.warns(SurrogateKeyWarning, match="per_entity_per_period"):
        c = PlotsimConfig(**raw)
    fct = next(t for t in c.tables if t.name == "fct_m1")
    assert fct.primary_key == "row_id"
    assert fct.primary_key_cols == ["row_id"]


def test_composite_pk_on_composite_grain_emits_no_warning():
    raw = _minimal_valid()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SurrogateKeyWarning)
        PlotsimConfig(**raw)


def test_single_pk_on_non_composite_grain_emits_no_warning():
    raw = _minimal_valid()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SurrogateKeyWarning)
        # dim_widget has grain=per_entity, single-column PK — should not warn
        PlotsimConfig(**raw)


def test_table_primary_key_cols_property():
    t_single = Table(
        name="dim_x", type="dim", grain="per_entity",
        columns=[{"name": "x_id", "dtype": "id", "source": "pk"}],
        primary_key="x_id",
    )
    assert t_single.primary_key_cols == ["x_id"]

    t_composite = Table(
        name="dim_y", type="dim", grain="per_period",
        columns=[
            {"name": "a", "dtype": "id", "source": "pk"},
            {"name": "b", "dtype": "id", "source": "pk"},
        ],
        primary_key=["a", "b"],
    )
    assert t_composite.primary_key_cols == ["a", "b"]


# --- Mission 001a: parse_source returns typed objects ---

def test_parse_source_pk():
    assert parse_source("pk") == PKSource()


def test_parse_source_fk():
    assert parse_source("fk:dim_user.user_id") == FKSource(
        table="dim_user", column="user_id"
    )


def test_parse_source_metric():
    assert parse_source("metric:engagement") == MetricSource(metric="engagement")


def test_parse_source_generated():
    assert parse_source("generated:faker.company") == GeneratedSource(
        provider="faker.company"
    )


def test_parse_source_static():
    assert parse_source("static:99.00") == StaticSource(value="99.00")


def test_parse_source_derived():
    assert parse_source("derived:size") == DerivedSource(field="size")


def test_parse_source_threshold():
    parsed = parse_source("threshold:churn_risk:above:0.7:for:3")
    assert parsed == ThresholdSource(
        metric="churn_risk", direction="above", value=0.7, consecutive=3
    )


def test_parse_source_threshold_below():
    parsed = parse_source("threshold:engagement:below:0.2:for:1")
    assert parsed == ThresholdSource(
        metric="engagement", direction="below", value=0.2, consecutive=1
    )


def test_parse_source_proportional():
    assert parse_source("proportional:engagement:scale:5") == ProportionalSource(
        metric="engagement", scale=5.0
    )


def test_parse_source_lag():
    assert parse_source("lag:engagement:periods:2") == LagSource(
        metric="engagement", periods=2
    )


# --- Mission 001a: parse_source rejects malformed inputs ---

def test_parse_source_unknown_prefix():
    with pytest.raises(ValueError):
        parse_source("wibble:something")


def test_parse_source_threshold_bad_direction():
    with pytest.raises(ValueError, match="above|below"):
        parse_source("threshold:churn_risk:sideways:0.7:for:3")


def test_parse_source_threshold_missing_for_keyword():
    with pytest.raises(ValueError, match="threshold"):
        parse_source("threshold:churn_risk:above:0.7:periods:3")


def test_parse_source_threshold_wrong_arity():
    with pytest.raises(ValueError, match="threshold"):
        parse_source("threshold:churn_risk:above:0.7")


def test_parse_source_threshold_non_numeric_value():
    with pytest.raises(ValueError, match="non-numeric"):
        parse_source("threshold:churn_risk:above:high:for:3")


def test_parse_source_threshold_non_integer_consecutive():
    with pytest.raises(ValueError, match="non-integer"):
        parse_source("threshold:churn_risk:above:0.7:for:three")


def test_parse_source_proportional_bad_format():
    with pytest.raises(ValueError, match="proportional"):
        parse_source("proportional:engagement:times:5")


def test_parse_source_proportional_non_numeric_scale():
    with pytest.raises(ValueError, match="non-numeric"):
        parse_source("proportional:engagement:scale:big")


def test_parse_source_lag_bad_format():
    with pytest.raises(ValueError, match="lag"):
        parse_source("lag:engagement:delay:2")


def test_parse_source_lag_non_integer_periods():
    with pytest.raises(ValueError, match="non-integer"):
        parse_source("lag:engagement:periods:two")


def test_parse_source_threshold_consecutive_min_one():
    # consecutive=0 is rejected by Pydantic Field(ge=1)
    with pytest.raises(Exception):
        parse_source("threshold:churn_risk:above:0.7:for:0")


def test_parse_source_proportional_scale_must_be_positive():
    with pytest.raises(Exception):
        parse_source("proportional:engagement:scale:0")


def test_parse_source_lag_periods_min_one():
    with pytest.raises(Exception):
        parse_source("lag:engagement:periods:0")


# --- Mission 001a: new column source types accepted in Column validator ---

def test_column_accepts_threshold_source():
    raw = _minimal_valid()
    raw["tables"][1]["columns"].append({
        "name": "flag", "dtype": "boolean",
        "source": "threshold:m1:above:0.5:for:2",
    })
    c = PlotsimConfig(**raw)
    fct = next(t for t in c.tables if t.name == "fct_m1")
    assert any(col.source.startswith("threshold:") for col in fct.columns)


def test_column_accepts_lag_source():
    raw = _minimal_valid()
    raw["tables"][1]["columns"].append({
        "name": "lagged", "dtype": "float",
        "source": "lag:m1:periods:2",
    })
    PlotsimConfig(**raw)  # does not raise


def test_threshold_source_unknown_metric_rejected():
    raw = _minimal_valid()
    raw["tables"][1]["columns"].append({
        "name": "flag", "dtype": "boolean",
        "source": "threshold:ghost:above:0.5:for:2",
    })
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "ghost" in str(exc.value)


def test_lag_source_unknown_metric_rejected():
    raw = _minimal_valid()
    raw["tables"][1]["columns"].append({
        "name": "lagged", "dtype": "float",
        "source": "lag:ghost:periods:2",
    })
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "ghost" in str(exc.value)


# --- Mission 001a: Table.row_count_source ---

def _event_table(**overrides):
    base = {
        "name": "evt_thing",
        "type": "event",
        "grain": "variable",
        "columns": [{"name": "event_id", "dtype": "id", "source": "pk"}],
        "primary_key": "event_id",
    }
    base.update(overrides)
    return base


def test_row_count_source_on_event_table_accepted():
    raw = _minimal_valid()
    raw["tables"].append(_event_table(
        row_count_source="proportional:m1:scale:5",
    ))
    c = PlotsimConfig(**raw)
    evt = next(t for t in c.tables if t.name == "evt_thing")
    assert evt.row_count_source == "proportional:m1:scale:5"


def test_row_count_source_on_non_event_table_rejected():
    raw = _minimal_valid()
    raw["tables"][0]["row_count_source"] = "proportional:m1:scale:5"
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "row_count_source" in str(exc.value)


def test_row_count_source_bad_format_rejected():
    raw = _minimal_valid()
    raw["tables"].append(_event_table(
        row_count_source="not_a_source",
    ))
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_row_count_source_unknown_metric_rejected():
    raw = _minimal_valid()
    raw["tables"].append(_event_table(
        row_count_source="proportional:ghost:scale:5",
    ))
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "ghost" in str(exc.value)


def test_row_count_source_optional():
    # An event table without row_count_source still loads.
    raw = _minimal_valid()
    raw["tables"].append(_event_table())
    PlotsimConfig(**raw)  # no row_count_source — fine


# --- Mission 001a: causal_lag on metrics ---

def _metric(name: str, **overrides) -> dict:
    base = {
        "name": name,
        "label": name.upper(),
        "distribution": "lognorm",
        "params": {"s": 0.5, "scale": 1.0},
        "polarity": "positive",
        "default_curve": "sigmoid",
    }
    base.update(overrides)
    return base


def test_causal_lag_valid():
    raw = _minimal_valid()
    raw["metrics"].append(_metric("m2"))
    raw["metrics"][0]["causal_lag"] = {"driver": "m2", "lag_periods": 2}
    c = PlotsimConfig(**raw)
    m1 = next(m for m in c.metrics if m.name == "m1")
    assert m1.causal_lag is not None
    assert m1.causal_lag.driver == "m2"
    assert m1.causal_lag.lag_periods == 2


def test_causal_lag_unknown_driver_rejected():
    raw = _minimal_valid()
    raw["metrics"][0]["causal_lag"] = {"driver": "ghost", "lag_periods": 1}
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "ghost" in str(exc.value)


def test_causal_lag_self_reference_rejected():
    raw = _minimal_valid()
    raw["metrics"][0]["causal_lag"] = {"driver": "m1", "lag_periods": 1}
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "cannot lag itself" in str(exc.value)


def test_causal_lag_circular_two_cycle_rejected():
    # m1 lags m2, m2 lags m1 — a 2-cycle.
    raw = _minimal_valid()
    raw["metrics"].append(_metric("m2"))
    raw["metrics"][0]["causal_lag"] = {"driver": "m2", "lag_periods": 1}
    raw["metrics"][1]["causal_lag"] = {"driver": "m1", "lag_periods": 1}
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "circular" in str(exc.value)


def test_causal_lag_circular_three_cycle_rejected():
    # m1 → m2 → m3 → m1.
    raw = _minimal_valid()
    raw["metrics"].append(_metric("m2"))
    raw["metrics"].append(_metric("m3"))
    raw["metrics"][0]["causal_lag"] = {"driver": "m2", "lag_periods": 1}
    raw["metrics"][1]["causal_lag"] = {"driver": "m3", "lag_periods": 1}
    raw["metrics"][2]["causal_lag"] = {"driver": "m1", "lag_periods": 1}
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "circular" in str(exc.value)


def test_causal_lag_chain_without_cycle_allowed():
    # m1 → m2 → m3, no back edge. Legal.
    raw = _minimal_valid()
    raw["metrics"].append(_metric("m2"))
    raw["metrics"].append(_metric("m3"))
    raw["metrics"][0]["causal_lag"] = {"driver": "m2", "lag_periods": 1}
    raw["metrics"][1]["causal_lag"] = {"driver": "m3", "lag_periods": 1}
    PlotsimConfig(**raw)


def test_causal_lag_periods_must_be_positive():
    raw = _minimal_valid()
    raw["metrics"].append(_metric("m2"))
    raw["metrics"][0]["causal_lag"] = {"driver": "m2", "lag_periods": 0}
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


# --- Mission 001a: stages top-level section ---

def _valid_stages() -> dict:
    return {
        "field": "m1",
        "sequence": [
            {"name": "s1", "threshold_enter": 0.0, "threshold_exit": 0.3},
            {"name": "s2", "threshold_enter": 0.3, "threshold_exit": 0.7},
            {"name": "s3", "threshold_enter": 0.7, "threshold_exit": None},
        ],
        "enforce_order": True,
    }


def test_stages_valid():
    raw = _minimal_valid()
    raw["stages"] = _valid_stages()
    c = PlotsimConfig(**raw)
    assert c.stages is not None
    assert c.stages.field == "m1"
    assert len(c.stages.sequence) == 3
    assert c.stages.sequence[-1].threshold_exit is None
    assert c.stages.enforce_order is True


def test_stages_optional_omitted():
    raw = _minimal_valid()
    c = PlotsimConfig(**raw)
    assert c.stages is None


def test_stages_field_unknown_metric_rejected():
    raw = _minimal_valid()
    stages = _valid_stages()
    stages["field"] = "ghost"
    raw["stages"] = stages
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "ghost" in str(exc.value)


def test_stages_too_few_rejected():
    raw = _minimal_valid()
    stages = _valid_stages()
    stages["sequence"] = [
        {"name": "only", "threshold_enter": 0.0, "threshold_exit": None},
    ]
    raw["stages"] = stages
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_stages_last_not_terminal_rejected():
    raw = _minimal_valid()
    stages = _valid_stages()
    stages["sequence"][-1]["threshold_exit"] = 0.9
    raw["stages"] = stages
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "terminal" in str(exc.value)


def test_stages_non_terminal_with_null_exit_rejected():
    raw = _minimal_valid()
    stages = _valid_stages()
    stages["sequence"][1]["threshold_exit"] = None
    raw["stages"] = stages
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "terminal" in str(exc.value) or "threshold_exit" in str(exc.value)


def test_stages_overlap_rejected():
    # Stage 1's exit (0.5) is after stage 2's enter (0.3) — overlap.
    raw = _minimal_valid()
    stages = _valid_stages()
    stages["sequence"] = [
        {"name": "s1", "threshold_enter": 0.0, "threshold_exit": 0.5},
        {"name": "s2", "threshold_enter": 0.3, "threshold_exit": 0.7},
        {"name": "s3", "threshold_enter": 0.7, "threshold_exit": None},
    ]
    raw["stages"] = stages
    with pytest.raises(ValidationError) as exc:
        PlotsimConfig(**raw)
    assert "overlap" in str(exc.value)


def test_stages_contiguous_boundaries_allowed():
    # Exit of N == enter of N+1 is a contiguous boundary; allowed.
    raw = _minimal_valid()
    stages = _valid_stages()
    # _valid_stages already has exit_N == enter_(N+1).
    raw["stages"] = stages
    PlotsimConfig(**raw)


def test_stages_enter_ge_exit_within_stage_rejected():
    raw = _minimal_valid()
    stages = _valid_stages()
    stages["sequence"][0] = {
        "name": "s1", "threshold_enter": 0.5, "threshold_exit": 0.3,
    }
    raw["stages"] = stages
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


def test_stages_threshold_out_of_range_rejected():
    raw = _minimal_valid()
    stages = _valid_stages()
    stages["sequence"][0]["threshold_enter"] = 1.5
    raw["stages"] = stages
    with pytest.raises(ValidationError):
        PlotsimConfig(**raw)


# --- Mission 001a: sample configs still load (backwards compat + new fields) ---

def test_saas_sample_loads_new_fields():
    c = load_config(SAAS_YAML)
    # stages present
    assert c.stages is not None
    assert c.stages.field == "churn_risk"
    assert [s.name for s in c.stages.sequence] == [
        "onboarding", "active", "at_risk", "churned",
    ]
    assert c.stages.sequence[-1].threshold_exit is None
    # causal_lag on support_tickets
    support = next(m for m in c.metrics if m.name == "support_tickets")
    assert support.causal_lag is not None
    assert support.causal_lag.driver == "engagement"
    assert support.causal_lag.lag_periods == 2
    # row_count_source on evt_login
    evt_login = next(t for t in c.tables if t.name == "evt_login")
    assert evt_login.row_count_source == "proportional:engagement:scale:5"
    # threshold column on evt_churn
    evt_churn = next(t for t in c.tables if t.name == "evt_churn")
    flag = next(col for col in evt_churn.columns if col.name == "churn_flag")
    assert flag.source == "threshold:churn_risk:above:0.7:for:3"


def test_hr_sample_loads_new_fields():
    c = load_config(HR_YAML)
    assert c.stages is not None
    assert c.stages.field == "attrition_risk"
    assert [s.name for s in c.stages.sequence] == [
        "new_hire", "established", "disengaging", "exited",
    ]
    absence = next(m for m in c.metrics if m.name == "absence_rate")
    assert absence.causal_lag is not None
    assert absence.causal_lag.driver == "engagement_index"
    assert absence.causal_lag.lag_periods == 1


def test_saas_sample_roundtrip_preserves_new_fields():
    c1 = load_config(SAAS_YAML)
    dumped = dump_config(c1)
    c2 = PlotsimConfig(**yaml.safe_load(dumped))
    assert c1 == c2
    assert c2.stages is not None
    assert c2.stages.sequence[-1].threshold_exit is None


def test_hr_sample_roundtrip_preserves_new_fields():
    c1 = load_config(HR_YAML)
    dumped = dump_config(c1)
    c2 = PlotsimConfig(**yaml.safe_load(dumped))
    assert c1 == c2
