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
    FakerSource,
    GeneratedSource,
    LagSource,
    MetricSource,
    PlotsimConfig,
    PERFECTLY_CLEAN,
    PKSource,
    ProportionalSource,
    REALISTIC,
    SLIGHTLY_MESSY,
    StageDefinition,
    StageSequence,
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
    assert DIRTY.gaussian_sigma == 0.10


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


def test_non_psd_correlation_projects_at_load():
    # M111: the classic 3-cycle (A↔B=0.9, A↔C=0.9, B↔C=-0.9) breaks the
    # triangle inequality and is not PD. Pre-M111 (FIX-F04) this raised
    # a pydantic ValidationError at load. M111 replaces the raise with
    # Higham nearest-PD projection: the validator emits a UserWarning
    # listing every adjusted pair, stashes the adjustments on
    # ``config._correlation_adjustments``, and returns a valid config.
    import warnings as _warnings

    raw = _minimal_valid()
    base_metric = raw["metrics"][0]
    raw["metrics"] = [
        {**base_metric, "name": "m_a", "label": "A"},
        {**base_metric, "name": "m_b", "label": "B"},
        {**base_metric, "name": "m_c", "label": "C"},
    ]
    raw["entities"][0]["archetype"] = "a1"
    fact_cols = raw["tables"][1]["columns"]
    fact_cols[2]["name"] = "m_a"
    fact_cols[2]["source"] = "metric:m_a"
    fact_cols.append({"name": "m_b", "dtype": "float", "source": "metric:m_b"})
    fact_cols.append({"name": "m_c", "dtype": "float", "source": "metric:m_c"})
    raw["correlations"] = [
        {"metric_a": "m_a", "metric_b": "m_b", "coefficient": 0.9},
        {"metric_a": "m_a", "metric_b": "m_c", "coefficient": 0.9},
        {"metric_a": "m_b", "metric_b": "m_c", "coefficient": -0.9},
    ]
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        cfg = PlotsimConfig(**raw)
    m111 = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "Correlation matrix was not positive definite" in str(w.message)
    ]
    assert len(m111) == 1
    assert cfg._correlation_adjustments is not None
    assert len(cfg._correlation_adjustments) == 3


def test_valid_psd_correlation_loads():
    # FIX-F04 boundary: a PSD correlation matrix passes load as before.
    # The shipped saas template's correlations satisfy PSD (see 007a in
    # SEQUENCE.md).
    c = load_config(SAAS_YAML)
    assert isinstance(c, PlotsimConfig)
    assert c.correlations


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


def test_parse_source_generated_non_faker():
    assert parse_source("generated:timestamp") == GeneratedSource(
        provider="timestamp"
    )


def test_parse_source_generated_faker_unparameterized():
    # FIX-05: bare faker providers parse as FakerSource with empty kwargs.
    assert parse_source("generated:faker.company") == FakerSource(
        method="company", kwargs={}
    )


def test_parse_source_generated_faker_parameterized():
    # FIX-05 / MF-2: kwargs are parsed out of the colon-delimited grammar.
    parsed = parse_source(
        "generated:faker.date_between:start_date:2020-01-01:end_date:2024-12-31"
    )
    assert parsed == FakerSource(
        method="date_between",
        kwargs={"start_date": "2020-01-01", "end_date": "2024-12-31"},
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


# --- FIX-03 acceptance: Column.pii_note schema field + README PII section ----


def test_pii_note_field_accepted_in_config():
    """FIX-03 / SF-4: Column accepts an optional pii_note metadata string and
    preserves it through load/dump round-trip."""
    data = _minimal_valid()
    data["tables"][0]["columns"].append({
        "name": "full_name",
        "dtype": "string",
        "source": "generated:faker.name",
        "pii_note": "Faker-generated name; may collide with real people.",
    })
    cfg = PlotsimConfig(**data)
    name_col = next(c for c in cfg.tables[0].columns if c.name == "full_name")
    assert name_col.pii_note == (
        "Faker-generated name; may collide with real people."
    )
    # Round-trip through YAML preserves the field.
    dumped = dump_config(cfg)
    cfg2 = PlotsimConfig(**yaml.safe_load(dumped))
    name_col2 = next(c for c in cfg2.tables[0].columns if c.name == "full_name")
    assert name_col2.pii_note == name_col.pii_note


def test_pii_note_defaults_to_none_when_omitted():
    """Backwards-compat: existing configs without pii_note still load."""
    cfg = PlotsimConfig(**_minimal_valid())
    for tbl in cfg.tables:
        for col in tbl.columns:
            assert col.pii_note is None


# --- FIX-04 acceptance: Column.distribution + Entity.cross_dim_fks schema ----


def test_column_distribution_string_uniform_normalizes():
    """FIX-04: bare string 'uniform' coerces to FKDistribution(weights=None)."""
    from plotsim.config import Column, FKDistribution
    col = Column(
        name="plan_id", dtype="id", source="fk:dim_plan.plan_id",
        distribution="uniform",
    )
    assert isinstance(col.distribution, FKDistribution)
    assert col.distribution.weights is None


def test_column_distribution_weighted_loads():
    """FIX-04: weighted distribution preserves keys and values."""
    from plotsim.config import Column
    col = Column(
        name="plan_id", dtype="id", source="fk:dim_plan.plan_id",
        distribution={"weights": {"starter": 0.5, "pro": 0.3, "enterprise": 0.2}},
    )
    assert col.distribution is not None
    assert col.distribution.weights == {
        "starter": 0.5, "pro": 0.3, "enterprise": 0.2,
    }


def test_column_distribution_unknown_string_rejected():
    """FIX-04: any string other than 'uniform' is invalid."""
    from plotsim.config import Column
    with pytest.raises(ValidationError):
        Column(
            name="x", dtype="id", source="fk:y.z",
            distribution="bogus",
        )


def test_fk_distribution_negative_weight_rejected():
    """FIX-04: weights must be non-negative."""
    from plotsim.config import FKDistribution
    with pytest.raises(ValidationError):
        FKDistribution(weights={"a": -1.0, "b": 0.5})


def test_fk_distribution_zero_sum_weights_rejected():
    """FIX-04: weights must sum to a positive value."""
    from plotsim.config import FKDistribution
    with pytest.raises(ValidationError):
        FKDistribution(weights={"a": 0.0, "b": 0.0})


def test_entity_cross_dim_fks_defaults_empty():
    """FIX-04: Entity.cross_dim_fks defaults to {} so existing configs load."""
    from plotsim.config import Entity
    e = Entity(name="x", archetype="a", size=1)
    assert e.cross_dim_fks == {}


def test_entity_cross_dim_fks_accepts_mapping():
    """FIX-04: per-cohort FK pinning round-trips."""
    from plotsim.config import Entity
    e = Entity(
        name="enterprise_accounts", archetype="expansion_champion",
        size=10, cross_dim_fks={"plan_id": "enterprise"},
    )
    assert e.cross_dim_fks == {"plan_id": "enterprise"}


def test_column_distribution_roundtrip_through_yaml():
    """FIX-04: distribution and pii_note both survive dump_config round-trip."""
    data = _minimal_valid()
    data["tables"][0]["columns"].append({
        "name": "plan_id",
        "dtype": "id",
        "source": "fk:dim_widget.widget_id",  # fake FK; cross-ref only checks tables
        "distribution": {"weights": {"a": 0.7, "b": 0.3}},
    })
    cfg = PlotsimConfig(**data)
    dumped = dump_config(cfg)
    cfg2 = PlotsimConfig(**yaml.safe_load(dumped))
    plan_col = next(
        c for c in cfg2.tables[0].columns if c.name == "plan_id"
    )
    assert plan_col.distribution is not None
    assert plan_col.distribution.weights == {"a": 0.7, "b": 0.3}


# --- FIX-05 acceptance: locale + allow_outside_window ------------------------


def test_locale_defaults_to_en_us():
    cfg = PlotsimConfig(**_minimal_valid())
    assert cfg.locale == "en_US"


def test_locale_single_string_accepted():
    data = _minimal_valid()
    data["locale"] = "ja_JP"
    cfg = PlotsimConfig(**data)
    assert cfg.locale == "ja_JP"


def test_locale_list_accepted():
    data = _minimal_valid()
    data["locale"] = ["en_US", "de_DE"]
    cfg = PlotsimConfig(**data)
    assert cfg.locale == ["en_US", "de_DE"]


def test_locale_round_trips_through_yaml():
    data = _minimal_valid()
    data["locale"] = ["en_US", "de_DE"]
    cfg = PlotsimConfig(**data)
    dumped = dump_config(cfg)
    cfg2 = PlotsimConfig(**yaml.safe_load(dumped))
    assert cfg2.locale == ["en_US", "de_DE"]


def test_column_allow_outside_window_defaults_false():
    cfg = PlotsimConfig(**_minimal_valid())
    for tbl in cfg.tables:
        for col in tbl.columns:
            assert col.allow_outside_window is False


def test_column_allow_outside_window_roundtrips():
    data = _minimal_valid()
    data["tables"][0]["columns"].append({
        "name": "birth_date",
        "dtype": "date",
        "source": "generated:faker.date_of_birth",
        "allow_outside_window": True,
    })
    cfg = PlotsimConfig(**data)
    dumped = dump_config(cfg)
    cfg2 = PlotsimConfig(**yaml.safe_load(dumped))
    col = next(c for c in cfg2.tables[0].columns if c.name == "birth_date")
    assert col.allow_outside_window is True


def test_parameterized_faker_rejects_odd_param_list():
    # Grammar requires matched key:value pairs after the method.
    with pytest.raises(ValueError, match="matched 'key:value' pairs"):
        parse_source("generated:faker.date_between:start_date")


# --- FIX-06 acceptance: StageSequence.downgrade_delay ------------------------


def test_downgrade_delay_defaults_to_none():
    seq = StageSequence(
        field="m1",
        sequence=[
            StageDefinition(name="low", threshold_enter=0.0, threshold_exit=0.5),
            StageDefinition(name="high", threshold_enter=0.5, threshold_exit=None),
        ],
    )
    assert seq.downgrade_delay is None


def test_enforce_order_defaults_to_false():
    """``enforce_order`` defaults to False — free-mode per-period assignment.
    Monotonic stage walks must be opted into explicitly with
    ``enforce_order: true``. Irreversible transitions live in SCD Type 2;
    stages reflect current lifecycle state."""
    seq = StageSequence(
        field="m1",
        sequence=[
            StageDefinition(name="low", threshold_enter=0.0, threshold_exit=0.5),
            StageDefinition(name="high", threshold_enter=0.5, threshold_exit=None),
        ],
    )
    assert seq.enforce_order is False


def test_downgrade_delay_accepts_positive_int():
    seq = StageSequence(
        field="m1",
        sequence=[
            StageDefinition(name="low", threshold_enter=0.0, threshold_exit=0.5),
            StageDefinition(name="high", threshold_enter=0.5, threshold_exit=None),
        ],
        downgrade_delay=3,
    )
    assert seq.downgrade_delay == 3


def test_downgrade_delay_rejects_zero_and_negative():
    with pytest.raises(ValidationError):
        StageSequence(
            field="m1",
            sequence=[
                StageDefinition(name="low", threshold_enter=0.0, threshold_exit=0.5),
                StageDefinition(name="high", threshold_enter=0.5, threshold_exit=None),
            ],
            downgrade_delay=0,
        )
    with pytest.raises(ValidationError):
        StageSequence(
            field="m1",
            sequence=[
                StageDefinition(name="low", threshold_enter=0.0, threshold_exit=0.5),
                StageDefinition(name="high", threshold_enter=0.5, threshold_exit=None),
            ],
            downgrade_delay=-1,
        )


# --- SEC-02: Table.name / Column.name identifier validator -------------------

from plotsim.config import Column as _Col  # local alias for the SEC-02 block


def test_column_name_rejects_parent_traversal():
    with pytest.raises(ValidationError, match="column name"):
        _Col(name="../escape", dtype="string", source="pk")


def test_column_name_rejects_absolute_path():
    with pytest.raises(ValidationError, match="column name"):
        _Col(name="/etc/shadow", dtype="string", source="pk")


def test_column_name_rejects_leading_digit():
    with pytest.raises(ValidationError, match="column name"):
        _Col(name="123starts_with_digit", dtype="string", source="pk")


def test_column_name_rejects_overlong():
    with pytest.raises(ValidationError, match="column name"):
        _Col(name="a" * 200, dtype="string", source="pk")


def test_column_name_accepts_snake_case_identifier():
    col = _Col(name="company_id", dtype="id", source="pk")
    assert col.name == "company_id"


def test_table_name_rejects_parent_traversal():
    with pytest.raises(ValidationError, match="table name"):
        Table(
            name="../escape",
            type="dim",
            grain="per_entity",
            columns=[_Col(name="x", dtype="id", source="pk")],
            primary_key="x",
        )


def test_table_name_rejects_absolute_path():
    with pytest.raises(ValidationError, match="table name"):
        Table(
            name="/etc/shadow",
            type="dim",
            grain="per_entity",
            columns=[_Col(name="x", dtype="id", source="pk")],
            primary_key="x",
        )


def test_table_name_rejects_overlong():
    with pytest.raises(ValidationError, match="table name"):
        Table(
            name="t" * 200,
            type="dim",
            grain="per_entity",
            columns=[_Col(name="x", dtype="id", source="pk")],
            primary_key="x",
        )


def test_table_name_rejects_leading_digit():
    with pytest.raises(ValidationError, match="table name"):
        Table(
            name="123bad",
            type="dim",
            grain="per_entity",
            columns=[_Col(name="x", dtype="id", source="pk")],
            primary_key="x",
        )


def test_table_name_accepts_standard_identifier():
    tbl = Table(
        name="dim_company",
        type="dim",
        grain="per_entity",
        columns=[_Col(name="company_id", dtype="id", source="pk")],
        primary_key="company_id",
    )
    assert tbl.name == "dim_company"


def test_all_bundled_templates_pass_name_validation():
    """SEC-02: the five shipped templates must continue to validate cleanly
    after the identifier regex lands — no table or column name needed a rename.
    """
    import warnings as _w
    configs_dir = ROOT / "plotsim" / "configs"
    for stem in ("saas", "hr", "education", "retail", "marketing"):
        path = configs_dir / f"sample_{stem}.yaml"
        with _w.catch_warnings():
            _w.simplefilter("ignore", SurrogateKeyWarning)
            load_config(path)


# --- Category B Layer 1: per-field bounds ------------------------------------

from plotsim.config import (
    Archetype,
    CausalLag,
    CurveSegment,
    Entity,
    NoiseConfig,
    ProportionalSource,
    TimeWindow,
)


def test_entity_size_at_limit_passes():
    e = Entity(name="e", archetype="a", size=5_000)
    assert e.size == 5_000


def test_entity_size_above_limit_fails():
    with pytest.raises(ValidationError, match="size"):
        Entity(name="e", archetype="a", size=5_001)


def test_proportional_source_scale_at_limit_passes():
    ps = ProportionalSource(metric="m", scale=100.0)
    assert ps.scale == 100.0


def test_proportional_source_scale_above_limit_fails():
    with pytest.raises(ValidationError, match="scale"):
        ProportionalSource(metric="m", scale=100.1)


def test_noise_gaussian_sigma_at_limit_passes():
    n = NoiseConfig(gaussian_sigma=5.0)
    assert n.gaussian_sigma == 5.0


def test_noise_gaussian_sigma_above_limit_fails():
    with pytest.raises(ValidationError, match="gaussian_sigma"):
        NoiseConfig(gaussian_sigma=5.1)


def test_causal_lag_periods_at_limit_passes():
    # F10 (M102): field-level cap moved from le=120 to le=10_000.
    # The authoritative per-granularity cap is enforced at the
    # PlotsimConfig level (covered by tests/test_lag_period_cap.py).
    # Direct CausalLag construction still catches obvious garbage
    # (e.g. typos producing five- or six-digit values).
    cl = CausalLag(driver="x", lag_periods=120)
    assert cl.lag_periods == 120
    cl_at_field_cap = CausalLag(driver="x", lag_periods=10_000)
    assert cl_at_field_cap.lag_periods == 10_000


def test_causal_lag_periods_above_limit_fails():
    # F10 (M102): bumped from 121 to 10_001 to match the new
    # field-level sanity cap. The previous 121 boundary moved to
    # the per-granularity model-level cap on PlotsimConfig (verified
    # by tests/test_lag_period_cap.py::test_above_cap_rejected).
    with pytest.raises(ValidationError, match="lag_periods"):
        CausalLag(driver="x", lag_periods=10_001)


def test_downgrade_delay_at_limit_passes():
    ss = StageSequence(
        field="m1",
        sequence=[
            StageDefinition(name="low", threshold_enter=0.0, threshold_exit=0.5),
            StageDefinition(name="high", threshold_enter=0.5, threshold_exit=None),
        ],
        downgrade_delay=120,
    )
    assert ss.downgrade_delay == 120


def test_downgrade_delay_above_limit_fails():
    with pytest.raises(ValidationError, match="downgrade_delay"):
        StageSequence(
            field="m1",
            sequence=[
                StageDefinition(name="low", threshold_enter=0.0, threshold_exit=0.5),
                StageDefinition(name="high", threshold_enter=0.5, threshold_exit=None),
            ],
            downgrade_delay=121,
        )


# --- Layer 1: list-length bounds ---------------------------------------------


def _bulk_metric(i: int) -> dict:
    return {
        "name": f"m{i}",
        "label": f"M{i}",
        "distribution": "lognorm",
        "params": {"s": 0.5, "scale": 1.0},
        "polarity": "positive",
    }


def _bulk_archetype(i: int) -> dict:
    return {
        "name": f"a{i}",
        "label": f"A{i}",
        "description": "x",
        "curve_segments": [
            {"curve": "plateau", "params": {"level": 0.5},
             "start_pct": 0.0, "end_pct": 1.0},
        ],
    }


def test_metrics_list_at_limit_passes():
    raw = _minimal_valid()
    raw["metrics"] = [_bulk_metric(i) for i in range(50)]
    # keep fct_m1 column's metric source pointing at an existing metric
    raw["tables"][1]["columns"][2]["source"] = "metric:m0"
    c = PlotsimConfig(**raw)
    assert len(c.metrics) == 50


def test_metrics_list_above_limit_fails():
    raw = _minimal_valid()
    raw["metrics"] = [_bulk_metric(i) for i in range(51)]
    raw["tables"][1]["columns"][2]["source"] = "metric:m0"
    with pytest.raises(ValidationError, match="metrics"):
        PlotsimConfig(**raw)


def test_archetypes_list_at_limit_passes():
    raw = _minimal_valid()
    raw["archetypes"] = [_bulk_archetype(i) for i in range(20)]
    raw["entities"][0]["archetype"] = "a0"
    c = PlotsimConfig(**raw)
    assert len(c.archetypes) == 20


def test_archetypes_list_above_limit_fails():
    raw = _minimal_valid()
    raw["archetypes"] = [_bulk_archetype(i) for i in range(21)]
    raw["entities"][0]["archetype"] = "a0"
    with pytest.raises(ValidationError, match="archetypes"):
        PlotsimConfig(**raw)


def test_entities_list_at_limit_passes():
    raw = _minimal_valid()
    raw["entities"] = [
        {"name": f"e{i}", "archetype": "a1", "size": 1} for i in range(100)
    ]
    c = PlotsimConfig(**raw)
    assert len(c.entities) == 100


def test_entities_list_above_limit_fails():
    # M117: cap raised from 100 → 100,000 to accommodate per-segment
    # expansion in the builder path. The Pydantic length check fires
    # before the cell-count gate, so this test never materialises a
    # generation envelope — only the list length matters.
    raw = _minimal_valid()
    raw["entities"] = [
        {"name": f"e{i}", "archetype": "a1", "size": 1} for i in range(100_001)
    ]
    with pytest.raises(ValidationError, match="entities"):
        PlotsimConfig(**raw)


def test_empty_entities_rejected():
    # R-05 / FIX-F05: an empty entities list is almost certainly a config
    # typo and previously produced zero-row fact tables silently. The field
    # now carries min_length=1 so this raises at load time like every other
    # structural defect.
    raw = _minimal_valid()
    raw["entities"] = []
    with pytest.raises(ValidationError, match="entities"):
        PlotsimConfig(**raw)


def test_tables_list_at_limit_passes():
    # 3 load-bearing tables from _minimal_valid + 47 extra dim tables = 50
    raw = _minimal_valid()
    extras = [
        {
            "name": f"dim_extra_{i}",
            "type": "dim",
            "grain": "per_reference",
            "columns": [{"name": "extra_id", "dtype": "id", "source": "pk"}],
            "primary_key": "extra_id",
        }
        for i in range(47)
    ]
    raw["tables"].extend(extras)
    c = PlotsimConfig(**raw)
    assert len(c.tables) == 50


def test_tables_list_above_limit_fails():
    raw = _minimal_valid()
    extras = [
        {
            "name": f"dim_extra_{i}",
            "type": "dim",
            "grain": "per_reference",
            "columns": [{"name": "extra_id", "dtype": "id", "source": "pk"}],
            "primary_key": "extra_id",
        }
        for i in range(48)
    ]
    raw["tables"].extend(extras)
    with pytest.raises(ValidationError, match="tables"):
        PlotsimConfig(**raw)


def test_r09_redundant_zero_correlation_warns():
    """R-09 / F-01: an explicit ``coefficient: 0.0`` entry emits
    ``RedundantCorrelationWarning`` but load still succeeds — unlisted pairs
    are already zero by default, so the entry is a no-op. We warn instead of
    rejecting because the entry remains structurally valid and rejecting a
    previously-accepted config would be a silent breaking change.
    """
    from plotsim.config import RedundantCorrelationWarning
    raw = _minimal_valid()
    # Add a second metric so the zero-correlation pair references two known
    # names; the _minimal_valid fixture only ships one.
    raw["metrics"].append({
        "name": "m2", "label": "M2", "distribution": "lognorm",
        "params": {"s": 0.5, "scale": 1.0}, "polarity": "positive",
    })
    raw["correlations"] = [
        {"metric_a": "m1", "metric_b": "m2", "coefficient": 0.0},
    ]
    with pytest.warns(RedundantCorrelationWarning, match="already the default"):
        cfg = PlotsimConfig(**raw)
    assert len(cfg.correlations) == 1
    assert cfg.correlations[0].coefficient == 0.0


def test_correlations_list_above_limit_fails():
    raw = _minimal_valid()
    raw["metrics"] = [_bulk_metric(i) for i in range(50)]
    raw["tables"][1]["columns"][2]["source"] = "metric:m0"
    # 1226 correlation entries (> 1225 cap).
    raw["correlations"] = [
        {"metric_a": "m0", "metric_b": "m1", "coefficient": 0.0}
        for _ in range(1_226)
    ]
    with pytest.raises(ValidationError, match="correlations"):
        PlotsimConfig(**raw)


def test_table_columns_at_limit_passes():
    # Dim table with 100 columns (1 PK + 99 static) should pass.
    raw = _minimal_valid()
    extra_cols = [
        {"name": f"col_{i}", "dtype": "string", "source": "static:x"}
        for i in range(99)
    ]
    raw["tables"][0]["columns"].extend(extra_cols)
    c = PlotsimConfig(**raw)
    assert len(c.tables[0].columns) == 100


def test_table_columns_above_limit_fails():
    raw = _minimal_valid()
    extra_cols = [
        {"name": f"col_{i}", "dtype": "string", "source": "static:x"}
        for i in range(100)
    ]
    raw["tables"][0]["columns"].extend(extra_cols)
    with pytest.raises(ValidationError, match="columns"):
        PlotsimConfig(**raw)


def test_archetype_curve_segments_at_limit_passes():
    segs = []
    step = 1.0 / 10
    for i in range(10):
        segs.append(CurveSegment(
            curve="plateau", params={"level": 0.5},
            start_pct=i * step, end_pct=(i + 1) * step,
        ))
    a = Archetype(name="a", label="A", description="x", curve_segments=segs)
    assert len(a.curve_segments) == 10


def test_archetype_curve_segments_above_limit_fails():
    segs = []
    step = 1.0 / 11
    for i in range(11):
        segs.append({
            "curve": "plateau", "params": {"level": 0.5},
            "start_pct": i * step, "end_pct": (i + 1) * step,
        })
    with pytest.raises(ValidationError, match="curve_segments"):
        Archetype(
            name="a", label="A", description="x", curve_segments=segs,
        )


def test_stage_sequence_at_limit_passes():
    # 10 stages: 9 non-terminal (enter in [0.0, 0.9) with overlap-safe exits)
    # + 1 terminal at 0.9.
    stages = []
    for i in range(9):
        stages.append(StageDefinition(
            name=f"s{i}", threshold_enter=i / 10,
            threshold_exit=(i + 1) / 10,
        ))
    stages.append(StageDefinition(
        name="terminal", threshold_enter=0.9, threshold_exit=None,
    ))
    ss = StageSequence(field="m1", sequence=stages)
    assert len(ss.sequence) == 10


def test_stage_sequence_above_limit_fails():
    stages = []
    for i in range(10):
        stages.append(StageDefinition(
            name=f"s{i}", threshold_enter=i / 11,
            threshold_exit=(i + 1) / 11,
        ))
    stages.append(StageDefinition(
        name="terminal", threshold_enter=10 / 11, threshold_exit=None,
    ))
    with pytest.raises(ValidationError, match="sequence"):
        StageSequence(field="m1", sequence=stages)


# --- Layer 1: TimeWindow span + total entity size ----------------------------


def test_time_window_monthly_at_limit_passes():
    tw = TimeWindow(start="2020-01", end="2049-12", granularity="monthly")
    assert tw.period_count() == 360


def test_time_window_monthly_above_limit_fails():
    with pytest.raises(ValidationError, match="360"):
        TimeWindow(start="2020-01", end="2050-01", granularity="monthly")


def test_time_window_daily_above_limit_fails():
    # 2020-01 to 2029-12 daily is 3,653 periods — one over the 3,650 cap.
    with pytest.raises(ValidationError, match="daily"):
        TimeWindow(start="2020-01", end="2029-12", granularity="daily")


def test_time_window_weekly_above_limit_fails():
    # 2020-01 to 2049-12 weekly overflows the 1,560 cap.
    with pytest.raises(ValidationError, match="weekly"):
        TimeWindow(start="2020-01", end="2049-12", granularity="weekly")


def test_time_window_daily_at_limit_passes():
    # Pick an end month where the inclusive daily span is <= 3,650.
    tw = TimeWindow(start="2020-01", end="2029-10", granularity="daily")
    assert tw.period_count() <= 3_650


def test_time_window_period_count_matches_trajectory_engine():
    # Parity check against compute_time_steps — the whole point of the span
    # validator is that it rejects what the engine would actually build.
    from plotsim.trajectory import compute_time_steps
    for start, end, gran in (
        ("2020-01", "2022-06", "monthly"),
        ("2020-01", "2020-12", "daily"),
        ("2020-01", "2021-06", "weekly"),
    ):
        tw = TimeWindow(start=start, end=end, granularity=gran)
        assert tw.period_count() == len(compute_time_steps(tw))


def test_total_entity_size_at_limit_passes():
    raw = _minimal_valid()
    # 100 cohorts × 1000 each = 100,000 at the limit.
    raw["entities"] = [
        {"name": f"e{i}", "archetype": "a1", "size": 1000} for i in range(100)
    ]
    c = PlotsimConfig(**raw)
    assert sum(e.size for e in c.entities) == 100_000


def test_total_entity_size_above_limit_fails():
    raw = _minimal_valid()
    # 100 cohorts × 1001 each = 100,100, just over.
    raw["entities"] = [
        {"name": f"e{i}", "archetype": "a1", "size": 1001} for i in range(100)
    ]
    with pytest.raises(ValidationError, match="100,000"):
        PlotsimConfig(**raw)


def test_all_bundled_templates_pass_layer1_bounds():
    # The five shipped templates were audited against every Layer 1 bound
    # before the cap landed. Regression test that they keep loading cleanly.
    import warnings as _w
    configs_dir = ROOT / "plotsim" / "configs"
    for stem in ("saas", "hr", "education", "retail", "marketing"):
        with _w.catch_warnings():
            _w.simplefilter("ignore", SurrogateKeyWarning)
            load_config(configs_dir / f"sample_{stem}.yaml")


# --- Category B Layer 2: config-time estimator -------------------------------


def _cells(n_entities_total: int, n_periods: int, raw: "dict | None" = None) -> dict:
    """Shape ``_minimal_valid`` so that sum(entity.size) × period_count hits
    ``n_entities_total * n_periods``. Uses 50 cohort groups × size=K so the
    individual bounds (100 cohort cap, 5_000 size cap) remain satisfied for
    any totals <= 250_000.
    """
    if raw is None:
        raw = _minimal_valid()
    k_cohorts = 50
    # Ceiling division so the actual total is >= n_entities_total (tests that
    # want "above threshold" stay above after rounding).
    per_cohort = max(1, -(-n_entities_total // k_cohorts))
    raw["entities"] = [
        {"name": f"e{i}", "archetype": "a1", "size": per_cohort}
        for i in range(k_cohorts)
    ]
    # Monthly span sized so period_count == n_periods.
    # _start_before_end requires start < end, so n_periods >= 2.
    assert n_periods >= 2
    months_past = n_periods - 1
    start_year = 2020
    start_month = 1
    end_month_total = start_month - 1 + months_past
    end_year = start_year + end_month_total // 12
    end_month = end_month_total % 12 + 1
    raw["time_window"] = {
        "start": f"{start_year}-{start_month:02d}",
        "end": f"{end_year}-{end_month:02d}",
        "granularity": "monthly",
    }
    return raw


def test_estimator_prints_summary_at_any_scale(capsys):
    PlotsimConfig(**_minimal_valid())
    captured = capsys.readouterr()
    assert "Config summary" in captured.err
    assert "cells" in captured.err
    assert captured.out == ""  # must NOT go to stdout


def test_estimator_summary_not_on_stdout(capsys):
    PlotsimConfig(**_minimal_valid())
    out = capsys.readouterr().out
    assert "Config summary" not in out


def test_estimator_below_warn_threshold_no_warning(capsys):
    # 499,950 cells — below the 500,000 warn threshold.
    raw = _cells(n_entities_total=25_000, n_periods=20)
    PlotsimConfig(**raw)
    err = capsys.readouterr().err
    assert "Config summary" in err
    assert "Warning" not in err


def test_estimator_warns_above_500k(capsys):
    # 500,050 cells — one small step over the warn threshold.
    raw = _cells(n_entities_total=25_002, n_periods=20)
    PlotsimConfig(**raw)
    err = capsys.readouterr().err
    assert "Warning" in err
    # 25_002 rounded up: 50 cohorts × 500 = 25_000 (integer div) — not exact.
    # Use the actually-emitted cell count from the summary; the warning line
    # should mention the threshold or the cell count.
    assert "500,000" in err or "cells" in err


def test_estimator_passes_at_2m_cells_boundary(capsys):
    # Exactly 2_000_000 cells — boundary is exclusive (reject above, not at).
    # 50 cohorts × 400 = 20_000 entities; 20_000 × 100 = 2_000_000.
    raw = _cells(n_entities_total=20_000, n_periods=100)
    PlotsimConfig(**raw)
    err = capsys.readouterr().err
    assert "Config summary" in err
    assert "exceeds the maximum" not in err


def test_estimator_rejects_above_2m_cells():
    # 2_100_000 cells — 50 cohorts × 420 = 21_000 entities; 21_000 × 100.
    raw = _cells(n_entities_total=21_000, n_periods=100)
    with pytest.raises(ValidationError, match="2,000,000"):
        PlotsimConfig(**raw)


def test_estimator_event_upper_bound_prints_when_derivable(capsys):
    # Sample SaaS declares a proportional event with a ValueRange-bearing
    # driver; its summary line should include an event-row estimate.
    from plotsim.config import load_config as _lc
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", SurrogateKeyWarning)
        _lc(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
    err = capsys.readouterr().err
    assert "Expected event rows" in err


def test_estimator_event_upper_bound_absent_when_no_range(capsys):
    # _minimal_valid has no event tables, so no event estimate should appear.
    PlotsimConfig(**_minimal_valid())
    err = capsys.readouterr().err
    assert "Expected event rows" not in err


def test_estimator_all_bundled_templates_have_no_warning(capsys):
    import warnings as _w
    configs_dir = ROOT / "plotsim" / "configs"
    for stem in ("saas", "hr", "education", "retail", "marketing"):
        with _w.catch_warnings():
            _w.simplefilter("ignore", SurrogateKeyWarning)
            load_config(configs_dir / f"sample_{stem}.yaml")
    err = capsys.readouterr().err
    assert "Warning:" not in err
    assert "exceeds the maximum" not in err
