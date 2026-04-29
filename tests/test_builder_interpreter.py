"""Interpreter translation tests.

Mirrors the acceptance criteria in mission-115-builder.md ::Interpreter::
section. Each test pinpoints one translation rule (recipe pick, override
threading, schema vocabulary, FK/PK convention) so a regression localises
to the exact step that broke.

The interpreter wraps PlotsimConfig.model_validate at the end — passing
construction here means the engine itself accepts the shape, not just
that the interpreter returned a python object.
"""
from __future__ import annotations

import warnings
from typing import Any

import pytest

from plotsim.builder.input import UserInput
from plotsim.builder.interpreter import (
    UNIT_FAKER_MAP,
    interpret,
)
from plotsim.builder.recipes import (
    AMOUNT_BETA_PARAMS,
    AMOUNT_LOGNORM_S,
    INDEX_DISTRIBUTION,
    INDEX_SIGMA_FRACTION,
    METRIC_RECIPES,
    RELATIONSHIP_RECIPES,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _minimal_input(**overrides: Any) -> UserInput:
    base: dict[str, Any] = {
        "about": "test domain",
        "unit": "company",
        "window": {"start": "2023-01", "end": "2024-12", "every": "monthly"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {"name": "mrr", "type": "amount", "polarity": "positive",
             "range": [100, 50000]},
        ],
        "segments": [
            {"name": "alpha", "count": 10, "archetype": "growth"},
            {"name": "beta", "count": 10, "archetype": "decline"},
        ],
    }
    base.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence semantic warnings during tests
        return UserInput.model_validate(base)


# ── Bare-minimum acceptance ─────────────────────────────────────────────────


def test_bare_minimum_input_produces_valid_plotsim_config():
    cfg = interpret(_minimal_input())
    # PlotsimConfig validation already ran inside interpret(); reaching here
    # means the engine accepted the shape.
    assert cfg.domain.entity_type == "company"
    assert len(cfg.metrics) == 2
    assert len(cfg.archetypes) == 2
    assert sum(e.size for e in cfg.entities) == 20


# ── Auto-generated schema ───────────────────────────────────────────────────


def test_auto_generated_schema_contains_dim_date_dim_unit_fct_unit():
    cfg = interpret(_minimal_input())
    table_names = [t.name for t in cfg.tables]
    assert "dim_date" in table_names
    assert "dim_company" in table_names
    assert "fct_company" in table_names


def test_auto_generated_fact_table_carries_all_metrics():
    cfg = interpret(_minimal_input())
    fact = next(t for t in cfg.tables if t.name == "fct_company")
    metric_cols = {c.name for c in fact.columns if c.source.startswith("metric:")}
    assert metric_cols == {"engagement", "mrr"}


def test_auto_generated_dim_unit_uses_unit_faker_map_for_company():
    cfg = interpret(_minimal_input(unit="company"))
    dim = next(t for t in cfg.tables if t.name == "dim_company")
    name_col = next(c for c in dim.columns if c.name == "company_name")
    assert "faker.company" in name_col.source


def test_auto_generated_dim_unit_uses_unit_faker_map_for_employee():
    cfg = interpret(_minimal_input(unit="employee"))
    dim = next(t for t in cfg.tables if t.name == "dim_employee")
    name_col = next(c for c in dim.columns if c.name == "employee_name")
    assert "faker.name" in name_col.source


def test_auto_generated_dim_unit_uses_unit_faker_map_for_customer():
    cfg = interpret(_minimal_input(unit="customer"))
    dim = next(t for t in cfg.tables if t.name == "dim_customer")
    name_col = next(c for c in dim.columns if c.name == "customer_name")
    assert "faker.name" in name_col.source


def test_auto_generated_dim_unit_unknown_unit_falls_back_to_company():
    cfg = interpret(_minimal_input(unit="rocketship"))
    dim = next(t for t in cfg.tables if t.name == "dim_rocketship")
    name_col = next(c for c in dim.columns if c.name == "rocketship_name")
    assert "faker.company" in name_col.source


def test_unit_faker_map_documents_known_units():
    assert "company" in UNIT_FAKER_MAP
    assert "employee" in UNIT_FAKER_MAP
    assert "customer" in UNIT_FAKER_MAP


# ── Metric type → distribution recipes ──────────────────────────────────────


def test_score_metric_picks_beta():
    ui = _minimal_input(metrics=[
        {"name": "x", "type": "score", "polarity": "positive"},
        {"name": "y", "type": "score", "polarity": "negative"},
    ])
    cfg = interpret(ui)
    m = next(m for m in cfg.metrics if m.name == "x")
    assert m.distribution == "beta"
    assert m.params == METRIC_RECIPES["score"]["params"]
    assert m.value_range.min == 0.0
    assert m.value_range.max == 1.0


def test_count_metric_picks_poisson_with_no_value_range():
    ui = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "tickets", "type": "count", "polarity": "negative"},
    ])
    cfg = interpret(ui)
    m = next(m for m in cfg.metrics if m.name == "tickets")
    assert m.distribution == "poisson"
    assert m.value_range is None


def test_index_metric_picks_normal_with_mu_at_midpoint():
    ui = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "nps", "type": "index", "polarity": "positive",
         "range": [-100, 100]},
    ])
    cfg = interpret(ui)
    m = next(m for m in cfg.metrics if m.name == "nps")
    assert m.distribution == INDEX_DISTRIBUTION
    assert m.params["mu"] == pytest.approx(0.0)
    # sigma = (max-min) * INDEX_SIGMA_FRACTION = 200/6
    assert m.params["sigma"] == pytest.approx(200.0 * INDEX_SIGMA_FRACTION)


def test_amount_with_high_ratio_picks_lognorm():
    # ratio = 50000 / 100 = 500× → lognorm
    ui = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "mrr", "type": "amount", "polarity": "positive",
         "range": [100, 50000]},
    ])
    cfg = interpret(ui)
    m = next(m for m in cfg.metrics if m.name == "mrr")
    assert m.distribution == "lognorm"
    assert m.params["s"] == AMOUNT_LOGNORM_S
    assert m.params["scale"] == pytest.approx(25050.0)  # midpoint


def test_amount_with_min_zero_picks_lognorm():
    ui = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "revenue", "type": "amount", "polarity": "positive",
         "range": [0, 10000]},
    ])
    cfg = interpret(ui)
    m = next(m for m in cfg.metrics if m.name == "revenue")
    assert m.distribution == "lognorm"


def test_amount_with_low_ratio_picks_beta():
    # ratio = 500 / 100 = 5× → below threshold (10) → beta
    ui = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "ticket_size", "type": "amount", "polarity": "positive",
         "range": [100, 500]},
    ])
    cfg = interpret(ui)
    m = next(m for m in cfg.metrics if m.name == "ticket_size")
    assert m.distribution == "beta"
    assert m.params == AMOUNT_BETA_PARAMS


# ── Causal lag threading ────────────────────────────────────────────────────


def test_follows_delay_translates_to_causal_lag():
    ui = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "tickets", "type": "count", "polarity": "negative",
         "follows": "engagement", "delay": 2},
    ])
    cfg = interpret(ui)
    m = next(m for m in cfg.metrics if m.name == "tickets")
    assert m.causal_lag is not None
    assert m.causal_lag.driver == "engagement"
    assert m.causal_lag.lag_periods == 2


def test_no_follows_means_no_causal_lag():
    cfg = interpret(_minimal_input())
    for m in cfg.metrics:
        assert m.causal_lag is None


# ── Baseline → MetricOverride.value_range ───────────────────────────────────


def test_baseline_high_restricts_value_range_to_upper_third():
    ui = _minimal_input(segments=[
        {"name": "alpha", "count": 10, "archetype": "growth",
         "baseline": {"mrr": "high"}},
        {"name": "beta", "count": 10, "archetype": "decline"},
    ])
    cfg = interpret(ui)
    arc = next(a for a in cfg.archetypes if a.name == "alpha")
    override = arc.metric_overrides["mrr"]
    # mrr range = [100, 50000]; high = upper third → [33,400; 50,000]
    assert override.value_range is not None
    assert override.value_range.min == pytest.approx(100 + 49900 * 2 / 3)
    assert override.value_range.max == pytest.approx(50000.0)


def test_baseline_low_restricts_value_range_to_lower_third():
    ui = _minimal_input(segments=[
        {"name": "alpha", "count": 10, "archetype": "growth",
         "baseline": {"engagement": "low"}},
        {"name": "beta", "count": 10, "archetype": "decline"},
    ])
    cfg = interpret(ui)
    arc = next(a for a in cfg.archetypes if a.name == "alpha")
    override = arc.metric_overrides["engagement"]
    # engagement range = [0, 1]; low = [0, 1/3]
    assert override.value_range.min == pytest.approx(0.0)
    assert override.value_range.max == pytest.approx(1.0 / 3.0)


def test_baseline_on_count_metric_silently_skipped():
    # count metrics have no value_range; baseline label can't restrict
    # what isn't there. The interpreter skips rather than raising.
    ui = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "tickets", "type": "count", "polarity": "negative"},
    ], segments=[
        {"name": "alpha", "count": 10, "archetype": "growth",
         "baseline": {"tickets": "high", "engagement": "low"}},
        {"name": "beta", "count": 10, "archetype": "decline"},
    ])
    cfg = interpret(ui)
    arc = next(a for a in cfg.archetypes if a.name == "alpha")
    assert "tickets" not in arc.metric_overrides
    assert "engagement" in arc.metric_overrides


# ── Connections → CorrelationPair ──────────────────────────────────────────


def test_connection_translates_to_correlation_pair_with_recipe_coefficient():
    ui = _minimal_input(connections=[
        "engagement driven_by mrr",
    ])
    cfg = interpret(ui)
    assert len(cfg.correlations) == 1
    pair = cfg.correlations[0]
    assert pair.metric_a == "engagement"
    assert pair.metric_b == "mrr"
    assert pair.coefficient == RELATIONSHIP_RECIPES["driven_by"]


def test_negative_connection_produces_negative_coefficient():
    ui = _minimal_input(connections=["engagement opposes mrr"])
    cfg = interpret(ui)
    assert cfg.correlations[0].coefficient < 0


def test_independent_connection_skipped_to_avoid_redundant_zero_warning():
    ui = _minimal_input(connections=["engagement independent mrr"])
    cfg = interpret(ui)
    assert cfg.correlations == []


# ── Lifecycle → StageSequence ──────────────────────────────────────────────


def test_lifecycle_translates_to_stage_sequence_with_enforce_order_false():
    ui = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "churn_risk", "type": "score", "polarity": "negative"},
    ], lifecycle={
        "track": "churn_risk",
        "stages": [
            {"onboarding": 0.0},
            {"active": 0.2},
            {"at_risk": 0.5},
            {"churned": 0.8},
        ],
    })
    cfg = interpret(ui)
    assert cfg.stages is not None
    assert cfg.stages.field == "churn_risk"
    assert cfg.stages.enforce_order is False
    assert [s.name for s in cfg.stages.sequence] == [
        "onboarding", "active", "at_risk", "churned",
    ]
    # Legacy mode: each non-terminal stage's threshold_exit equals the
    # next stage's threshold_enter; terminal stage has threshold_exit=None.
    assert cfg.stages.sequence[0].threshold_enter == 0.0
    assert cfg.stages.sequence[0].threshold_exit == 0.2
    assert cfg.stages.sequence[-1].threshold_exit is None


def test_no_lifecycle_means_no_stages():
    cfg = interpret(_minimal_input())
    assert cfg.stages is None


# ── Schema vocabulary translation ───────────────────────────────────────────


def _saas_like_input(**overrides: Any) -> UserInput:
    """An input with explicit dimensions/facts/events to exercise the
    vocabulary translator."""
    base: dict[str, Any] = {
        "about": "test",
        "unit": "company",
        "window": {"start": "2023-01", "end": "2024-12", "every": "monthly"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {"name": "mrr", "type": "amount", "polarity": "positive",
             "range": [100, 50000]},
            {"name": "churn_risk", "type": "score", "polarity": "negative"},
        ],
        "segments": [
            {"name": "alpha", "count": 10, "archetype": "growth"},
            {"name": "beta", "count": 10, "archetype": "decline"},
        ],
        "dimensions": [
            {"name": "dim_date", "per": "period", "columns": [
                {"name": "date_key", "type": "id"},
                {"name": "date", "type": "date"},
                {"name": "year", "type": "int"},
            ]},
            {"name": "dim_company", "per": "unit", "columns": [
                {"name": "company_id", "type": "id"},
                {"name": "company_name", "type": "faker.company"},
                {"name": "cohort_size", "type": "segment.count"},
                {"name": "plan_tier", "type": "scd",
                 "tracks": "mrr",
                 "tiers": ["starter", "growth", "enterprise"],
                 "at": [0.4, 0.7]},
            ]},
            {"name": "dim_plan", "reference": True, "columns": [
                {"name": "plan_id", "type": "id"},
                {"name": "plan_name", "type": "static.starter"},
                {"name": "monthly_price", "type": "static.99.00"},
            ]},
        ],
        "facts": [
            {"name": "fct_revenue", "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "plan_id", "type": "ref.dim_plan"},
                {"name": "mrr", "type": "metric.mrr"},
            ]},
            {"name": "fct_engagement", "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "engagement_score", "type": "metric.engagement"},
                {"name": "customer_sentiment", "type": "bucket",
                 "labels": ["at_risk", "satisfied", "delighted"]},
            ]},
        ],
        "events": [
            {"name": "evt_churn",
             "trigger": "threshold", "metric": "churn_risk",
             "above": 0.7, "for": 3,
             "columns": [
                 {"name": "event_id", "type": "id"},
                 {"name": "date_key", "type": "ref.dim_date"},
                 {"name": "company_id", "type": "ref.dim_company"},
                 {"name": "churn_reason", "type": "faker.sentence"},
                 {"name": "churn_flag", "type": "flag"},
             ]},
            {"name": "evt_login",
             "trigger": "proportional", "driver": "engagement", "scale": 5,
             "columns": [
                 {"name": "event_id", "type": "id"},
                 {"name": "date_key", "type": "ref.dim_date"},
                 {"name": "company_id", "type": "ref.dim_company"},
                 {"name": "event_ts", "type": "timestamp"},
             ]},
        ],
    }
    base.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return UserInput.model_validate(base)


def test_id_column_translates_to_pk():
    cfg = interpret(_saas_like_input())
    dim = next(t for t in cfg.tables if t.name == "dim_company")
    pk_col = next(c for c in dim.columns if c.name == "company_id")
    assert pk_col.dtype == "id"
    assert pk_col.source == "pk"


def test_ref_column_translates_to_fk_with_target_pk_resolved():
    cfg = interpret(_saas_like_input())
    fact = next(t for t in cfg.tables if t.name == "fct_revenue")
    fk_col = next(c for c in fact.columns if c.name == "company_id")
    assert fk_col.dtype == "id"
    assert fk_col.source == "fk:dim_company.company_id"


def test_metric_column_translates_with_correct_dtype_for_count_metric():
    ui = _minimal_input(metrics=[
        {"name": "engagement", "type": "score", "polarity": "positive"},
        {"name": "tickets", "type": "count", "polarity": "negative"},
    ], dimensions=[
        {"name": "dim_date", "per": "period", "columns": [
            {"name": "date_key", "type": "id"},
            {"name": "date", "type": "date"},
        ]},
        {"name": "dim_company", "per": "unit", "columns": [
            {"name": "company_id", "type": "id"},
        ]},
    ], facts=[
        {"name": "fct_x", "columns": [
            {"name": "date_key", "type": "ref.dim_date"},
            {"name": "company_id", "type": "ref.dim_company"},
            {"name": "ticket_count", "type": "metric.tickets"},
            {"name": "engagement_score", "type": "metric.engagement"},
        ]},
    ])
    cfg = interpret(ui)
    fact = next(t for t in cfg.tables if t.name == "fct_x")
    ticket_col = next(c for c in fact.columns if c.name == "ticket_count")
    eng_col = next(c for c in fact.columns if c.name == "engagement_score")
    assert ticket_col.dtype == "int"   # poisson → int
    assert eng_col.dtype == "float"   # beta → float


def test_faker_column_special_case_year_dtype_int():
    ui = _saas_like_input()
    cfg = interpret(ui)
    # The integration template uses faker.sentence and faker.company; neither
    # is "year". Build a dedicated minimal fixture instead.
    extra = _minimal_input(dimensions=[
        {"name": "dim_date", "per": "period", "columns": [
            {"name": "date_key", "type": "id"},
            {"name": "date", "type": "date"},
        ]},
        {"name": "dim_company", "per": "unit", "columns": [
            {"name": "company_id", "type": "id"},
            {"name": "founded_year", "type": "faker.year"},
            {"name": "company_name", "type": "faker.company"},
        ]},
    ], facts=[
        {"name": "fct_x", "columns": [
            {"name": "date_key", "type": "ref.dim_date"},
            {"name": "company_id", "type": "ref.dim_company"},
            {"name": "engagement_score", "type": "metric.engagement"},
            {"name": "mrr", "type": "metric.mrr"},
        ]},
    ])
    cfg2 = interpret(extra)
    dim = next(t for t in cfg2.tables if t.name == "dim_company")
    yr_col = next(c for c in dim.columns if c.name == "founded_year")
    assert yr_col.dtype == "int"
    assert yr_col.source == "generated:faker.year"
    nm_col = next(c for c in dim.columns if c.name == "company_name")
    assert nm_col.dtype == "string"


def test_static_column_numeric_dtype_float():
    cfg = interpret(_saas_like_input())
    dim = next(t for t in cfg.tables if t.name == "dim_plan")
    price_col = next(c for c in dim.columns if c.name == "monthly_price")
    assert price_col.dtype == "float"
    assert price_col.source == "static:99.00"


def test_static_column_string_value_dtype_string():
    cfg = interpret(_saas_like_input())
    dim = next(t for t in cfg.tables if t.name == "dim_plan")
    name_col = next(c for c in dim.columns if c.name == "plan_name")
    assert name_col.dtype == "string"
    assert name_col.source == "static:starter"


def test_segment_count_column_translates_to_derived_size():
    cfg = interpret(_saas_like_input())
    dim = next(t for t in cfg.tables if t.name == "dim_company")
    cohort_col = next(c for c in dim.columns if c.name == "cohort_size")
    assert cohort_col.dtype == "int"
    assert cohort_col.source == "derived:size"


def test_timestamp_column_translates_correctly():
    cfg = interpret(_saas_like_input())
    evt = next(t for t in cfg.tables if t.name == "evt_login")
    ts_col = next(c for c in evt.columns if c.name == "event_ts")
    assert ts_col.dtype == "date"
    assert ts_col.source == "generated:timestamp"


def test_dim_date_dtype_words_translate_with_generated_date_key_source():
    cfg = interpret(_saas_like_input())
    dim_date = next(t for t in cfg.tables if t.name == "dim_date")
    date_col = next(c for c in dim_date.columns if c.name == "date")
    year_col = next(c for c in dim_date.columns if c.name == "year")
    assert date_col.dtype == "date"
    assert date_col.source == "generated:date_key"
    assert year_col.dtype == "int"
    assert year_col.source == "generated:date_key"


def test_dtype_word_outside_dim_date_rejected():
    # `int` on a non-dim_date column has no source; reject with a guidance.
    ui_dict = {
        "about": "test", "unit": "company",
        "window": {"start": "2023-01", "end": "2024-12"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
        ],
        "segments": [
            {"name": "a", "count": 10, "archetype": "growth"},
            {"name": "b", "count": 10, "archetype": "decline"},
        ],
        "dimensions": [
            {"name": "dim_date", "per": "period", "columns": [
                {"name": "date_key", "type": "id"},
                {"name": "date", "type": "date"},
            ]},
            {"name": "dim_company", "per": "unit", "columns": [
                {"name": "company_id", "type": "id"},
                {"name": "anomaly_count", "type": "int"},  # bare dtype here
            ]},
        ],
    }
    with pytest.raises(ValueError, match="dtype-only type"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interpret(UserInput.model_validate(ui_dict))


def test_bucket_column_translates_with_inline_labels():
    cfg = interpret(_saas_like_input())
    fact = next(t for t in cfg.tables if t.name == "fct_engagement")
    bucket_col = next(c for c in fact.columns if c.name == "customer_sentiment")
    assert bucket_col.dtype == "string"
    assert bucket_col.source.startswith("text:bucket:[")
    assert "at_risk" in bucket_col.source
    assert "delighted" in bucket_col.source


def test_scd_column_resolves_trigger_metric_to_fact_table():
    cfg = interpret(_saas_like_input())
    dim = next(t for t in cfg.tables if t.name == "dim_company")
    scd_col = next(c for c in dim.columns if c.name == "plan_tier")
    assert scd_col.dtype == "string"
    assert scd_col.source == "scd_type2"
    assert scd_col.scd_type2 is not None
    # mrr is emitted by fct_revenue → trigger_metric = "fct_revenue.mrr"
    assert scd_col.scd_type2.trigger_metric == "fct_revenue.mrr"
    assert scd_col.scd_type2.thresholds == (0.4, 0.7)
    assert scd_col.scd_type2.labels == ("starter", "growth", "enterprise")


def test_scd_tracks_unknown_metric_rejected():
    ui_dict = {
        "about": "test", "unit": "company",
        "window": {"start": "2023-01", "end": "2024-12"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
        ],
        "segments": [
            {"name": "a", "count": 10, "archetype": "growth"},
            {"name": "b", "count": 10, "archetype": "decline"},
        ],
        "dimensions": [
            {"name": "dim_company", "per": "unit", "columns": [
                {"name": "company_id", "type": "id"},
                {"name": "tier", "type": "scd",
                 "tracks": "phantom",  # no fact emits this
                 "tiers": ["a", "b"], "at": [0.5]},
            ]},
        ],
        "facts": [
            {"name": "fct_x", "columns": [
                {"name": "date_key", "type": "id"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "engagement_score", "type": "metric.engagement"},
            ]},
        ],
    }
    with pytest.raises(ValueError, match="phantom"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interpret(UserInput.model_validate(ui_dict))


# ── PK / FK conventions ────────────────────────────────────────────────────


def test_fact_pk_excludes_reference_dim_fk():
    cfg = interpret(_saas_like_input())
    fact = next(t for t in cfg.tables if t.name == "fct_revenue")
    # plan_id is a ref to dim_plan (reference: true) → not in PK
    assert fact.primary_key == ["date_key", "company_id"]
    # but plan_id is still in foreign_keys
    assert any("dim_plan" in fk for fk in fact.foreign_keys)


def test_fact_pk_includes_non_reference_dim_fks():
    cfg = interpret(_saas_like_input())
    fact = next(t for t in cfg.tables if t.name == "fct_engagement")
    assert fact.primary_key == ["date_key", "company_id"]


# ── Event tables ────────────────────────────────────────────────────────────


def test_proportional_event_uses_row_count_source():
    cfg = interpret(_saas_like_input())
    evt = next(t for t in cfg.tables if t.name == "evt_login")
    assert evt.row_count_source is not None
    assert evt.row_count_source.startswith("proportional:engagement:scale:")


def test_proportional_event_has_no_flag_column():
    cfg = interpret(_saas_like_input())
    evt = next(t for t in cfg.tables if t.name == "evt_login")
    assert all(c.dtype != "boolean" for c in evt.columns)


def test_threshold_event_emits_flag_column_with_threshold_source():
    cfg = interpret(_saas_like_input())
    evt = next(t for t in cfg.tables if t.name == "evt_churn")
    flag_col = next(c for c in evt.columns if c.name == "churn_flag")
    assert flag_col.dtype == "boolean"
    assert flag_col.source.startswith("threshold:churn_risk:above:0.7:for:")


def test_threshold_event_has_no_row_count_source():
    cfg = interpret(_saas_like_input())
    evt = next(t for t in cfg.tables if t.name == "evt_churn")
    assert evt.row_count_source is None


# ── Sub-entity dim ──────────────────────────────────────────────────────────


def test_sub_entity_dim_with_default_count_one_uses_variable_grain():
    ui = _minimal_input(dimensions=[
        {"name": "dim_company", "per": "unit", "columns": [
            {"name": "company_id", "type": "id"},
        ]},
        {"name": "dim_user", "per": "unit", "columns": [
            {"name": "user_id", "type": "id"},
            {"name": "company_id", "type": "ref.dim_company"},
        ]},
        {"name": "dim_date", "per": "period", "columns": [
            {"name": "date_key", "type": "id"},
            {"name": "date", "type": "date"},
        ]},
    ], facts=[
        {"name": "fct_x", "columns": [
            {"name": "date_key", "type": "ref.dim_date"},
            {"name": "company_id", "type": "ref.dim_company"},
            {"name": "engagement_score", "type": "metric.engagement"},
            {"name": "mrr", "type": "metric.mrr"},
        ]},
    ])
    cfg = interpret(ui)
    user_dim = next(t for t in cfg.tables if t.name == "dim_user")
    assert user_dim.grain == "variable"


# ── Domain pluralisation ────────────────────────────────────────────────────


def test_domain_entity_label_pluralised_for_company():
    cfg = interpret(_minimal_input(unit="company"))
    assert cfg.domain.entity_label == "Companies"


def test_domain_entity_label_pluralised_for_employee():
    cfg = interpret(_minimal_input(unit="employee"))
    assert cfg.domain.entity_label == "Employees"


# ── Determinism (interpreter is deterministic apart from seed) ─────────────


def test_seed_is_present_and_within_uint32_range():
    cfg = interpret(_minimal_input())
    assert 0 <= cfg.seed < 2**32


def test_two_interprets_produce_structurally_identical_configs_modulo_seed():
    a = interpret(_minimal_input())
    b = interpret(_minimal_input())
    # Same metrics, archetypes, entities, tables structure (compare names
    # and counts; other fields are deterministic given input).
    assert [m.name for m in a.metrics] == [m.name for m in b.metrics]
    assert [t.name for t in a.tables] == [t.name for t in b.tables]
    assert [(e.name, e.size) for e in a.entities] == [(e.name, e.size) for e in b.entities]


# ── Output config defaults ─────────────────────────────────────────────────


def test_output_defaults_to_csv_format_in_output_directory():
    cfg = interpret(_minimal_input())
    assert cfg.output.format == "csv"
    assert cfg.output.directory == "output"


def test_noise_defaults_to_zero():
    cfg = interpret(_minimal_input())
    assert cfg.noise.gaussian_sigma == 0.0
    assert cfg.noise.outlier_rate == 0.0
    assert cfg.noise.mcar_rate == 0.0
