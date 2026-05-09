"""Tests for plotsim.metrics global seasonal modulation — Mission 119.

Covers the mission's acceptance criteria:

  - Basic seasonality lifts/dips metric centers in the configured months.
  - Empty ``seasonal_effects`` produces output byte-identical to pre-M119.
  - Overlapping effects sum at the global level before sensitivities apply.
  - Per-metric ``seasonal_sensitivity`` (1.0 / 0.0 / negative) shapes the
    effective multiplier.
  - Per-entity ``seasonal_sensitivity`` likewise. Two entities sharing an
    archetype but differing in sensitivity have different seasonal
    amplitudes and identical trajectory positions.
  - Composition: ``effective = global × metric_sens × entity_sens``.
  - Modulated center clamped to ``value_range`` BEFORE distribution sampling.
  - Trajectory positions unchanged by seasonality (trajectory-first
    invariant).
  - Granularity (monthly / weekly / daily) — the right calendar months
    select the right periods.
  - ``trace_metric_cell`` exposes ``seasonal_factor`` + ``modulated_center``
    matching ``base_center × (1 + seasonal_factor)`` (modulo clamp).
  - Determinism: same (config, seed) → byte-identical output.
  - Bundled templates load with default-empty ``seasonal_effects``.
  - Builder layer (UserInput.seasonality + per-segment sensitivity)
    translates 1:1 into the engine config.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from plotsim.config import (
    Archetype,
    Column,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    OutputConfig,
    PlotsimConfig,
    SeasonalEffect,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
    ValueRange,
    load_config,
)
from plotsim.inspect import trace_metric_cell
from plotsim.tables import _build_seasonal_factors, generate_tables_with_state


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "plotsim" / "configs"


# --- Fixtures ---------------------------------------------------------------


def _flat_archetype() -> Archetype:
    """Flat trajectory at position 0.5 — every period has the same base center."""
    return Archetype(
        name="flat",
        label="flat",
        description="constant 0.5 plateau",
        curve_segments=[
            CurveSegment(
                curve="plateau",
                params={"level": 0.5},
                start_pct=0.0,
                end_pct=1.0,
            ),
        ],
    )


def _normal_metric(
    *,
    name: str = "m",
    mu: float = 10.0,
    sigma: float = 0.001,
    seasonal_sensitivity: float = 1.0,
    value_range: ValueRange | None = None,
    polarity: str = "positive",
) -> Metric:
    """Tight normal so realised samples cluster on the center.

    ``mu=10`` means ``position_to_center(0.5)`` returns 5.0. ``sigma=0.001``
    keeps the mean of any column near the center to several decimal places,
    so seasonal lifts/dips show up as direct mean shifts.
    """
    return Metric(
        name=name,
        label=name,
        distribution="normal",
        params={"mu": mu, "sigma": sigma},
        polarity=polarity,  # type: ignore[arg-type]
        value_range=value_range,
        seasonal_sensitivity=seasonal_sensitivity,
    )


def _config(
    *,
    metrics: list[Metric] | None = None,
    entities: list[Entity] | None = None,
    seasonal_effects: list[SeasonalEffect] | None = None,
    granularity: str = "monthly",
    start: str = "2024-01",
    end: str = "2024-12",
    seed: int = 42,
) -> PlotsimConfig:
    if metrics is None:
        metrics = [_normal_metric()]
    if entities is None:
        entities = [Entity(name="e1", archetype="flat", size=1)]
    if seasonal_effects is None:
        seasonal_effects = []
    fact_cols = [
        Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
        Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
    ]
    for m in metrics:
        fact_cols.append(
            Column(name=m.name, dtype="float", source=f"metric:{m.name}"),
        )
    fct = Table(
        name="fct_m",
        type="fact",
        grain="per_entity_per_period",
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=fact_cols,
    )
    dim_date = Table(
        name="dim_date",
        type="dim",
        grain="per_period",
        primary_key="date_key",
        columns=[
            Column(name="date_key", dtype="id", source="pk"),
            Column(name="date", dtype="date", source="generated:date_key"),
        ],
    )
    dim_entity = Table(
        name="dim_entity",
        type="dim",
        grain="per_entity",
        primary_key="entity_id",
        columns=[
            Column(name="entity_id", dtype="id", source="pk"),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="t",
                description="t",
                entity_type="entity",
                entity_label="Entities",
            ),
            time_window=TimeWindow(
                start=start,
                end=end,
                granularity=granularity,  # type: ignore[arg-type]
            ),
            seed=seed,
            metrics=metrics,
            archetypes=[_flat_archetype()],
            entities=entities,
            tables=[dim_date, dim_entity, fct],
            seasonal_effects=seasonal_effects,
            output=OutputConfig(format="csv", directory="out/m119"),
        )


# --- SeasonalEffect engine model validation ---------------------------------


def test_seasonal_effect_months_in_range():
    SeasonalEffect(months=(1, 6, 12), strength=0.1)
    with pytest.raises(ValidationError) as exc:
        SeasonalEffect(months=(0, 6), strength=0.1)
    assert "out of range" in str(exc.value).lower()
    with pytest.raises(ValidationError) as exc:
        SeasonalEffect(months=(13,), strength=0.1)
    assert "out of range" in str(exc.value).lower()


def test_seasonal_effect_months_unique_within_one_effect():
    with pytest.raises(ValidationError) as exc:
        SeasonalEffect(months=(11, 12, 12), strength=0.1)
    assert "unique" in str(exc.value).lower()


def test_seasonal_effect_negative_strength_allowed():
    eff = SeasonalEffect(months=(6, 7, 8), strength=-0.15)
    assert eff.strength == -0.15


def test_plotsim_config_seasonal_effects_defaults_empty():
    cfg = _config()
    assert cfg.seasonal_effects == []


# --- Period → calendar-month resolver ---------------------------------------


def test_period_calendar_months_monthly_one_per_calendar_month():
    tw = TimeWindow(start="2024-01", end="2024-12", granularity="monthly")
    assert tw.period_calendar_months() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def test_period_calendar_months_monthly_year_boundary():
    tw = TimeWindow(start="2023-11", end="2024-02", granularity="monthly")
    assert tw.period_calendar_months() == [11, 12, 1, 2]


def test_period_calendar_months_daily_matches_each_day():
    tw = TimeWindow(start="2024-01", end="2024-02", granularity="daily")
    months = tw.period_calendar_months()
    assert months[0] == 1 and months[-1] == 2
    assert months.count(1) == 31
    assert months.count(2) == 29  # 2024 leap year


def test_period_calendar_months_weekly_uses_first_in_window_day():
    """Weekly: a week belongs to the month of its first in-window date."""
    tw = TimeWindow(start="2024-12", end="2025-01", granularity="weekly")
    months = tw.period_calendar_months()
    # Most weeks fall fully inside Dec or Jan; the first week starts 2024-12-01
    # (a Sunday), so its first in-window day is 2024-12-01 → month 12.
    assert months[0] == 12
    assert any(m == 1 for m in months)
    assert all(m in (12, 1) for m in months)


# --- Backward compatibility (no seasonal_effects) ---------------------------


def test_empty_seasonal_effects_byte_identical():
    """A config with no seasonality must produce identical output to the
    same config that explicitly declares ``seasonal_effects=[]`` — both go
    through the ``_build_seasonal_factors → None`` short-circuit."""
    cfg_a = _config()
    cfg_b = _config(seasonal_effects=[])
    tables_a, _ = generate_tables_with_state(cfg_a, np.random.default_rng(7))
    tables_b, _ = generate_tables_with_state(cfg_b, np.random.default_rng(7))
    np.testing.assert_array_equal(
        tables_a["fct_m"]["m"].to_numpy(),
        tables_b["fct_m"]["m"].to_numpy(),
    )


def test_build_seasonal_factors_returns_none_when_empty():
    cfg = _config()
    assert _build_seasonal_factors(cfg, n_periods=12) is None


def test_build_seasonal_factors_sums_overlapping_effects():
    cfg = _config(
        seasonal_effects=[
            SeasonalEffect(months=(12,), strength=0.3),
            SeasonalEffect(months=(11, 12), strength=0.1),
        ]
    )
    factors = _build_seasonal_factors(cfg, n_periods=12)
    assert factors is not None
    # months 1..10 → 0.0; month 11 → 0.1; month 12 → 0.4
    assert factors[10] == pytest.approx(0.1)
    assert factors[11] == pytest.approx(0.4)
    assert all(factors[i] == pytest.approx(0.0) for i in range(10))


# --- Trace-level center math (exact, free of distribution noise) ------------


def test_trace_seasonal_factor_zero_when_no_effects():
    cfg = _config()
    r = trace_metric_cell(cfg, "e1", period_index=11, metric_name="m")
    assert r.seasonal_factor == 0.0
    assert r.modulated_center == pytest.approx(r.distribution_center)


def test_trace_december_lift_thirty_percent():
    """AC: ``[{months: [12], strength: 0.3}]`` raises Dec center ~30% over base."""
    cfg = _config(
        seasonal_effects=[
            SeasonalEffect(months=(12,), strength=0.3),
        ]
    )
    r_nov = trace_metric_cell(cfg, "e1", period_index=10, metric_name="m")
    r_dec = trace_metric_cell(cfg, "e1", period_index=11, metric_name="m")
    assert r_nov.seasonal_factor == pytest.approx(0.0)
    assert r_dec.seasonal_factor == pytest.approx(0.3)
    assert r_dec.modulated_center == pytest.approx(r_dec.distribution_center * 1.3)
    # Period 10 (November) should be unchanged from base.
    assert r_nov.modulated_center == pytest.approx(r_nov.distribution_center)


def test_trace_overlapping_effects_sum_at_global_level():
    """AC: Dec gets +0.4, Nov gets +0.1 when effects overlap."""
    cfg = _config(
        seasonal_effects=[
            SeasonalEffect(months=(12,), strength=0.3),
            SeasonalEffect(months=(11, 12), strength=0.1),
        ]
    )
    r_oct = trace_metric_cell(cfg, "e1", period_index=9, metric_name="m")
    r_nov = trace_metric_cell(cfg, "e1", period_index=10, metric_name="m")
    r_dec = trace_metric_cell(cfg, "e1", period_index=11, metric_name="m")
    assert r_oct.seasonal_factor == pytest.approx(0.0)
    assert r_nov.seasonal_factor == pytest.approx(0.1)
    assert r_dec.seasonal_factor == pytest.approx(0.4)


# --- Per-metric sensitivity --------------------------------------------------


def test_per_metric_sensitivity_negative_half():
    """metric_sens=-0.5 during +0.3 global → effective -0.15."""
    cfg = _config(
        metrics=[_normal_metric(seasonal_sensitivity=-0.5)],
        seasonal_effects=[SeasonalEffect(months=(12,), strength=0.3)],
    )
    r = trace_metric_cell(cfg, "e1", period_index=11, metric_name="m")
    assert r.seasonal_factor == pytest.approx(-0.15)
    assert r.modulated_center == pytest.approx(r.distribution_center * 0.85)


def test_per_metric_sensitivity_zero_immune():
    cfg = _config(
        metrics=[_normal_metric(seasonal_sensitivity=0.0)],
        seasonal_effects=[SeasonalEffect(months=(12,), strength=0.3)],
    )
    r = trace_metric_cell(cfg, "e1", period_index=11, metric_name="m")
    assert r.seasonal_factor == pytest.approx(0.0)
    assert r.modulated_center == pytest.approx(r.distribution_center)


def test_per_metric_sensitivity_one_follows_global():
    cfg = _config(
        metrics=[_normal_metric(seasonal_sensitivity=1.0)],
        seasonal_effects=[SeasonalEffect(months=(12,), strength=0.3)],
    )
    r = trace_metric_cell(cfg, "e1", period_index=11, metric_name="m")
    assert r.seasonal_factor == pytest.approx(0.3)


# --- Per-entity (segment) sensitivity ---------------------------------------


def test_per_entity_sensitivity_one_and_a_half_lifts_more():
    cfg = _config(
        entities=[
            Entity(
                name="e1",
                archetype="flat",
                size=1,
                seasonal_sensitivity=1.5,
            )
        ],
        seasonal_effects=[SeasonalEffect(months=(12,), strength=0.3)],
    )
    r = trace_metric_cell(cfg, "e1", period_index=11, metric_name="m")
    assert r.seasonal_factor == pytest.approx(0.45)
    assert r.modulated_center == pytest.approx(r.distribution_center * 1.45)


def test_per_entity_sensitivity_zero_immune():
    cfg = _config(
        entities=[
            Entity(
                name="e1",
                archetype="flat",
                size=1,
                seasonal_sensitivity=0.0,
            )
        ],
        seasonal_effects=[SeasonalEffect(months=(12,), strength=0.3)],
    )
    r = trace_metric_cell(cfg, "e1", period_index=11, metric_name="m")
    assert r.seasonal_factor == pytest.approx(0.0)
    assert r.modulated_center == pytest.approx(r.distribution_center)


def test_two_entities_same_archetype_different_amplitudes_same_trajectory():
    """AC: Two segments sharing an archetype but with different sensitivities
    produce different seasonal amplitudes and identical trajectory shapes."""
    cfg = _config(
        entities=[
            Entity(name="e_low", archetype="flat", size=1, seasonal_sensitivity=0.5),
            Entity(name="e_high", archetype="flat", size=1, seasonal_sensitivity=2.0),
        ],
        seasonal_effects=[SeasonalEffect(months=(12,), strength=0.3)],
    )
    r_low = trace_metric_cell(cfg, "e_low", period_index=11, metric_name="m")
    r_high = trace_metric_cell(cfg, "e_high", period_index=11, metric_name="m")
    # Different seasonal amplitudes:
    assert r_low.seasonal_factor == pytest.approx(0.15)
    assert r_high.seasonal_factor == pytest.approx(0.6)
    # Same trajectory shape — flat archetype puts both at position 0.5.
    assert r_low.trajectory_position == pytest.approx(r_high.trajectory_position)
    assert r_low.distribution_center == pytest.approx(r_high.distribution_center)


# --- Composition + clamp ----------------------------------------------------


def test_composition_metric_and_entity_multiply_with_global():
    """AC: metric=-0.5 × segment=1.5 × global=0.3 → effective = -0.225."""
    cfg = _config(
        metrics=[_normal_metric(seasonal_sensitivity=-0.5)],
        entities=[
            Entity(
                name="e1",
                archetype="flat",
                size=1,
                seasonal_sensitivity=1.5,
            )
        ],
        seasonal_effects=[SeasonalEffect(months=(12,), strength=0.3)],
    )
    r = trace_metric_cell(cfg, "e1", period_index=11, metric_name="m")
    assert r.seasonal_factor == pytest.approx(0.3 * -0.5 * 1.5)
    assert r.modulated_center == pytest.approx(r.distribution_center * (1 + 0.3 * -0.5 * 1.5))


def test_modulated_center_clamped_to_value_range():
    """A wildly negative effective strength would push center below
    value_range.min — the clamp must trigger BEFORE distribution sampling."""
    cfg = _config(
        metrics=[
            _normal_metric(
                mu=10.0,
                sigma=0.001,
                value_range=ValueRange(min=4.0, max=20.0),
            )
        ],
        seasonal_effects=[SeasonalEffect(months=(12,), strength=-0.9)],
    )
    r = trace_metric_cell(cfg, "e1", period_index=11, metric_name="m")
    # Pre-clamp would be 5.0 * (1 - 0.9) = 0.5 < value_range.min=4.0.
    assert r.distribution_center == pytest.approx(5.0)
    assert r.modulated_center == pytest.approx(4.0)
    assert r.seasonal_factor == pytest.approx(-0.9)


def test_trajectory_unchanged_by_seasonality():
    """Trajectory-first invariant — adding seasonality does NOT shift the
    underlying trajectory positions for any entity."""
    cfg_clean = _config()
    cfg_seasonal = _config(
        seasonal_effects=[
            SeasonalEffect(months=(12,), strength=0.5),
        ]
    )
    for p in range(12):
        r_clean = trace_metric_cell(cfg_clean, "e1", p, "m")
        r_season = trace_metric_cell(cfg_seasonal, "e1", p, "m")
        assert r_clean.trajectory_position == r_season.trajectory_position
        assert r_clean.effective_position == r_season.effective_position


# --- Granularity ------------------------------------------------------------


def test_monthly_dec_period_only_lifted():
    cfg = _config(
        granularity="monthly",
        start="2024-01",
        end="2024-12",
        seasonal_effects=[SeasonalEffect(months=(12,), strength=0.3)],
    )
    factors = _build_seasonal_factors(cfg, n_periods=12)
    assert factors is not None
    assert factors[11] == pytest.approx(0.3)
    for i in range(11):
        assert factors[i] == pytest.approx(0.0)


def test_daily_dec_days_lifted():
    cfg = _config(
        granularity="daily",
        start="2024-11",
        end="2024-12",
        seasonal_effects=[SeasonalEffect(months=(12,), strength=0.3)],
    )
    n_days = cfg.time_window.period_count()
    factors = _build_seasonal_factors(cfg, n_periods=n_days)
    months = cfg.time_window.period_calendar_months()
    assert factors is not None
    for i, m in enumerate(months):
        if m == 12:
            assert factors[i] == pytest.approx(0.3)
        else:
            assert factors[i] == pytest.approx(0.0)


def test_weekly_weeks_starting_in_dec_lifted():
    cfg = _config(
        granularity="weekly",
        start="2024-11",
        end="2024-12",
        seasonal_effects=[SeasonalEffect(months=(12,), strength=0.3)],
    )
    n_weeks = cfg.time_window.period_count()
    factors = _build_seasonal_factors(cfg, n_periods=n_weeks)
    months = cfg.time_window.period_calendar_months()
    assert factors is not None
    # At least one week starts in December.
    assert 12 in months
    for i, m in enumerate(months):
        expected = 0.3 if m == 12 else 0.0
        assert factors[i] == pytest.approx(expected)


# --- End-to-end fact-table effect -------------------------------------------


def test_fact_table_december_mean_lifts_thirty_percent():
    """Generate a tight-sigma normal column over a 24-month window. With a
    +0.3 December effect, the mean of December rows should sit near
    1.3× the mean of non-December rows."""
    cfg = _config(
        start="2023-01",
        end="2024-12",
        seasonal_effects=[SeasonalEffect(months=(12,), strength=0.3)],
    )
    tables, _ = generate_tables_with_state(cfg, np.random.default_rng(11))
    # Cross-reference month by joining on date_key.
    fct = tables["fct_m"]
    dim_date = tables["dim_date"]
    merged = fct.merge(dim_date[["date_key", "month"]], on="date_key")
    dec_mean = merged.loc[merged["month"] == 12, "m"].mean()
    other_mean = merged.loc[merged["month"] != 12, "m"].mean()
    # With sigma=0.001 the realized mean is essentially the modulated center.
    assert dec_mean == pytest.approx(other_mean * 1.3, rel=1e-2)


# --- Determinism ------------------------------------------------------------


def test_same_seed_byte_identical_with_seasonality():
    cfg = _config(
        seasonal_effects=[
            SeasonalEffect(months=(11, 12), strength=0.2),
        ]
    )
    a, _ = generate_tables_with_state(cfg, np.random.default_rng(13))
    b, _ = generate_tables_with_state(cfg, np.random.default_rng(13))
    np.testing.assert_array_equal(
        a["fct_m"]["m"].to_numpy(),
        b["fct_m"]["m"].to_numpy(),
    )


def test_seasonality_changes_values_not_structure():
    cfg_clean = _config()
    cfg_season = _config(
        seasonal_effects=[
            SeasonalEffect(months=(12,), strength=0.5),
        ]
    )
    a, _ = generate_tables_with_state(cfg_clean, np.random.default_rng(2))
    b, _ = generate_tables_with_state(cfg_season, np.random.default_rng(2))
    assert list(a) == list(b)
    assert a["fct_m"].columns.tolist() == b["fct_m"].columns.tolist()
    assert len(a["fct_m"]) == len(b["fct_m"])
    # Same row count + columns, but cell values differ in the December rows.
    assert not np.allclose(
        a["fct_m"]["m"].to_numpy(),
        b["fct_m"]["m"].to_numpy(),
    )


# --- Bundled engine templates byte-identical (no seasonal_effects) -----------


@pytest.mark.parametrize("yaml_name", ["sample_saas.yaml"])
def test_bundled_engine_templates_load_with_default_empty_effects(yaml_name):
    """Bundled templates do not declare ``seasonal_effects`` — they must
    load and produce ``seasonal_effects = []`` so the metrics pipeline
    short-circuits and output is byte-identical to pre-M119."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = load_config(CONFIGS / yaml_name)
    assert cfg.seasonal_effects == []
    # Default Metric.seasonal_sensitivity == 1.0.
    for m in cfg.metrics:
        assert m.seasonal_sensitivity == 1.0
    # Default Entity.seasonal_sensitivity == 1.0.
    for e in cfg.entities:
        assert e.seasonal_sensitivity == 1.0


# --- Builder layer ----------------------------------------------------------


def test_builder_seasonal_effect_input_validates_months():
    from plotsim.builder.input import SeasonalEffectInput

    SeasonalEffectInput(months=(11, 12), strength=0.2)
    with pytest.raises(ValidationError):
        SeasonalEffectInput(months=(0,), strength=0.2)
    with pytest.raises(ValidationError):
        SeasonalEffectInput(months=(13,), strength=0.2)
    with pytest.raises(ValidationError):
        SeasonalEffectInput(months=(12, 12), strength=0.2)


def test_builder_user_input_accepts_seasonality_field():
    from plotsim.builder.input import (
        MetricInput,
        SegmentInput,
        SeasonalEffectInput,
        UserInput,
    )

    ui = UserInput(
        about="seasonal sandbox",
        unit="store",
        window=("2024-01", "2024-12", "monthly"),
        metrics=[
            MetricInput(
                name="sales",
                type="amount",
                polarity="positive",
                range=(0.0, 100.0),
                seasonal_sensitivity=-0.5,
            ),
        ],
        segments=[
            SegmentInput(name="cohort_a", count=3, archetype="flat", seasonal_sensitivity=1.5),
            SegmentInput(name="cohort_b", count=3, archetype="flat", seasonal_sensitivity=0.0),
        ],
        seasonality=[
            SeasonalEffectInput(months=(11, 12), strength=0.2),
            SeasonalEffectInput(months=(6, 7, 8), strength=-0.1),
        ],
    )
    assert len(ui.seasonality) == 2
    assert ui.seasonality[0].months == (11, 12)
    assert ui.seasonality[0].strength == 0.2
    assert ui.metrics[0].seasonal_sensitivity == -0.5
    assert ui.segments[0].seasonal_sensitivity == 1.5
    assert ui.segments[1].seasonal_sensitivity == 0.0


def test_builder_interpreter_translates_seasonality_to_engine_config():
    """End-to-end builder path: ``UserInput.seasonality`` →
    ``PlotsimConfig.seasonal_effects``; per-metric and per-segment
    sensitivities land on every ``Metric`` / expanded ``Entity``."""
    from plotsim.builder import create

    # ``flat`` (plateau) archetype is in the builder's parser vocabulary.
    cfg = create(
        about="seasonal sandbox",
        unit="store",
        window=("2024-01", "2024-12", "monthly"),
        metrics=[
            {
                "name": "sales",
                "type": "amount",
                "polarity": "positive",
                "range": (0.0, 100.0),
                "seasonal_sensitivity": -0.5,
            },
        ],
        segments=[
            {"name": "cohort_a", "count": 3, "archetype": "flat", "seasonal_sensitivity": 1.5},
            {"name": "cohort_b", "count": 3, "archetype": "flat", "seasonal_sensitivity": 0.0},
        ],
        seasonality=[
            {"months": (11, 12), "strength": 0.2},
            {"months": (6, 7, 8), "strength": -0.1},
        ],
    )
    assert len(cfg.seasonal_effects) == 2
    assert cfg.seasonal_effects[0].months == (11, 12)
    assert cfg.seasonal_effects[0].strength == 0.2
    sales = next(m for m in cfg.metrics if m.name == "sales")
    assert sales.seasonal_sensitivity == -0.5
    cohort_a_entities = [e for e in cfg.entities if e.archetype == "cohort_a"]
    cohort_b_entities = [e for e in cfg.entities if e.archetype == "cohort_b"]
    assert len(cohort_a_entities) == 3
    assert len(cohort_b_entities) == 3
    assert all(e.seasonal_sensitivity == 1.5 for e in cohort_a_entities)
    assert all(e.seasonal_sensitivity == 0.0 for e in cohort_b_entities)


def test_builder_default_no_seasonality_produces_empty_effects():
    """A builder config without ``seasonality`` must produce an empty
    engine ``seasonal_effects`` and default-1.0 sensitivities."""
    from plotsim.builder import create

    cfg = create(
        about="seasonal sandbox",
        unit="store",
        window=("2024-01", "2024-12", "monthly"),
        metrics=[
            {"name": "sales", "type": "amount", "polarity": "positive", "range": (0.0, 100.0)},
        ],
        segments=[
            {"name": "cohort_a", "count": 3, "archetype": "flat"},
        ],
    )
    assert cfg.seasonal_effects == []
    assert all(m.seasonal_sensitivity == 1.0 for m in cfg.metrics)
    assert all(e.seasonal_sensitivity == 1.0 for e in cfg.entities)
