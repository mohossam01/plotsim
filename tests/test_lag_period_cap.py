"""F10 regression — granularity-aware causal_lag.lag_periods cap (M102).

Pre-fix: ``CausalLag.lag_periods`` carried a flat ``Field(ge=1, le=120)``
constraint regardless of the configured granularity. 120 reads
differently in each:

* monthly: 120 = 10 years (overshoots most plausible business cases)
* weekly:  120 ≈ 2.3 years (tighter than the monthly equivalent)
* daily:   120 ≈ 4 months (blocks quarterly / multi-quarter lags)

Mission 100 flagged the cap as granularity-blind. Mission 102 / F10
moves the cap to a per-granularity dict that targets ~10 years at
each granularity:

    {"monthly": 120, "weekly": 520, "daily": 3_650}

Implementation: the field-level Pydantic constraint is relaxed to
``le=3_650`` (the daily ceiling); a model-level validator on
``PlotsimConfig`` enforces the per-granularity bound and produces an
error message naming the granularity that rejected the value.

Tests:

* ``test_<granularity>_at_cap_loads`` — for each of monthly, weekly,
  daily, a config with ``lag_periods`` exactly at the granularity's
  cap loads cleanly.
* ``test_<granularity>_above_cap_rejected`` — cap+1 raises
  ``ValidationError`` with the granularity and cap named in the
  message.
* ``test_lag_period_field_cap_still_caps_extreme`` — values above the
  daily ceiling (3650) are rejected at the field level under any
  granularity (defense-in-depth).
* ``test_pre_f10_blocked_values_now_load`` — a daily config with
  ``lag_periods=200`` (rejected pre-F10 by ``le=120``) loads
  successfully post-fix.
"""
from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from plotsim.config import (
    Archetype,
    CausalLag,
    Column,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    OutputConfig,
    PlotsimConfig,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
)


# Per-granularity cap as wired in plotsim.config._LAG_PERIOD_LIMITS.
CAPS = {"monthly": 120, "weekly": 520, "daily": 3_650}


def _build_config(granularity: str, lag_periods: int) -> PlotsimConfig:
    """Two-metric config with one causal_lag chain. ``end`` is sized so
    the time window can hold ``lag_periods`` periods at the requested
    granularity (otherwise the time-window span limit might fire first
    on the high end of the daily cap)."""
    if granularity == "monthly":
        # Daily cap of 3650 ≈ 10 years; for monthly we only need
        # 121 max — a 10-year window comfortably fits.
        time_window = TimeWindow(
            start="2024-01", end="2033-12", granularity="monthly",
        )
    elif granularity == "weekly":
        # 521 weeks ≈ 10 years; bumped to a 12-year span for headroom.
        time_window = TimeWindow(
            start="2020-01", end="2031-12", granularity="weekly",
        )
    else:  # daily
        # 3651 days ≈ 10 years — exactly at the configured time-window
        # span cap (3_650 daily). Push to 3651 by extending end into
        # next year is not allowed (daily span cap = 3_650). We
        # don't need the time window to literally hold the lag
        # periods — period_index < lag_periods just means the lag
        # falls back to the unmodified current position. Use a
        # smaller window for the daily tests; the validator runs
        # on the lag_periods value, not on whether the window can
        # hold it.
        time_window = TimeWindow(
            start="2020-01", end="2024-12", granularity="daily",
        )

    metrics = [
        Metric(
            name="driver", label="driver",
            distribution="normal", params={"mu": 1.0, "sigma": 0.1},
            polarity="positive",
        ),
        Metric(
            name="follower", label="follower",
            distribution="normal", params={"mu": 1.0, "sigma": 0.1},
            polarity="positive",
            causal_lag=CausalLag(
                driver="driver", lag_periods=lag_periods, blend_weight=1.0,
            ),
        ),
    ]
    arch = Archetype(
        name="flat", label="flat",
        description="constant 0.5 plateau",
        curve_segments=[
            CurveSegment(
                curve="plateau", params={"level": 0.5},
                start_pct=0.0, end_pct=1.0,
            ),
        ],
    )
    fct = Table(
        name="fct_metrics", type="fact", grain="per_entity_per_period",
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
            Column(name="driver", dtype="float", source="metric:driver"),
            Column(name="follower", dtype="float", source="metric:follower"),
        ],
    )
    dim_date = Table(
        name="dim_date", type="dim", grain="per_period",
        primary_key="date_key",
        columns=[
            Column(name="date_key", dtype="id", source="pk"),
            Column(name="date", dtype="date", source="generated:date_key"),
        ],
    )
    dim_entity = Table(
        name="dim_entity", type="dim", grain="per_entity",
        primary_key="entity_id",
        columns=[
            Column(name="entity_id", dtype="id", source="pk"),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="t", description="t",
                entity_type="entity", entity_label="Entities",
            ),
            time_window=time_window,
            seed=0,
            metrics=metrics,
            archetypes=[arch],
            entities=[Entity(name="e1", archetype="flat", size=1)],
            tables=[dim_date, dim_entity, fct],
            output=OutputConfig(format="csv", directory="out/f10"),
        )


# --- At-cap loads (one test per granularity) --------------------------------


@pytest.mark.parametrize("granularity, cap", list(CAPS.items()))
def test_at_cap_loads(granularity, cap):
    """A config with lag_periods exactly at the granularity's cap
    loads cleanly post-F10."""
    cfg = _build_config(granularity, cap)
    assert cfg.metrics[1].causal_lag is not None
    assert cfg.metrics[1].causal_lag.lag_periods == cap


# --- Above-cap rejected -----------------------------------------------------


@pytest.mark.parametrize("granularity, cap", list(CAPS.items()))
def test_above_cap_rejected(granularity, cap):
    """cap+1 raises ValidationError with the granularity and cap
    named in the message."""
    with pytest.raises(ValidationError) as exc_info:
        _build_config(granularity, cap + 1)
    msg = str(exc_info.value)
    assert granularity in msg, f"granularity not in message: {msg}"
    assert str(cap) in msg, f"cap not in message: {msg}"
    assert str(cap + 1) in msg, f"violating value not in message: {msg}"


# --- Field-level extreme cap (defense-in-depth) -----------------------------


def test_lag_period_field_cap_still_caps_extreme():
    """Values above the field-level sanity bound (10_000) are
    rejected by Pydantic before the model-level granularity check
    fires. The model-level check is authoritative for values in
    the [3651, 10000] range under monthly/weekly granularity (where
    those values exceed the per-granularity cap but the field-level
    bound still passes)."""
    with pytest.raises(ValidationError):
        CausalLag(driver="d", lag_periods=10_001, blend_weight=1.0)
    # 3650 sits inside the field-level bound; CausalLag-only
    # construction (no PlotsimConfig context) accepts it. The
    # per-granularity validator only runs at the model level.
    ok = CausalLag(driver="d", lag_periods=3_650, blend_weight=1.0)
    assert ok.lag_periods == 3_650


# --- Pre-F10 vs post-F10 behavioral delta -----------------------------------


def test_pre_f10_blocked_values_now_load():
    """A daily config with lag_periods=200 was rejected by the
    pre-F10 flat ``le=120`` field cap. Post-F10 it loads
    (200 ≤ daily cap 3650)."""
    cfg = _build_config("daily", 200)
    assert cfg.metrics[1].causal_lag is not None
    assert cfg.metrics[1].causal_lag.lag_periods == 200


def test_pre_f10_borderline_monthly_value_still_loads():
    """A monthly config with lag_periods=120 (the only value the
    pre-F10 flat cap permitted at the upper end) still loads
    post-F10 — exactly at the new monthly cap."""
    cfg = _build_config("monthly", 120)
    assert cfg.metrics[1].causal_lag is not None
    assert cfg.metrics[1].causal_lag.lag_periods == 120


# --- Bundled-template parity ------------------------------------------------


def test_bundled_saas_lag_within_monthly_cap():
    """sample_saas.yaml uses monthly granularity with
    support_tickets.causal_lag.lag_periods=2 — well below the
    monthly cap of 120. Loads post-F10 unchanged."""
    from plotsim.config import load_config
    from pathlib import Path
    cfg_path = (
        Path(__file__).resolve().parent.parent
        / "plotsim" / "configs" / "sample_saas.yaml"
    )
    cfg = load_config(cfg_path)
    for m in cfg.metrics:
        if m.causal_lag is not None:
            assert m.causal_lag.lag_periods <= CAPS["monthly"]


def test_bundled_hr_lag_within_monthly_cap():
    """sample_hr.yaml uses monthly granularity with
    absence_rate.causal_lag.lag_periods=1 — well below the cap."""
    from plotsim.config import load_config
    from pathlib import Path
    cfg_path = (
        Path(__file__).resolve().parent.parent
        / "plotsim" / "configs" / "sample_hr.yaml"
    )
    cfg = load_config(cfg_path)
    for m in cfg.metrics:
        if m.causal_lag is not None:
            assert m.causal_lag.lag_periods <= CAPS["monthly"]
