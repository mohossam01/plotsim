"""0.6-M19 Fix 6 — event timestamp within-period distribution.

Pre-fix, ``generated:timestamp`` on an event or variable-grain row
returned the period anchor (1st-of-month for monthly granularity,
the Monday for weekly, the day itself for daily). Every row inside
period P had the *same* timestamp, which made event tables look
synthetic and broke downstream joins/queries that filter by event
time-of-day or day-of-week.

This module covers the post-fix surface:

  * Proportional events on monthly granularity emit timestamps
    spread across the calendar month.
  * Same on weekly and daily granularities (week-spread and
    hour-spread respectively).
  * Variable-grain facts and per_parent_row children route through
    the same within-period draw.
  * Threshold events distribute the single firing-row timestamp
    inside the firing period rather than emitting the anchor.
  * Per_entity_per_period facts keep anchor-only timestamps —
    the mission spec restricts the change to event and
    variable-grain tables, and these facts have one row per
    (entity, period) so an in-period draw would just inject noise.
  * Same (config, seed) reproduces identical timestamps across two
    runs — the new rng draws are deterministic.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from typing import Any

import numpy as np
import pytest

from plotsim import create, generate_tables, load_template


# --- Helpers ----------------------------------------------------------------


def _saas_like_config(
    *,
    granularity: str = "monthly",
    window: tuple[str, str] | None = None,
    seed: int = 27272,
) -> Any:
    """Builder config with a monthly/weekly/daily window, one per_entity
    dim, a per_entity_per_period fact carrying a metric, and a
    proportional event with a ``timestamp`` column. Window defaults
    are chosen so each granularity sees enough periods that an
    in-period spread shows multiple distinct days.
    """
    if window is None:
        # _compute_n_periods uses YYYY-MM as YYYY-MM-01 — daily windows
        # need an end month different from start to give the archetype
        # validator more than one period to compose.
        window = {
            "monthly": ("2024-01", "2024-03"),
            "weekly": ("2024-01", "2024-02"),
            "daily": ("2024-01", "2024-02"),
        }[granularity]
    base = {
        "about": "event timestamp regression",
        "unit": "customer",
        "seed": seed,
        "window": (*window, granularity),
        "metrics": [
            {
                "name": "purchases",
                "type": "amount",
                "polarity": "positive",
                "range": [3, 5],
            },
        ],
        "segments": [
            {"name": "g", "count": 8, "archetype": "growth"},
        ],
        "dimensions": [
            {
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                    {"name": "customer_name", "type": "faker.name"},
                ],
            },
        ],
        "facts": [
            {
                "name": "fct_visit",
                "metrics": ["purchases"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "purchases", "type": "metric.purchases"},
                ],
            },
        ],
        "events": [
            {
                "name": "evt_action",
                "trigger": "proportional",
                "driver": "purchases",
                "scale": 1.0,
                "columns": [
                    {"name": "event_id", "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "event_ts", "type": "timestamp"},
                ],
            }
        ],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return create(**base)


def _to_date(ts: Any) -> _dt.date:
    if isinstance(ts, _dt.datetime):
        return ts.date()
    return ts


# --- Proportional events: within-period distribution ----------------------


def test_monthly_event_timestamps_span_multiple_days_per_month():
    """On monthly granularity, the same period's event rows must land
    on different calendar days. Pre-fix every row in a period landed
    on day 1.
    """
    cfg = _saas_like_config(granularity="monthly")
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    evt = tables["evt_action"]
    assert len(evt) > 0
    # Group by period via the day_key value (YYYYMM01 for monthly anchor).
    # Per-month distinct-day count must exceed 1; with ~40+ rows/month
    # and 30+ days/month this is essentially deterministic.
    by_month = evt.assign(_d=evt["event_ts"].map(_to_date))
    by_month["_ym"] = by_month["_d"].map(lambda d: (d.year, d.month))
    per_month_distinct_days = by_month.groupby("_ym")["_d"].nunique()
    assert (per_month_distinct_days > 1).all(), (
        f"some monthly period collapsed to a single day; "
        f"per-month distinct day counts: "
        f"{per_month_distinct_days.to_dict()!r}"
    )


def test_monthly_event_timestamps_stay_within_their_month():
    """A row's timestamp must fall inside the month it claims via
    date_key, never spill into the next/previous month.
    """
    cfg = _saas_like_config(granularity="monthly")
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    evt = tables["evt_action"]
    # date_key is YYYYMMDD int (anchor day 1).
    for _, row in evt.iterrows():
        anchor_y = int(row["date_key"]) // 10000
        anchor_m = (int(row["date_key"]) // 100) % 100
        ts_date = _to_date(row["event_ts"])
        assert (ts_date.year, ts_date.month) == (anchor_y, anchor_m), (
            f"event_ts {row['event_ts']!r} fell outside its declared "
            f"period (date_key={row['date_key']!r})"
        )


def test_weekly_event_timestamps_span_multiple_days_per_week():
    cfg = _saas_like_config(granularity="weekly")
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    evt = tables["evt_action"]
    assert len(evt) > 0
    by_week = evt.assign(_d=evt["event_ts"].map(_to_date))
    by_week["_week"] = by_week["_d"].map(lambda d: d.isocalendar()[:2])
    distinct = by_week.groupby("_week")["_d"].nunique()
    assert (distinct > 1).any(), (
        f"weekly events collapsed to single days in every week; "
        f"distinct-day counts: {distinct.to_dict()!r}"
    )


def test_daily_event_timestamps_span_multiple_hours_per_day():
    """On daily granularity, period length is 24h, so within-period
    spread shows up as multiple distinct hours-of-day per period.
    """
    cfg = _saas_like_config(granularity="daily")
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    evt = tables["evt_action"]
    assert len(evt) > 0
    # Group rows by their date_key (anchor day). Each daily period
    # should contain timestamps at varying hours.
    distinct_hours_per_day = (
        evt.assign(_d=evt["event_ts"].map(_to_date), _h=evt["event_ts"].map(lambda t: t.hour))
        .groupby("_d")["_h"]
        .nunique()
    )
    assert (distinct_hours_per_day > 1).any(), (
        f"daily events all landed on the same hour-of-day; "
        f"distinct hours per day: {distinct_hours_per_day.to_dict()!r}"
    )


# --- Determinism -----------------------------------------------------------


def test_within_period_timestamps_are_deterministic_under_seed():
    cfg_a = _saas_like_config(granularity="monthly")
    cfg_b = _saas_like_config(granularity="monthly")
    tables_a = generate_tables(cfg_a, np.random.default_rng(cfg_a.seed))
    tables_b = generate_tables(cfg_b, np.random.default_rng(cfg_b.seed))
    assert (
        tables_a["evt_action"]["event_ts"].tolist() == tables_b["evt_action"]["event_ts"].tolist()
    )


# --- Per_entity_per_period fact: anchor preserved --------------------------


def test_per_entity_per_period_fact_timestamp_stays_anchored():
    """The mission spec restricts the change to event and variable-
    grain tables. A per_entity_per_period fact with a ``timestamp``
    column must still emit the period anchor (1st of month here) —
    backward-compat invariant.
    """
    base = {
        "about": "anchor preservation",
        "unit": "customer",
        "seed": 27272,
        "window": ("2024-01", "2024-03", "monthly"),
        "metrics": [
            {
                "name": "purchases",
                "type": "amount",
                "polarity": "positive",
                "range": [1, 5],
            },
        ],
        "segments": [
            {"name": "g", "count": 4, "archetype": "growth"},
        ],
        "dimensions": [
            {
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                ],
            },
        ],
        "facts": [
            {
                "name": "fct_visit",
                "metrics": ["purchases"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "purchases", "type": "metric.purchases"},
                    {"name": "period_ts", "type": "timestamp"},
                ],
            },
        ],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = create(**base)
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    fct = tables["fct_visit"]
    for _, row in fct.iterrows():
        ts_date = _to_date(row["period_ts"])
        assert ts_date.day == 1, (
            f"per_entity_per_period fact dropped anchor behavior — "
            f"row period_ts={row['period_ts']!r} is not 1st-of-month"
        )


# --- Variable-grain fact + per_parent_row child ---------------------------


def test_orders_template_per_parent_row_timestamps_distributed():
    """``orders`` template ships fct_orders (variable-grain) +
    fct_order_items (per_parent_row child). Variable-grain rows route
    through ``_emit_proportional_rows`` (within-period draw); child
    rows route through ``_build_per_parent_row_fact`` (within-period
    draw). Neither has a ``timestamp`` column in the bundled template,
    so add one via a custom builder config that mirrors the structure.
    """
    base = {
        "about": "variable + child timestamp",
        "unit": "customer",
        "seed": 27272,
        "window": ("2024-01", "2024-03", "monthly"),
        "metrics": [
            {
                "name": "order_volume",
                "type": "amount",
                "polarity": "positive",
                "range": [3, 6],
            },
        ],
        "segments": [
            {"name": "g", "count": 6, "archetype": "growth"},
        ],
        "dimensions": [
            {
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                ],
            },
        ],
        "facts": [
            {
                "name": "fct_orders",
                "row_count_driver": "order_volume",
                "row_count_scale": 1.0,
                "columns": [
                    {"name": "order_id", "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "order_date", "type": "ref.dim_date"},
                    {"name": "order_ts", "type": "timestamp"},
                ],
            },
            {
                "name": "fct_order_items",
                "parent_table": "fct_orders",
                "children_per_row": [1, 3],
                "columns": [
                    {"name": "item_id", "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "order_date", "type": "ref.dim_date"},
                    {"name": "item_ts", "type": "timestamp"},
                ],
            },
        ],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = create(**base)
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))

    # Variable-grain parent: timestamps distributed across the month.
    parent = tables["fct_orders"]
    assert len(parent) > 0
    parent_days = parent["order_ts"].map(lambda t: _to_date(t).day).unique()
    assert len(parent_days) > 1, (
        f"variable-grain fct_orders.order_ts collapsed to a single "
        f"day-of-month: {sorted(parent_days)!r}"
    )

    # Per_parent_row child: timestamps distributed too.
    child = tables["fct_order_items"]
    assert len(child) > 0
    child_days = child["item_ts"].map(lambda t: _to_date(t).day).unique()
    assert len(child_days) > 1, (
        f"per_parent_row fct_order_items.item_ts collapsed to a "
        f"single day-of-month: {sorted(child_days)!r}"
    )

    # Each child row's timestamp must fall in the same month its
    # parent's order_date claims — the period inherits down.
    child["_d"] = child["item_ts"].map(_to_date)
    child["_ym"] = child["_d"].map(lambda d: (d.year, d.month))
    parent_ym = parent.assign(
        _d=parent["order_ts"].map(_to_date),
        _ym=parent["order_ts"].map(lambda t: (_to_date(t).year, _to_date(t).month)),
    )
    parent_ym_by_id = dict(zip(parent_ym["order_id"], parent_ym["_ym"]))
    for _, row in child.iterrows():
        assert row["_ym"] == parent_ym_by_id[row["order_id"]], (
            f"child item_ts month {row['_ym']!r} doesn't match its "
            f"parent {row['order_id']!r} period {parent_ym_by_id[row['order_id']]!r}"
        )


# --- Threshold event timestamp --------------------------------------------


def test_threshold_event_timestamp_distributed_within_firing_period():
    """A threshold event emits a single firing row per entity at the
    streak emergence period. That single timestamp must be drawn
    within the firing period rather than the period anchor.
    """
    base = {
        "about": "threshold event timestamp",
        "unit": "customer",
        "seed": 27272,
        "window": ("2024-01", "2024-12", "monthly"),
        "metrics": [
            {
                "name": "score",
                "type": "score",
                "polarity": "positive",
                "range": [0.0, 1.0],
            },
        ],
        "segments": [
            {"name": "g", "count": 30, "archetype": "growth"},
        ],
        "dimensions": [
            {
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                ],
            },
        ],
        "facts": [
            {
                "name": "fct_score",
                "metrics": ["score"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "score", "type": "metric.score"},
                ],
            },
        ],
        "events": [
            {
                "name": "evt_milestone",
                "trigger": "threshold",
                "metric": "score",
                "above": 0.1,
                "for": 1,
                "columns": [
                    {"name": "event_id", "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "date_key", "type": "ref.dim_date"},
                    # ``flag`` is the trigger column — without it the
                    # event builder leaves row_count_source unset and
                    # never fires.
                    {"name": "crossed", "type": "flag"},
                    {"name": "fired_at", "type": "timestamp"},
                ],
            },
        ],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = create(**base)
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    evt = tables["evt_milestone"]
    if len(evt) < 2:
        pytest.skip(
            f"threshold event fired <2 rows ({len(evt)}); test requires "
            f"multiple rows to assert distribution"
        )
    days = evt["fired_at"].map(lambda t: _to_date(t).day).tolist()
    # At least one row must NOT be day-1 (the old anchor behavior).
    assert any(d != 1 for d in days), (
        f"every threshold-event firing landed on day-1-of-month; "
        f"timestamps appear anchor-locked: {days!r}"
    )


# --- End-to-end: bundled templates still build ----------------------------


def test_saas_template_event_ts_distributed():
    """The bundled saas template has evt_login with an event_ts
    column. After the fix it should produce distributed timestamps.
    """
    cfg = load_template("saas")
    tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
    evt = tables["evt_login"]
    assert len(evt) > 0
    days = evt["event_ts"].map(lambda t: _to_date(t).day).unique()
    assert len(days) > 1, (
        f"saas evt_login.event_ts collapsed to a single day-of-month: " f"{sorted(days)!r}"
    )
