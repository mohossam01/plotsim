"""F7 regression — duplicate correlation entries (M102).

Pre-fix: ``PlotsimConfig`` accepted any list of ``CorrelationPair``
entries without checking for duplicate ``(metric_a, metric_b)``
keys. Downstream, ``_build_correlation_matrix`` reduced the list to
an ``(a, b) → coefficient`` mapping with last-write-wins (assignment
into ``mat[i, j]`` overwrites). A user who configured the same pair
twice with conflicting coefficients got whichever entry happened to
land last, with no warning or error.

Post-fix: ``PlotsimConfig._cross_reference_integrity`` raises
``ValidationError`` when two correlation entries reference the same
unordered pair, citing both coefficients and the pair name. The
check runs before the PSD gate so the duplicate report has priority
over any matrix-degeneracy error the contradictory entries might
otherwise have produced.

Tests:

* ``test_duplicate_same_order_raises`` — ``(a, b)`` configured twice
  with different coefficients.
* ``test_duplicate_opposite_order_raises`` — ``(a, b)`` then
  ``(b, a)`` (pair is unordered).
* ``test_duplicate_with_same_coefficient_still_raises`` — even
  identical coefficients are a config defect (the entry is
  redundant; one of the two is intended to be a different pair and
  the user typoed).
* ``test_distinct_pairs_load`` — three distinct pairs load
  successfully.
* ``test_single_correlation_loads`` — single entry, no duplicates.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from plotsim.config import (
    Archetype,
    Column,
    CorrelationPair,
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


def _build_three_metric_config(
    correlations: list[CorrelationPair],
) -> PlotsimConfig:
    """Three normal metrics + minimal table set; correlations are the
    only degree of freedom."""
    metrics = [
        Metric(
            name=name,
            label=name,
            distribution="normal",
            params={"mu": 10.0, "sigma": 2.0},
            polarity="positive",
        )
        for name in ("a", "b", "c")
    ]
    arch = Archetype(
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
    fct = Table(
        name="fct_metrics",
        type="fact",
        grain="per_entity_per_period",
        primary_key=["date_key", "user_id"],
        foreign_keys=["dim_date.date_key", "dim_user.user_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="user_id", dtype="id", source="fk:dim_user.user_id"),
            Column(name="a", dtype="float", source="metric:a"),
            Column(name="b", dtype="float", source="metric:b"),
            Column(name="c", dtype="float", source="metric:c"),
        ],
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
    dim_user = Table(
        name="dim_user",
        type="dim",
        grain="per_entity",
        primary_key="user_id",
        columns=[
            Column(name="user_id", dtype="id", source="pk"),
            Column(name="user_name", dtype="string", source="generated:faker.name"),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="t",
                description="t",
                entity_type="user",
                entity_label="Users",
            ),
            time_window=TimeWindow(
                start="2024-01",
                end="2024-12",
                granularity="monthly",
            ),
            seed=0,
            metrics=metrics,
            archetypes=[arch],
            entities=[Entity(name="u01", archetype="flat", size=1)],
            tables=[dim_date, dim_user, fct],
            correlations=correlations,
            output=OutputConfig(format="csv", directory="out/f7"),
        )


def test_duplicate_same_order_raises():
    """Two entries on (a, b) with different coefficients must raise
    with both coefficients and the pair name in the message."""
    correlations = [
        CorrelationPair(metric_a="a", metric_b="b", coefficient=0.7),
        CorrelationPair(metric_a="a", metric_b="b", coefficient=0.3),
    ]
    with pytest.raises(ValidationError) as exc_info:
        _build_three_metric_config(correlations)
    msg = str(exc_info.value)
    assert "duplicate" in msg.lower(), f"error message does not say 'duplicate': {msg}"
    assert "0.7" in msg, f"missing first coefficient in message: {msg}"
    assert "0.3" in msg, f"missing second coefficient in message: {msg}"
    assert "'a'" in msg and "'b'" in msg, f"pair names missing from message: {msg}"


def test_duplicate_opposite_order_raises():
    """Pair is unordered: (a, b) and (b, a) reference the same pair."""
    correlations = [
        CorrelationPair(metric_a="a", metric_b="b", coefficient=0.7),
        CorrelationPair(metric_a="b", metric_b="a", coefficient=0.3),
    ]
    with pytest.raises(ValidationError) as exc_info:
        _build_three_metric_config(correlations)
    msg = str(exc_info.value)
    assert "duplicate" in msg.lower()
    assert "0.7" in msg and "0.3" in msg


def test_duplicate_with_same_coefficient_still_raises():
    """Even identical coefficients on the same pair are a redundant
    entry — likely the second was intended to reference a different
    pair and the user typoed. Reject so the typo surfaces."""
    correlations = [
        CorrelationPair(metric_a="a", metric_b="b", coefficient=0.5),
        CorrelationPair(metric_a="a", metric_b="b", coefficient=0.5),
    ]
    with pytest.raises(ValidationError) as exc_info:
        _build_three_metric_config(correlations)
    assert "duplicate" in str(exc_info.value).lower()


def test_distinct_pairs_load():
    """Three distinct pairs (a,b), (a,c), (b,c) — no duplicates — load."""
    correlations = [
        CorrelationPair(metric_a="a", metric_b="b", coefficient=0.4),
        CorrelationPair(metric_a="a", metric_b="c", coefficient=0.3),
        CorrelationPair(metric_a="b", metric_b="c", coefficient=0.5),
    ]
    cfg = _build_three_metric_config(correlations)
    assert len(cfg.correlations) == 3


def test_single_correlation_loads():
    """One CorrelationPair on (a, b) — trivially no duplicates."""
    correlations = [
        CorrelationPair(metric_a="a", metric_b="b", coefficient=0.5),
    ]
    cfg = _build_three_metric_config(correlations)
    assert len(cfg.correlations) == 1
