"""F12 regression — dtype × source cross-check validator (M102).

Mission 101 spot-checks Q3 found that the schema accepted
``dtype: boolean`` on a ``MetricSource`` or ``LagSource`` column
even though the resulting cell is ``bool(continuous_metric_value)``
— near-constant ``True`` for any positive-skewed distribution
(poisson with λ > 0, lognorm, gamma, weibull). No template,
fixture, or doc exercised this combination, but nothing rejected
it either.

Mission 102 / F12 originally combined the validator with removing
the dead ``has_bool_metric`` scalar-fallback predicate at
``tables.py:353-356``. The predicate-removal step landed in F3
(commit `2ccfdbe`) when the vectorized path was extended to
coerce boolean columns correctly via
``_coerce_array_for_dtype``. F12 narrows to the validator that
prevents the bad combination from loading in the first place.

Tests:

* ``test_metric_source_with_boolean_dtype_rejected`` — ``dtype:
  boolean`` on a ``metric:foo`` source raises at config load with
  the column name and source in the message.
* ``test_lag_source_with_boolean_dtype_rejected`` — same on a
  ``lag:foo`` source.
* ``test_metric_source_with_int_dtype_loads`` — the natural
  poisson-output combination.
* ``test_metric_source_with_float_dtype_loads`` — the natural
  continuous-distribution combination.
* ``test_threshold_source_with_boolean_dtype_loads`` — confirms
  the validator does not block ``ThresholdSource``, which
  produces booleans by design (saas's ``churn_flag``).
* ``test_bundled_templates_load_under_validator`` — every
  bundled template still loads (none of them violate the
  combination).
"""

from __future__ import annotations

import warnings
from pathlib import Path

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
    load_config,
)


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "plotsim" / "configs"


def _build_config(
    *,
    fct_columns: list[Column],
    extra_metrics: list[Metric] | None = None,
) -> PlotsimConfig:
    """Build a minimal config with a customisable fact table column list."""
    metrics = [
        Metric(
            name="m1",
            label="m1",
            distribution="poisson",
            params={"lambda": 5.0},
            polarity="positive",
        ),
        Metric(
            name="m2",
            label="m2",
            distribution="lognorm",
            params={"s": 0.5, "loc": 0.0, "scale": 50.0},
            polarity="positive",
        ),
    ]
    if extra_metrics:
        metrics.extend(extra_metrics)
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
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
            *fct_columns,
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
                start="2024-01",
                end="2024-12",
                granularity="monthly",
            ),
            seed=0,
            metrics=metrics,
            archetypes=[arch],
            entities=[Entity(name="e1", archetype="flat", size=1)],
            tables=[dim_date, dim_entity, fct],
            output=OutputConfig(format="csv", directory="out/f12"),
        )


# --- Rejection paths --------------------------------------------------------


def test_metric_source_with_boolean_dtype_rejected():
    """``dtype: boolean`` on a metric: source raises with column name
    and source in the message."""
    with pytest.raises(ValidationError) as exc_info:
        _build_config(
            fct_columns=[
                Column(name="m1_bool", dtype="boolean", source="metric:m1"),
            ]
        )
    msg = str(exc_info.value)
    assert "boolean" in msg.lower()
    assert "m1_bool" in msg
    assert "metric:m1" in msg
    # The error names the source kind for clarity.
    assert "metric" in msg.lower()


def test_lag_source_with_boolean_dtype_rejected():
    """Same gate on lag: sources. m3 carries a causal_lag chain so
    a lag:m3:n source is well-formed; the dtype:boolean clash is
    what the validator catches."""
    extras = [
        Metric(
            name="m3",
            label="m3",
            distribution="normal",
            params={"mu": 10.0, "sigma": 2.0},
            polarity="positive",
            causal_lag=CausalLag(
                driver="m1",
                lag_periods=2,
                blend_weight=1.0,
            ),
        ),
    ]
    with pytest.raises(ValidationError) as exc_info:
        _build_config(
            fct_columns=[
                Column(name="m3_bool", dtype="boolean", source="lag:m3:periods:1"),
            ],
            extra_metrics=extras,
        )
    msg = str(exc_info.value)
    assert "boolean" in msg.lower()
    assert "m3_bool" in msg
    assert "lag:m3:periods:1" in msg


# --- Acceptance paths -------------------------------------------------------


def test_metric_source_with_int_dtype_loads():
    """``dtype: int`` on a poisson metric source — the natural
    integer-output combination — loads cleanly."""
    cfg = _build_config(
        fct_columns=[
            Column(name="m1_int", dtype="int", source="metric:m1"),
        ]
    )
    assert cfg is not None


def test_metric_source_with_float_dtype_loads():
    """``dtype: float`` on a continuous metric source — the natural
    continuous-output combination — loads cleanly."""
    cfg = _build_config(
        fct_columns=[
            Column(name="m2_float", dtype="float", source="metric:m2"),
        ]
    )
    assert cfg is not None


def test_threshold_source_with_boolean_dtype_loads():
    """ThresholdSource is allowed to carry dtype: boolean — that's
    its design (the cell value is the threshold predicate). Verifies
    F12 didn't over-blacklist."""
    cfg = _build_config(
        fct_columns=[
            Column(
                name="m1_threshold",
                dtype="boolean",
                source="threshold:m1:above:3.0:for:2",
            ),
        ]
    )
    assert cfg is not None


# --- Bundled-template parity ------------------------------------------------


@pytest.mark.parametrize(
    "stem",
    ["saas", "hr", "education", "retail", "marketing"],
)
def test_bundled_templates_load_under_validator(stem):
    """Every bundled template loads. None of them combine dtype: boolean
    with metric: or lag: sources; this guards the migration path
    against a hidden YAML reference that would surface as a
    validator rejection."""
    cfg = load_config(CONFIGS / f"sample_{stem}.yaml")
    # Sanity: any boolean column must have a non-MetricSource,
    # non-LagSource source.
    for tbl in cfg.tables:
        for col in tbl.columns:
            if col.dtype == "boolean":
                assert (
                    "metric:" not in col.source
                ), f"{stem}: {tbl.name}.{col.name} violates F12 (boolean × metric:source)"
                assert not col.source.startswith(
                    "lag:"
                ), f"{stem}: {tbl.name}.{col.name} violates F12 (boolean × lag:source)"
