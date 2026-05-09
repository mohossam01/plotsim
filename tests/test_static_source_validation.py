"""F11 regression — _coerce_static / static-source ISO date validation (M102).

Pre-fix: a column declared with ``dtype: date`` and
``source: "static:not-a-date"`` loaded silently. At generation time,
``dimensions._coerce_static`` caught the ``ValueError`` from
``datetime.fromisoformat`` and returned the raw string, so the
resulting dim table held a column declared as date but containing
``str`` values. Downstream type-coerced consumers (CSV readers
expecting ``parse_dates=`` to work, dashboard joins on ISO date
keys, etc.) saw silent type corruption.

Post-fix:

* ``PlotsimConfig._cross_reference_integrity`` rejects malformed
  static dates at config load with a message naming the column,
  the bad value, and the expected ISO format. Multi-value statics
  (``"static:2024-01-01,2024-02-01,2024-03-01"``) are split on
  commas and each candidate is validated.
* ``dimensions._coerce_static`` raises ``ValueError`` rather than
  returning the raw string, providing defense-in-depth for
  programmatic ``PlotsimConfig`` construction that bypasses
  YAML-load validators.

Tests:

* ``test_static_date_with_invalid_format_rejected`` — config with
  ``static:not-a-date`` raises ``ValidationError`` at load.
* ``test_static_date_with_valid_iso_date_loads`` — single ISO date
  loads cleanly.
* ``test_static_date_csv_with_one_invalid_member_rejected`` —
  multi-value ``"2024-01-01,bogus,2024-03-01"`` raises with the
  bad token named.
* ``test_static_date_csv_all_valid_loads`` — multi-value of
  three valid ISO dates loads.
* ``test_static_non_date_dtype_with_invalid_string_loads`` —
  validation only fires on ``dtype: date``; a string-dtype column
  with ``static:not-a-date`` is unaffected.
* ``test_coerce_static_raises_on_malformed_date`` — direct unit on
  ``dimensions._coerce_static``.
"""

from __future__ import annotations

import warnings
from datetime import date

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
    SurrogateKeyWarning,
    Table,
    TimeWindow,
)
from plotsim.dimensions import _coerce_static


def _config_with_static_column(
    *,
    static_value: str,
    dtype: str = "date",
) -> PlotsimConfig:
    """Build a minimal config carrying one static-source column with the
    given dtype and value. Lives in dim_reference (a per-period dim with
    a single static column drives the row count via _split_static)."""
    metric = Metric(
        name="m",
        label="m",
        distribution="normal",
        params={"mu": 1.0, "sigma": 0.1},
        polarity="positive",
    )
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
        name="fct_m",
        type="fact",
        grain="per_entity_per_period",
        primary_key=["date_key", "entity_id"],
        foreign_keys=["dim_date.date_key", "dim_entity.entity_id"],
        columns=[
            Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_entity.entity_id"),
            Column(name="m", dtype="float", source="metric:m"),
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
    # The static column under test lives in a reference dim.
    dim_ref = Table(
        name="dim_ref",
        type="dim",
        grain="variable",
        primary_key="ref_id",
        columns=[
            Column(name="ref_id", dtype="id", source="pk"),
            Column(
                name="static_field",
                dtype=dtype,  # type: ignore[arg-type]
                source=f"static:{static_value}",
            ),
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
                end="2024-03",
                granularity="monthly",
            ),
            seed=0,
            metrics=[metric],
            archetypes=[arch],
            entities=[Entity(name="e1", archetype="flat", size=1)],
            tables=[dim_date, dim_entity, dim_ref, fct],
            output=OutputConfig(format="csv", directory="out/f11"),
        )


# --- Load-time validation ---------------------------------------------------


def test_static_date_with_invalid_format_rejected():
    """``static:not-a-date`` on a dtype:date column raises at load."""
    with pytest.raises(ValidationError) as exc_info:
        _config_with_static_column(static_value="not-a-date", dtype="date")
    msg = str(exc_info.value)
    assert "not-a-date" in msg, f"bad value not in message: {msg}"
    assert (
        "iso" in msg.lower() or "yyyy-mm-dd" in msg.lower()
    ), f"format hint missing from message: {msg}"
    assert "static_field" in msg


def test_static_date_with_valid_iso_date_loads():
    """A single ISO-format date loads."""
    cfg = _config_with_static_column(static_value="2024-06-15", dtype="date")
    assert any(any(c.name == "static_field" for c in tbl.columns) for tbl in cfg.tables)


def test_static_date_csv_with_one_invalid_member_rejected():
    """Multi-value statics with ANY invalid member raise."""
    with pytest.raises(ValidationError) as exc_info:
        _config_with_static_column(
            static_value="2024-01-01,bogus,2024-03-01",
            dtype="date",
        )
    assert "bogus" in str(exc_info.value)


def test_static_date_csv_all_valid_loads():
    """Multi-value of three valid ISO dates loads."""
    cfg = _config_with_static_column(
        static_value="2024-01-01,2024-02-01,2024-03-01",
        dtype="date",
    )
    assert cfg is not None


def test_static_non_date_dtype_with_invalid_string_loads():
    """The F11 validator only fires on dtype=date; a string-dtype
    column with the same raw value is unaffected."""
    cfg = _config_with_static_column(static_value="not-a-date", dtype="string")
    assert cfg is not None


# --- Direct unit on _coerce_static ------------------------------------------


def test_coerce_static_raises_on_malformed_date():
    """Defense-in-depth: programmatic construction that reaches
    generation time without going through PlotsimConfig still surfaces
    the malformed-date condition rather than silently returning a string."""
    with pytest.raises(ValueError) as exc_info:
        _coerce_static("not-a-date", "date")
    msg = str(exc_info.value)
    assert "not-a-date" in msg
    assert "iso" in msg.lower() or "yyyy-mm-dd" in msg.lower()


def test_coerce_static_returns_date_for_valid_input():
    """Pre-fix behavior preserved on the happy path."""
    result = _coerce_static("2024-06-15", "date")
    assert result == date(2024, 6, 15)


def test_coerce_static_unaffected_for_other_dtypes():
    """int / float / boolean / string paths unchanged."""
    assert _coerce_static("42", "int") == 42
    assert _coerce_static("3.14", "float") == pytest.approx(3.14)
    assert _coerce_static("true", "boolean") is True
    assert _coerce_static("not-a-date", "string") == "not-a-date"
