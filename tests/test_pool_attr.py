"""M122 — pool.{attr} column type and segment.attributes value pools.

Covers the AC subset under "pool.{attr}" in
``project/missions/mission-122-builder-feature-parity.md``:

  * ``pool.industry`` resolves to a ``pool:industry`` source with a
    ``value_pool`` keyed by every expanded entity name.
  * Each entity's pool entry matches its segment's attribute list.
  * Scalar attribute values wrap into a single-element list.
  * Segment missing a declared attribute → ``ValueError`` at the
    column-translate step.
  * Auto-schema with attributes-on-every-segment emits one
    ``pool:{attr}`` column per attribute, alphabetically ordered.
  * Auto-schema without attributes is unchanged from the pre-M122
    baseline.
  * The pre-M117 ``segment.count`` (now ``pool:cohort_size``) path is
    unaffected by the new vocabulary entry.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from plotsim.builder import create
from plotsim.builder.input import UserInput
from plotsim.builder.interpreter import interpret


def _input(**overrides: Any) -> UserInput:
    base: dict[str, Any] = {
        "about": "M122 pool.attr",
        "unit": "company",
        "window": {"start": "2024-01", "end": "2024-12"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
        ],
        "segments": [
            {"name": "alpha", "count": 5, "archetype": "growth"},
            {"name": "beta", "count": 5, "archetype": "flat"},
        ],
    }
    base.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return UserInput.model_validate(base)


# ── pool.{attr} on an explicit dim column ──────────────────────────────────


def test_pool_attr_resolves_to_pool_source_with_value_pool():
    cfg = interpret(
        _input(
            segments=[
                {
                    "name": "alpha",
                    "count": 3,
                    "archetype": "growth",
                    "attributes": {"industry": ["Tech", "Finance"]},
                },
                {
                    "name": "beta",
                    "count": 3,
                    "archetype": "flat",
                    "attributes": {"industry": ["Healthcare"]},
                },
            ],
            dimensions=[
                {
                    "name": "dim_company",
                    "per": "unit",
                    "columns": [
                        {"name": "company_id", "type": "id"},
                        {"name": "industry", "type": "pool.industry"},
                    ],
                },
            ],
        )
    )
    dim = next(t for t in cfg.tables if t.name == "dim_company")
    industry_col = next(c for c in dim.columns if c.name == "industry")
    assert industry_col.source == "pool:industry"
    assert industry_col.dtype == "string"
    assert industry_col.value_pool is not None
    assert set(industry_col.value_pool.keys()) == {f"alpha_{i:04d}" for i in range(3)} | {
        f"beta_{i:04d}" for i in range(3)
    }


def test_pool_attr_each_entity_pool_matches_segment_attribute_list():
    cfg = interpret(
        _input(
            segments=[
                {
                    "name": "alpha",
                    "count": 3,
                    "archetype": "growth",
                    "attributes": {"industry": ["Tech", "Finance"]},
                },
                {
                    "name": "beta",
                    "count": 3,
                    "archetype": "flat",
                    "attributes": {"industry": ["Healthcare", "Retail"]},
                },
            ],
            dimensions=[
                {
                    "name": "dim_company",
                    "per": "unit",
                    "columns": [
                        {"name": "company_id", "type": "id"},
                        {"name": "industry", "type": "pool.industry"},
                    ],
                },
            ],
        )
    )
    industry_col = next(
        c for t in cfg.tables if t.name == "dim_company" for c in t.columns if c.name == "industry"
    )
    pool = industry_col.value_pool
    for i in range(3):
        assert pool[f"alpha_{i:04d}"] == ["Tech", "Finance"]
    for i in range(3):
        assert pool[f"beta_{i:04d}"] == ["Healthcare", "Retail"]


def test_pool_attr_scalar_attribute_wrapped_in_single_element_list():
    cfg = interpret(
        _input(
            segments=[
                {
                    "name": "alpha",
                    "count": 3,
                    "archetype": "growth",
                    "attributes": {"tier": "enterprise"},
                },
                {
                    "name": "beta",
                    "count": 3,
                    "archetype": "flat",
                    "attributes": {"tier": "starter"},
                },
            ],
            dimensions=[
                {
                    "name": "dim_company",
                    "per": "unit",
                    "columns": [
                        {"name": "company_id", "type": "id"},
                        {"name": "tier", "type": "pool.tier"},
                    ],
                },
            ],
        )
    )
    tier_col = next(
        c for t in cfg.tables if t.name == "dim_company" for c in t.columns if c.name == "tier"
    )
    pool = tier_col.value_pool
    assert pool["alpha_0000"] == ["enterprise"]
    assert pool["beta_0000"] == ["starter"]


def test_pool_attr_numeric_scalar_stringified():
    """PoolSource columns are dtype=string; numeric attribute values
    must be stringified in the value_pool so the engine writes string
    cells without dtype drift.
    """
    cfg = interpret(
        _input(
            segments=[
                {
                    "name": "alpha",
                    "count": 3,
                    "archetype": "growth",
                    "attributes": {"founded_year": 2010},
                },
                {
                    "name": "beta",
                    "count": 3,
                    "archetype": "flat",
                    "attributes": {"founded_year": 2018},
                },
            ],
            dimensions=[
                {
                    "name": "dim_company",
                    "per": "unit",
                    "columns": [
                        {"name": "company_id", "type": "id"},
                        {"name": "founded_year", "type": "pool.founded_year"},
                    ],
                },
            ],
        )
    )
    col = next(
        c
        for t in cfg.tables
        if t.name == "dim_company"
        for c in t.columns
        if c.name == "founded_year"
    )
    assert col.value_pool["alpha_0000"] == ["2010"]
    assert col.value_pool["beta_0000"] == ["2018"]


def test_pool_attr_missing_on_some_segments_raises():
    """Strict: every segment must declare the attribute referenced by a
    ``pool.{attr}`` column. Missing it on any segment makes the auto-schema
    omit the attribute, and an explicit ``pool.{attr}`` column then sees
    an empty pool entry — caught at column-translate time.
    """
    with pytest.raises(ValueError, match="not declared on every segment"):
        interpret(
            _input(
                segments=[
                    {
                        "name": "alpha",
                        "count": 3,
                        "archetype": "growth",
                        "attributes": {"industry": ["Tech"]},
                    },
                    {
                        "name": "beta",
                        "count": 3,
                        "archetype": "flat",
                        # `industry` deliberately missing on this segment
                        "attributes": {"region": ["EMEA"]},
                    },
                ],
                dimensions=[
                    {
                        "name": "dim_company",
                        "per": "unit",
                        "columns": [
                            {"name": "company_id", "type": "id"},
                            {"name": "industry", "type": "pool.industry"},
                        ],
                    },
                ],
            )
        )


def test_pool_attr_on_per_entity_per_period_fact_accepted():
    """``pool.{attr}`` on a per_entity_per_period fact column now
    interprets cleanly and wires through ``_fact_scalar_pool`` /
    ``_fact_vec_pool``. The widening over the M19-Fix-1 baseline
    (which accepted variable-grain facts, per_parent_row children,
    and events) covers the most common fact grain — one row per
    (entity, period).
    """
    cfg = interpret(
        _input(
            segments=[
                {
                    "name": "alpha",
                    "count": 3,
                    "archetype": "growth",
                    "attributes": {"industry": ["Tech", "Finance"]},
                },
                {
                    "name": "beta",
                    "count": 3,
                    "archetype": "flat",
                    "attributes": {"industry": ["Healthcare"]},
                },
            ],
            dimensions=[
                {
                    "name": "dim_date",
                    "per": "period",
                    "columns": [
                        {"name": "date_key", "type": "id"},
                        {"name": "date", "type": "date"},
                    ],
                },
                {
                    "name": "dim_company",
                    "per": "unit",
                    "columns": [
                        {"name": "company_id", "type": "id"},
                    ],
                },
            ],
            facts=[
                {
                    "name": "fct_company",
                    "metrics": ["engagement"],
                    "columns": [
                        {"name": "date_key", "type": "ref.dim_date"},
                        {"name": "company_id", "type": "ref.dim_company"},
                        {"name": "engagement", "type": "metric.engagement"},
                        {"name": "industry", "type": "pool.industry"},
                    ],
                },
            ],
        )
    )
    fct = next(t for t in cfg.tables if t.name == "fct_company")
    industry_col = next(c for c in fct.columns if c.name == "industry")
    assert industry_col.source == "pool:industry"
    # Builder expands each segment into per-entity rows (alpha_0000, ...).
    # Every entity in a segment shares that segment's attribute pool.
    pool = industry_col.value_pool
    assert pool is not None
    alpha_keys = sorted(k for k in pool if k.startswith("alpha"))
    beta_keys = sorted(k for k in pool if k.startswith("beta"))
    assert len(alpha_keys) == 3 and len(beta_keys) == 3
    assert all(pool[k] == ["Tech", "Finance"] for k in alpha_keys)
    assert all(pool[k] == ["Healthcare"] for k in beta_keys)


# ── Auto-schema attribute columns ──────────────────────────────────────────


def test_auto_schema_with_attributes_adds_pool_columns_alphabetical():
    cfg = interpret(
        _input(
            segments=[
                {
                    "name": "alpha",
                    "count": 3,
                    "archetype": "growth",
                    "attributes": {
                        "tier": "enterprise",
                        "industry": ["Tech", "Finance"],
                        "region": ["US", "EMEA"],
                    },
                },
                {
                    "name": "beta",
                    "count": 3,
                    "archetype": "flat",
                    "attributes": {
                        "tier": "starter",
                        "industry": ["Healthcare"],
                        "region": ["APAC"],
                    },
                },
            ],
        )
    )
    dim = next(t for t in cfg.tables if t.name == "dim_company")
    # Expected: company_id, company_name, then attribute columns alphabetically
    pool_cols = [c for c in dim.columns if c.source.startswith("pool:")]
    assert [c.name for c in pool_cols] == ["industry", "region", "tier"]
    for col in pool_cols:
        assert col.dtype == "string"
        assert col.value_pool is not None
        assert len(col.value_pool) == 6  # 3 alpha + 3 beta


def test_auto_schema_without_attributes_unchanged():
    """No attributes → no pool columns on auto-generated dim_{unit}."""
    cfg = interpret(_input())  # no attributes
    dim = next(t for t in cfg.tables if t.name == "dim_company")
    assert all(not c.source.startswith("pool:") for c in dim.columns)
    assert {c.name for c in dim.columns} == {"company_id", "company_name"}


def test_auto_schema_partial_attribute_omitted_from_dim():
    """Only attributes declared on EVERY segment surface in the auto-schema.

    A partial attribute (declared on some segments, missing on others)
    cannot back a ``pool:`` column without leaving entities with no pool
    entry — the engine's ``validate_value_pool_coverage`` would reject
    that. The auto-schema simply omits these.
    """
    cfg = interpret(
        _input(
            segments=[
                {
                    "name": "alpha",
                    "count": 3,
                    "archetype": "growth",
                    "attributes": {"industry": ["Tech"], "region": ["US"]},
                },
                {
                    "name": "beta",
                    "count": 3,
                    "archetype": "flat",
                    # Only `industry`; `region` is partial.
                    "attributes": {"industry": ["Healthcare"]},
                },
            ],
        )
    )
    dim = next(t for t in cfg.tables if t.name == "dim_company")
    pool_cols = {c.name for c in dim.columns if c.source.startswith("pool:")}
    assert pool_cols == {"industry"}  # `region` excluded


# ── M117 segment.count path unaffected ──────────────────────────────────────


def test_segment_count_pool_still_works_after_m122():
    """M117's ``segment.count`` (resolves to ``pool:cohort_size``) must
    keep working after ``pool.{attr}`` lands.
    """
    cfg = interpret(
        _input(
            dimensions=[
                {
                    "name": "dim_company",
                    "per": "unit",
                    "columns": [
                        {"name": "company_id", "type": "id"},
                        {"name": "cohort_size", "type": "segment.count"},
                    ],
                },
            ],
        )
    )
    col = next(
        c
        for t in cfg.tables
        if t.name == "dim_company"
        for c in t.columns
        if c.name == "cohort_size"
    )
    assert col.source == "pool:cohort_size"
    assert col.value_pool is not None
    assert col.value_pool["alpha_0000"] == ["5"]


# ── Attribute validation surface (UserInput level) ─────────────────────────


def test_segment_attribute_value_must_be_scalar_or_list_of_scalars():
    with pytest.raises(ValueError, match="must be a scalar"):
        UserInput.model_validate(
            {
                "about": "x",
                "unit": "x",
                "window": {"start": "2024-01", "end": "2024-12"},
                "metrics": [{"name": "m", "type": "score", "polarity": "positive"}],
                "segments": [
                    {
                        "name": "a",
                        "count": 3,
                        "archetype": "growth",
                        "attributes": {"industry": {"nested": "dict"}},
                    },
                ],
            }
        )


def test_segment_attribute_empty_list_rejected():
    with pytest.raises(ValueError, match="list must be non-empty"):
        UserInput.model_validate(
            {
                "about": "x",
                "unit": "x",
                "window": {"start": "2024-01", "end": "2024-12"},
                "metrics": [{"name": "m", "type": "score", "polarity": "positive"}],
                "segments": [
                    {
                        "name": "a",
                        "count": 3,
                        "archetype": "growth",
                        "attributes": {"industry": []},
                    },
                ],
            }
        )


# ── End-to-end: builder → engine load passes ───────────────────────────────


def test_pool_attr_end_to_end_via_create():
    """Top-level ``create()`` accepts pool.{attr} input and produces a
    PlotsimConfig that the engine validates clean.
    """
    cfg = create(
        about="e2e",
        unit="company",
        window={"start": "2024-01", "end": "2024-12"},
        metrics=[
            {"name": "engagement", "type": "score", "polarity": "positive"},
        ],
        segments=[
            {
                "name": "alpha",
                "count": 3,
                "archetype": "growth",
                "attributes": {"industry": ["Tech", "Finance"]},
            },
            {
                "name": "beta",
                "count": 3,
                "archetype": "flat",
                "attributes": {"industry": ["Healthcare"]},
            },
        ],
    )
    industry_col = next(
        c for t in cfg.tables if t.name == "dim_company" for c in t.columns if c.name == "industry"
    )
    assert industry_col.source == "pool:industry"
