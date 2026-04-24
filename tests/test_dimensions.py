"""Tests for plotsim.dimensions — Mission 005 acceptance criteria.

Covers:
  - dim_date: monthly/weekly/daily row counts, YYYYMMDD keys, sequential
    period_index, spot checks on quarter/month, daily-only columns.
  - dim_entity: 1-row-per-entity, unique PKs, derived:archetype/size, faker
    generator, Faker method dispatch and typo surfacing.
  - dim_subentity: total row count = sum(entity.size), correct parent FK
    per block, PK uniqueness across the whole table, FK integrity.
  - dim_reference: row count driven by longest static-CSV column, broadcast
    of single-value columns, dtype coercion.
  - build_all_dimensions: routing, build order (reference → entity → sub-
    entity), orchestration against the two sample YAMLs.
  - Determinism: same seed → byte-identical frames incl. faker output;
    different seeds → different faker values but identical structure/row
    counts.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotsim import load_config
from plotsim.config import (
    Column,
    Entity,
    Table,
    TimeWindow,
)
from plotsim.dimensions import (
    build_all_dimensions,
    build_dim_date,
    build_dim_entity,
    build_dim_reference,
    build_dim_subentity,
)

ROOT = Path(__file__).resolve().parent.parent
SAAS_YAML = ROOT / "plotsim" / "configs" / "sample_saas.yaml"
HR_YAML = ROOT / "plotsim" / "configs" / "sample_hr.yaml"


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- dim_date ----------------------------------------------------------------


def test_dim_date_monthly_row_count_inclusive():
    tw = TimeWindow(start="2023-01", end="2024-12", granularity="monthly")
    df = build_dim_date(tw)
    assert len(df) == 24  # 12 + 12 months inclusive


def test_dim_date_monthly_single_month_span():
    tw = TimeWindow(start="2024-01", end="2024-02", granularity="monthly")
    df = build_dim_date(tw)
    assert len(df) == 2
    assert df["period_label"].tolist() == ["2024-01", "2024-02"]


def test_dim_date_monthly_date_key_is_yyyymmdd_first_of_month():
    tw = TimeWindow(start="2023-01", end="2023-03", granularity="monthly")
    df = build_dim_date(tw)
    assert df["date_key"].tolist() == [20230101, 20230201, 20230301]
    assert df["date"].tolist() == [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)]


def test_dim_date_monthly_quarter_and_month_correct():
    tw = TimeWindow(start="2024-01", end="2024-12", granularity="monthly")
    df = build_dim_date(tw)
    # 2024-03 (row 2, 0-indexed) → Q1, month 3
    row_march = df.iloc[2]
    assert row_march["month"] == 3
    assert row_march["quarter"] == 1
    # 2024-07 → Q3
    assert df.iloc[6]["quarter"] == 3
    # Month name
    assert row_march["month_name"] == "March"


def test_dim_date_period_index_zero_based_sequential():
    tw = TimeWindow(start="2023-01", end="2024-06", granularity="monthly")
    df = build_dim_date(tw)
    assert df["period_index"].tolist() == list(range(18))


def test_dim_date_date_key_unique():
    tw = TimeWindow(start="2022-01", end="2024-12", granularity="monthly")
    df = build_dim_date(tw)
    assert df["date_key"].is_unique


def test_dim_date_monthly_no_daily_columns():
    tw = TimeWindow(start="2024-01", end="2024-12", granularity="monthly")
    df = build_dim_date(tw)
    for col in ("day_of_week", "day_of_month", "is_weekend"):
        assert col not in df.columns


def test_dim_date_weekly_row_count_matches_iso_weeks():
    tw = TimeWindow(start="2024-01", end="2024-03", granularity="weekly")
    df = build_dim_date(tw)
    # Jan 1, 2024 is ISO week 2024-W01; Mar 31, 2024 is ISO week 2024-W13.
    assert len(df) == 13
    assert df["period_label"].iloc[0] == "2024-W01"
    # Date is Monday of that ISO week.
    assert df["date"].iloc[0] == date(2024, 1, 1)


def test_dim_date_weekly_no_daily_columns():
    tw = TimeWindow(start="2024-01", end="2024-02", granularity="weekly")
    df = build_dim_date(tw)
    for col in ("day_of_week", "day_of_month", "is_weekend"):
        assert col not in df.columns


def test_dim_date_daily_row_count_matches_days():
    # TimeWindow requires strict start < end, so widen to two months and
    # slice in-test. Feb 2024 has 29 days (leap year), Jan has 31.
    tw = TimeWindow(start="2024-01", end="2024-02", granularity="daily")
    df = build_dim_date(tw)
    assert len(df) == 31 + 29
    feb = df[df["date"].apply(lambda d: d.month == 2)].reset_index(drop=True)
    assert len(feb) == 29
    assert feb["date"].iloc[0] == date(2024, 2, 1)
    assert feb["date"].iloc[-1] == date(2024, 2, 29)


def test_dim_date_daily_has_daily_columns():
    tw = TimeWindow(start="2024-01", end="2024-02", granularity="daily")
    df = build_dim_date(tw)
    assert "day_of_week" in df.columns
    assert "day_of_month" in df.columns
    assert "is_weekend" in df.columns
    # 2024-01-06 is a Saturday (index 5).
    row = df[df["date"] == date(2024, 1, 6)].iloc[0]
    assert row["day_of_week"] == "Saturday"
    assert bool(row["is_weekend"]) is True
    # 2024-01-08 is a Monday.
    row_mon = df[df["date"] == date(2024, 1, 8)].iloc[0]
    assert row_mon["day_of_week"] == "Monday"
    assert bool(row_mon["is_weekend"]) is False


# --- dim_entity --------------------------------------------------------------


def _saas_entities() -> list[Entity]:
    return [
        Entity(name="acme_corp_cohort", archetype="rocket_then_cliff", size=50),
        Entity(name="globex_cohort", archetype="steady_grower", size=30),
        Entity(name="hooli_cohort", archetype="zombie_account", size=10),
    ]


def _table(name: str, grain: str, columns: list[Column], pk: str | list[str],
           ttype: str = "dim", fks: list[str] | None = None) -> Table:
    return Table(
        name=name,
        type=ttype,  # type: ignore[arg-type]
        grain=grain,  # type: ignore[arg-type]
        columns=columns,
        primary_key=pk,
        foreign_keys=fks or [],
    )


def test_dim_entity_one_row_per_entity():
    tbl = _table(
        "dim_company", "per_entity",
        [
            Column(name="company_id", dtype="id", source="pk"),
            Column(name="company_name", dtype="string", source="generated:faker.company"),
            Column(name="archetype_name", dtype="string", source="derived:archetype"),
            Column(name="cohort_size", dtype="int", source="derived:size"),
        ],
        "company_id",
    )
    entities = _saas_entities()
    df = build_dim_entity(tbl, entities, _rng(1))
    assert len(df) == 3


def test_dim_entity_pk_values_unique_and_prefixed():
    tbl = _table(
        "dim_company", "per_entity",
        [Column(name="company_id", dtype="id", source="pk"),
         Column(name="x", dtype="string", source="static:v")],
        "company_id",
    )
    df = build_dim_entity(tbl, _saas_entities(), _rng(0))
    assert df["company_id"].is_unique
    assert df["company_id"].iloc[0].startswith("c-")


def test_dim_entity_derived_fields_populated():
    tbl = _table(
        "dim_company", "per_entity",
        [
            Column(name="company_id", dtype="id", source="pk"),
            Column(name="cohort_size", dtype="int", source="derived:size"),
            Column(name="archetype_name", dtype="string", source="derived:archetype"),
        ],
        "company_id",
    )
    df = build_dim_entity(tbl, _saas_entities(), _rng(0))
    assert df["cohort_size"].tolist() == [50, 30, 10]
    assert df["archetype_name"].tolist() == [
        "rocket_then_cliff", "steady_grower", "zombie_account",
    ]


def test_dim_entity_entity_name_generator_populated():
    tbl = _table(
        "dim_company", "per_entity",
        [
            Column(name="company_id", dtype="id", source="pk"),
            Column(name="entity_name", dtype="string", source="generated:entity_name"),
        ],
        "company_id",
    )
    df = build_dim_entity(tbl, _saas_entities(), _rng(0))
    assert df["entity_name"].tolist() == [
        "acme_corp_cohort", "globex_cohort", "hooli_cohort",
    ]


def test_dim_entity_faker_values_non_null_non_empty():
    tbl = _table(
        "dim_company", "per_entity",
        [
            Column(name="company_id", dtype="id", source="pk"),
            Column(name="company_name", dtype="string", source="generated:faker.company"),
        ],
        "company_id",
    )
    df = build_dim_entity(tbl, _saas_entities(), _rng(7))
    assert df["company_name"].notna().all()
    assert (df["company_name"].str.len() > 0).all()


def test_dim_entity_unsupported_faker_provider_raises_at_build():
    tbl = _table(
        "dim_company", "per_entity",
        [
            Column(name="company_id", dtype="id", source="pk"),
            Column(name="bogus", dtype="string", source="generated:faker.not_a_method"),
        ],
        "company_id",
    )
    with pytest.raises(ValueError, match="not_a_method"):
        build_dim_entity(tbl, _saas_entities(), _rng(0))


def test_dim_entity_unsupported_derived_field_raises():
    tbl = _table(
        "dim_company", "per_entity",
        [
            Column(name="company_id", dtype="id", source="pk"),
            Column(name="weird", dtype="string", source="derived:nonsense"),
        ],
        "company_id",
    )
    with pytest.raises(ValueError, match="derived field"):
        build_dim_entity(tbl, _saas_entities(), _rng(0))


# --- dim_reference -----------------------------------------------------------


def test_dim_reference_row_count_from_longest_static_column():
    tbl = _table(
        "dim_plan", "per_reference",
        [
            Column(name="plan_id", dtype="id", source="pk"),
            Column(name="plan_name", dtype="string",
                   source="static:starter,professional,enterprise"),
            Column(name="monthly_price", dtype="float", source="static:99.0"),
        ],
        "plan_id",
    )
    df = build_dim_reference(tbl, _rng(0))
    assert len(df) == 3
    assert df["plan_name"].tolist() == ["starter", "professional", "enterprise"]
    # Single-value column broadcasts to all rows.
    assert df["monthly_price"].tolist() == [99.0, 99.0, 99.0]


def test_dim_reference_single_value_gives_single_row():
    tbl = _table(
        "dim_plan", "per_reference",
        [
            Column(name="plan_id", dtype="id", source="pk"),
            Column(name="plan_name", dtype="string", source="static:starter"),
            Column(name="monthly_price", dtype="float", source="static:99.00"),
        ],
        "plan_id",
    )
    df = build_dim_reference(tbl, _rng(0))
    assert len(df) == 1
    assert df["plan_name"].iloc[0] == "starter"
    assert df["monthly_price"].iloc[0] == 99.0


def test_dim_reference_pk_unique():
    tbl = _table(
        "dim_plan", "per_reference",
        [
            Column(name="plan_id", dtype="id", source="pk"),
            Column(name="plan_name", dtype="string", source="static:a,b,c,d"),
        ],
        "plan_id",
    )
    df = build_dim_reference(tbl, _rng(0))
    assert df["plan_id"].is_unique
    assert df["plan_id"].tolist() == ["p-001", "p-002", "p-003", "p-004"]


def test_dim_reference_dtype_coercion():
    tbl = _table(
        "dim_flags", "per_reference",
        [
            Column(name="flag_id", dtype="id", source="pk"),
            Column(name="is_on", dtype="boolean", source="static:true,false"),
            Column(name="count", dtype="int", source="static:1,2"),
        ],
        "flag_id",
    )
    df = build_dim_reference(tbl, _rng(0))
    assert df["is_on"].tolist() == [True, False]
    assert df["count"].tolist() == [1, 2]


# --- dim_subentity -----------------------------------------------------------


def _dim_user_table() -> Table:
    return _table(
        "dim_user", "variable",
        [
            Column(name="user_id", dtype="id", source="pk"),
            Column(name="company_id", dtype="id", source="fk:dim_company.company_id"),
            Column(name="user_name", dtype="string", source="generated:faker.name"),
            Column(name="role", dtype="string", source="static:member"),
        ],
        "user_id",
        fks=["dim_company.company_id"],
    )


def _dim_company_table() -> Table:
    return _table(
        "dim_company", "per_entity",
        [
            Column(name="company_id", dtype="id", source="pk"),
            Column(name="company_name", dtype="string", source="generated:faker.company"),
        ],
        "company_id",
    )


def test_dim_subentity_total_rows_equals_sum_of_sizes():
    entities = _saas_entities()
    parent = build_dim_entity(_dim_company_table(), entities, _rng(0))
    df = build_dim_subentity(_dim_user_table(), entities, parent, _rng(1))
    assert len(df) == 50 + 30 + 10


def test_dim_subentity_fk_values_match_parent_per_block():
    entities = _saas_entities()
    parent = build_dim_entity(_dim_company_table(), entities, _rng(0))
    df = build_dim_subentity(_dim_user_table(), entities, parent, _rng(1))
    parent_ids = parent["company_id"].tolist()
    # Block 1: 50 rows with parent_ids[0].
    assert df["company_id"].iloc[:50].tolist() == [parent_ids[0]] * 50
    # Block 2: 30 rows with parent_ids[1].
    assert df["company_id"].iloc[50:80].tolist() == [parent_ids[1]] * 30
    # Block 3: 10 rows with parent_ids[2].
    assert df["company_id"].iloc[80:].tolist() == [parent_ids[2]] * 10


def test_dim_subentity_pk_unique_across_full_table():
    entities = _saas_entities()
    parent = build_dim_entity(_dim_company_table(), entities, _rng(0))
    df = build_dim_subentity(_dim_user_table(), entities, parent, _rng(1))
    assert df["user_id"].is_unique


def test_dim_subentity_fk_integrity_with_parent():
    entities = _saas_entities()
    parent = build_dim_entity(_dim_company_table(), entities, _rng(0))
    df = build_dim_subentity(_dim_user_table(), entities, parent, _rng(1))
    assert set(df["company_id"]).issubset(set(parent["company_id"]))


def test_dim_subentity_static_value_broadcast():
    entities = _saas_entities()
    parent = build_dim_entity(_dim_company_table(), entities, _rng(0))
    df = build_dim_subentity(_dim_user_table(), entities, parent, _rng(1))
    assert (df["role"] == "member").all()


# --- build_all_dimensions ----------------------------------------------------


def test_build_all_dimensions_saas_produces_expected_tables():
    config = load_config(SAAS_YAML)
    dims = build_all_dimensions(config, _rng(config.seed))
    assert set(dims.keys()) == {"dim_date", "dim_company", "dim_user", "dim_plan"}


def test_build_all_dimensions_hr_produces_expected_tables():
    config = load_config(HR_YAML)
    dims = build_all_dimensions(config, _rng(config.seed))
    assert set(dims.keys()) == {"dim_date", "dim_employee", "dim_department"}


def test_build_all_dimensions_saas_row_counts():
    config = load_config(SAAS_YAML)
    dims = build_all_dimensions(config, _rng(config.seed))
    # 2023-01 to 2024-12 = 24 months.
    assert len(dims["dim_date"]) == 24
    # 3 cohort-entities.
    assert len(dims["dim_company"]) == 3
    # sum(50, 30, 10) sub-entity rows.
    assert len(dims["dim_user"]) == 90
    # dim_plan: static:starter + static:99.00 → 1 row.
    assert len(dims["dim_plan"]) == 1


def test_build_all_dimensions_hr_row_counts():
    config = load_config(HR_YAML)
    dims = build_all_dimensions(config, _rng(config.seed))
    # 2022-01 to 2024-12 = 36 months.
    assert len(dims["dim_date"]) == 36
    # 4 employee-entities (per_entity grain).
    assert len(dims["dim_employee"]) == 4
    # dim_department: 1 static row.
    assert len(dims["dim_department"]) == 1


def test_build_all_dimensions_saas_cross_table_fk_integrity():
    config = load_config(SAAS_YAML)
    dims = build_all_dimensions(config, _rng(config.seed))
    company_ids = set(dims["dim_company"]["company_id"])
    user_fks = set(dims["dim_user"]["company_id"])
    assert user_fks.issubset(company_ids)
    assert user_fks == company_ids  # every company has at least one user


def test_build_all_dimensions_hr_fk_backfill_to_reference():
    config = load_config(HR_YAML)
    dims = build_all_dimensions(config, _rng(config.seed))
    # dim_employee has FK to dim_department → should be filled, not null.
    assert dims["dim_employee"]["dept_id"].notna().all()
    assert set(dims["dim_employee"]["dept_id"]).issubset(
        set(dims["dim_department"]["dept_id"])
    )


# --- Determinism -------------------------------------------------------------


def test_same_seed_identical_output_including_faker():
    config = load_config(SAAS_YAML)
    a = build_all_dimensions(config, _rng(config.seed))
    b = build_all_dimensions(config, _rng(config.seed))
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_different_faker_same_structure():
    config = load_config(SAAS_YAML)
    a = build_all_dimensions(config, _rng(1))
    b = build_all_dimensions(config, _rng(2))
    # Same shape, same columns, same row counts.
    for name in a:
        assert list(a[name].columns) == list(b[name].columns)
        assert len(a[name]) == len(b[name])
    # Faker-generated company names should differ at least once across seeds.
    assert not (a["dim_company"]["company_name"]
                .eq(b["dim_company"]["company_name"]).all())
