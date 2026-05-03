"""M117: entity expansion + Table.count sub-entity multiplier.

Mirrors the acceptance criteria in mission-117-entity-expansion.md. The
core invariant under test: each segment with ``count: N`` produces N
individual ``Entity(size=1)`` objects, and any sub-entity row multiplier
travels via ``Table.count`` (default 1) — the two compose multiplicatively
in ``dimensions.build_dim_subentity`` so engine-direct and builder paths
share one code path.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from plotsim import (
    create,
    create_from_yaml,
    generate_tables,
    load_config,
    validate,
)
from plotsim.config import (
    Column,
    Entity,
    PlotsimConfig,
    PoolSource,
    Table,
    parse_source,
)
from plotsim.validation import validate_value_pool_coverage


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_YAML = REPO_ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml"
BUNDLED_TEMPLATES = (
    "sample_education.yaml",
    "sample_hr.yaml",
    "sample_marketing.yaml",
    "sample_retail.yaml",
    "sample_saas.yaml",
)


def _silent_create(**kwargs: Any) -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create(**kwargs)


def _silent_create_from_yaml(path: Path) -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create_from_yaml(str(path))


def _generate(cfg: PlotsimConfig, seed: int = 42) -> dict[str, pd.DataFrame]:
    return generate_tables(cfg, np.random.default_rng(seed))


# ── Entity expansion (auto-schema) ──────────────────────────────────────────


def test_auto_schema_single_segment_count_80_produces_80_entities():
    cfg = _silent_create(
        about="single-cohort",
        unit="company",
        window=("2024-01", "2024-12"),
        metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
        segments=[{"name": "alpha", "count": 80, "archetype": "growth"}],
    )
    assert len(cfg.entities) == 80
    assert all(e.size == 1 for e in cfg.entities)
    assert all(e.archetype == "alpha" for e in cfg.entities)


def test_expanded_entity_names_are_zero_padded():
    cfg = _silent_create(
        about="naming",
        unit="company",
        window=("2024-01", "2024-06"),
        metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
        segments=[{"name": "alpha", "count": 80, "archetype": "growth"}],
    )
    names = [e.name for e in cfg.entities]
    assert names[0] == "alpha_0000"
    assert names[79] == "alpha_0079"
    # Every name follows the {segment}_{i:04d} pattern.
    for i, n in enumerate(names):
        assert n == f"alpha_{i:04d}"


def test_auto_schema_dim_unit_row_count_matches_expansion():
    cfg = _silent_create(
        about="row-count",
        unit="company",
        window=("2024-01", "2024-12"),
        metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
        segments=[{"name": "alpha", "count": 80, "archetype": "growth"}],
    )
    tables = _generate(cfg)
    assert len(tables["dim_company"]) == 80


def test_auto_schema_fct_unit_row_count_is_entities_times_periods():
    cfg = _silent_create(
        about="fact-rows",
        unit="company",
        window=("2024-01", "2024-12"),
        metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
        segments=[{"name": "alpha", "count": 80, "archetype": "growth"}],
    )
    n_periods = cfg.time_window.period_count()
    tables = _generate(cfg)
    assert len(tables["fct_company"]) == 80 * n_periods


def test_bug_reproduction_three_segments_180_entities_4320_fact_rows():
    """Exact reproduction of the bug report: 3 segments [80, 60, 40] over a
    24-month window → dim_vehicle 180 rows, fct_vehicle 4320 rows."""
    cfg = _silent_create(
        about="vehicle fleet",
        unit="vehicle",
        window=("2023-01", "2024-12"),  # 24 monthly periods
        metrics=[{"name": "utilization", "type": "score", "polarity": "positive"}],
        segments=[
            {"name": "new_fleet", "count": 80, "archetype": "growth"},
            {"name": "mid_life",  "count": 60, "archetype": "flat"},
            {"name": "retiring",  "count": 40, "archetype": "decline"},
        ],
    )
    assert len(cfg.entities) == 180
    assert cfg.time_window.period_count() == 24
    tables = _generate(cfg)
    assert len(tables["dim_vehicle"]) == 180
    assert len(tables["fct_vehicle"]) == 4320  # 180 × 24


def test_entities_from_same_segment_share_archetype():
    cfg = _silent_create(
        about="multi-cohort",
        unit="customer",
        window=("2024-01", "2024-06"),
        metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
        segments=[
            {"name": "loyal", "count": 30, "archetype": "growth"},
            {"name": "churned", "count": 20, "archetype": "decline"},
        ],
    )
    by_arch: dict[str, list[str]] = {}
    for e in cfg.entities:
        by_arch.setdefault(e.archetype, []).append(e.name)
    assert sorted(by_arch.keys()) == ["churned", "loyal"]
    assert all(n.startswith("loyal_") for n in by_arch["loyal"])
    assert all(n.startswith("churned_") for n in by_arch["churned"])
    assert len(by_arch["loyal"]) == 30
    assert len(by_arch["churned"]) == 20


# ── Entity expansion (explicit schema) ──────────────────────────────────────


def test_saas_template_yaml_loads():
    cfg = _silent_create_from_yaml(TEMPLATE_YAML)
    assert len(cfg.entities) == 95  # 20 + 25 + 15 + 15 + 10 + 10
    assert all(e.size == 1 for e in cfg.entities)


def test_saas_template_validate_returns_ok():
    cfg = _silent_create_from_yaml(TEMPLATE_YAML)
    tables = _generate(cfg)
    report = validate(cfg, tables)
    assert report.ok, [
        f"{i.check}: {i.message}" for i in report.errors
    ]


def test_saas_template_dim_user_has_95_rows_default_count():
    cfg = _silent_create_from_yaml(TEMPLATE_YAML)
    # dim_user is sub-entity (variable grain, FK to dim_company); default
    # Table.count=1 means one row per parent entity → 95 rows.
    dim_user_tbl = next(t for t in cfg.tables if t.name == "dim_user")
    assert dim_user_tbl.count == 1
    assert dim_user_tbl.grain == "variable"
    tables = _generate(cfg)
    assert len(tables["dim_user"]) == 95


def test_saas_template_fact_tables_have_2280_rows_each():
    cfg = _silent_create_from_yaml(TEMPLATE_YAML)
    tables = _generate(cfg)
    # 95 entities × 24 periods.
    expected = 95 * 24
    for fact_name in ("fct_engagement", "fct_revenue", "fct_support_tickets"):
        assert len(tables[fact_name]) == expected, (
            f"{fact_name}: expected {expected}, got {len(tables[fact_name])}"
        )


# ── Table.count sub-entity multiplier ───────────────────────────────────────


def test_table_count_3_on_variable_dim_produces_3x_rows():
    """A builder-authored sub-entity dim with ``count: 3`` produces three
    rows per parent entity. ``DimInput.count: 3`` → ``Table.count=3``,
    composed with ``Entity.size=1`` → 3 rows per entity in dim_user."""
    cfg = _silent_create(
        about="users-per-company",
        unit="company",
        window=("2024-01", "2024-06"),
        metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
        segments=[{"name": "team", "count": 5, "archetype": "growth"}],
        dimensions=[
            {"name": "dim_date", "per": "period", "columns": [
                {"name": "date_key", "type": "id"},
            ]},
            {"name": "dim_company", "per": "unit", "columns": [
                {"name": "company_id", "type": "id"},
                {"name": "company_name", "type": "faker.company"},
            ]},
            {"name": "dim_user", "per": "unit", "count": 3, "columns": [
                {"name": "user_id", "type": "id"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "user_name", "type": "faker.name"},
            ]},
        ],
        facts=[
            {"name": "fct_engagement", "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "engagement", "type": "metric.engagement"},
            ]},
        ],
    )
    dim_user_tbl = next(t for t in cfg.tables if t.name == "dim_user")
    assert dim_user_tbl.count == 3
    tables = _generate(cfg)
    # 5 entities × 3 = 15 dim_user rows.
    assert len(tables["dim_user"]) == 15


def test_table_count_gt_1_on_non_variable_dim_rejects_at_load():
    """``Table.count > 1`` is only meaningful on variable-grain dims; the
    engine validator rejects the misconfiguration."""
    with pytest.raises(ValueError, match="count.*variable"):
        Table(
            name="dim_company",
            type="dim",
            grain="per_entity",
            columns=[Column(name="company_id", dtype="id", source="pk")],
            primary_key="company_id",
            count=2,
        )


def test_engine_direct_config_with_entity_size_unaffected():
    """An engine-direct config with ``Entity.size=N`` and the default
    ``Table.count=1`` still produces N sub-entity rows per parent.
    Backward compatibility for the bundled engine templates."""
    cfg = load_config(str(REPO_ROOT / "plotsim" / "configs" / "sample_saas.yaml"))
    # sample_saas has dim_user as a sub-entity dim of dim_company; entity
    # sizes are 10, 30, 50 (sum 90).
    assert all(t.count == 1 for t in cfg.tables if t.type == "dim")
    sizes = [e.size for e in cfg.entities]
    assert sum(sizes) > len(sizes)  # at least one Entity.size > 1
    tables = generate_tables(cfg, np.random.default_rng(42))
    assert len(tables["dim_user"]) == sum(sizes)


def test_multiplicative_size_times_count():
    """Engine-direct config with ``Entity.size=5`` + ``Table.count=3``
    should yield 15 sub-entity rows per parent. Validates the
    multiplicative composition without going through the builder."""
    from plotsim.config import (
        Archetype, CurveSegment, Domain, Metric, OutputConfig,
        PlotsimConfig, TimeWindow, ValueRange,
    )
    cfg = PlotsimConfig(
        domain=Domain(
            name="multi", description="multi", entity_type="company",
            entity_label="Companies",
        ),
        time_window=TimeWindow(start="2024-01", end="2024-06", granularity="monthly"),
        seed=42,
        metrics=[Metric(
            name="engagement", label="Engagement", distribution="beta",
            params={"alpha": 2.0, "beta": 5.0}, polarity="positive",
            value_range=ValueRange(min=0.0, max=1.0),
        )],
        archetypes=[Archetype(
            name="growth", label="Growth", description="growth",
            curve_segments=[CurveSegment(
                curve="sigmoid", start_pct=0.0, end_pct=1.0,
                params={"midpoint": 0.5, "steepness": 5.0},
            )],
        )],
        entities=[Entity(name="cohort", archetype="growth", size=5)],
        tables=[
            Table(
                name="dim_date", type="dim", grain="per_period",
                columns=[Column(name="date_key", dtype="id", source="pk")],
                primary_key="date_key",
            ),
            Table(
                name="dim_company", type="dim", grain="per_entity",
                columns=[
                    Column(name="company_id", dtype="id", source="pk"),
                    Column(name="company_name", dtype="string",
                           source="generated:faker.company"),
                ],
                primary_key="company_id",
            ),
            Table(
                name="dim_user", type="dim", grain="variable",
                columns=[
                    Column(name="user_id", dtype="id", source="pk"),
                    Column(name="company_id", dtype="id",
                           source="fk:dim_company.company_id"),
                ],
                primary_key="user_id",
                foreign_keys=["dim_company.company_id"],
                count=3,
            ),
            Table(
                name="fct_engagement", type="fact",
                grain="per_entity_per_period",
                columns=[
                    Column(name="date_key", dtype="id",
                           source="fk:dim_date.date_key"),
                    Column(name="company_id", dtype="id",
                           source="fk:dim_company.company_id"),
                    Column(name="engagement", dtype="float",
                           source="metric:engagement"),
                ],
                primary_key=["date_key", "company_id"],
                foreign_keys=["dim_date.date_key", "dim_company.company_id"],
            ),
        ],
        output=OutputConfig(format="csv", directory="output"),
    )
    tables = generate_tables(cfg, np.random.default_rng(0))
    # 1 Entity, size=5, Table.count=3 → 15 dim_user rows.
    assert len(tables["dim_user"]) == 15


# ── PoolSource and attributes (segment.count column) ────────────────────────


def test_segment_count_column_translates_to_pool_source():
    """The builder's ``segment.count`` column type now resolves to a
    PoolSource — pre-M117 it was ``derived:size``, which after expansion
    would have emitted 1 for every row."""
    cfg = _silent_create_from_yaml(TEMPLATE_YAML)
    dim_company = next(t for t in cfg.tables if t.name == "dim_company")
    cohort_col = next(c for c in dim_company.columns if c.name == "cohort_size")
    parsed = parse_source(cohort_col.source)
    assert isinstance(parsed, PoolSource)
    assert cohort_col.value_pool is not None


def test_segment_count_value_pool_keyed_by_expanded_entity_names():
    cfg = _silent_create_from_yaml(TEMPLATE_YAML)
    dim_company = next(t for t in cfg.tables if t.name == "dim_company")
    cohort_col = next(c for c in dim_company.columns if c.name == "cohort_size")
    pool_keys = set(cohort_col.value_pool.keys())
    entity_names = {e.name for e in cfg.entities}
    assert pool_keys == entity_names


def test_segment_count_value_pool_carries_original_cohort_size():
    """Every expanded entity in a segment maps to that segment's original
    ``count`` — full coverage, not a sample of prefixes."""
    cfg = _silent_create_from_yaml(TEMPLATE_YAML)
    dim_company = next(t for t in cfg.tables if t.name == "dim_company")
    cohort_col = next(c for c in dim_company.columns if c.name == "cohort_size")

    # Re-read the saas template's segments from the YAML so the assertion
    # can't drift if the template changes counts; iterate over all 6
    # segments and assert each expanded entity's pool list = [str(count)].
    import yaml as _yaml
    src = _yaml.safe_load(TEMPLATE_YAML.read_text(encoding="utf-8"))
    segment_counts = {s["name"]: s["count"] for s in src["segments"]}
    assert len(segment_counts) == 6, (
        f"saas template should have 6 segments; got {sorted(segment_counts)}"
    )

    seen_segments: set[str] = set()
    for entity_name, pool_values in cohort_col.value_pool.items():
        # Recover the segment from the entity name. Segment names contain
        # underscores (``promising_client``, ``steady_enterprise``), so a
        # naive ``rsplit('_', 1)[0]`` works because the suffix is always
        # the 4-digit zero-padded index appended by the M117 expansion.
        prefix, _suffix = entity_name.rsplit("_", 1)
        assert prefix in segment_counts, (
            f"{entity_name}: prefix {prefix!r} not in template segments "
            f"{sorted(segment_counts)}"
        )
        expected = [str(segment_counts[prefix])]
        assert pool_values == expected, (
            f"{entity_name}: expected pool {expected}, got {pool_values}"
        )
        seen_segments.add(prefix)
    # Every declared segment is represented in the pool.
    assert seen_segments == set(segment_counts), (
        f"missing segments: {sorted(set(segment_counts) - seen_segments)}"
    )


def test_validate_value_pool_coverage_passes_with_expanded_names():
    cfg = _silent_create_from_yaml(TEMPLATE_YAML)
    errors = validate_value_pool_coverage(cfg)
    assert errors == [], f"value_pool coverage errors: {errors}"


def test_segment_count_column_resolves_to_cohort_size_in_dim_rows():
    """Sanity: the generated dim_company.cohort_size cells carry the
    original cohort population, not the expanded ``Entity.size``=1."""
    cfg = _silent_create_from_yaml(TEMPLATE_YAML)
    tables = _generate(cfg)
    dim_company = tables["dim_company"]
    # dim_company is SCD-expanded; restrict to is_current rows for a
    # 1:1 mapping with entities.
    current = dim_company[dim_company["is_current"].astype(bool)].reset_index(drop=True)
    assert len(current) == 95
    # Build expected map: company_id (in cfg.entities order) → cohort_size.
    expected = []
    for s in (
        ("promising_client", 20), ("steady_enterprise", 25),
        ("slow_churn", 15), ("seasonal_accounts", 15),
        ("dormant", 10), ("turnaround", 10),
    ):
        expected.extend([s[1]] * s[1])
    actual = current["cohort_size"].tolist()
    assert actual == expected, (
        f"cohort_size mismatch: first 5 expected={expected[:5]} actual={actual[:5]}"
    )


# ── Backward compatibility ──────────────────────────────────────────────────


@pytest.mark.parametrize("template", BUNDLED_TEMPLATES)
def test_bundled_engine_templates_still_load_and_validate(template):
    """All five bundled engine-direct templates load via ``load_config``
    (interpreter bypassed), generate, and validate without errors."""
    path = REPO_ROOT / "plotsim" / "configs" / template
    if not path.exists():
        pytest.skip(f"{template} not present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load_config(str(path))
    tables = generate_tables(cfg, np.random.default_rng(42))
    report = validate(cfg, tables)
    assert report.ok, [f"{i.check}: {i.message}" for i in report.errors]


def test_engine_direct_config_table_count_defaults_to_1():
    """Loading an engine-direct config without an explicit ``count`` field
    on its tables sets ``Table.count == 1`` everywhere."""
    cfg = load_config(str(REPO_ROOT / "plotsim" / "configs" / "sample_saas.yaml"))
    assert all(t.count == 1 for t in cfg.tables)


# ── Determinism ─────────────────────────────────────────────────────────────


def test_same_input_same_seed_byte_identical_output():
    """Two runs of ``generate_tables`` with the same config + seed produce
    identical DataFrames."""
    cfg = _silent_create(
        about="determinism",
        unit="company",
        window=("2024-01", "2024-06"),
        metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
        segments=[
            {"name": "alpha", "count": 30, "archetype": "growth"},
            {"name": "beta", "count": 20, "archetype": "decline"},
        ],
    )
    a = generate_tables(cfg, np.random.default_rng(42))
    b = generate_tables(cfg, np.random.default_rng(42))
    assert sorted(a.keys()) == sorted(b.keys())
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_yields_different_values_same_structure():
    cfg = _silent_create(
        about="seed-variation",
        unit="company",
        window=("2024-01", "2024-06"),
        metrics=[{"name": "engagement", "type": "score", "polarity": "positive"}],
        segments=[
            {"name": "alpha", "count": 30, "archetype": "growth"},
            {"name": "beta", "count": 20, "archetype": "decline"},
        ],
    )
    a = generate_tables(cfg, np.random.default_rng(1))
    b = generate_tables(cfg, np.random.default_rng(7))
    # Same shape.
    for name in a:
        assert a[name].shape == b[name].shape
        assert list(a[name].columns) == list(b[name].columns)
    # At least one fact column differs.
    fct_a = a["fct_company"]["engagement"].to_numpy()
    fct_b = b["fct_company"]["engagement"].to_numpy()
    assert not np.array_equal(fct_a, fct_b), (
        "different seeds produced identical fact column"
    )
