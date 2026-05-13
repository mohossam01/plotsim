"""Tests for the multi-source / overlap mode (0.6-M13).

Covers the 8 mission acceptance criteria:

  1. 2-source config produces 2 per-source dim tables plus the canonical.
  2. Name drift visible in source dims at the configured rate.
  3. ID schemes differ across sources.
  4. Attribute conflicts present at the configured rate.
  5. Manifest ``source_entity_mappings`` has one record per
     (entity, source) with all drifted fields listed.
  6. Single-source (no multi_source block) configs unchanged.
  7. Deterministic under seed.
  8. New template passes ``plotsim run`` / ``plotsim validate``.

The 8th criterion is exercised against the bundled
``crm_billing_overlap`` template; the rest run against minimal
hand-rolled configs so each AC has a dedicated, narrow assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import plotsim
from plotsim.config import (
    Archetype,
    Column,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    MultiSourceConfig,
    OutputConfig,
    PlotsimConfig,
    SourceDeclaration,
    Table,
    TimeWindow,
    ValueRange,
)
from plotsim.dimensions import build_all_dimensions
from plotsim.manifest import (
    MANIFEST_SCHEMA_VERSION,
    SourceEntityMapping,
    build_manifest,
)
from plotsim.multi_source import (
    _apply_name_drift,
    _generate_source_ids,
    apply_source_drift,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_config(
    *,
    n_entities: int = 8,
    multi_source: MultiSourceConfig | None = None,
) -> PlotsimConfig:
    """Minimal engine-direct config: one metric, one archetype, one dim_company.

    Three string-typed columns on ``dim_company`` so the drift module has a
    name-drift candidate (``company_name`` via ``faker.company``) plus a
    distinct attribute-drift candidate (``industry`` via
    ``faker.industry``). PK column is ``company_id`` (canonical
    ``c-NNN``).
    """
    return PlotsimConfig(
        domain=Domain(
            name="t",
            description="t",
            entity_type="company",
            entity_label="Companies",
        ),
        time_window=TimeWindow(start="2024-01", end="2024-04", granularity="monthly"),
        seed=2026,
        metrics=[
            Metric(
                name="mrr",
                label="MRR",
                distribution="lognorm",
                params={"s": 1.0, "scale": 1000.0},
                polarity="positive",
                value_range=ValueRange(min=0.0, max=10_000.0),
            )
        ],
        archetypes=[
            Archetype(
                name="g",
                label="g",
                description="g",
                curve_segments=[
                    CurveSegment(
                        start_pct=0.0,
                        end_pct=1.0,
                        curve="sigmoid",
                        params={"midpoint": 0.5, "steepness": 1.0},
                    )
                ],
            )
        ],
        entities=[Entity(name=f"e_{i:03d}", archetype="g", size=1) for i in range(n_entities)],
        tables=[
            Table(
                name="dim_date",
                type="dim",
                grain="per_period",
                columns=[
                    Column(name="date_key", dtype="id", source="pk"),
                    Column(name="date", dtype="date", source="generated:date_key"),
                ],
                primary_key="date_key",
            ),
            Table(
                name="dim_company",
                type="dim",
                grain="per_entity",
                columns=[
                    Column(name="company_id", dtype="id", source="pk"),
                    Column(
                        name="company_name",
                        dtype="string",
                        source="generated:faker.company",
                    ),
                    Column(
                        name="industry",
                        dtype="string",
                        source="generated:faker.industry",
                    ),
                ],
                primary_key="company_id",
            ),
        ],
        output=OutputConfig(format="csv", directory="output"),
        multi_source=multi_source,
    )


# ── AC1: 2 per-source dims + 1 canonical ───────────────────────────────────


def test_two_sources_produce_canonical_plus_two_per_source_dims():
    cfg = _make_config(
        multi_source=MultiSourceConfig(
            sources=[
                SourceDeclaration(name="crm"),
                SourceDeclaration(name="billing", key_scheme="numeric"),
            ]
        )
    )
    dims = build_all_dimensions(cfg, np.random.default_rng(cfg.seed))
    company_dims = sorted(d for d in dims if d.startswith("dim_company"))
    assert company_dims == [
        "dim_company",
        "dim_company_billing",
        "dim_company_crm",
    ]
    # All three carry the same row count.
    n_entities = len(cfg.entities)
    for name in company_dims:
        assert len(dims[name]) == n_entities


# ── AC2: name drift visible at the configured rate ─────────────────────────


def test_name_drift_visible_at_configured_rate():
    cfg = _make_config(
        n_entities=20,
        multi_source=MultiSourceConfig(
            sources=[
                SourceDeclaration(name="crm", name_drift_rate=1.0),
                # MultiSourceConfig.min_length=2 — the second source is a
                # placeholder we don't assert on here.
                SourceDeclaration(name="billing", name_drift_rate=0.0),
            ]
        ),
    )
    dims = build_all_dimensions(cfg, np.random.default_rng(cfg.seed))
    canonical = dims["dim_company"]
    drifted = dims["dim_company_crm"]
    # name_drift_rate=1.0 → every row's company_name should differ.
    n_drifted = sum(
        1
        for canonical_name, drifted_name in zip(canonical["company_name"], drifted["company_name"])
        if canonical_name != drifted_name
    )
    assert n_drifted == len(canonical)


def test_name_drift_zero_rate_leaves_names_intact():
    cfg = _make_config(
        n_entities=20,
        multi_source=MultiSourceConfig(
            sources=[
                SourceDeclaration(name="crm", name_drift_rate=0.0),
                SourceDeclaration(name="billing", name_drift_rate=0.0),
            ]
        ),
    )
    dims = build_all_dimensions(cfg, np.random.default_rng(cfg.seed))
    canonical_names = list(dims["dim_company"]["company_name"])
    drifted_names = list(dims["dim_company_crm"]["company_name"])
    assert canonical_names == drifted_names


# ── AC3: ID schemes differ across sources ──────────────────────────────────


def test_id_schemes_differ_across_sources():
    cfg = _make_config(
        n_entities=5,
        multi_source=MultiSourceConfig(
            sources=[
                SourceDeclaration(name="crm", key_scheme="prefix_padded"),
                SourceDeclaration(name="billing", key_scheme="numeric"),
                SourceDeclaration(name="archive", key_scheme="uuid_short"),
            ]
        ),
    )
    dims = build_all_dimensions(cfg, np.random.default_rng(cfg.seed))
    crm_ids = list(dims["dim_company_crm"]["company_id_crm"])
    billing_ids = list(dims["dim_company_billing"]["company_id_billing"])
    archive_ids = list(dims["dim_company_archive"]["company_id_archive"])
    # Prefix-padded shape: starts with the upper-case entity type.
    assert all(s.startswith("COMPANY-") for s in crm_ids)
    # Numeric: all-digits, no dash, no prefix.
    assert all(s.isdigit() for s in billing_ids)
    # uuid_short: 5-char lowercase hex.
    assert all(len(s) == 5 and all(c in "0123456789abcdef" for c in s) for s in archive_ids)


# ── AC4: attribute conflicts present at configured rate ────────────────────


def test_attribute_conflicts_at_configured_rate():
    cfg = _make_config(
        n_entities=20,
        multi_source=MultiSourceConfig(
            sources=[
                SourceDeclaration(
                    name="billing",
                    attribute_drift_rate=1.0,
                    # Disable name drift to isolate the attribute signal.
                    name_drift_rate=0.0,
                ),
                # MultiSourceConfig.min_length=2 — second source is a
                # placeholder we don't assert on here.
                SourceDeclaration(name="crm"),
            ]
        ),
    )
    dims = build_all_dimensions(cfg, np.random.default_rng(cfg.seed))
    canonical_industry = list(dims["dim_company"]["industry"])
    drifted_industry = list(dims["dim_company_billing"]["industry"])
    # attribute_drift_rate=1.0 → every row's industry should differ.
    n_drifted = sum(1 for a, b in zip(canonical_industry, drifted_industry) if a != b)
    assert n_drifted == len(canonical_industry)


# ── AC5: manifest mapping records one per (entity, source) ────────────────


def test_manifest_source_entity_mappings_complete():
    n_entities = 6
    cfg = _make_config(
        n_entities=n_entities,
        multi_source=MultiSourceConfig(
            sources=[
                SourceDeclaration(name="crm", name_drift_rate=0.5),
                SourceDeclaration(name="billing", attribute_drift_rate=0.5),
            ]
        ),
    )
    # Minimum end-to-end pipeline: dim builder → manifest builder.
    rng = np.random.default_rng(cfg.seed)
    dims = build_all_dimensions(cfg, rng)
    # ``build_manifest`` needs trajectories + tables; we fake the minimum
    # shape it expects since the multi-source mappings are derived from
    # ``config._source_entity_mappings`` rather than the runtime data.
    n_periods = len(dims["dim_date"])
    trajectories = {e.name: np.linspace(0.1, 0.9, n_periods) for e in cfg.entities}
    manifest = build_manifest(cfg, trajectories=trajectories, tables=dims)

    # 6 entities × 2 sources × 1 dim_company = 12 records.
    assert len(manifest.source_entity_mappings) == n_entities * 2
    # Each entity gets exactly one record per source.
    by_entity_source: dict[tuple[str, str], SourceEntityMapping] = {}
    for rec in manifest.source_entity_mappings:
        key = (rec.entity, rec.source)
        assert key not in by_entity_source, f"duplicate mapping for {key}"
        by_entity_source[key] = rec
    expected_keys = {
        (entity.name, src.name) for entity in cfg.entities for src in cfg.multi_source.sources
    }
    assert set(by_entity_source) == expected_keys
    # ``drifted_fields`` is a list of column names from the canonical dim.
    canonical_columns = {"company_id", "company_name", "industry"}
    for rec in manifest.source_entity_mappings:
        for field in rec.drifted_fields:
            assert field in canonical_columns


def test_manifest_schema_bumped_to_1_6():
    # 1.5 introduced the source_entity_mappings list (0.6-M13); 1.6 adds
    # the parent_child_relations list (0.6-M18). This module's contract
    # tracks the pin at the schema level, not the field semantics.
    assert MANIFEST_SCHEMA_VERSION == "1.6"


# ── AC6: single-source configs unchanged (no multi_source block) ──────────


def test_no_multi_source_block_leaves_dims_unchanged():
    cfg = _make_config(multi_source=None)
    dims = build_all_dimensions(cfg, np.random.default_rng(cfg.seed))
    # Only the canonical dim — no per-source emission.
    assert sorted(d for d in dims if d.startswith("dim_company")) == ["dim_company"]
    # No mappings stashed.
    assert cfg._source_entity_mappings is None


def test_no_multi_source_block_manifest_has_empty_mapping_list():
    cfg = _make_config(multi_source=None)
    rng = np.random.default_rng(cfg.seed)
    dims = build_all_dimensions(cfg, rng)
    n_periods = len(dims["dim_date"])
    trajectories = {e.name: np.linspace(0.1, 0.9, n_periods) for e in cfg.entities}
    manifest = build_manifest(cfg, trajectories=trajectories, tables=dims)
    assert manifest.source_entity_mappings == []


# ── AC7: deterministic under seed ─────────────────────────────────────────


def test_deterministic_under_seed():
    """Two builds with the same seed produce byte-identical dim frames and mappings."""

    def _build():
        cfg = _make_config(
            n_entities=10,
            multi_source=MultiSourceConfig(
                sources=[
                    SourceDeclaration(
                        name="crm",
                        name_drift_rate=0.5,
                        attribute_drift_rate=0.5,
                    ),
                    SourceDeclaration(
                        name="billing",
                        key_scheme="numeric",
                        name_drift_rate=0.3,
                        attribute_drift_rate=0.3,
                    ),
                ]
            ),
        )
        rng = np.random.default_rng(cfg.seed)
        dims = build_all_dimensions(cfg, rng)
        return cfg, dims

    cfg1, dims1 = _build()
    cfg2, dims2 = _build()

    # Each per-source dim is byte-identical across builds.
    for name in ("dim_company", "dim_company_crm", "dim_company_billing"):
        assert dims1[name].equals(dims2[name]), f"non-deterministic on {name}"

    # The mapping stash is byte-identical across builds.
    assert cfg1._source_entity_mappings == cfg2._source_entity_mappings


# ── AC8: bundled template passes plotsim run + validate ───────────────────


def test_bundled_template_loads_and_validates(tmp_path: Path):
    from plotsim.tables import generate_tables_with_state

    cfg = plotsim.load_template("crm_billing_overlap")
    rng = np.random.default_rng(cfg.seed)
    tables, gen_state = generate_tables_with_state(cfg, rng)
    report = plotsim.validate(cfg, tables)
    assert report.ok, f"validation errors: {report.errors}"

    manifest = build_manifest(
        cfg,
        gen_state.trajectories,
        tables,
        scd_state=gen_state.scd,
        bridge_state=gen_state.bridges,
    )
    out_dir = tmp_path / "out"
    plotsim.write_tables(tables, cfg, report=report, output_dir=out_dir, manifest=manifest)
    assert (out_dir / "dim_company.csv").is_file()
    assert (out_dir / "dim_company_crm.csv").is_file()
    assert (out_dir / "dim_company_billing.csv").is_file()
    manifest_payload = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["schema_version"] == "1.6"
    # 20 entities × 2 sources = 40 mapping records.
    assert len(manifest_payload["source_entity_mappings"]) == 40


# ── Engine validator coverage ──────────────────────────────────────────────


def test_multi_source_without_per_entity_dim_raises():
    """The cross-reference validator gates ``multi_source`` on the
    presence of at least one per_entity dim."""
    cfg_kwargs = dict(
        domain=Domain(
            name="t",
            description="t",
            entity_type="company",
            entity_label="Companies",
        ),
        time_window=TimeWindow(start="2024-01", end="2024-04", granularity="monthly"),
        seed=42,
        metrics=[
            Metric(
                name="mrr",
                label="MRR",
                distribution="lognorm",
                params={"s": 1.0, "scale": 1000.0},
                polarity="positive",
                value_range=ValueRange(min=0.0, max=10_000.0),
            )
        ],
        archetypes=[
            Archetype(
                name="g",
                label="g",
                description="g",
                curve_segments=[
                    CurveSegment(
                        start_pct=0.0,
                        end_pct=1.0,
                        curve="sigmoid",
                        params={"midpoint": 0.5, "steepness": 1.0},
                    )
                ],
            )
        ],
        entities=[Entity(name="e_0", archetype="g", size=1)],
        # NO per_entity dim — only dim_date.
        tables=[
            Table(
                name="dim_date",
                type="dim",
                grain="per_period",
                columns=[
                    Column(name="date_key", dtype="id", source="pk"),
                    Column(name="date", dtype="date", source="generated:date_key"),
                ],
                primary_key="date_key",
            ),
        ],
        output=OutputConfig(format="csv", directory="output"),
        multi_source=MultiSourceConfig(
            sources=[
                SourceDeclaration(name="crm"),
                SourceDeclaration(name="billing"),
            ]
        ),
    )
    with pytest.raises(ValueError, match="no per_entity dim tables are declared"):
        PlotsimConfig(**cfg_kwargs)


def test_multi_source_emission_collision_raises():
    """A source name that would emit a colliding dim table name is
    rejected at config load."""
    cfg_kwargs = dict(
        domain=Domain(
            name="t",
            description="t",
            entity_type="company",
            entity_label="Companies",
        ),
        time_window=TimeWindow(start="2024-01", end="2024-04", granularity="monthly"),
        seed=42,
        metrics=[
            Metric(
                name="mrr",
                label="MRR",
                distribution="lognorm",
                params={"s": 1.0, "scale": 1000.0},
                polarity="positive",
                value_range=ValueRange(min=0.0, max=10_000.0),
            )
        ],
        archetypes=[
            Archetype(
                name="g",
                label="g",
                description="g",
                curve_segments=[
                    CurveSegment(
                        start_pct=0.0,
                        end_pct=1.0,
                        curve="sigmoid",
                        params={"midpoint": 0.5, "steepness": 1.0},
                    )
                ],
            )
        ],
        entities=[Entity(name="e_0", archetype="g", size=1)],
        # dim_company plus a second dim whose name collides with the
        # would-be ``dim_company_shadow`` per-source emission.
        tables=[
            Table(
                name="dim_date",
                type="dim",
                grain="per_period",
                columns=[
                    Column(name="date_key", dtype="id", source="pk"),
                    Column(name="date", dtype="date", source="generated:date_key"),
                ],
                primary_key="date_key",
            ),
            Table(
                name="dim_company",
                type="dim",
                grain="per_entity",
                columns=[
                    Column(name="company_id", dtype="id", source="pk"),
                ],
                primary_key="company_id",
            ),
            Table(
                name="dim_company_shadow",
                type="dim",
                grain="per_reference",
                columns=[
                    Column(name="shadow_id", dtype="id", source="pk"),
                    Column(name="label", dtype="string", source="static:x"),
                ],
                primary_key="shadow_id",
            ),
        ],
        output=OutputConfig(format="csv", directory="output"),
        multi_source=MultiSourceConfig(
            sources=[
                SourceDeclaration(name="shadow"),
                SourceDeclaration(name="other"),
            ]
        ),
    )
    with pytest.raises(ValueError, match="collides with an existing table"):
        PlotsimConfig(**cfg_kwargs)


def test_multi_source_requires_at_least_two_sources():
    """``MultiSourceConfig`` rejects single-source declarations at the
    model layer — engine fail-fast before the dim builder runs."""
    with pytest.raises(ValueError):
        MultiSourceConfig(sources=[SourceDeclaration(name="crm")])


def test_multi_source_rejects_duplicate_source_names():
    with pytest.raises(ValueError, match="duplicate source name"):
        MultiSourceConfig(
            sources=[
                SourceDeclaration(name="crm"),
                SourceDeclaration(name="crm"),
            ]
        )


# ── Drift-module unit tests ────────────────────────────────────────────────


def test_apply_name_drift_swap_changes_value():
    rng = np.random.default_rng(0)
    out = _apply_name_drift("Acme Corporation", "swap", rng)
    assert out != "Acme Corporation"
    assert len(out) == len("Acme Corporation")


def test_apply_name_drift_casing_inverts_alpha_chars():
    out = _apply_name_drift("Acme Corp", "casing", np.random.default_rng(0))
    assert out == "aCME cORP"


def test_apply_name_drift_abbreviate_takes_initials():
    out = _apply_name_drift(
        "Acme Industries Inc",
        "abbreviate",
        np.random.default_rng(0),
    )
    assert out == "A.I.I."


def test_apply_name_drift_handles_empty_and_short_strings():
    rng = np.random.default_rng(0)
    assert _apply_name_drift("", "swap", rng) == ""
    assert _apply_name_drift("a", "swap", rng) == "a"


def test_generate_source_ids_prefix_padded_format():
    ids = _generate_source_ids("prefix_padded", "company", 3, np.random.default_rng(0))
    assert ids == ["COMPANY-001", "COMPANY-002", "COMPANY-003"]


def test_generate_source_ids_numeric_is_sequential():
    ids = _generate_source_ids("numeric", "company", 4, np.random.default_rng(0))
    assert all(s.isdigit() for s in ids)
    # Sequential — each is previous + 1.
    nums = [int(s) for s in ids]
    assert all(b == a + 1 for a, b in zip(nums, nums[1:]))


def test_generate_source_ids_uuid_short_is_lowercase_hex():
    ids = _generate_source_ids("uuid_short", "company", 5, np.random.default_rng(0))
    assert all(len(s) == 5 and all(c in "0123456789abcdef" for c in s) for s in ids)
    # 5 chars of hex = 16**5 = 1M possible values, so 5 distinct draws
    # are overwhelmingly likely to be unique.
    assert len(set(ids)) == 5


def test_apply_source_drift_returns_renamed_pk_column():
    """The drift function renames the canonical PK column to
    ``<entity_type>_id_<source>`` and removes the canonical PK."""
    import pandas as pd

    canonical = pd.DataFrame(
        {
            "company_id": ["c-001", "c-002"],
            "company_name": ["Acme Corp", "Globex Inc"],
        }
    )
    columns = [
        Column(name="company_id", dtype="id", source="pk"),
        Column(name="company_name", dtype="string", source="generated:faker.company"),
    ]
    drifted, source_pk, mappings = apply_source_drift(
        canonical_df=canonical,
        canonical_columns=columns,
        canonical_pk_column="company_id",
        source=SourceDeclaration(name="crm"),
        entity_type="company",
        rng=np.random.default_rng(0),
    )
    assert source_pk == "company_id_crm"
    assert "company_id_crm" in drifted.columns
    assert "company_id" not in drifted.columns
    assert len(mappings) == 2
