"""M106 — SCD Type 2 tests.

Covers the full mission's acceptance criteria:

  * Config model: extra=forbid, threshold/label/trigger_metric validation
  * Cross-reference: trigger_metric must resolve to a fact table + metric
  * Pairing: source 'scd_type2' and Column.scd_type2 are paired or both absent
  * Dim expansion: per-entity versioned rows with dim_row_id + validity window
  * Hysteresis: re-entry into a previously-visited band does NOT add a row
  * Fact FK resolution: dim_row_id column appended; joins are 1:1 + orphan-free
  * Non-SCD facts: unchanged, no dim_row_id added
  * Manifest: SCDEvent list populated with full crossing metadata
  * Determinism: same seed → byte-identical dim and fact rows
  * Schema export: plotsim-schema.json contains SCDType2Config
  * Bundled saas template: SCD plan_tier upgrade flow demonstrates end-to-end
  * All other 4 bundled templates: empty SCD section in manifest, no regressions
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from plotsim import (
    PlotsimConfig,
    SurrogateKeyWarning,
    build_manifest,
    generate_tables_with_state,
    load_config,
    validate,
    write_tables,
)
from plotsim.config import SCDType2Config, parse_source
from plotsim.manifest import (
    MANIFEST_FILENAME,
    ManifestSchema,
    SCDEvent,
)
from plotsim.tables import (
    SCD_VALID_TO_SENTINEL,
    SCDState,
    SCDVersion,
    _compute_scd_versions,
)


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "plotsim" / "configs"
SAAS_YAML = CONFIGS_DIR / "sample_saas.yaml"


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def saas_cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return load_config(SAAS_YAML)


@pytest.fixture
def saas_run(saas_cfg):
    rng = np.random.default_rng(saas_cfg.seed)
    tables, state = generate_tables_with_state(saas_cfg, rng)
    return saas_cfg, tables, state


def _saas_yaml_dict():
    with SAAS_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_cfg_from_dict(d: dict) -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(**d)


# --- 1. SCDType2Config model -----------------------------------------------


def test_scd_type2_config_extra_forbid():
    with pytest.raises(ValidationError, match="extra"):
        SCDType2Config(
            trigger_metric="fct_revenue.mrr",
            thresholds=(0.5,),
            labels=("a", "b"),
            unknown_field=42,
        )


def test_scd_type2_trigger_metric_format_required():
    with pytest.raises(ValidationError, match="trigger_metric"):
        SCDType2Config(
            trigger_metric="no_dot_here",
            thresholds=(0.5,),
            labels=("a", "b"),
        )


def test_scd_type2_trigger_metric_non_empty_sides():
    with pytest.raises(ValidationError, match="non-empty"):
        SCDType2Config(
            trigger_metric=".mrr",
            thresholds=(0.5,),
            labels=("a", "b"),
        )
    with pytest.raises(ValidationError, match="non-empty"):
        SCDType2Config(
            trigger_metric="fct_revenue.",
            thresholds=(0.5,),
            labels=("a", "b"),
        )


def test_scd_type2_thresholds_strictly_increasing():
    with pytest.raises(ValidationError, match="strictly increasing"):
        SCDType2Config(
            trigger_metric="fct_revenue.mrr",
            thresholds=(0.5, 0.3),
            labels=("a", "b", "c"),
        )
    with pytest.raises(ValidationError, match="strictly increasing"):
        SCDType2Config(
            trigger_metric="fct_revenue.mrr",
            thresholds=(0.5, 0.5),
            labels=("a", "b", "c"),
        )


def test_scd_type2_thresholds_open_interval_zero_one():
    for bad in (0.0, -0.1, 1.0, 1.5):
        with pytest.raises(ValidationError, match="open interval"):
            SCDType2Config(
                trigger_metric="fct_revenue.mrr",
                thresholds=(bad,),
                labels=("a", "b"),
            )


def test_scd_type2_label_count_must_be_thresholds_plus_one():
    with pytest.raises(ValidationError, match="expected 3"):
        SCDType2Config(
            trigger_metric="fct_revenue.mrr",
            thresholds=(0.3, 0.7),
            labels=("a", "b"),
        )


def test_scd_type2_labels_must_be_unique():
    with pytest.raises(ValidationError, match="duplicate"):
        SCDType2Config(
            trigger_metric="fct_revenue.mrr",
            thresholds=(0.5,),
            labels=("dup", "dup"),
        )


# --- 2. Source pairing on Column -------------------------------------------


def test_parse_source_scd_marker():
    parsed = parse_source("scd_type2")
    from plotsim.config import SCDType2Source

    assert isinstance(parsed, SCDType2Source)


def test_column_with_scd_source_requires_scd_config():
    d = _saas_yaml_dict()
    # Add an SCD source column to dim_company without the scd_type2 block.
    for tbl in d["tables"]:
        if tbl["name"] == "dim_company":
            tbl["columns"].append(
                {
                    "name": "broken_tier",
                    "dtype": "string",
                    "source": "scd_type2",
                }
            )
            break
    with pytest.raises(ValidationError, match="scd_type2"):
        _build_cfg_from_dict(d)


def test_column_with_scd_config_requires_scd_source():
    d = _saas_yaml_dict()
    for tbl in d["tables"]:
        if tbl["name"] == "dim_company":
            tbl["columns"].append(
                {
                    "name": "broken_tier",
                    "dtype": "string",
                    "source": "static:placeholder",
                    "scd_type2": {
                        "trigger_metric": "fct_revenue.mrr",
                        "thresholds": [0.5],
                        "labels": ["lo", "hi"],
                    },
                }
            )
            break
    with pytest.raises(ValidationError, match="scd_type2"):
        _build_cfg_from_dict(d)


# --- 3. PlotsimConfig cross-reference --------------------------------------


def _add_scd_column(cfg_dict, *, on_dim, scd_block):
    """Append an SCD column on the named dim, returning the modified dict."""
    out = copy.deepcopy(cfg_dict)
    for tbl in out["tables"]:
        if tbl["name"] == on_dim:
            tbl["columns"].append(
                {
                    "name": "extra_tier",
                    "dtype": "string",
                    "source": "scd_type2",
                    "scd_type2": scd_block,
                }
            )
            break
    return out


def test_scd_trigger_metric_unknown_metric_rejected():
    d = _saas_yaml_dict()
    # Replace the existing SCD block with one pointing at a missing metric.
    for tbl in d["tables"]:
        if tbl["name"] == "dim_company":
            for col in tbl["columns"]:
                if col.get("source") == "scd_type2":
                    col["scd_type2"]["trigger_metric"] = "fct_revenue.no_such_metric"
                    break
            break
    with pytest.raises(ValidationError, match="unknown metric"):
        _build_cfg_from_dict(d)


def test_scd_trigger_metric_unknown_table_rejected():
    d = _saas_yaml_dict()
    for tbl in d["tables"]:
        if tbl["name"] == "dim_company":
            for col in tbl["columns"]:
                if col.get("source") == "scd_type2":
                    col["scd_type2"]["trigger_metric"] = "no_such_table.mrr"
                    break
            break
    with pytest.raises(ValidationError, match="unknown table"):
        _build_cfg_from_dict(d)


def test_scd_trigger_metric_dim_table_rejected():
    d = _saas_yaml_dict()
    for tbl in d["tables"]:
        if tbl["name"] == "dim_company":
            for col in tbl["columns"]:
                if col.get("source") == "scd_type2":
                    # Point at a dim table — must be rejected (fact required).
                    col["scd_type2"]["trigger_metric"] = "dim_company.mrr"
                    break
            break
    with pytest.raises(ValidationError):
        _build_cfg_from_dict(d)


def test_scd_only_one_column_per_dim_table():
    d = _add_scd_column(
        _saas_yaml_dict(),
        on_dim="dim_company",
        scd_block={
            "trigger_metric": "fct_revenue.mrr",
            "thresholds": [0.5],
            "labels": ["a", "b"],
        },
    )
    with pytest.raises(ValidationError, match="at most one"):
        _build_cfg_from_dict(d)


def test_scd_rejected_on_non_per_entity_dim():
    d = _saas_yaml_dict()
    # Try adding SCD to dim_plan (per_reference grain).
    for tbl in d["tables"]:
        if tbl["name"] == "dim_plan":
            tbl["columns"].append(
                {
                    "name": "tier",
                    "dtype": "string",
                    "source": "scd_type2",
                    "scd_type2": {
                        "trigger_metric": "fct_revenue.mrr",
                        "thresholds": [0.5],
                        "labels": ["a", "b"],
                    },
                }
            )
            break
    with pytest.raises(ValidationError, match="per_entity"):
        _build_cfg_from_dict(d)


# --- 4. Pure SCD-version computation ---------------------------------------


def _date_keys(n_periods: int) -> np.ndarray:
    """Synthesize date_keys 20230101, 20230201, ... for a synthetic config."""
    return np.array(
        [20230000 + (i + 1) * 100 + 1 for i in range(n_periods)],
        dtype=np.int64,
    )


def test_compute_scd_versions_monotonic_climb():
    cfg = SCDType2Config(
        trigger_metric="fct_revenue.mrr",
        thresholds=(0.3, 0.7),
        labels=("low", "mid", "high"),
    )
    traj = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    dks = _date_keys(len(traj))
    versions = _compute_scd_versions(traj, cfg, dks, starting_dim_row_id=1)
    # Three bands visited: low (p0), mid (p2), high (p4).
    assert [v.band_label for v in versions] == ["low", "mid", "high"]
    assert [v.dim_row_id for v in versions] == [1, 2, 3]
    assert [v.valid_from_period for v in versions] == [0, 2, 4]
    # valid_to_period: next segment's start, sentinel for last.
    assert versions[0].valid_to_period == 2
    assert versions[1].valid_to_period == 4
    assert versions[2].valid_to == SCD_VALID_TO_SENTINEL
    assert versions[2].is_current is True
    assert versions[0].is_current is False
    # Crossing positions: starting band has None; subsequent ones carry value.
    assert versions[0].crossing_position is None
    assert versions[1].crossing_position == pytest.approx(0.4)
    assert versions[2].crossing_position == pytest.approx(0.8)


def test_compute_scd_versions_no_crossing():
    cfg = SCDType2Config(
        trigger_metric="fct_revenue.mrr",
        thresholds=(0.3, 0.7),
        labels=("low", "mid", "high"),
    )
    traj = np.array([0.1, 0.15, 0.2, 0.18, 0.12])
    dks = _date_keys(len(traj))
    versions = _compute_scd_versions(traj, cfg, dks, starting_dim_row_id=1)
    assert len(versions) == 1
    assert versions[0].band_label == "low"
    assert versions[0].is_current is True
    assert versions[0].valid_from == int(dks[0])
    assert versions[0].valid_to == SCD_VALID_TO_SENTINEL


def test_compute_scd_versions_hysteresis_no_demote_row():
    """An entity that crosses up then drops back stays in the higher band."""
    cfg = SCDType2Config(
        trigger_metric="fct_revenue.mrr",
        thresholds=(0.4,),
        labels=("low", "high"),
    )
    # Climbs to high at p2, drops back to low at p3 — one upgrade only.
    traj = np.array([0.1, 0.2, 0.6, 0.1, 0.05, 0.6])
    dks = _date_keys(len(traj))
    versions = _compute_scd_versions(traj, cfg, dks, starting_dim_row_id=1)
    assert [v.band_label for v in versions] == ["low", "high"]
    # No "second high" version from the second crossing.
    assert len(versions) == 2


def test_compute_scd_versions_skip_band_jump():
    """A trajectory that leaps two bands at once registers each band crossed."""
    cfg = SCDType2Config(
        trigger_metric="fct_revenue.mrr",
        thresholds=(0.3, 0.7),
        labels=("low", "mid", "high"),
    )
    # Jumps from low straight to high at p2.
    traj = np.array([0.1, 0.2, 0.9, 0.95, 0.98])
    dks = _date_keys(len(traj))
    versions = _compute_scd_versions(traj, cfg, dks, starting_dim_row_id=1)
    # The accumulator records the new max band only; intermediate "mid" is
    # skipped because the entity never ran through it.
    assert [v.band_label for v in versions] == ["low", "high"]


# --- 5. SAAS end-to-end SCD shape ------------------------------------------


def test_saas_dim_company_has_scd_columns(saas_run):
    _cfg, tables, _state = saas_run
    dim = tables["dim_company"]
    for col in ("dim_row_id", "valid_from", "valid_to", "is_current", "plan_tier"):
        assert col in dim.columns, f"dim_company missing {col}"


def test_saas_dim_company_multiple_versions_per_active_entity(saas_run):
    _cfg, tables, _state = saas_run
    dim = tables["dim_company"]
    counts = dim["company_id"].value_counts()
    # rocket_then_cliff and steady_grower both produce upgrades; zombie has
    # exactly one row. So at least two entities must hold > 1 version.
    multi = (counts > 1).sum()
    assert multi >= 2, f"expected >=2 entities with multiple versions, got {multi}"


def test_saas_dim_row_id_is_unique(saas_run):
    _cfg, tables, _state = saas_run
    dim = tables["dim_company"]
    assert dim["dim_row_id"].is_unique


def test_saas_each_entity_has_exactly_one_current(saas_run):
    cfg, tables, _state = saas_run
    dim = tables["dim_company"]
    n_current = dim["is_current"].astype(bool).sum()
    assert n_current == len(cfg.entities)
    # current rows have the sentinel valid_to.
    current_rows = dim[dim["is_current"]]
    assert (current_rows["valid_to"] == SCD_VALID_TO_SENTINEL).all()


def test_saas_zombie_entity_has_single_version(saas_run):
    _cfg, tables, _state = saas_run
    dim = tables["dim_company"]
    # The "zombie_account" archetype is plateau at 0.15 — never crosses 0.4.
    # Find which company_id maps to that cohort by reading one row's id off
    # entity-order: the third entity (hooli_cohort) is the zombie.
    pks = dim.drop_duplicates(subset="company_id", keep="first")["company_id"].tolist()
    zombie_pk = pks[2]
    zombie_rows = dim[dim["company_id"] == zombie_pk]
    assert len(zombie_rows) == 1
    only = zombie_rows.iloc[0]
    assert only["plan_tier"] == "starter"
    assert bool(only["is_current"]) is True
    assert int(only["valid_to"]) == SCD_VALID_TO_SENTINEL


def test_saas_validity_windows_cover_every_period_per_entity(saas_run):
    _cfg, tables, _state = saas_run
    dim = tables["dim_company"]
    dim_date = tables["dim_date"]
    dks = dim_date["date_key"].tolist()
    for company_id, grp in dim.groupby("company_id"):
        for dk in dks:
            covered = grp[(grp["valid_from"] <= dk) & (dk < grp["valid_to"])]
            assert len(covered) == 1, (
                f"company {company_id} period {dk}: expected one active "
                f"version, got {len(covered)}"
            )


# --- 6. Fact dim_row_id resolution -----------------------------------------


def test_saas_fact_revenue_has_dim_row_id(saas_run):
    _cfg, tables, _state = saas_run
    fct = tables["fct_revenue"]
    assert "dim_row_id" in fct.columns


def test_saas_fact_engagement_has_dim_row_id(saas_run):
    _cfg, tables, _state = saas_run
    fct = tables["fct_engagement"]
    assert "dim_row_id" in fct.columns


def test_saas_fact_join_on_dim_row_id_is_one_to_one(saas_run):
    _cfg, tables, _state = saas_run
    facts = tables["fct_revenue"]
    dim = tables["dim_company"][["dim_row_id", "company_id", "plan_tier"]]
    merged = pd.merge(facts, dim, on="dim_row_id", suffixes=("", "_dim"))
    # No fan-out: same row count after merge.
    assert len(merged) == len(facts)
    # No orphans: every fact row matched a dim row.
    assert merged["plan_tier"].notna().all()
    # The dim's company_id matches the fact's company_id (entity business key
    # remains intact alongside the SCD surrogate).
    assert (merged["company_id"] == merged["company_id_dim"]).all()


def test_saas_fact_dim_row_id_advances_with_period(saas_run):
    _cfg, tables, _state = saas_run
    fct = tables["fct_revenue"].sort_values(["company_id", "date_key"])
    # Per entity, dim_row_id is non-decreasing across periods (advancement
    # only — never goes back).
    for _eid, grp in fct.groupby("company_id"):
        ids = grp["dim_row_id"].tolist()
        assert all(
            ids[i] <= ids[i + 1] for i in range(len(ids) - 1)
        ), f"dim_row_id regressed within entity: {ids}"


# --- 7. Non-SCD facts unaffected -------------------------------------------


def test_non_scd_template_no_dim_row_id_added():
    # The HR template has no SCD columns; no fact should sprout dim_row_id.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(CONFIGS_DIR / "sample_hr.yaml")
    tables, state = generate_tables_with_state(cfg, np.random.default_rng(cfg.seed))
    assert state.scd.is_empty
    for name, df in tables.items():
        assert "dim_row_id" not in df.columns, f"{name} unexpectedly has a dim_row_id column"


# --- 8. Manifest SCD events ------------------------------------------------


def test_manifest_scd_events_populated_for_saas(saas_run):
    cfg, tables, state = saas_run
    m = build_manifest(cfg, state.trajectories, tables, scd_state=state.scd)
    assert len(m.scd_events) > 0
    for e in m.scd_events:
        assert isinstance(e, SCDEvent)
        assert e.dim_table == "dim_company"
        assert e.trigger_metric == "fct_revenue.mrr"
        assert 0 <= e.period_index < len(tables["dim_date"])
        assert 0.0 <= e.trigger_position <= 1.0
        assert e.old_dim_row_id != e.new_dim_row_id
        assert e.old_label != e.new_label


def test_manifest_scd_events_match_dim_row_id_jumps(saas_run):
    cfg, tables, state = saas_run
    m = build_manifest(cfg, state.trajectories, tables, scd_state=state.scd)
    dim = tables["dim_company"]
    # Every (old_dim_row_id, new_dim_row_id) pair in the manifest must appear
    # in the dim table (chained: new-id row exists, old-id row exists).
    dim_ids = set(dim["dim_row_id"].tolist())
    for e in m.scd_events:
        assert e.old_dim_row_id in dim_ids
        assert e.new_dim_row_id in dim_ids


def test_manifest_no_scd_events_for_non_scd_templates():
    # Post-M112 the only bundled template without an SCD-typed dim is hr.
    # saas/dim_company.plan_tier, education/dim_student.academic_standing,
    # retail/dim_customer.customer_tier, and marketing/dim_customer.customer_tier
    # all declare scd_type2.
    for name in ("hr",):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SurrogateKeyWarning)
            cfg = load_config(CONFIGS_DIR / f"sample_{name}.yaml")
        tables, state = generate_tables_with_state(
            cfg,
            np.random.default_rng(cfg.seed),
        )
        m = build_manifest(cfg, state.trajectories, tables, scd_state=state.scd)
        assert m.scd_events == [], f"{name} unexpectedly produced SCD events"


def test_manifest_scd_events_round_trip_through_disk(saas_run, tmp_path):
    cfg, tables, state = saas_run
    m = build_manifest(cfg, state.trajectories, tables, scd_state=state.scd)
    write_tables(tables, cfg, output_dir=tmp_path, manifest=m)
    payload = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert "scd_events" in payload
    assert len(payload["scd_events"]) == len(m.scd_events)
    # Re-validate via the schema model: forward-compat round trip.
    parsed = ManifestSchema(**payload)
    assert len(parsed.scd_events) == len(m.scd_events)


# --- 9. Validation suite -----------------------------------------------------


def test_saas_validation_clean_with_scd(saas_run):
    cfg, tables, _state = saas_run
    report = validate(cfg, tables)
    assert report.ok, [f"{i.check}: {i.message}" for i in report.errors]


def test_validation_detects_missing_dim_row_id_on_scd_dim(saas_run):
    cfg, tables, _state = saas_run
    broken = dict(tables)
    broken["dim_company"] = tables["dim_company"].drop(columns=["dim_row_id"])
    report = validate(cfg, broken)
    assert not report.ok
    assert any(i.check == "scd_integrity" for i in report.errors)


# --- 10. Determinism --------------------------------------------------------


def _run_saas_seeded(cfg):
    return generate_tables_with_state(cfg, np.random.default_rng(cfg.seed))


def test_saas_dim_company_byte_identical_across_runs(saas_cfg):
    tables_a, _ = _run_saas_seeded(saas_cfg)
    tables_b, _ = _run_saas_seeded(saas_cfg)
    pd.testing.assert_frame_equal(
        tables_a["dim_company"],
        tables_b["dim_company"],
    )


def test_saas_fact_dim_row_id_byte_identical_across_runs(saas_cfg):
    tables_a, _ = _run_saas_seeded(saas_cfg)
    tables_b, _ = _run_saas_seeded(saas_cfg)
    a = tables_a["fct_revenue"]["dim_row_id"]
    b = tables_b["fct_revenue"]["dim_row_id"]
    pd.testing.assert_series_equal(a, b)


def test_saas_manifest_scd_events_deterministic(saas_cfg):
    tables_a, state_a = _run_saas_seeded(saas_cfg)
    tables_b, state_b = _run_saas_seeded(saas_cfg)
    ma = build_manifest(saas_cfg, state_a.trajectories, tables_a, scd_state=state_a.scd)
    mb = build_manifest(saas_cfg, state_b.trajectories, tables_b, scd_state=state_b.scd)
    assert [e.model_dump() for e in ma.scd_events] == [e.model_dump() for e in mb.scd_events]


# --- 11. Schema export ------------------------------------------------------


def test_plotsim_schema_includes_scd_type2_config():
    schema_path = ROOT / "plotsim-schema.json"
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    defs = payload.get("$defs") or payload.get("definitions") or {}
    assert "SCDType2Config" in defs


# --- 12. SCDState dataclass --------------------------------------------------


def test_scd_state_is_empty_when_no_scd_columns():
    state = SCDState(dims={})
    assert state.is_empty


def test_scd_state_carries_per_entity_versions(saas_run):
    _cfg, _tables, state = saas_run
    assert "dim_company" in state.scd.dims
    dim_state = state.scd.dims["dim_company"]
    # acme_corp_cohort upgrades twice → 3 versions; zombie hooli_cohort → 1.
    assert len(dim_state.versions["acme_corp_cohort"]) == 3
    assert len(dim_state.versions["hooli_cohort"]) == 1
    # SCDVersion shape integrity.
    v0 = dim_state.versions["acme_corp_cohort"][0]
    assert isinstance(v0, SCDVersion)
    assert v0.dim_row_id >= 1
