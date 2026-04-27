"""M107 Track B — post-generation quality injection.

Tests cover the four layers the mission spec calls out:

  1. **Config validation** — ``QualityConfig`` rejects bad targets (FK
     columns, period/date_key columns, dim/bridge tables, unknown
     columns), enforces ``rate ∈ [0, 1]``, rejects unknown issue types,
     and accepts the ``"*"`` sentinel only when alone.
  2. **Pure pipeline** — ``quality.apply_issues`` is filesystem-free, the
     clean tables passed in are not mutated, and each issue type
     produces the documented corruption shape (null counts,
     duplicated row counts, ``_arrival_period`` / ``{col}_v2``
     columns).
  3. **Determinism** — same config + seed → byte-identical corrupted
     output and identical ground-truth records.
  4. **Manifest integration** — when ``write_tables`` runs with
     ``quality_issues`` configured the on-disk manifest's
     ``quality_injections`` field carries the per-issue ground truth
     and the on-disk tables hold the corrupted data, while the
     in-memory ``tables`` dict the caller passed in stays clean.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from plotsim import (
    build_manifest,
    generate_tables_with_state,
    load_config,
    validate_tables,
    write_tables,
)
from plotsim.config import PlotsimConfig
from plotsim.quality import apply_issues


SAAS_TEMPLATE = "plotsim/configs/sample_saas.yaml"


# ---------------------------------------------------------------------------
# Config-level validation
# ---------------------------------------------------------------------------


def _saas_dict() -> dict:
    with open(SAAS_TEMPLATE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _add_issue(cfg: dict, issue: dict) -> dict:
    cfg.setdefault("quality", {"quality_issues": []})
    cfg["quality"]["quality_issues"].append(issue)
    return cfg


def test_quality_rejects_dim_target_table():
    cfg = _saas_dict()
    _add_issue(cfg, {
        "type": "null_injection",
        "target_table": "dim_company",
        "target_columns": ["company_name"],
        "rate": 0.1,
    })
    with pytest.raises(ValidationError, match="fact and event tables only"):
        PlotsimConfig(**cfg)


def test_quality_rejects_bridge_target_table():
    cfg = _saas_dict()
    cfg["bridges"] = [{
        "name": "bridge_company_plan",
        "type": "bridge",
        "connects": ["dim_company", "dim_plan"],
        "cardinality": {"min": 1, "max": 1},
        "trajectory_driven": False,
        "metrics": [],
    }]
    _add_issue(cfg, {
        "type": "null_injection",
        "target_table": "bridge_company_plan",
        "target_columns": ["plan_id"],
        "rate": 0.1,
    })
    with pytest.raises(ValidationError, match="bridge table; quality"):
        PlotsimConfig(**cfg)


def test_quality_rejects_unknown_target_table():
    cfg = _saas_dict()
    _add_issue(cfg, {
        "type": "null_injection",
        "target_table": "nonexistent_table",
        "target_columns": ["x"],
        "rate": 0.1,
    })
    with pytest.raises(ValidationError, match="not a known table"):
        PlotsimConfig(**cfg)


def test_quality_rejects_fk_column_target():
    cfg = _saas_dict()
    _add_issue(cfg, {
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["company_id"],  # FK column
        "rate": 0.1,
    })
    with pytest.raises(ValidationError, match="FK or period"):
        PlotsimConfig(**cfg)


def test_quality_rejects_date_key_target():
    cfg = _saas_dict()
    _add_issue(cfg, {
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["date_key"],
        "rate": 0.1,
    })
    with pytest.raises(ValidationError, match="FK or period"):
        PlotsimConfig(**cfg)


def test_quality_rejects_unknown_column():
    cfg = _saas_dict()
    _add_issue(cfg, {
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["does_not_exist"],
        "rate": 0.1,
    })
    with pytest.raises(ValidationError, match="not present on table"):
        PlotsimConfig(**cfg)


def test_quality_rejects_unknown_issue_type():
    cfg = _saas_dict()
    _add_issue(cfg, {
        "type": "unknown_corruption",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 0.1,
    })
    with pytest.raises(ValidationError):
        PlotsimConfig(**cfg)


def test_quality_rejects_rate_above_one():
    cfg = _saas_dict()
    _add_issue(cfg, {
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 1.5,
    })
    with pytest.raises(ValidationError):
        PlotsimConfig(**cfg)


def test_quality_rejects_negative_seed_offset():
    cfg = _saas_dict()
    _add_issue(cfg, {
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 0.1,
        "seed_offset": -1,
    })
    with pytest.raises(ValidationError):
        PlotsimConfig(**cfg)


def test_quality_rejects_star_mixed_with_explicit():
    cfg = _saas_dict()
    _add_issue(cfg, {
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["*", "mrr"],
        "rate": 0.1,
    })
    with pytest.raises(ValidationError, match="mixes the '\\*'"):
        PlotsimConfig(**cfg)


def test_empty_quality_issues_default_clean_output():
    """Saas template default (no quality block) produces clean output."""
    cfg = load_config(SAAS_TEMPLATE)
    assert cfg.quality.quality_issues == []


# ---------------------------------------------------------------------------
# Pure pipeline — apply_issues
# ---------------------------------------------------------------------------


def _saas_with_quality(issues: list[dict]) -> PlotsimConfig:
    cfg = _saas_dict()
    cfg["quality"] = {"quality_issues": issues}
    return PlotsimConfig(**cfg)


def test_null_injection_sets_expected_count():
    cfg = _saas_with_quality([{
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 0.1,
    }])
    tables, _ = generate_tables_with_state(cfg)
    n_orig = len(tables["fct_revenue"])
    expected_nulls = int(0.1 * n_orig)
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    new_nulls = corrupted["fct_revenue"]["mrr"].isna().sum()
    orig_nulls = tables["fct_revenue"]["mrr"].isna().sum()
    assert new_nulls - orig_nulls == expected_nulls
    assert len(gt) == 1
    assert gt[0].issue_type == "null_injection"
    assert gt[0].column == "mrr"
    assert len(gt[0].row_indices) == expected_nulls


def test_apply_issues_does_not_mutate_input():
    """The clean ``tables`` dict the caller passed in stays clean."""
    cfg = _saas_with_quality([{
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 0.5,
    }])
    tables, _ = generate_tables_with_state(cfg)
    snapshot = tables["fct_revenue"].copy(deep=True)
    apply_issues(tables, cfg, base_seed=cfg.seed)
    pd.testing.assert_frame_equal(tables["fct_revenue"], snapshot)


def test_duplicate_rows_grows_table():
    cfg = _saas_with_quality([{
        "type": "duplicate_rows",
        "target_table": "fct_revenue",
        "target_columns": ["*"],
        "rate": 0.1,
    }])
    tables, _ = generate_tables_with_state(cfg)
    n_orig = len(tables["fct_revenue"])
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    expected_extra = int(0.1 * n_orig)
    assert len(corrupted["fct_revenue"]) == n_orig + expected_extra
    assert gt[0].issue_type == "duplicate_rows"
    assert gt[0].column == "_rows"


def test_type_mismatch_changes_dtype_to_object():
    cfg = _saas_with_quality([{
        "type": "type_mismatch",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 0.05,
    }])
    tables, _ = generate_tables_with_state(cfg)
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    assert corrupted["fct_revenue"]["mrr"].dtype == object
    assert gt[0].issue_type == "type_mismatch"
    # At least one cell is a string in the corrupted column.
    series = corrupted["fct_revenue"]["mrr"]
    assert any(isinstance(v, str) for v in series.tolist())


def test_late_arrival_adds_arrival_period_column():
    cfg = _saas_with_quality([{
        "type": "late_arrival",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 0.1,
    }])
    tables, _ = generate_tables_with_state(cfg)
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    assert "_arrival_period" in corrupted["fct_revenue"].columns
    # Original date_key column unchanged.
    assert "date_key" in corrupted["fct_revenue"].columns
    pd.testing.assert_series_equal(
        corrupted["fct_revenue"]["date_key"].reset_index(drop=True),
        tables["fct_revenue"]["date_key"].reset_index(drop=True),
        check_names=False,
    )
    # Approximately rate fraction of rows have non-null _arrival_period.
    n_arrival = corrupted["fct_revenue"]["_arrival_period"].notna().sum()
    expected = int(0.1 * len(tables["fct_revenue"]))
    assert n_arrival == expected
    assert gt[0].issue_type == "late_arrival"


def test_schema_drift_adds_v2_column():
    cfg = _saas_with_quality([{
        "type": "schema_drift",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 0.1,
    }])
    tables, _ = generate_tables_with_state(cfg)
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    assert "mrr_v2" in corrupted["fct_revenue"].columns
    assert gt[0].issue_type == "schema_drift"
    # For every affected row, the original column is null and v2 holds the value.
    affected = gt[0].row_indices
    df = corrupted["fct_revenue"]
    for i, ridx in enumerate(affected):
        assert pd.isna(df["mrr"].iloc[ridx])
        assert df["mrr_v2"].iloc[ridx] == gt[0].clean_values[i]


def test_apply_issues_is_deterministic():
    cfg = _saas_with_quality([{
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 0.1,
    }])
    tables, _ = generate_tables_with_state(cfg)
    corrupted_a, gt_a = apply_issues(tables, cfg, base_seed=cfg.seed)
    corrupted_b, gt_b = apply_issues(tables, cfg, base_seed=cfg.seed)
    pd.testing.assert_frame_equal(
        corrupted_a["fct_revenue"], corrupted_b["fct_revenue"],
    )
    assert gt_a[0].row_indices == gt_b[0].row_indices
    assert gt_a[0].clean_values == gt_b[0].clean_values


def test_seed_offset_changes_affected_rows():
    issue_template = {
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 0.1,
    }
    cfg_a = _saas_with_quality([{**issue_template, "seed_offset": 0}])
    cfg_b = _saas_with_quality([{**issue_template, "seed_offset": 100}])
    tables, _ = generate_tables_with_state(cfg_a)
    _, gt_a = apply_issues(tables, cfg_a, base_seed=cfg_a.seed)
    _, gt_b = apply_issues(tables, cfg_b, base_seed=cfg_b.seed)
    assert gt_a[0].row_indices != gt_b[0].row_indices


def test_empty_quality_issues_short_circuits():
    cfg = load_config(SAAS_TEMPLATE)
    tables, _ = generate_tables_with_state(cfg)
    out, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    assert out is tables  # same dict object — no copy taken
    assert gt == []


def test_star_sentinel_excludes_fk_and_period_columns():
    cfg = _saas_with_quality([{
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["*"],
        "rate": 0.5,
    }])
    tables, _ = generate_tables_with_state(cfg)
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    affected_cols = {r.column for r in gt}
    # FK and period/PK columns must not appear among corrupted columns.
    assert "company_id" not in affected_cols
    assert "date_key" not in affected_cols
    assert "plan_id" not in affected_cols
    # Metric columns ARE corrupted.
    assert "mrr" in affected_cols


# ---------------------------------------------------------------------------
# Manifest + write_tables integration
# ---------------------------------------------------------------------------


def test_write_tables_writes_corrupted_data_and_clean_manifest_truth(tmp_path):
    cfg_dict = _saas_dict()
    cfg_dict["output"]["directory"] = str(tmp_path / "out")
    cfg_dict["quality"] = {
        "quality_issues": [{
            "type": "null_injection",
            "target_table": "fct_revenue",
            "target_columns": ["mrr"],
            "rate": 0.1,
            "seed_offset": 0,
        }],
    }
    cfg = PlotsimConfig(**cfg_dict)
    tables, state = generate_tables_with_state(cfg)
    n_orig_nulls = int(tables["fct_revenue"]["mrr"].isna().sum())
    manifest = build_manifest(
        cfg, state.trajectories, tables,
        scd_state=state.scd, bridge_state=state.bridges,
    )
    out_dir = write_tables(tables, cfg, manifest=manifest)

    # The on-disk manifest carries the quality_injections ground truth.
    manifest_text = (Path(out_dir) / "manifest.json").read_text(encoding="utf-8")
    manifest_obj = json.loads(manifest_text)
    qi = manifest_obj["quality_injections"]
    assert len(qi) == 1
    assert qi[0]["issue_type"] == "null_injection"
    assert qi[0]["table"] == "fct_revenue"
    assert qi[0]["column"] == "mrr"
    assert len(qi[0]["row_indices"]) > 0

    # The on-disk fct_revenue.csv has more nulls than the in-memory clean copy.
    written_df = pd.read_csv(Path(out_dir) / "fct_revenue.csv")
    on_disk_nulls = int(written_df["mrr"].isna().sum())
    assert on_disk_nulls > n_orig_nulls

    # The in-memory ``tables`` dict the caller passed in stays clean.
    assert int(tables["fct_revenue"]["mrr"].isna().sum()) == n_orig_nulls


def test_write_tables_no_quality_block_produces_baseline(tmp_path):
    """Saas template without uncommenting the quality block writes
    output identical to a pre-M107 baseline (clean fct_revenue)."""
    cfg = load_config(SAAS_TEMPLATE)
    out_dir_a = tmp_path / "a"
    out_dir_b = tmp_path / "b"
    tables_a, state_a = generate_tables_with_state(cfg)
    manifest_a = build_manifest(
        cfg, state_a.trajectories, tables_a,
        scd_state=state_a.scd, bridge_state=state_a.bridges,
    )
    write_tables(tables_a, cfg, manifest=manifest_a, output_dir=out_dir_a)
    tables_b, state_b = generate_tables_with_state(cfg)
    manifest_b = build_manifest(
        cfg, state_b.trajectories, tables_b,
        scd_state=state_b.scd, bridge_state=state_b.bridges,
    )
    write_tables(tables_b, cfg, manifest=manifest_b, output_dir=out_dir_b)
    # Both manifests should have empty quality_injections.
    m_a = json.loads((out_dir_a / "manifest.json").read_text(encoding="utf-8"))
    m_b = json.loads((out_dir_b / "manifest.json").read_text(encoding="utf-8"))
    assert m_a["quality_injections"] == []
    assert m_b["quality_injections"] == []
    # Two runs at same seed → same output.
    assert (
        (out_dir_a / "fct_revenue.csv").read_bytes()
        == (out_dir_b / "fct_revenue.csv").read_bytes()
    )


def test_corrupted_output_does_not_break_validate_clean():
    """``validate_tables`` runs on the in-memory clean dict, before
    corruption — so adding quality_issues should not produce errors."""
    cfg = _saas_with_quality([{
        "type": "null_injection",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 0.05,
    }])
    tables, _ = generate_tables_with_state(cfg)
    report = validate_tables(cfg, tables)
    assert report.ok, [issue.message for issue in report.errors]


def test_late_arrival_offsets_in_documented_range():
    cfg = _saas_with_quality([{
        "type": "late_arrival",
        "target_table": "fct_revenue",
        "target_columns": ["mrr"],
        "rate": 0.2,
    }])
    tables, _ = generate_tables_with_state(cfg)
    corrupted, _ = apply_issues(tables, cfg, base_seed=cfg.seed)
    arr = corrupted["fct_revenue"]["_arrival_period"]
    # Affected rows: period + uniform(1, 5). Compare with date_key.
    affected = arr.notna()
    if affected.any():
        diffs = (
            arr[affected].astype(int).values
            - corrupted["fct_revenue"]["date_key"][affected].astype(int).values
        )
        assert (diffs >= 1).all() and (diffs <= 5).all()
