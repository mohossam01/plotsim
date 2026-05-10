"""0.6-M9a — volume_anomaly quality issue.

Spike or drop fact / event rows at one or more target periods. Mirrors
the surface of ``duplicate_rows`` (row-level, ``column="_rows"`` manifest
sentinel) and adds two row-level operations the existing five issue
types don't cover:

  * ``mode="spike"`` — duplicate ``floor(rate * N_at_period)`` rows AT
    each target period, append at the end.
  * ``mode="drop"`` — remove the same count.

Tests cover the same four layers as ``test_quality_injection.py``:

  1. **Config validation** — engine-level ``QualityIssue`` model
     enforces (a) ``mode`` set when type='volume_anomaly', (b) exactly
     one of ``target_period`` / ``target_periods`` set, (c) those
     fields rejected on non-VA issue types.
  2. **Cross-ref validation** — ``validate_tables_config`` errors when
     target_columns is not ``["*"]`` for volume_anomaly.
  3. **Pure pipeline** — ``apply_issues`` produces correct row-count
     changes, manifest records, and never mutates the input dict.
  4. **Builder passthrough** — ``QualityIssueInput`` accepts the new
     ``mode`` / ``period`` / ``periods`` shape and the interpreter
     routes them to the engine config.

The saas template emits 3 rows per period on fct_revenue, which lets
``rate=1.0`` produce a deterministic ``±3``/``±6`` row delta the tests
can pin without fragility to seed shifts.
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
    create,
    generate_tables_with_state,
    write_tables,
)
from plotsim.config import PlotsimConfig, QualityIssue
from plotsim.quality import apply_issues


SAAS_TEMPLATE = "plotsim/configs/sample_saas.yaml"
ROWS_PER_PERIOD = 3  # saas: 3 segments × 1 row each per period on fct_revenue


def _saas_dict() -> dict:
    with open(SAAS_TEMPLATE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _saas_with_quality(issues: list[dict]) -> PlotsimConfig:
    cfg = _saas_dict()
    cfg["quality"] = {"quality_issues": issues}
    return PlotsimConfig(**cfg)


# ---------------------------------------------------------------------------
# Engine config-level validation (QualityIssue model)
# ---------------------------------------------------------------------------


def test_quality_volume_anomaly_accepts_canonical_shape():
    """Spike at a single target period — minimal happy path."""
    issue = QualityIssue(
        type="volume_anomaly",
        target_table="fct_revenue",
        target_columns=["*"],
        rate=0.5,
        mode="spike",
        target_period=5,
    )
    assert issue.type == "volume_anomaly"
    assert issue.mode == "spike"
    assert issue.target_period == 5
    assert issue.target_periods is None


def test_quality_volume_anomaly_accepts_target_periods_list():
    issue = QualityIssue(
        type="volume_anomaly",
        target_table="fct_revenue",
        target_columns=["*"],
        rate=0.5,
        mode="drop",
        target_periods=[3, 7, 11],
    )
    assert issue.target_periods == [3, 7, 11]
    assert issue.target_period is None


def test_quality_volume_anomaly_rejects_missing_mode():
    with pytest.raises(ValidationError, match="requires.*mode"):
        QualityIssue(
            type="volume_anomaly",
            target_table="fct_revenue",
            target_columns=["*"],
            rate=0.5,
            target_period=5,
        )


def test_quality_volume_anomaly_rejects_missing_target():
    with pytest.raises(ValidationError, match="target_period.*target_periods"):
        QualityIssue(
            type="volume_anomaly",
            target_table="fct_revenue",
            target_columns=["*"],
            rate=0.5,
            mode="spike",
        )


def test_quality_volume_anomaly_rejects_both_target_fields():
    with pytest.raises(ValidationError, match="exactly one"):
        QualityIssue(
            type="volume_anomaly",
            target_table="fct_revenue",
            target_columns=["*"],
            rate=0.5,
            mode="spike",
            target_period=5,
            target_periods=[7, 9],
        )


def test_quality_volume_anomaly_rejects_negative_target_period():
    with pytest.raises(ValidationError):
        QualityIssue(
            type="volume_anomaly",
            target_table="fct_revenue",
            target_columns=["*"],
            rate=0.5,
            mode="spike",
            target_period=-1,
        )


def test_quality_volume_anomaly_rejects_empty_target_periods():
    with pytest.raises(ValidationError, match="non-empty"):
        QualityIssue(
            type="volume_anomaly",
            target_table="fct_revenue",
            target_columns=["*"],
            rate=0.5,
            mode="spike",
            target_periods=[],
        )


def test_quality_volume_anomaly_rejects_negative_in_target_periods():
    with pytest.raises(ValidationError, match="non-negative"):
        QualityIssue(
            type="volume_anomaly",
            target_table="fct_revenue",
            target_columns=["*"],
            rate=0.5,
            mode="drop",
            target_periods=[3, -1, 5],
        )


def test_quality_volume_anomaly_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        QualityIssue(
            type="volume_anomaly",
            target_table="fct_revenue",
            target_columns=["*"],
            rate=0.5,
            mode="invert",  # type: ignore[arg-type]
            target_period=5,
        )


def test_quality_volume_anomaly_fields_rejected_on_non_va_types():
    """`mode`, `target_period`, `target_periods` are only valid when
    type='volume_anomaly'."""
    with pytest.raises(ValidationError, match="only valid when type='volume_anomaly'"):
        QualityIssue(
            type="null_injection",
            target_table="fct_revenue",
            target_columns=["mrr"],
            rate=0.1,
            mode="spike",
        )
    with pytest.raises(ValidationError, match="only valid when type='volume_anomaly'"):
        QualityIssue(
            type="duplicate_rows",
            target_table="fct_revenue",
            target_columns=["*"],
            rate=0.1,
            target_period=5,
        )


def test_quality_volume_anomaly_rejects_explicit_target_columns_via_validator():
    """Cross-ref validation: volume_anomaly's target_columns must be ['*']."""
    cfg = _saas_dict()
    cfg["quality"] = {
        "quality_issues": [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["mrr"],
                "rate": 0.5,
                "mode": "spike",
                "target_period": 5,
            }
        ]
    }
    with pytest.raises(ValidationError, match="row-level corruption"):
        PlotsimConfig(**cfg)


# ---------------------------------------------------------------------------
# Pure pipeline — apply_issues with volume_anomaly
# ---------------------------------------------------------------------------


def test_volume_anomaly_spike_grows_by_period_row_count():
    cfg = _saas_with_quality(
        [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 1.0,
                "mode": "spike",
                "target_period": 5,
            }
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    n_orig = len(tables["fct_revenue"])
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    assert len(corrupted["fct_revenue"]) == n_orig + ROWS_PER_PERIOD
    assert len(gt) == 1
    assert gt[0].issue_type == "volume_anomaly"
    assert gt[0].column == "_rows"
    assert len(gt[0].row_indices) == ROWS_PER_PERIOD
    assert gt[0].clean_values == []


def test_volume_anomaly_drop_shrinks_by_period_row_count():
    cfg = _saas_with_quality(
        [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 1.0,
                "mode": "drop",
                "target_period": 5,
            }
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    n_orig = len(tables["fct_revenue"])
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    assert len(corrupted["fct_revenue"]) == n_orig - ROWS_PER_PERIOD
    assert gt[0].issue_type == "volume_anomaly"
    assert gt[0].column == "_rows"


def test_volume_anomaly_target_periods_list_aggregates_per_period():
    cfg = _saas_with_quality(
        [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 1.0,
                "mode": "drop",
                "target_periods": [3, 7, 11],
            }
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    n_orig = len(tables["fct_revenue"])
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    assert len(corrupted["fct_revenue"]) == n_orig - 3 * ROWS_PER_PERIOD
    assert len(gt[0].row_indices) == 3 * ROWS_PER_PERIOD


def test_volume_anomaly_spike_rows_are_actual_duplicates():
    """Appended duplicates carry the same column values as the source rows."""
    cfg = _saas_with_quality(
        [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 1.0,
                "mode": "spike",
                "target_period": 5,
            }
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    src_idxs = gt[0].row_indices
    src_rows = tables["fct_revenue"].iloc[src_idxs].reset_index(drop=True)
    n_orig = len(tables["fct_revenue"])
    appended = corrupted["fct_revenue"].iloc[n_orig:].reset_index(drop=True)
    pd.testing.assert_frame_equal(src_rows, appended)


def test_volume_anomaly_drop_removes_correct_rows():
    """The rows recorded in row_indices are exactly the rows missing
    from the corrupted output."""
    cfg = _saas_with_quality(
        [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 1.0,
                "mode": "drop",
                "target_period": 5,
            }
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    dropped_idxs = gt[0].row_indices
    expected_kept = tables["fct_revenue"].drop(index=dropped_idxs).reset_index(drop=True)
    pd.testing.assert_frame_equal(corrupted["fct_revenue"], expected_kept)


def test_volume_anomaly_target_period_out_of_range_is_noop():
    cfg = _saas_with_quality(
        [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 1.0,
                "mode": "spike",
                "target_period": 999,
            }
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    assert len(corrupted["fct_revenue"]) == len(tables["fct_revenue"])
    assert gt == []


def test_volume_anomaly_rate_zero_is_noop():
    cfg = _saas_with_quality(
        [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 0.0,
                "mode": "spike",
                "target_period": 5,
            }
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    assert len(corrupted["fct_revenue"]) == len(tables["fct_revenue"])
    assert gt == []


def test_volume_anomaly_does_not_mutate_input():
    cfg = _saas_with_quality(
        [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 1.0,
                "mode": "spike",
                "target_period": 5,
            }
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    snapshot = tables["fct_revenue"].copy(deep=True)
    apply_issues(tables, cfg, base_seed=cfg.seed)
    pd.testing.assert_frame_equal(tables["fct_revenue"], snapshot)


def test_volume_anomaly_is_deterministic():
    cfg = _saas_with_quality(
        [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 0.5,  # partial rate exercises the rng draw
                "mode": "spike",
                "target_period": 5,
            }
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    a, gt_a = apply_issues(tables, cfg, base_seed=cfg.seed)
    b, gt_b = apply_issues(tables, cfg, base_seed=cfg.seed)
    pd.testing.assert_frame_equal(a["fct_revenue"], b["fct_revenue"])
    assert gt_a[0].row_indices == gt_b[0].row_indices


def test_volume_anomaly_seed_offset_changes_affected_rows():
    """At rate=0.5 across 3 rows, floor gives 1 row picked — different
    seed offsets should generally pick different rows. Multi-period
    sampling makes the difference robust."""
    template = {
        "type": "volume_anomaly",
        "target_table": "fct_revenue",
        "target_columns": ["*"],
        "rate": 0.5,
        "mode": "spike",
        "target_periods": [3, 7, 11, 15, 19],
    }
    cfg_a = _saas_with_quality([{**template, "seed_offset": 0}])
    cfg_b = _saas_with_quality([{**template, "seed_offset": 1000}])
    tables, _ = generate_tables_with_state(cfg_a)
    _, gt_a = apply_issues(tables, cfg_a, base_seed=cfg_a.seed)
    _, gt_b = apply_issues(tables, cfg_b, base_seed=cfg_b.seed)
    assert gt_a[0].row_indices != gt_b[0].row_indices


def test_volume_anomaly_partial_rate_picks_floor():
    """At rate=0.5 against 3 rows per period, floor(0.5 * 3) = 1 row
    per period."""
    cfg = _saas_with_quality(
        [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 0.5,
                "mode": "drop",
                "target_periods": [3, 7],
            }
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    n_orig = len(tables["fct_revenue"])
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    assert n_orig - len(corrupted["fct_revenue"]) == 2
    assert len(gt[0].row_indices) == 2


def test_volume_anomaly_coexists_with_other_issues():
    """Multi-issue config: null_injection + volume_anomaly drop. Each
    issue applies to its own target_table copy, manifest records both."""
    cfg = _saas_with_quality(
        [
            {
                "type": "null_injection",
                "target_table": "fct_engagement",
                "target_columns": ["engagement_score"],
                "rate": 0.1,
                "seed_offset": 0,
            },
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 1.0,
                "mode": "drop",
                "target_period": 5,
                "seed_offset": 1,
            },
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    n_orig_rev = len(tables["fct_revenue"])
    n_orig_eng = len(tables["fct_engagement"])
    corrupted, gt = apply_issues(tables, cfg, base_seed=cfg.seed)
    assert len(corrupted["fct_revenue"]) == n_orig_rev - ROWS_PER_PERIOD
    assert len(corrupted["fct_engagement"]) == n_orig_eng  # unchanged
    types = {r.issue_type for r in gt}
    assert types == {"null_injection", "volume_anomaly"}


# ---------------------------------------------------------------------------
# Manifest + write_tables integration
# ---------------------------------------------------------------------------


def test_volume_anomaly_manifest_written_to_disk(tmp_path):
    cfg_dict = _saas_dict()
    cfg_dict["output"]["directory"] = str(tmp_path / "out")
    cfg_dict["quality"] = {
        "quality_issues": [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 1.0,
                "mode": "spike",
                "target_period": 5,
            }
        ]
    }
    cfg = PlotsimConfig(**cfg_dict)
    tables, state = generate_tables_with_state(cfg)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    out_dir = write_tables(tables, cfg, manifest=manifest)

    payload = json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))
    qi = payload["quality_injections"]
    assert len(qi) == 1
    assert qi[0]["issue_type"] == "volume_anomaly"
    assert qi[0]["column"] == "_rows"
    assert qi[0]["table"] == "fct_revenue"
    assert qi[0]["clean_values"] == []
    assert len(qi[0]["row_indices"]) == ROWS_PER_PERIOD


# ---------------------------------------------------------------------------
# Holdout interaction (existing gate; volume_anomaly should also trip it)
# ---------------------------------------------------------------------------


def test_volume_anomaly_blocks_holdout_at_load():
    cfg = _saas_dict()
    cfg["quality"] = {
        "quality_issues": [
            {
                "type": "volume_anomaly",
                "target_table": "fct_revenue",
                "target_columns": ["*"],
                "rate": 1.0,
                "mode": "drop",
                "target_period": 5,
            }
        ]
    }
    cfg["holdout"] = {
        "enabled": True,
        "target_metric": "mrr",
        "holdout_periods": 3,
    }
    with pytest.raises(ValidationError):
        PlotsimConfig(**cfg)


# ---------------------------------------------------------------------------
# Builder layer — QualityIssueInput + interpreter passthrough
# ---------------------------------------------------------------------------


def _builder_kwargs() -> dict:
    """Minimal builder input for quality-issue routing tests.

    Mirrors ``test_builder_power_features._explicit_input``: explicit
    dim/fact schema lets the builder run with the volume_anomaly
    issue without depending on auto-schema fact-name conventions.
    """
    return {
        "about": "volume_anomaly routing demo",
        "unit": "company",
        "window": {"start": "2024-01", "end": "2024-12"},
        "metrics": [
            {"name": "mrr", "type": "amount", "polarity": "positive", "range": [100, 50_000]},
        ],
        "segments": [
            {"name": "alpha", "count": 5, "archetype": "growth"},
        ],
        "dimensions": [
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
                    {"name": "company_name", "type": "faker.company"},
                ],
            },
        ],
        "facts": [
            {
                "name": "fct_company",
                "metrics": ["mrr"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "company_id", "type": "ref.dim_company"},
                    {"name": "mrr", "type": "metric.mrr"},
                ],
            },
        ],
    }


def test_builder_volume_anomaly_routes_through_interpreter():
    """The builder's UserInput shape (mode/period/periods) must round-
    trip into engine config (mode/target_period/target_periods)."""
    cfg = create(
        **_builder_kwargs(),
        quality=[
            {
                "table": "fct_company",
                "issue": "volume_anomaly",
                "rate": 1.0,
                "mode": "spike",
                "period": 5,
            }
        ],
    )
    issues = cfg.quality.quality_issues
    assert len(issues) == 1
    assert issues[0].type == "volume_anomaly"
    assert issues[0].mode == "spike"
    assert issues[0].target_period == 5
    assert issues[0].target_columns == ["*"]


def test_builder_volume_anomaly_periods_list_routes_through():
    cfg = create(
        **_builder_kwargs(),
        quality=[
            {
                "table": "fct_company",
                "issue": "volume_anomaly",
                "rate": 0.5,
                "mode": "drop",
                "periods": [3, 7, 11],
            }
        ],
    )
    issue = cfg.quality.quality_issues[0]
    assert issue.mode == "drop"
    assert issue.target_periods == [3, 7, 11]
    assert issue.target_period is None


def test_builder_quality_issue_input_rejects_volume_anomaly_with_column():
    from plotsim.builder.input import QualityIssueInput

    with pytest.raises(ValidationError, match="row-level"):
        QualityIssueInput(
            table="fct_x",
            issue="volume_anomaly",
            rate=0.5,
            mode="spike",
            period=5,
            column="mrr",
        )


def test_builder_quality_issue_input_rejects_mode_on_null_injection():
    from plotsim.builder.input import QualityIssueInput

    with pytest.raises(ValidationError, match="does not accept"):
        QualityIssueInput(
            table="fct_x",
            issue="null_injection",
            rate=0.1,
            column="mrr",
            mode="spike",
        )


def test_builder_quality_issue_input_rejects_volume_anomaly_without_mode():
    from plotsim.builder.input import QualityIssueInput

    with pytest.raises(ValidationError, match="requires `mode`"):
        QualityIssueInput(
            table="fct_x",
            issue="volume_anomaly",
            rate=0.5,
            period=5,
        )


def test_builder_quality_issue_input_rejects_volume_anomaly_without_period():
    from plotsim.builder.input import QualityIssueInput

    with pytest.raises(ValidationError, match="`period`.*`periods`"):
        QualityIssueInput(
            table="fct_x",
            issue="volume_anomaly",
            rate=0.5,
            mode="spike",
        )
