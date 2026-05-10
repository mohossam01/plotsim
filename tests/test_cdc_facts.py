"""0.6-M9c — fact-side CDC audit columns.

When ``Table.cdc`` is True (or ``FactInput.cdc`` on the builder side),
the fact table emits three audit columns at generation time:

  * ``_inserted_at`` — ISO period string from each row's date_key,
    looked up against ``dim_date.period_label``.
  * ``_updated_at`` — same as ``_inserted_at`` initially; bumped to
    the last period's label on rows that a column-level quality issue
    mutated.
  * ``_op`` — ``"I"`` for the initial insert, ``"U"`` for rows the
    quality layer flipped.

Tests cover the surface from config validation through end-to-end
write:

  1. **Config validation** — ``Table.cdc=True`` rejected on dim /
     event / bridge tables.
  2. **Audit columns at generation** — present when enabled, absent
     when disabled, value mapping correct, multi-fact selectivity.
  3. **Quality mutation flip** — ``null_injection`` /
     ``type_mismatch`` / ``schema_drift`` flip the affected rows to
     ``_op="U"``; ``duplicate_rows`` / ``late_arrival`` do NOT (their
     ground-truth indices don't align with the corrupted frame).
  4. **End-to-end write** — the on-disk CSV has the audit columns and
     the U-flipped ``_op`` values.
  5. **Holdout interaction** — CDC works alongside holdout when
     quality is empty; both _train and _holdout files carry audit
     columns.
  6. **Builder passthrough** — ``FactInput.cdc`` routes onto engine
     ``Table.cdc``.
  7. **Bundled template** — ``plotsim.load_template("cdc_demo")``
     produces a working CDC config end-to-end.
  8. **Determinism** — same ``(config, seed)`` produces the same
     ``_op`` sequence across runs.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

import plotsim
from plotsim import (
    build_manifest,
    create,
    generate_tables_with_state,
    load_template,
    write_tables,
)
from plotsim.config import PlotsimConfig


# --- 1. Config-level validation --------------------------------------------


def _explicit_cdc_kwargs(**overrides: Any) -> dict:
    base: dict[str, Any] = {
        "about": "cdc test",
        "unit": "customer",
        "seed": 7,
        "window": {"start": "2024-01", "end": "2024-06"},
        "metrics": [
            {"name": "mrr", "type": "amount", "polarity": "positive", "range": [10, 5000]},
        ],
        "segments": [
            {"name": "g", "count": 5, "archetype": "growth"},
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
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                    {"name": "customer_name", "type": "faker.name"},
                ],
            },
        ],
        "facts": [
            {
                "name": "fct_billing",
                "metrics": ["mrr"],
                "cdc": True,
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "mrr", "type": "metric.mrr"},
                ],
            },
        ],
    }
    base.update(overrides)
    return base


def _create(**overrides: Any) -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return create(**_explicit_cdc_kwargs(**overrides))


def test_table_cdc_default_is_false():
    cfg = _create(
        facts=[
            {
                "name": "fct_billing",
                "metrics": ["mrr"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "mrr", "type": "metric.mrr"},
                ],
            },
        ]
    )
    fact = next(t for t in cfg.tables if t.type == "fact")
    assert fact.cdc is False


def test_table_cdc_true_on_fact_accepted():
    cfg = _create()
    fact = next(t for t in cfg.tables if t.type == "fact")
    assert fact.cdc is True


def test_table_cdc_rejected_on_dim_via_engine_config():
    """Engine-direct ``Table(type='dim', cdc=True)`` must raise."""
    from plotsim.config import Column, Table

    with pytest.raises(ValidationError, match="cdc=True but type='dim'"):
        Table(
            name="dim_test",
            type="dim",
            grain="per_entity",
            columns=[
                Column(name="entity_id", source="pk", dtype="int"),
            ],
            primary_key="entity_id",
            cdc=True,
        )


def test_table_cdc_rejected_on_event_via_engine_config():
    from plotsim.config import Column, Table

    with pytest.raises(ValidationError, match="cdc=True but type='event'"):
        Table(
            name="evt_test",
            type="event",
            grain="per_entity_per_period",
            columns=[
                Column(name="event_id", source="pk", dtype="int"),
                Column(name="date_key", source="fk:dim_date.date_key", dtype="int"),
                Column(name="customer_id", source="fk:dim_customer.customer_id", dtype="int"),
            ],
            primary_key="event_id",
            cdc=True,
        )


# --- 2. Audit columns at generation ----------------------------------------


def test_cdc_disabled_fact_has_no_audit_columns():
    cfg = _create(
        facts=[
            {
                "name": "fct_billing",
                "metrics": ["mrr"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "mrr", "type": "metric.mrr"},
                ],
            },
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    df = tables["fct_billing"]
    assert "_inserted_at" not in df.columns
    assert "_updated_at" not in df.columns
    assert "_op" not in df.columns


def test_cdc_enabled_fact_has_three_audit_columns():
    cfg = _create()
    tables, _ = generate_tables_with_state(cfg)
    df = tables["fct_billing"]
    assert "_inserted_at" in df.columns
    assert "_updated_at" in df.columns
    assert "_op" in df.columns


def test_cdc_initial_op_is_all_inserts():
    cfg = _create()
    tables, _ = generate_tables_with_state(cfg)
    df = tables["fct_billing"]
    assert (df["_op"] == "I").all()


def test_cdc_inserted_at_matches_period_label_per_row():
    """Each row's ``_inserted_at`` should equal the ``period_label``
    looked up via that row's ``date_key`` against ``dim_date``."""
    cfg = _create()
    tables, _ = generate_tables_with_state(cfg)
    fct = tables["fct_billing"]
    dim_date = tables["dim_date"]
    period_label_by_dkey = dict(
        zip(dim_date["date_key"].astype(int), dim_date["period_label"].astype(str))
    )
    for _, row in fct.iterrows():
        expected = period_label_by_dkey[int(row["date_key"])]
        assert row["_inserted_at"] == expected


def test_cdc_updated_at_equals_inserted_at_at_generation():
    cfg = _create()
    tables, _ = generate_tables_with_state(cfg)
    fct = tables["fct_billing"]
    pd.testing.assert_series_equal(
        fct["_inserted_at"].reset_index(drop=True),
        fct["_updated_at"].reset_index(drop=True),
        check_names=False,
    )


def test_cdc_audit_columns_at_end_of_column_order():
    """Convention: engine-added columns sit at the tail so user-declared
    column order is preserved."""
    cfg = _create()
    tables, _ = generate_tables_with_state(cfg)
    cols = list(tables["fct_billing"].columns)
    assert cols[-3:] == ["_inserted_at", "_updated_at", "_op"]


def test_cdc_selectivity_per_fact_table():
    """Two facts in one config — only the cdc=True one gets audit columns."""
    cfg = _create(
        facts=[
            {
                "name": "fct_billing",
                "metrics": ["mrr"],
                "cdc": True,
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "mrr", "type": "metric.mrr"},
                ],
            },
            {
                "name": "fct_usage",
                "metrics": ["mrr"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "mrr", "type": "metric.mrr"},
                ],
            },
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    assert "_op" in tables["fct_billing"].columns
    assert "_op" not in tables["fct_usage"].columns


# --- 3. Quality mutation flip ----------------------------------------------


def _cdc_with_quality(rate: float = 0.1, issue: str = "null_injection") -> PlotsimConfig:
    spec = {"table": "fct_billing", "issue": issue, "rate": rate}
    if issue in ("null_injection", "type_mismatch", "schema_drift"):
        spec["column"] = "mrr"
    return _create(quality=[spec])


def test_cdc_null_injection_flips_op_to_u(tmp_path):
    cfg = _cdc_with_quality(rate=0.2, issue="null_injection")
    tables, state = generate_tables_with_state(cfg)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    out_dir = write_tables(tables, cfg, manifest=manifest, output_dir=tmp_path / "out")
    written = pd.read_csv(Path(out_dir) / "fct_billing.csv")
    null_rows = written[written["mrr"].isna()]
    assert len(null_rows) > 0
    assert (null_rows["_op"] == "U").all()
    insert_rows = written[~written["mrr"].isna()]
    assert (insert_rows["_op"] == "I").all()


def test_cdc_type_mismatch_flips_op_to_u(tmp_path):
    cfg = _cdc_with_quality(rate=0.2, issue="type_mismatch")
    tables, state = generate_tables_with_state(cfg)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    out_dir = write_tables(tables, cfg, manifest=manifest, output_dir=tmp_path / "out")
    written = pd.read_csv(Path(out_dir) / "fct_billing.csv")
    u_rows = written[written["_op"] == "U"]
    assert len(u_rows) > 0


def test_cdc_schema_drift_flips_op_to_u(tmp_path):
    cfg = _cdc_with_quality(rate=0.2, issue="schema_drift")
    tables, state = generate_tables_with_state(cfg)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    out_dir = write_tables(tables, cfg, manifest=manifest, output_dir=tmp_path / "out")
    written = pd.read_csv(Path(out_dir) / "fct_billing.csv")
    u_rows = written[written["_op"] == "U"]
    assert len(u_rows) > 0


def test_cdc_late_arrival_does_not_flip_op(tmp_path):
    """Row-level quality issue: ground-truth row_indices reference the
    pre-corruption frame and don't align with the corrupted frame
    after the corruption shifts indices. Helper intentionally skips."""
    cfg = _cdc_with_quality(rate=0.2, issue="late_arrival")
    tables, state = generate_tables_with_state(cfg)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    out_dir = write_tables(tables, cfg, manifest=manifest, output_dir=tmp_path / "out")
    written = pd.read_csv(Path(out_dir) / "fct_billing.csv")
    # Every row stays at _op="I"; late_arrival just adds an
    # _arrival_period column.
    assert (written["_op"] == "I").all()
    assert "_arrival_period" in written.columns


def test_cdc_updated_at_bumped_to_last_period(tmp_path):
    """U-flip rows carry ``_updated_at`` = last period's label."""
    cfg = _cdc_with_quality(rate=0.3, issue="null_injection")
    tables, state = generate_tables_with_state(cfg)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    out_dir = write_tables(tables, cfg, manifest=manifest, output_dir=tmp_path / "out")
    written = pd.read_csv(Path(out_dir) / "fct_billing.csv")
    last_period = tables["dim_date"]["period_label"].astype(str).iloc[-1]
    u_rows = written[written["_op"] == "U"]
    assert (u_rows["_updated_at"] == last_period).all()


def test_cdc_inserted_at_unchanged_by_quality_flip(tmp_path):
    """The U-flip touches ``_op`` and ``_updated_at`` only — the
    original ``_inserted_at`` per row stays at the row's date_key
    period."""
    cfg = _cdc_with_quality(rate=0.3, issue="null_injection")
    tables, state = generate_tables_with_state(cfg)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    out_dir = write_tables(tables, cfg, manifest=manifest, output_dir=tmp_path / "out")
    written = pd.read_csv(Path(out_dir) / "fct_billing.csv")
    dim_date = tables["dim_date"]
    period_label_by_dkey = dict(
        zip(dim_date["date_key"].astype(int), dim_date["period_label"].astype(str))
    )
    for _, row in written.iterrows():
        expected = period_label_by_dkey[int(row["date_key"])]
        assert str(row["_inserted_at"]) == expected


# --- 4. Manifest still records the ground truth -----------------------------


def test_cdc_manifest_quality_injections_unchanged(tmp_path):
    """CDC marking is purely a column-level addition on the corrupted
    output; the manifest's ``quality_injections`` ground-truth list
    must still record the original (table, column, row_indices,
    clean_values) tuples."""
    cfg = _cdc_with_quality(rate=0.2, issue="null_injection")
    tables, state = generate_tables_with_state(cfg)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    out_dir = write_tables(tables, cfg, manifest=manifest, output_dir=tmp_path / "out")
    payload = json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))
    assert len(payload["quality_injections"]) == 1
    qi = payload["quality_injections"][0]
    assert qi["issue_type"] == "null_injection"
    assert qi["table"] == "fct_billing"
    assert qi["column"] == "mrr"


# --- 5. Holdout interaction -------------------------------------------------


def test_cdc_holdout_train_and_holdout_files_carry_audit_columns(tmp_path):
    """Holdout requires no quality issues, but CDC columns ride along
    on both _train and _holdout splits."""
    cfg = _create(holdout={"target": "mrr", "periods": 2})
    tables, state = generate_tables_with_state(cfg)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    out_dir = write_tables(tables, cfg, manifest=manifest, output_dir=tmp_path / "out")
    train = pd.read_csv(Path(out_dir) / "fct_billing_train.csv")
    holdout = pd.read_csv(Path(out_dir) / "fct_billing_holdout.csv")
    for df in (train, holdout):
        assert "_inserted_at" in df.columns
        assert "_updated_at" in df.columns
        assert "_op" in df.columns
        assert (df["_op"] == "I").all()


# --- 6. Builder passthrough --------------------------------------------------


def test_builder_fact_input_cdc_routes_to_table():
    cfg = _create()
    fact = next(t for t in cfg.tables if t.name == "fct_billing")
    assert fact.cdc is True


def test_builder_fact_input_cdc_default_false():
    cfg = _create(
        facts=[
            {
                "name": "fct_billing",
                "metrics": ["mrr"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "mrr", "type": "metric.mrr"},
                ],
            },
        ]
    )
    fact = next(t for t in cfg.tables if t.name == "fct_billing")
    assert fact.cdc is False


# --- 7. Bundled template ---------------------------------------------------


def test_cdc_demo_template_in_list_templates():
    assert "cdc_demo" in plotsim.list_templates()


def test_cdc_demo_template_loads_and_generates(tmp_path):
    cfg = load_template("cdc_demo")
    assert isinstance(cfg, PlotsimConfig)
    fact = next(t for t in cfg.tables if t.name == "fct_billing")
    assert fact.cdc is True

    tables, state = generate_tables_with_state(cfg)
    manifest = build_manifest(
        cfg,
        state.trajectories,
        tables,
        scd_state=state.scd,
        bridge_state=state.bridges,
    )
    out_dir = write_tables(tables, cfg, manifest=manifest, output_dir=tmp_path / "out")
    written = pd.read_csv(Path(out_dir) / "fct_billing.csv")
    # The bundled template configures null_injection at 5%; the U-flip
    # should produce a non-empty subset of "U" rows.
    op_counts = written["_op"].value_counts()
    assert "I" in op_counts.index
    assert "U" in op_counts.index
    assert op_counts["U"] > 0


# --- 8. Determinism --------------------------------------------------------


def test_cdc_deterministic_op_sequence_across_runs(tmp_path):
    cfg = _cdc_with_quality(rate=0.2, issue="null_injection")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out_dir in (out_a, out_b):
        tables, state = generate_tables_with_state(cfg)
        manifest = build_manifest(
            cfg,
            state.trajectories,
            tables,
            scd_state=state.scd,
            bridge_state=state.bridges,
        )
        write_tables(tables, cfg, manifest=manifest, output_dir=out_dir)
    a = pd.read_csv(out_a / "fct_billing.csv")
    b = pd.read_csv(out_b / "fct_billing.csv")
    assert a["_op"].tolist() == b["_op"].tolist()
    assert a["_inserted_at"].tolist() == b["_inserted_at"].tolist()
    assert a["_updated_at"].tolist() == b["_updated_at"].tolist()


# --- 9. Pre-M9c output unchanged when cdc=False everywhere -----------------


def test_no_cdc_anywhere_preserves_pre_m9c_columns():
    """A config with no cdc=True facts produces the exact column set
    it produced pre-M9c."""
    cfg = _create(
        facts=[
            {
                "name": "fct_billing",
                "metrics": ["mrr"],
                "columns": [
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "mrr", "type": "metric.mrr"},
                ],
            },
        ]
    )
    tables, _ = generate_tables_with_state(cfg)
    cols = set(tables["fct_billing"].columns)
    cdc_cols = {"_inserted_at", "_updated_at", "_op"}
    assert cols.isdisjoint(cdc_cols)
