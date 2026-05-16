"""0.6-M19 — FK-target resolution across three call sites.

Three independent fixes that all share the same shape: resolve a
referenced column by its FK target, not by literal name match.

  1. **Cross-dim FK uniform draw on variable-grain facts**
     (``_emit_proportional_rows``). Pre-fix, a fact column FKing
     into a true reference dim (no entity back-link) fell through
     to ``parent.iloc[0]`` — every row referenced the same dim
     row. Fixed: uniform RNG draw across the dim.

  2. **CDC date-FK column resolution** (``_apply_cdc_audit_columns``).
     Pre-fix, the audit-column augmentation looked for a literal
     ``date_key`` column on the fact. Facts whose date column was
     renamed (e.g. ``billing_period``) silently lost CDC. Fixed:
     resolve via ``_find_date_fk_column``.

  3. **partition_by FK-target resolution** (``OutputConfig`` +
     ``write_single_table``). Pre-fix, ``partition_by: date_key``
     required a literal ``date_key`` column on every fact that
     should partition. Facts with renamed date columns silently
     wrote single files. Fixed: literal match takes precedence,
     FK target is the fallback at both config-validation and
     write time.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from plotsim import (
    build_manifest,
    create,
    generate_tables,
    generate_tables_with_state,
    write_tables,
)
from plotsim.builder import create_from_yaml
from plotsim.config import PlotsimConfig


ROOT = Path(__file__).resolve().parent.parent


# --- Fix #3: cross-dim FK uniform draw on variable-grain facts -------------


def _orders_with_payment_dim_config(**overrides: Any) -> PlotsimConfig:
    """Variable-grain ``fct_orders`` with a cross-dim FK into a true
    reference dim (no entity back-link). This is the configuration
    shape that pre-fix collapsed onto ``parent.iloc[0]``: ``back_link``
    in ``_emit_proportional_rows`` is None because ``dim_payment_method``
    has no column FKing back to ``dim_customer``.
    """
    base: dict[str, Any] = {
        "about": "variable-grain cross-dim FK test",
        "unit": "customer",
        "seed": 19001,
        "window": ("2024-01", "2024-06", "monthly"),
        "metrics": [
            {
                "name": "order_volume",
                "type": "amount",
                "polarity": "positive",
                "range": [3, 30],
            },
        ],
        "segments": [
            {"name": "growers", "count": 6, "archetype": "growth"},
            {"name": "decliners", "count": 6, "archetype": "decline"},
        ],
        "dimensions": [
            {
                "name": "dim_customer",
                "per": "unit",
                "columns": [
                    {"name": "customer_id", "type": "id"},
                    {"name": "customer_name", "type": "faker.name"},
                ],
            },
            {
                "name": "dim_payment_method",
                "reference": True,
                "columns": [
                    {"name": "payment_method_id", "type": "id"},
                    {
                        "name": "method_name",
                        "type": "static.cash,credit,debit,paypal,wire",
                    },
                ],
            },
        ],
        "facts": [
            {
                "name": "fct_orders",
                "row_count_driver": "order_volume",
                "row_count_scale": 1.0,
                "columns": [
                    {"name": "order_id", "type": "id"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "order_date", "type": "ref.dim_date"},
                    {
                        "name": "payment_method_id",
                        "type": "ref.dim_payment_method",
                    },
                ],
            },
        ],
    }
    base.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return create(**base)


class TestVariableGrainCrossDimFKUniformDraw:
    """Acceptance: cross-dim FK on variable-grain fact distributes
    across the dim rows (not always row 0)."""

    def test_cross_dim_fk_spans_multiple_dim_rows(self):
        cfg = _orders_with_payment_dim_config()
        tables = generate_tables(cfg)
        fct_orders = tables["fct_orders"]
        dim_payment = tables["dim_payment_method"]
        assert len(fct_orders) > 50, (
            "test precondition: enough rows to exercise the RNG draw " f"(got {len(fct_orders)})"
        )
        # All payment_method_id values must reference real dim rows.
        valid_ids = set(dim_payment["payment_method_id"].tolist())
        observed = set(fct_orders["payment_method_id"].tolist())
        assert observed.issubset(valid_ids), f"observed FK values {observed - valid_ids} not in dim"
        # Pre-fix behavior would collapse every row onto a single
        # dim PK. Post-fix, with 5 dim rows and ~hundreds of fact
        # rows, the empirical span must be > 1.
        assert len(observed) > 1, (
            "cross-dim FK on variable-grain fact collapsed onto a "
            "single dim row (pre-M19 behavior); expected uniform "
            "draw across the dim"
        )

    def test_cross_dim_fk_approximately_uniform(self):
        """With 5 dim rows and uniform draw, the empirical
        distribution should be roughly balanced — every dim row gets
        non-trivial mass."""
        cfg = _orders_with_payment_dim_config()
        tables = generate_tables(cfg)
        fct_orders = tables["fct_orders"]
        counts = fct_orders["payment_method_id"].value_counts()
        # All 5 payment methods should appear at least once across
        # the full fact (uniform draw on hundreds of rows).
        assert len(counts) == 5, (
            f"expected all 5 payment methods to appear; got " f"{counts.to_dict()}"
        )
        # Loose bound: no single value claims more than 50% of rows.
        max_share = counts.max() / counts.sum()
        assert max_share < 0.5, (
            f"one payment method claims {max_share:.0%} of rows; "
            f"draw is not approximately uniform"
        )

    def test_cross_dim_fk_deterministic_under_seed(self):
        cfg_a = _orders_with_payment_dim_config()
        cfg_b = _orders_with_payment_dim_config()
        tables_a = generate_tables(cfg_a)
        tables_b = generate_tables(cfg_b)
        assert (
            tables_a["fct_orders"]["payment_method_id"].tolist()
            == tables_b["fct_orders"]["payment_method_id"].tolist()
        )


# --- Fix #4: CDC date-FK column resolution ---------------------------------


def _cdc_with_renamed_date_fk_config(**overrides: Any) -> PlotsimConfig:
    """Like the cdc_demo template but the fact's date FK column is
    named ``billing_period`` instead of the literal ``date_key``.
    Pre-fix, ``_apply_cdc_audit_columns`` looked for ``date_key`` on
    the DataFrame and silently skipped — the fact emitted without
    ``_inserted_at`` / ``_updated_at`` / ``_op`` columns.
    """
    base: dict[str, Any] = {
        "about": "cdc renamed date FK test",
        "unit": "customer",
        "seed": 19002,
        "window": ("2024-01", "2024-06", "monthly"),
        "metrics": [
            {
                "name": "mrr",
                "type": "amount",
                "polarity": "positive",
                "range": [10, 5000],
            },
        ],
        "segments": [
            {"name": "g", "count": 5, "archetype": "growth"},
        ],
        "dimensions": [
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
                    {"name": "billing_period", "type": "ref.dim_date"},
                    {"name": "customer_id", "type": "ref.dim_customer"},
                    {"name": "mrr", "type": "metric.mrr"},
                ],
            },
        ],
    }
    base.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return create(**base)


class TestCDCRenamedDateFK:
    """Acceptance: CDC audit columns populate correctly when the date
    FK column is named anything other than the literal ``date_key``."""

    def test_audit_columns_present_with_renamed_date_fk(self):
        cfg = _cdc_with_renamed_date_fk_config()
        tables, _ = generate_tables_with_state(cfg)
        df = tables["fct_billing"]
        assert "billing_period" in df.columns, "test precondition: fact column should be renamed"
        assert "date_key" not in df.columns
        assert "_inserted_at" in df.columns
        assert "_updated_at" in df.columns
        assert "_op" in df.columns

    def test_inserted_at_resolves_via_fk_target(self):
        """Each row's ``_inserted_at`` is the period_label looked up
        via the row's ``billing_period`` value (the FK target is still
        ``dim_date.date_key``)."""
        cfg = _cdc_with_renamed_date_fk_config()
        tables, _ = generate_tables_with_state(cfg)
        fct = tables["fct_billing"]
        dim_date = tables["dim_date"]
        period_label_by_dkey = dict(
            zip(
                dim_date["date_key"].astype(int),
                dim_date["period_label"].astype(str),
            )
        )
        for _, row in fct.iterrows():
            expected = period_label_by_dkey[int(row["billing_period"])]
            assert row["_inserted_at"] == expected
        assert (fct["_op"] == "I").all()

    def test_audit_columns_survive_to_disk_write(self, tmp_path):
        cfg = _cdc_with_renamed_date_fk_config()
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
        assert "_inserted_at" in written.columns
        assert "_updated_at" in written.columns
        assert "_op" in written.columns
        assert (written["_op"] == "I").all()


# --- Fix #5: partition_by FK-target resolution -----------------------------


pq = pytest.importorskip("pyarrow.parquet")


def _orders_template_parquet_partitioned(tmp_path: Path) -> PlotsimConfig:
    """Orders template with parquet output + partition_by: date_key.
    The template's ``fct_orders`` declares ``order_date`` (ref.dim_date)
    rather than a literal ``date_key`` column — exactly the case the
    FK-target resolution unlocks."""
    cfg = create_from_yaml(ROOT / "tests" / "configs" / "orders_template.yaml")
    return cfg.model_copy(
        update={
            "output": cfg.output.model_copy(
                update={
                    "format": "parquet",
                    "directory": str(tmp_path),
                    "partition_by": "date_key",
                }
            ),
        }
    )


class TestPartitionByFKTargetResolution:
    """Acceptance: ``partition_by: date_key`` partitions facts whose
    date column is renamed (e.g. ``order_date``) via FK target. Literal
    name match still takes precedence on tables that declare it."""

    def test_validator_accepts_fk_target_only_config(self, tmp_path):
        """Pre-fix this raised: no literal ``date_key`` column exists
        anywhere in the orders template — only FK targets to it."""
        cfg = _orders_template_parquet_partitioned(tmp_path)
        # Loading + model_copy through the validator did not raise.
        assert cfg.output.partition_by == "date_key"

    def test_renamed_date_fk_partitions_via_local_column(self, tmp_path):
        """``fct_orders.order_date`` is the FK to ``dim_date.date_key``;
        the partitioned writer should create
        ``fct_orders/order_date=<value>/`` directories."""
        cfg = _orders_template_parquet_partitioned(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg)
            write_tables(tables, cfg, output_dir=tmp_path)
        dataset_dir = tmp_path / "fct_orders"
        assert dataset_dir.is_dir(), (
            "fct_orders should partition via FK-target resolution "
            "even though it has no literal 'date_key' column"
        )
        # Directories use the LOCAL column name (order_date), which
        # matches the actual partitioning column.
        partition_dirs = [child.name for child in dataset_dir.iterdir() if child.is_dir()]
        assert all(d.startswith("order_date=") for d in partition_dirs), (
            f"expected order_date=<value>/ partition dirs, got " f"{partition_dirs[:5]}"
        )

    def test_dim_date_partitions_via_literal_match(self, tmp_path):
        """``dim_date`` carries a literal ``date_key`` column — that
        path stays primary (literal match takes precedence)."""
        cfg = _orders_template_parquet_partitioned(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg)
            write_tables(tables, cfg, output_dir=tmp_path)
        dim_dir = tmp_path / "dim_date"
        assert dim_dir.is_dir()
        partition_dirs = [child.name for child in dim_dir.iterdir() if child.is_dir()]
        assert all(d.startswith("date_key=") for d in partition_dirs)

    def test_table_without_date_fk_stays_single_file(self, tmp_path):
        """``dim_customer`` has neither a literal ``date_key`` nor an
        FK to ``dim_date`` — it must write as a single Parquet file."""
        cfg = _orders_template_parquet_partitioned(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg)
            write_tables(tables, cfg, output_dir=tmp_path)
        single_file = tmp_path / "dim_customer.parquet"
        dir_form = tmp_path / "dim_customer"
        assert single_file.is_file()
        assert not dir_form.exists()

    def test_validator_rejects_unmatched_partition_key(self):
        """Pre- and post-fix: a partition_by with no literal match AND
        no FK target match still raises at load."""
        cfg = create_from_yaml(ROOT / "tests" / "configs" / "orders_template.yaml")
        payload = cfg.model_dump()
        payload["output"]["format"] = "parquet"
        payload["output"]["partition_by"] = "definitely_nonexistent_col"
        with pytest.raises(Exception, match="does not match any"):
            PlotsimConfig.model_validate(payload)

    def test_fk_target_partition_round_trips(self, tmp_path):
        cfg = _orders_template_parquet_partitioned(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg)
            write_tables(tables, cfg, output_dir=tmp_path)
        recovered = pd.read_parquet(tmp_path / "fct_orders")
        original = tables["fct_orders"]
        assert len(recovered) == len(original)
        assert set(recovered.columns) == set(original.columns)
