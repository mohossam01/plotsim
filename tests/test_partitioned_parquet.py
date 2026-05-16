"""Tests for 0.6-M16a partitioned Parquet output.

When ``output.partition_by`` is set on a parquet config, every table
that carries a column with that name is written as a directory of
Parquet files under
``<output_dir>/<table_name>/<partition_by>=<value>/...`` via
``pyarrow.parquet.write_to_dataset``. Tables without the column fall
back to single-file Parquet. The streaming-Parquet row-group writer
(M121b) bypasses cleanly when partitioning is on — partitioning wins
because it's the user-visible knob, streaming is an internal memory
tactic.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotsim.builder import create_from_yaml
from plotsim.config import OutputConfig, PlotsimConfig
from plotsim.output import (
    _streaming_parquet_eligible,
    write_tables,
)
from plotsim.tables import generate_tables


pq = pytest.importorskip("pyarrow.parquet")


ROOT = Path(__file__).resolve().parent.parent


# --- Helpers ---------------------------------------------------------------


def _saas_parquet_config(tmp_path: Path, *, partition_by: str | None = "date_key"):
    """Load the saas template and switch it to parquet output with the
    requested ``partition_by``. ``partition_by=None`` reverts to single-
    file parquet for the baseline-unchanged test."""
    cfg = create_from_yaml(ROOT / "tests" / "configs" / "saas_template.yaml")
    return cfg.model_copy(
        update={
            "output": cfg.output.model_copy(
                update={
                    "format": "parquet",
                    "directory": str(tmp_path),
                    "partition_by": partition_by,
                }
            ),
        }
    )


def _tables_for(cfg) -> dict[str, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_tables(cfg, np.random.default_rng(cfg.seed))


# --- Directory structure ---------------------------------------------------


class TestDirectoryStructure:
    """Partitioned tables land as ``<output>/<table>/<col>=<value>/*.parquet``
    directories; tables without the partition column stay as a single
    ``<output>/<table>.parquet`` file."""

    def test_partitioned_tables_become_directories(self, tmp_path):
        cfg = _saas_parquet_config(tmp_path)
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        for tbl in cfg.tables:
            has_col = any(c.name == "date_key" for c in tbl.columns)
            dataset_dir = tmp_path / tbl.name
            single_file = tmp_path / f"{tbl.name}.parquet"
            if has_col:
                assert dataset_dir.is_dir(), f"{tbl.name}: expected directory"
                assert not single_file.exists(), (
                    f"{tbl.name}: single-file parquet should not be written "
                    f"alongside the dataset directory"
                )
            else:
                assert single_file.is_file(), f"{tbl.name}: expected single file"
                assert (
                    not dataset_dir.exists()
                ), f"{tbl.name}: should not be partitioned (no date_key column)"

    def test_partition_directory_names_match_column_values(self, tmp_path):
        cfg = _saas_parquet_config(tmp_path)
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        fact_name = "fct_revenue"
        expected = {str(v) for v in tables[fact_name]["date_key"].unique()}
        dataset_dir = tmp_path / fact_name
        observed = set()
        for child in dataset_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            assert name.startswith("date_key="), name
            observed.add(name[len("date_key=") :])
        assert observed == expected, (
            f"{fact_name}: partition dirs {sorted(observed)[:5]}... "
            f"don't match column values {sorted(expected)[:5]}..."
        )


# --- Round-trip equality ---------------------------------------------------


class TestRoundTrip:
    """Reading the partitioned dataset back via ``pd.read_parquet``
    recovers every cell value (allowing partition column dtype drift,
    which pyarrow normalises on read)."""

    def test_partitioned_data_round_trips_to_original(self, tmp_path):
        cfg = _saas_parquet_config(tmp_path)
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        for fact_name in ("fct_engagement", "fct_revenue", "fct_support_tickets"):
            original = tables[fact_name]
            recovered = pd.read_parquet(tmp_path / fact_name)
            assert len(recovered) == len(original), fact_name
            assert set(recovered.columns) == set(original.columns), fact_name
            sort_keys = [
                c.name
                for c in next(t for t in cfg.tables if t.name == fact_name).columns
                if c.dtype == "id"
            ]
            o = original.sort_values(sort_keys).reset_index(drop=True)
            r = recovered[original.columns].sort_values(sort_keys).reset_index(drop=True)
            for col in original.columns:
                if pd.api.types.is_numeric_dtype(o[col]) and pd.api.types.is_numeric_dtype(r[col]):
                    o_arr = pd.to_numeric(o[col], errors="coerce").to_numpy()
                    r_arr = pd.to_numeric(r[col], errors="coerce").to_numpy()
                    np.testing.assert_array_equal(o_arr, r_arr, err_msg=f"{fact_name}.{col}")
                else:
                    assert (
                        o[col].astype(object).tolist() == r[col].astype(object).tolist()
                    ), f"{fact_name}.{col}"


# --- Baseline parity -------------------------------------------------------


class TestBaselineParity:
    """``partition_by=None`` produces output byte-identical to a config
    that omits the field entirely — guards against accidental behaviour
    leaks from the partitioning branch."""

    def test_partition_by_none_unchanged(self, tmp_path):
        cfg_a = _saas_parquet_config(tmp_path / "with_field", partition_by=None)
        cfg_b = create_from_yaml(ROOT / "tests" / "configs" / "saas_template.yaml").model_copy(
            update={
                "output": OutputConfig(
                    format="parquet",
                    directory=str(tmp_path / "without_field"),
                ),
            }
        )
        tables_a = _tables_for(cfg_a)
        tables_b = _tables_for(cfg_b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables_a, cfg_a, output_dir=tmp_path / "with_field")
            write_tables(tables_b, cfg_b, output_dir=tmp_path / "without_field")

        for tbl in cfg_a.tables:
            path_a = tmp_path / "with_field" / f"{tbl.name}.parquet"
            path_b = tmp_path / "without_field" / f"{tbl.name}.parquet"
            assert path_a.is_file(), tbl.name
            assert path_b.is_file(), tbl.name
            df_a = pd.read_parquet(path_a)
            df_b = pd.read_parquet(path_b)
            pd.testing.assert_frame_equal(df_a, df_b)


# --- Nested column compatibility -------------------------------------------


class TestNestedColumns:
    """Tables with ``struct`` / ``array`` columns survive partitioning —
    the explicit-schema path in ``_build_nested_pa_schema`` is shared
    between single-file and partitioned writers."""

    def test_struct_column_survives_partitioning(self, tmp_path):
        cfg = create_from_yaml(ROOT / "tests" / "configs" / "retail_template.yaml")
        cfg_p = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(
                    update={
                        "format": "parquet",
                        "directory": str(tmp_path),
                        "partition_by": "margin_tier",
                    }
                ),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg_p, np.random.default_rng(cfg_p.seed))
            write_tables(tables, cfg_p, output_dir=tmp_path)

        dataset_dir = tmp_path / "dim_product_category"
        assert dataset_dir.is_dir()
        recovered = pd.read_parquet(dataset_dir)
        original = tables["dim_product_category"]
        assert len(recovered) == len(original)
        sample = recovered["catalog_metadata"].dropna().iloc[0]
        assert isinstance(sample, dict), type(sample)
        assert set(sample.keys()) == set(original["catalog_metadata"].iloc[0].keys())


# --- Validator coverage ----------------------------------------------------


class TestValidators:
    """``output.partition_by`` is rejected at load when: format is not
    parquet; the column doesn't exist on any table; the column has a
    float / struct / array dtype. Tests use direct construction
    (``OutputConfig(...)``) and ``PlotsimConfig.model_validate(...)``
    because pydantic v2's ``model_copy`` does not re-run validators."""

    def test_requires_parquet_format(self):
        with pytest.raises(Exception, match="requires output.format='parquet'"):
            OutputConfig(format="csv", directory="x", partition_by="date_key")

    def test_rejects_unknown_column(self):
        cfg = create_from_yaml(ROOT / "tests" / "configs" / "saas_template.yaml")
        payload = cfg.model_dump()
        payload["output"]["format"] = "parquet"
        payload["output"]["partition_by"] = "nonexistent_col"
        with pytest.raises(Exception, match="does not match any column"):
            PlotsimConfig.model_validate(payload)

    def test_rejects_float_column(self):
        cfg = create_from_yaml(ROOT / "tests" / "configs" / "saas_template.yaml")
        payload = cfg.model_dump()
        payload["output"]["format"] = "parquet"
        # mrr on fct_revenue is dtype=float — should be rejected as a
        # partition key.
        payload["output"]["partition_by"] = "mrr"
        with pytest.raises(Exception, match="not a valid partition key type"):
            PlotsimConfig.model_validate(payload)


# --- Streaming bypass ------------------------------------------------------


class TestStreamingBypass:
    """Partitioning disables the M121b streaming-Parquet row-group writer
    so the partitioned-dataset path is the only Parquet writer on
    partitioned configs."""

    def test_streaming_eligibility_false_when_partitioned(self):
        cfg = create_from_yaml(ROOT / "tests" / "configs" / "saas_template.yaml")
        cfg_v = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(
                    update={"format": "parquet", "partition_by": "date_key"}
                ),
                "generation_mode": "vectorized",
            }
        )
        assert _streaming_parquet_eligible(cfg_v) is False

    def test_streaming_eligibility_true_without_partition(self):
        cfg = create_from_yaml(ROOT / "tests" / "configs" / "saas_template.yaml")
        cfg_v = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"format": "parquet"}),
                "generation_mode": "vectorized",
            }
        )
        assert _streaming_parquet_eligible(cfg_v) is True


# --- Sidecar behaviour per design Q2 ---------------------------------------


class TestSidecars:
    """Confirms the design-Q2 sidecar contract:
    - Denormalized wide tables: partition when column present.
    - Companions (config.yaml, validation_report.txt, manifest.json):
      always single files at the top level.
    """

    def test_denormalized_wide_partitions(self, tmp_path):
        cfg = create_from_yaml(ROOT / "tests" / "configs" / "saas_template.yaml")
        cfg_p = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(
                    update={
                        "format": "parquet",
                        "directory": str(tmp_path),
                        "partition_by": "date_key",
                        "denormalized": True,
                    }
                ),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg_p, np.random.default_rng(cfg_p.seed))
            write_tables(tables, cfg_p, output_dir=tmp_path)

        for fact in ("fct_engagement", "fct_revenue", "fct_support_tickets"):
            wide_dir = tmp_path / f"{fact}_wide"
            assert (
                wide_dir.is_dir()
            ), f"{fact}_wide should partition when date_key is on the wide frame"
            assert not (tmp_path / f"{fact}_wide.parquet").exists()

    def test_companions_always_single_files(self, tmp_path):
        """``write_tables`` only emits ``manifest.json`` when a manifest
        is passed in explicitly; this test asserts the two unconditional
        companions (``config.yaml``, ``validation_report.txt``) stay as
        top-level files under partitioning."""
        cfg = _saas_parquet_config(tmp_path)
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        assert (tmp_path / "config.yaml").is_file()
        assert (tmp_path / "validation_report.txt").is_file()
        # Neither is a directory — companions are never partitioned.
        assert not (tmp_path / "config.yaml").is_dir()
        assert not (tmp_path / "validation_report.txt").is_dir()
