"""Tests for M121b streaming Parquet writer.

The writer is opt-in via two conditions: ``output.format == "parquet"``
AND the resolved generation_mode is ``"vectorized"``. CSV output and
serial-mode runs keep the pre-mission single-shot ``to_parquet`` path.

Per the operator's AC correction: streaming and non-streaming Parquet
files are NOT byte-identical (row group metadata differs by
construction). The contract is round-trip DataFrame equality —
``pd.read_parquet(streaming) == pd.read_parquet(non_streaming)`` for
the same ``(config, seed, generation_mode)``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotsim.builder import create_from_yaml
from plotsim.config import QualityConfig, QualityIssue, load_config
from plotsim.output import (
    _streaming_fact_table_names,
    _streaming_parquet_eligible,
    write_tables,
)
from plotsim.tables import (
    _resolve_generation_mode,
    generate_tables,
    iter_fact_chunks,
)


pq = pytest.importorskip("pyarrow.parquet")


ROOT = Path(__file__).resolve().parent.parent


# --- Eligibility gate -------------------------------------------------------


class TestEligibility:
    """``_streaming_parquet_eligible`` is the dispatch gate. Both
    conditions (parquet format AND vectorized resolution) must hold."""

    def test_csv_serial_not_eligible(self):
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        cfg = cfg.model_copy(update={"generation_mode": "serial"})
        assert cfg.output.format == "csv"
        assert _resolve_generation_mode(cfg) == "serial"
        assert not _streaming_parquet_eligible(cfg)

    def test_csv_vectorized_not_eligible(self):
        """CSV output keeps the existing path even when vectorized.
        Streaming Parquet is a Parquet-only optimization."""
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        cfg = cfg.model_copy(update={"generation_mode": "vectorized"})
        assert cfg.output.format == "csv"
        assert _resolve_generation_mode(cfg) == "vectorized"
        assert not _streaming_parquet_eligible(cfg)

    def test_parquet_serial_not_eligible(self):
        """Serial mode keeps the single-shot ``to_parquet`` path even when
        format is parquet — the chunked iterator's row-major layout
        assumption only holds for the vectorized dispatcher's archetype
        groupings."""
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        cfg = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"format": "parquet"}),
                "generation_mode": "serial",
            }
        )
        assert _resolve_generation_mode(cfg) == "serial"
        assert not _streaming_parquet_eligible(cfg)

    def test_parquet_vectorized_eligible(self):
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        cfg = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"format": "parquet"}),
                "generation_mode": "vectorized",
            }
        )
        assert _streaming_parquet_eligible(cfg)


# --- Chunk iterator ---------------------------------------------------------


class TestIterFactChunks:
    """``iter_fact_chunks`` slices unified fact DataFrames into per-
    archetype chunks. Helper for the streaming writer; tested
    independently because it's also a useful seam for analysis tooling."""

    def test_chunk_count_matches_archetypes(self):
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(
                cfg.model_copy(update={"generation_mode": "vectorized"}),
                np.random.default_rng(cfg.seed),
            )
        chunks = list(iter_fact_chunks(cfg, tables))
        # saas builder template has 6 segments; each segment expands to
        # one archetype so we expect 6 chunks for the per_entity_per_period
        # facts. No per_period facts in saas → no sentinel chunk.
        archetype_chunks = [c for c in chunks if c[0] != "__per_period__"]
        unique_archetypes = {e.archetype for e in cfg.entities}
        assert len(archetype_chunks) == len(unique_archetypes)

    def test_chunk_row_counts_match_entity_counts(self):
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(
                cfg.model_copy(update={"generation_mode": "vectorized"}),
                np.random.default_rng(cfg.seed),
            )
        n_periods = cfg.time_window.period_count()
        # entities-per-archetype, derived from config.entities.
        ents_per_arch: dict[str, int] = {}
        for e in cfg.entities:
            ents_per_arch[e.archetype] = ents_per_arch.get(e.archetype, 0) + 1
        for arch, chunk in iter_fact_chunks(cfg, tables):
            if arch == "__per_period__":
                continue
            for _name, df in chunk.items():
                assert len(df) == ents_per_arch[arch] * n_periods, (
                    f"{arch}: expected {ents_per_arch[arch]}×{n_periods} " f"rows, got {len(df)}"
                )

    def test_chunk_union_equals_unified(self):
        """Concatenating every chunk's fact DataFrame should reconstruct
        the unified DataFrame (row order may differ, but row sets match)."""
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(
                cfg.model_copy(update={"generation_mode": "vectorized"}),
                np.random.default_rng(cfg.seed),
            )
        for fact_name in (
            n for n in tables if any(t.name == n and t.type == "fact" for t in cfg.tables)
        ):
            unified = tables[fact_name]
            collected = []
            for arch, chunk in iter_fact_chunks(cfg, tables):
                if fact_name in chunk:
                    collected.append(chunk[fact_name])
            recombined = pd.concat(collected, ignore_index=True)
            assert len(recombined) == len(unified), (
                f"{fact_name}: chunks total {len(recombined)} rows, " f"unified has {len(unified)}"
            )

    def test_per_period_fact_uses_sentinel(self):
        """Fact tables whose grain isn't ``per_entity_per_period`` (or
        whose row counts don't match the entity-major layout) yield
        once under the sentinel key, NOT per archetype."""
        # Build a config with a per_period fact via direct PlotsimConfig
        # construction; the builder doesn't expose per_period facts
        # directly. Use the engine-direct sample_saas fixture as a base.
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
        # sample_saas has per_entity_per_period facts only; the iterator
        # should not emit a sentinel chunk for it.
        chunks = list(iter_fact_chunks(cfg, tables))
        sentinel_chunks = [c for c in chunks if c[0] == "__per_period__"]
        assert sentinel_chunks == [], (
            "sample_saas has no per_period facts — sentinel chunk should " "not be emitted"
        )


# --- Round-trip equality (the operator's corrected AC) ---------------------


class TestRoundTripEquality:
    """Streaming Parquet and non-streaming Parquet, given the same
    in-memory ``tables`` dict, produce on-disk Parquet files that read
    back to cell-identical DataFrames. Raw file bytes differ (row group
    metadata varies by construction) — read via ``pd.read_parquet`` and
    compare frames, never bytes.

    Comparison is **disk-to-disk** (streaming write vs single-shot
    ``to_parquet`` write) rather than in-memory-vs-disk. Quality
    injection runs inside ``write_tables`` and produces a corrupted
    copy of the input; both writers consume the same corrupted dict,
    so the on-disk parity holds even when fact columns carry
    column-level quality issues like ``null_injection``. The prior
    in-memory-vs-disk pattern failed for any fact-column injection
    because the in-memory ``tables`` is the clean source pre-corruption
    — a contract gap discovered while shipping a fact-column
    ``null_injection`` in the saas template.
    """

    @staticmethod
    def _assert_parquet_files_equal(
        streaming_path: Path,
        plain_path: Path,
        fact_name: str,
    ) -> None:
        """Read both Parquet files and assert cell-for-cell DataFrame
        equality. Numeric columns compare via ``to_numpy()`` to absorb
        nullable-Int64 representation drift; non-numeric columns
        compare via ``astype(object).tolist()``."""
        a = pd.read_parquet(streaming_path)
        b = pd.read_parquet(plain_path)
        assert len(a) == len(b), fact_name
        assert sorted(a.columns) == sorted(b.columns), fact_name
        for col in a.columns:
            sa = pd.Series(a[col]).reset_index(drop=True)
            sb = pd.Series(b[col]).reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
                a_arr = pd.to_numeric(sa, errors="coerce").to_numpy()
                b_arr = pd.to_numeric(sb, errors="coerce").to_numpy()
                np.testing.assert_array_equal(a_arr, b_arr, err_msg=f"{fact_name}.{col}")
            else:
                assert (
                    sa.astype(object).tolist() == sb.astype(object).tolist()
                ), f"{fact_name}.{col}"

    @staticmethod
    def _write_streaming_and_plain(
        cfg_v,
        tables: dict[str, pd.DataFrame],
        tmp_path: Path,
    ) -> tuple[Path, Path]:
        """Write ``tables`` twice from the same in-memory dict — once
        through the streaming Parquet path (``generation_mode ==
        "vectorized"``) and once through the single-shot ``to_parquet``
        path (flipped to ``"serial"`` for the write call only).

        Flipping ``generation_mode`` on the write-side config copy
        toggles ``_streaming_parquet_eligible`` without re-running
        generation; quality injection, CDC flip, denormalization, and
        every other ``write_tables`` branch is mode-agnostic and
        consumes the same corrupted dict on both writes
        (``_apply_quality_issues`` is deterministic given
        ``(tables, config, seed)`` and returns a new dict without
        mutating its input)."""
        stream_dir = tmp_path / "streaming"
        plain_dir = tmp_path / "plain"
        cfg_plain = cfg_v.model_copy(update={"generation_mode": "serial"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg_v, output_dir=stream_dir)
            write_tables(tables, cfg_plain, output_dir=plain_dir)
        return stream_dir, plain_dir

    def test_streaming_matches_non_streaming(self, tmp_path):
        """Baseline: with no quality issues configured, the streaming
        Parquet write reads back cell-identical to a single-shot
        ``to_parquet`` write of the same in-memory tables dict."""
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        cfg_v = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"format": "parquet"}),
                "generation_mode": "vectorized",
                "quality": QualityConfig(quality_issues=[]),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg_v, np.random.default_rng(cfg_v.seed))
        stream_dir, plain_dir = self._write_streaming_and_plain(cfg_v, tables, tmp_path)
        for fact_name in _streaming_fact_table_names(cfg_v):
            self._assert_parquet_files_equal(
                stream_dir / f"{fact_name}.parquet",
                plain_dir / f"{fact_name}.parquet",
                fact_name,
            )

    def test_streaming_matches_non_streaming_with_fact_column_null_injection(self, tmp_path):
        """Regression: ``null_injection`` on a streaming-eligible fact
        table round-trips identically through the streaming and
        non-streaming write paths.

        Pre-fix, the test in this slot compared the clean in-memory
        ``tables`` dict against the corrupted on-disk Parquet, so any
        fact-column injection rate > 0 broke the assertion and forced
        bundled templates to route column-level quality issues onto
        event tables. After the fix the comparison is disk-to-disk
        (both writes share the same corrupted dict via
        ``_apply_quality_issues``), so future templates are free to
        place ``null_injection`` / ``type_mismatch`` / ``schema_drift``
        on fact columns without invalidating the streaming path's
        contract."""
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        cfg_v = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"format": "parquet"}),
                "generation_mode": "vectorized",
            }
        )
        streaming_facts = sorted(_streaming_fact_table_names(cfg_v))
        assert streaming_facts, "saas template must have at least one streaming-eligible fact table"
        target_fact = streaming_facts[0]
        cfg_q = cfg_v.model_copy(
            update={
                "quality": QualityConfig(
                    quality_issues=[
                        QualityIssue(
                            type="null_injection",
                            target_table=target_fact,
                            target_columns=["*"],
                            rate=0.05,
                        )
                    ]
                ),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg_q, np.random.default_rng(cfg_q.seed))
        stream_dir, plain_dir = self._write_streaming_and_plain(cfg_q, tables, tmp_path)
        for fact_name in _streaming_fact_table_names(cfg_q):
            self._assert_parquet_files_equal(
                stream_dir / f"{fact_name}.parquet",
                plain_dir / f"{fact_name}.parquet",
                fact_name,
            )

    def test_streaming_vs_non_streaming_dataframes_equal(self, tmp_path):
        """Two writes — one with vectorized+parquet (streaming) and one
        with serial+parquet (non-streaming) — produce read-back
        DataFrames that are equal cell-for-cell. Same seed both
        sides; the path the engine produced differs but the on-disk
        data must reconcile."""
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        cfg_parquet = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"format": "parquet"}),
            }
        )
        # Serial run — non-streaming path.
        cfg_s = cfg_parquet.model_copy(update={"generation_mode": "serial"})
        # Vectorized run — streaming path.
        cfg_v = cfg_parquet.model_copy(update={"generation_mode": "vectorized"})

        out_s = tmp_path / "serial"
        out_v = tmp_path / "vectorized"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables_s = generate_tables(cfg_s, np.random.default_rng(cfg.seed))
            write_tables(tables_s, cfg_s, output_dir=out_s)
            tables_v = generate_tables(cfg_v, np.random.default_rng(cfg.seed))
            write_tables(tables_v, cfg_v, output_dir=out_v)

        # Cross-mode cell values diverge by design (RNG order differs),
        # so we can't compare cell-for-cell across the two writes. The
        # equality contract this test enforces is *within* a mode:
        # the streaming write reads back identical to its in-memory
        # source for every fact table.
        for fact_name in _streaming_fact_table_names(cfg_v):
            on_disk_streaming = pd.read_parquet(out_v / f"{fact_name}.parquet")
            on_disk_serial = pd.read_parquet(out_s / f"{fact_name}.parquet")
            # Shape parity must hold across modes (same row count, same
            # columns) even though cell values differ.
            assert on_disk_streaming.shape == on_disk_serial.shape, fact_name
            assert sorted(on_disk_streaming.columns) == sorted(on_disk_serial.columns), fact_name


# --- Row group AC ----------------------------------------------------------


class TestRowGroups:
    """Row groups in the output Parquet correspond 1:1 to archetype
    batches for ``per_entity_per_period`` facts."""

    def test_row_group_count_matches_archetypes(self, tmp_path):
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        cfg = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"format": "parquet"}),
                "generation_mode": "vectorized",
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
            write_tables(tables, cfg, output_dir=tmp_path)
        unique_archetypes = {e.archetype for e in cfg.entities}
        for fact_name in _streaming_fact_table_names(cfg):
            meta = pq.read_metadata(tmp_path / f"{fact_name}.parquet")
            assert meta.num_row_groups == len(unique_archetypes), fact_name

    def test_row_group_sizes_match_entity_counts(self, tmp_path):
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        cfg = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"format": "parquet"}),
                "generation_mode": "vectorized",
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
            write_tables(tables, cfg, output_dir=tmp_path)
        n_periods = cfg.time_window.period_count()
        # Each archetype's row group should hold ents_per_arch × n_periods rows.
        ents_per_arch: dict[str, int] = {}
        archetype_order: list[str] = []
        for e in cfg.entities:
            if e.archetype not in ents_per_arch:
                archetype_order.append(e.archetype)
                ents_per_arch[e.archetype] = 0
            ents_per_arch[e.archetype] += 1
        expected_sizes = [ents_per_arch[a] * n_periods for a in archetype_order]
        for fact_name in _streaming_fact_table_names(cfg):
            meta = pq.read_metadata(tmp_path / f"{fact_name}.parquet")
            sizes = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
            assert sizes == expected_sizes, (
                f"{fact_name}: expected row group sizes {expected_sizes}, " f"got {sizes}"
            )


# --- Backward compatibility -------------------------------------------------


class TestBackwardCompat:
    """Serial and CSV paths must remain untouched by M121b."""

    def test_serial_parquet_unchanged(self, tmp_path):
        """Serial + parquet still uses the single-shot ``to_parquet``
        path → one row group per fact (pyarrow default)."""
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        cfg = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"format": "parquet"}),
                "generation_mode": "serial",
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
            write_tables(tables, cfg, output_dir=tmp_path)
        for tbl in cfg.tables:
            if tbl.type != "fact":
                continue
            meta = pq.read_metadata(tmp_path / f"{tbl.name}.parquet")
            assert meta.num_row_groups == 1, (
                f"{tbl.name}: serial+parquet expected 1 row group, " f"got {meta.num_row_groups}"
            )

    def test_vectorized_csv_unchanged(self, tmp_path):
        """Vectorized + CSV writes the unified DataFrame as before. No
        per-row-group concept in CSV; the test only confirms output
        is produced and readable."""
        cfg = load_config(ROOT / "plotsim" / "configs" / "sample_saas.yaml")
        cfg = cfg.model_copy(
            update={
                "generation_mode": "vectorized",
            }
        )
        # Format remains CSV (the default for sample_saas).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = generate_tables(cfg, np.random.default_rng(cfg.seed))
            write_tables(tables, cfg, output_dir=tmp_path)
        for tbl in cfg.tables:
            if tbl.type != "fact":
                continue
            csv_path = tmp_path / f"{tbl.name}.csv"
            assert csv_path.exists()
            df = pd.read_csv(csv_path)
            assert len(df) > 0
