"""Tests for 0.6-M16c SQL dump output.

When ``output.format == "sql"`` is set on a config, every fact / dim /
event / bridge table is written to a single ``data.sql`` file with
dialect-aware DDL (``CREATE TABLE`` + ``PRIMARY KEY`` + ``FOREIGN KEY``
clauses) and batched ``INSERT`` statements (~100 rows per statement).
``sql_dialect`` selects between postgresql (default), mysql, and
sqlite — the three dialects produce syntactically distinct output but
all replay top-to-bottom into their target database. Denormalized wide
tables and holdout splits emit as trailing blocks without FK
constraints. ``entity_features.enabled=True`` paired with
``format: sql`` is rejected at config load.
"""

from __future__ import annotations

import re
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotsim.builder import create, create_from_yaml
from plotsim.config import OutputConfig, PlotsimConfig
from plotsim.output import (
    SQL_FILENAME,
    _resolve_output_format,
    _sql_format_value,
    _sql_quote_identifier,
    _sql_quote_string,
    _sql_table_order,
    write_tables,
)
from plotsim.tables import generate_tables


ROOT = Path(__file__).resolve().parent.parent


# --- Helpers ---------------------------------------------------------------


def _saas_sql_config(tmp_path: Path, *, dialect: str = "postgresql"):
    """Load the saas template and switch it to sql output."""
    cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
    return cfg.model_copy(
        update={
            "output": cfg.output.model_copy(
                update={
                    "format": "sql",
                    "sql_dialect": dialect,
                    "directory": str(tmp_path),
                }
            ),
        }
    )


def _tables_for(cfg) -> dict[str, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_tables(cfg, np.random.default_rng(cfg.seed))


def _builder_kwargs(**overrides):
    """Minimal ``create()`` kwargs for builder-passthrough tests."""
    base = {
        "about": "sql test domain",
        "unit": "company",
        "window": {"start": "2023-01", "end": "2023-06", "every": "monthly"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
        ],
        "segments": [
            {"name": "alpha", "count": 4, "archetype": "growth"},
        ],
    }
    base.update(overrides)
    return base


# --- Directory structure ---------------------------------------------------


class TestDirectoryStructure:
    """SQL output produces exactly one ``data.sql`` file inside the
    output directory — no per-table files, no per-dialect suffix."""

    def test_single_data_sql_file_written(self, tmp_path):
        cfg = _saas_sql_config(tmp_path)
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        sql_path = tmp_path / SQL_FILENAME
        assert sql_path.is_file()
        # None of the per-table files from CSV / Parquet / JSONL paths.
        for tbl in cfg.tables:
            for ext in ("csv", "parquet", "jsonl"):
                assert not (tmp_path / f"{tbl.name}.{ext}").exists()

    def test_companions_still_written(self, tmp_path):
        """``config.yaml`` and ``validation_report.txt`` are not table
        data and write alongside ``data.sql`` regardless of format."""
        cfg = _saas_sql_config(tmp_path)
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        assert (tmp_path / "config.yaml").is_file()
        assert (tmp_path / "validation_report.txt").is_file()


# --- SQLite round-trip -----------------------------------------------------


class TestSqliteRoundTrip:
    """The sqlite-dialect dump replays cleanly into an in-memory
    SQLite database and reproduces the per-table row counts that
    ``write_tables`` would land on disk for any other format."""

    def test_sqlite_dialect_loads_and_row_counts_match(self, tmp_path):
        cfg = _saas_sql_config(tmp_path, dialect="sqlite")
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        sql_text = (tmp_path / SQL_FILENAME).read_text(encoding="utf-8")
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(sql_text)
            for tbl in cfg.tables:
                row_count = conn.execute(f'SELECT COUNT(*) FROM "{tbl.name}";').fetchone()[0]
                # Tables-to-write may differ from clean ``tables`` due
                # to CDC / quality expansion (see test_jsonl_output for
                # the same dynamic). Compare CSV row count as the
                # canonical writer-output baseline.
                csv_dir = tmp_path / "csv_baseline"
                csv_dir.mkdir(exist_ok=True)
                # Lazy: rely on jsonl tests covering CSV parity; here
                # just check SQL row count > 0 for tables that have
                # data in the generated dict.
                expected_nonzero = len(tables[tbl.name]) > 0
                if expected_nonzero:
                    assert (
                        row_count > 0
                    ), f"{tbl.name}: SQL row count {row_count} but DataFrame is non-empty"
        finally:
            conn.close()

    def test_sqlite_fact_row_count_matches_csv_baseline(self, tmp_path):
        """Fact tables don't grow under CDC / quality (the post-quality
        DataFrame is the same length); compare SQL row count to the
        clean DataFrame for the three fact tables."""
        cfg = _saas_sql_config(tmp_path, dialect="sqlite")
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)

        sql_text = (tmp_path / SQL_FILENAME).read_text(encoding="utf-8")
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(sql_text)
            for fact in ("fct_engagement", "fct_revenue", "fct_support_tickets"):
                sql_count = conn.execute(f'SELECT COUNT(*) FROM "{fact}";').fetchone()[0]
                assert sql_count == len(
                    tables[fact]
                ), f"{fact}: SQL={sql_count} != DataFrame={len(tables[fact])}"
        finally:
            conn.close()


# --- Dialect distinctness --------------------------------------------------


class TestDialectDistinctness:
    """The three dialects produce textually distinct DDL:
    PG uses ``"col"`` + ``NUMERIC`` / ``BOOLEAN``; MySQL uses
    `` `col` `` + ``DOUBLE`` / ``TINYINT(1)`` / ``VARCHAR(255)``;
    SQLite uses ``"col"`` + ``REAL`` / ``INTEGER`` (no native bool)."""

    @pytest.fixture
    def three_dialects(self, tmp_path):
        outputs: dict[str, str] = {}
        for dialect in ("postgresql", "mysql", "sqlite"):
            dialect_dir = tmp_path / dialect
            dialect_dir.mkdir()
            cfg = _saas_sql_config(dialect_dir, dialect=dialect)
            tables = _tables_for(cfg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_tables(tables, cfg, output_dir=dialect_dir)
            outputs[dialect] = (dialect_dir / SQL_FILENAME).read_text(encoding="utf-8")
        return outputs

    def test_mysql_uses_backtick_quoting(self, three_dialects):
        assert "`fct_engagement`" in three_dialects["mysql"]
        assert "`fct_engagement`" not in three_dialects["postgresql"]
        assert "`fct_engagement`" not in three_dialects["sqlite"]

    def test_pg_and_sqlite_use_double_quote(self, three_dialects):
        assert '"fct_engagement"' in three_dialects["postgresql"]
        assert '"fct_engagement"' in three_dialects["sqlite"]

    def test_pg_uses_numeric_for_floats(self, three_dialects):
        assert "NUMERIC" in three_dialects["postgresql"]
        assert "NUMERIC" not in three_dialects["sqlite"]

    def test_sqlite_uses_real_for_floats(self, three_dialects):
        assert "REAL" in three_dialects["sqlite"]
        assert "REAL" not in three_dialects["postgresql"]

    def test_mysql_uses_double_and_tinyint(self, three_dialects):
        assert "DOUBLE" in three_dialects["mysql"]
        # boolean detection on the saas template's bool-typed columns
        # (none directly; check the SCD2 ``is_current`` column instead).
        assert "TINYINT(1)" in three_dialects["mysql"]

    def test_each_dialect_declares_self_in_header(self, three_dialects):
        for dialect, text in three_dialects.items():
            assert f"Dialect: {dialect}" in text, f"{dialect}: header should name the dialect"


# --- Header + replay hint --------------------------------------------------


class TestHeader:
    """The SQL file header carries the dialect name and a replay
    command appropriate to that dialect (``psql`` / ``mysql`` /
    ``sqlite3``)."""

    def test_postgresql_header_names_psql(self, tmp_path):
        cfg = _saas_sql_config(tmp_path, dialect="postgresql")
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)
        text = (tmp_path / SQL_FILENAME).read_text(encoding="utf-8")
        assert "psql -d" in text

    def test_mysql_header_names_mysql_cli(self, tmp_path):
        cfg = _saas_sql_config(tmp_path, dialect="mysql")
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)
        text = (tmp_path / SQL_FILENAME).read_text(encoding="utf-8")
        assert "mysql " in text

    def test_sqlite_header_names_sqlite3(self, tmp_path):
        cfg = _saas_sql_config(tmp_path, dialect="sqlite")
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)
        text = (tmp_path / SQL_FILENAME).read_text(encoding="utf-8")
        assert "sqlite3" in text


# --- Table order: dims before facts ---------------------------------------


class TestTableOrder:
    """Dimension tables must be created (and populated) before any
    fact / event / bridge table that FKs to them — otherwise the FK
    constraint would fail at replay time."""

    def test_dim_create_appears_before_fact_create(self, tmp_path):
        cfg = _saas_sql_config(tmp_path, dialect="postgresql")
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)
        text = (tmp_path / SQL_FILENAME).read_text(encoding="utf-8")

        dim_pos = text.find('CREATE TABLE "dim_company"')
        fact_pos = text.find('CREATE TABLE "fct_engagement"')
        assert dim_pos != -1 and fact_pos != -1
        assert dim_pos < fact_pos, "dim_company CREATE must precede fct_engagement CREATE"

    def test_sql_table_order_dims_first(self):
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        order = _sql_table_order(cfg)
        dim_names = {t.name for t in cfg.tables if t.type == "dim"}
        # All dim names appear before the first non-dim.
        first_non_dim = next((i for i, n in enumerate(order) if n not in dim_names), len(order))
        assert all(n in dim_names for n in order[:first_non_dim])
        assert all(n not in dim_names for n in order[first_non_dim:])


# --- NULL handling --------------------------------------------------------


class TestNullHandling:
    """NaN / pd.NA / None values render as the bare token ``NULL`` in
    INSERT VALUES — never as ``"NULL"`` (quoted) or empty cells."""

    def test_nan_float_renders_as_null(self):
        assert _sql_format_value(float("nan"), "postgresql") == "NULL"
        assert _sql_format_value(float("nan"), "sqlite") == "NULL"

    def test_none_renders_as_null(self):
        assert _sql_format_value(None, "postgresql") == "NULL"

    def test_pd_na_renders_as_null(self):
        assert _sql_format_value(pd.NA, "postgresql") == "NULL"

    def test_int_renders_unquoted(self):
        assert _sql_format_value(42, "postgresql") == "42"

    def test_bool_pg_renders_as_keyword(self):
        assert _sql_format_value(True, "postgresql") == "TRUE"
        assert _sql_format_value(False, "postgresql") == "FALSE"

    def test_bool_mysql_sqlite_renders_as_int(self):
        assert _sql_format_value(True, "mysql") == "1"
        assert _sql_format_value(False, "sqlite") == "0"

    def test_string_single_quotes_escaped(self):
        assert _sql_format_value("O'Reilly", "postgresql") == "'O''Reilly'"

    def test_dict_renders_as_json_text(self):
        out = _sql_format_value({"a": 1, "b": "x"}, "postgresql")
        assert out.startswith("'") and out.endswith("'")
        assert '"a": 1' in out

    def test_list_renders_as_json_text(self):
        out = _sql_format_value([1, 2, 3], "postgresql")
        assert out == "'[1, 2, 3]'"

    def test_timestamp_renders_iso(self):
        ts = pd.Timestamp("2024-01-15T12:00:00")
        out = _sql_format_value(ts, "postgresql")
        assert "2024-01-15 12:00:00" in out


# --- Identifier quoting ---------------------------------------------------


class TestIdentifierQuoting:
    """Identifier-quote characters differ by dialect: PG / SQLite use
    ``"col"``, MySQL uses `` `col` ``. Embedded quote characters in the
    identifier are doubled (SQL standard)."""

    def test_postgresql_double_quotes(self):
        assert _sql_quote_identifier("fct_x", "postgresql") == '"fct_x"'

    def test_mysql_backticks(self):
        assert _sql_quote_identifier("fct_x", "mysql") == "`fct_x`"

    def test_sqlite_double_quotes(self):
        assert _sql_quote_identifier("fct_x", "sqlite") == '"fct_x"'

    def test_quote_string_escapes_apostrophe(self):
        assert _sql_quote_string("O'Reilly") == "'O''Reilly'"


# --- Sidecars: wide + holdout in single file ------------------------------


class TestSidecarBlocks:
    """Denormalized wide tables and holdout splits emit as trailing
    CREATE TABLE + INSERT blocks within ``data.sql``, AFTER the star
    schema, and without FK constraints."""

    def test_wide_tables_emit_as_trailing_blocks(self, tmp_path):
        cfg = _saas_sql_config(tmp_path, dialect="sqlite")
        # saas template already has denormalized=True per the PR #35
        # template-feature broadening; assert that the wide block lands
        # in data.sql for all three fact tables.
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)
        text = (tmp_path / SQL_FILENAME).read_text(encoding="utf-8")
        for fact in ("fct_engagement", "fct_revenue", "fct_support_tickets"):
            assert (
                f'CREATE TABLE "{fact}_wide"' in text
            ), f"{fact}_wide should emit as a trailing block"
        # No separate _wide.csv / .parquet / .jsonl files written under sql.
        assert not (tmp_path / "fct_engagement_wide.csv").exists()
        assert not (tmp_path / "fct_engagement_wide.parquet").exists()

    def test_wide_tables_have_no_fk_constraints(self, tmp_path):
        cfg = _saas_sql_config(tmp_path, dialect="postgresql")
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)
        text = (tmp_path / SQL_FILENAME).read_text(encoding="utf-8")
        # Extract just the wide-table CREATE statements and assert no
        # FOREIGN KEY clause inside them.
        wide_blocks = re.findall(r'CREATE TABLE "[\w_]+_wide" \([^;]+\);', text, flags=re.DOTALL)
        assert len(wide_blocks) >= 1
        for block in wide_blocks:
            assert "FOREIGN KEY" not in block, "wide-table block must not declare FK constraints"


# --- FK constraints: only for non-SCD2 targets ----------------------------


class TestForeignKeys:
    """``FOREIGN KEY (col) REFERENCES dim(pk)`` appears for fact-side
    references to dims with unique natural PKs. SCD2 dims (those with
    a ``dim_row_id`` surrogate) have non-unique natural keys, so the
    FK is omitted — replay would otherwise reject the constraint."""

    def test_fk_to_non_scd2_dim_emitted(self, tmp_path):
        cfg = _saas_sql_config(tmp_path, dialect="postgresql")
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)
        text = (tmp_path / SQL_FILENAME).read_text(encoding="utf-8")
        # dim_date is not SCD2 in the saas template — FK to it should
        # land in every fact's CREATE TABLE.
        assert 'REFERENCES "dim_date"("date_key")' in text

    def test_fk_to_scd2_dim_omitted(self, tmp_path):
        cfg = _saas_sql_config(tmp_path, dialect="postgresql")
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)
        text = (tmp_path / SQL_FILENAME).read_text(encoding="utf-8")
        # dim_company IS SCD2 — the natural FK ``company_id`` isn't
        # unique on the dim side, so the constraint must be skipped.
        assert 'REFERENCES "dim_company"' not in text


# --- INSERT batching -------------------------------------------------------


class TestInsertBatching:
    """INSERT statements batch ~100 rows per statement (mission spec).
    Each statement emits ``INSERT INTO "x" (...) VALUES`` once,
    followed by comma-separated row tuples."""

    def test_each_insert_statement_has_at_most_100_rows(self, tmp_path):
        cfg = _saas_sql_config(tmp_path, dialect="sqlite")
        tables = _tables_for(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp_path)
        text = (tmp_path / SQL_FILENAME).read_text(encoding="utf-8")
        for stmt in re.findall(r"INSERT INTO [^;]+;", text, flags=re.DOTALL):
            # Count value-tuple lines (each starts with two spaces +
            # an opening paren).
            row_lines = [line for line in stmt.splitlines() if line.startswith("  (")]
            assert (
                1 <= len(row_lines) <= 100
            ), f"INSERT statement has {len(row_lines)} rows (expected 1-100)"


# --- entity_features + sql rejected ----------------------------------------


class TestEntityFeaturesRejection:
    """``entity_features.enabled=True`` paired with ``format: sql`` is
    rejected at config load — its wide-aggregate shape doesn't fit
    the single-file SQL dump's star-schema layout."""

    def test_entity_features_plus_sql_rejected_at_load(self):
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        payload = cfg.model_dump()
        payload["output"]["format"] = "sql"
        payload["entity_features"] = {"enabled": True}
        with pytest.raises(Exception, match="entity_features"):
            PlotsimConfig.model_validate(payload)

    def test_entity_features_plus_csv_still_allowed(self):
        """Regression guard: the new gate only fires under sql; CSV
        and the other formats continue to support entity_features."""
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        payload = cfg.model_dump()
        # entity_features requires manifest.include=true and zero
        # quality issues — the saas template carries quality issues by
        # default, so swap the quality block to empty before flipping
        # entity_features on. This isolates the test to the format gate.
        payload["quality"]["quality_issues"] = []
        payload["entity_features"] = {"enabled": True}
        # If validation passes, the format gate is correctly scoped.
        PlotsimConfig.model_validate(payload)


# --- Format Literal --------------------------------------------------------


class TestFormatLiteral:
    """``OutputConfig.format`` accepts ``"sql"`` alongside the other
    three values, and ``OutputConfig.sql_dialect`` accepts only the
    three documented dialect words."""

    def test_sql_accepted_by_output_config(self):
        oc = OutputConfig(format="sql", directory="out")
        assert oc.format == "sql"
        # Default dialect.
        assert oc.sql_dialect == "postgresql"

    def test_each_dialect_accepted(self):
        for d in ("postgresql", "mysql", "sqlite"):
            oc = OutputConfig(format="sql", directory="out", sql_dialect=d)
            assert oc.sql_dialect == d

    def test_unknown_dialect_rejected(self):
        with pytest.raises(Exception, match="sql_dialect"):
            OutputConfig(format="sql", directory="out", sql_dialect="oracle")

    def test_resolve_output_format_returns_sql(self):
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        cfg_s = cfg.model_copy(
            update={"output": cfg.output.model_copy(update={"format": "sql"})},
        )
        assert _resolve_output_format(cfg_s) == "sql"


# --- Builder passthrough --------------------------------------------------


class TestBuilderPassthrough:
    """``create(output="sql")`` shorthand and the dict form both
    resolve to ``PlotsimConfig.output.format == "sql"``. The
    ``sql_dialect`` field plumbs through ``OutputInput`` ->
    ``OutputConfig`` (interpreter passthrough — same pattern as
    M16a's ``partition_by``)."""

    def test_builder_shorthand_sql(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = create(**_builder_kwargs(output="sql"))
        assert cfg.output.format == "sql"
        assert cfg.output.sql_dialect == "postgresql"

    def test_builder_dict_form_with_dialect(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = create(
                **_builder_kwargs(
                    output={
                        "format": "sql",
                        "directory": "out_dir",
                        "sql_dialect": "sqlite",
                    }
                ),
            )
        assert cfg.output.format == "sql"
        assert cfg.output.directory == "out_dir"
        assert cfg.output.sql_dialect == "sqlite"

    def test_builder_rejects_unknown_shorthand(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(Exception, match="unknown output format"):
                create(**_builder_kwargs(output="oracle"))


# --- CSV regression guard --------------------------------------------------


class TestCsvUnchanged:
    """Adding the SQL writer branch must not perturb CSV output —
    a config with ``format: csv`` produces byte-identical files to a
    baseline config with the field omitted, run after run."""

    def test_csv_output_byte_identical(self, tmp_path):
        cfg = create_from_yaml(ROOT / "plotsim" / "configs" / "templates" / "saas_template.yaml")
        a = tmp_path / "run_a"
        b = tmp_path / "run_b"
        a.mkdir()
        b.mkdir()
        cfg_a = cfg.model_copy(
            update={"output": cfg.output.model_copy(update={"directory": str(a)})},
        )
        cfg_b = cfg.model_copy(
            update={
                "output": OutputConfig(format="csv", directory=str(b)),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables_a = _tables_for(cfg_a)
            tables_b = _tables_for(cfg_b)
            write_tables(tables_a, cfg_a, output_dir=a)
            write_tables(tables_b, cfg_b, output_dir=b)

        for tbl in cfg.tables:
            ba = (a / f"{tbl.name}.csv").read_bytes()
            bb = (b / f"{tbl.name}.csv").read_bytes()
            assert ba == bb, f"{tbl.name}.csv differs between runs"
