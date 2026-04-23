"""Tests for plotsim.output — Mission 008 acceptance.

Covers:
  - write_tables end-to-end on both sample domains: every configured table
    produces a CSV, config.yaml and validation_report.txt are present, and
    the returned path points at the target directory.
  - CSV conventions: utf-8 encoding, no index column, float precision 4dp,
    NaN rendered as empty string, integer columns rendered without a ``.0``
    suffix, non-numeric fields quoted.
  - Column ordering: PK first, FKs next in config order, remaining config
    columns in declared order, and DataFrame-only columns (e.g. ``stage``
    added by assign_stages) appended last.
  - Config round-trip: write_config_copy emits valid YAML that load_config
    re-accepts; re-running generate_tables from the re-loaded config yields
    byte-identical CSVs (determinism under the same seed).
  - Validation report: header with counts + status, one line per issue with
    the check, table, message, and details; clean run is explicitly noted.
  - Directory handling: creates missing directories, overwrites pre-existing
    files, defaults to ``config.output.directory`` when no override is passed.
  - Write proceeds even when the validation report is invalid — the mission
    spec calls that out so users can inspect broken data.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotsim import dump_config, load_config
from plotsim.config import (
    PlotsimConfig,
    SurrogateKeyWarning,
)
from plotsim.output import (
    CONFIG_FILENAME,
    REPORT_FILENAME,
    write_config_copy,
    write_single_table,
    write_tables,
    write_validation_report,
)
from plotsim.tables import generate_tables
from plotsim.validation import (
    ValidationIssue,
    ValidationReport,
    validate_tables,
)


ROOT = Path(__file__).resolve().parent.parent
SAAS_YAML = ROOT / "plotsim" / "configs" / "sample_saas.yaml"
HR_YAML = ROOT / "plotsim" / "configs" / "sample_hr.yaml"


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture(scope="module")
def saas_cfg() -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return load_config(SAAS_YAML)


@pytest.fixture(scope="module")
def hr_cfg() -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return load_config(HR_YAML)


@pytest.fixture
def saas_bundle(saas_cfg):
    tables = generate_tables(saas_cfg, _rng(saas_cfg.seed))
    report = validate_tables(saas_cfg, tables)
    return saas_cfg, tables, report


@pytest.fixture
def hr_bundle(hr_cfg):
    tables = generate_tables(hr_cfg, _rng(hr_cfg.seed))
    report = validate_tables(hr_cfg, tables)
    return hr_cfg, tables, report


# --- End-to-end on sample domains --------------------------------------------


def test_write_tables_saas_produces_csv_per_table(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    out = write_tables(tables, cfg, report, output_dir=tmp_path)
    assert out == tmp_path
    for name in tables:
        assert (tmp_path / f"{name}.csv").exists(), f"{name}.csv missing"


def test_write_tables_hr_produces_csv_per_table(hr_bundle, tmp_path):
    cfg, tables, report = hr_bundle
    out = write_tables(tables, cfg, report, output_dir=tmp_path)
    assert out == tmp_path
    for name in tables:
        assert (tmp_path / f"{name}.csv").exists(), f"{name}.csv missing"


def test_write_tables_emits_config_and_report(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    write_tables(tables, cfg, report, output_dir=tmp_path)
    assert (tmp_path / CONFIG_FILENAME).exists()
    assert (tmp_path / REPORT_FILENAME).exists()


def test_write_tables_returns_the_output_path(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    nested = tmp_path / "deep" / "nested" / "dir"
    out = write_tables(tables, cfg, report, output_dir=nested)
    assert out == nested
    assert nested.is_dir()


def test_write_tables_defaults_to_config_output_directory(
    saas_bundle, tmp_path, monkeypatch,
):
    cfg, tables, report = saas_bundle
    # Swap cwd so the relative config.output.directory ("out/saas") lands in tmp.
    monkeypatch.chdir(tmp_path)
    out = write_tables(tables, cfg, report)
    expected = tmp_path / cfg.output.directory
    assert out.resolve() == expected.resolve()
    assert any(expected.glob("*.csv"))


def test_write_tables_overwrites_existing_files(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    # Pre-seed one of the target files with garbage.
    stale = tmp_path / "dim_company.csv"
    stale.write_text("this should be overwritten\n", encoding="utf-8")
    write_tables(tables, cfg, report, output_dir=tmp_path)
    content = stale.read_text(encoding="utf-8")
    assert "this should be overwritten" not in content
    assert content.startswith(("\"", "c", "company"))  # real CSV content


def test_write_tables_writes_even_when_report_invalid(saas_bundle, tmp_path):
    cfg, tables, _ = saas_bundle
    # Construct a fake-invalid report so we can verify CSVs are still written.
    bad = ValidationReport(issues=(
        ValidationIssue(
            check="fk_integrity",
            severity="error",
            table="fct_engagement",
            message="synthetic failure for test",
        ),
    ))
    write_tables(tables, cfg, bad, output_dir=tmp_path)
    for name in tables:
        assert (tmp_path / f"{name}.csv").exists()
    assert "INVALID" in (tmp_path / REPORT_FILENAME).read_text(encoding="utf-8")


# --- CSV convention tests ----------------------------------------------------


def test_csv_is_utf8(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    write_tables(tables, cfg, report, output_dir=tmp_path)
    path = tmp_path / "dim_company.csv"
    # Must decode cleanly as utf-8.
    path.read_text(encoding="utf-8")


def test_csv_has_no_dataframe_index_column(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    write_tables(tables, cfg, report, output_dir=tmp_path)
    loaded = pd.read_csv(tmp_path / "dim_date.csv")
    # First column should be the real first column of dim_date, not a dumped index.
    assert "Unnamed: 0" not in loaded.columns
    assert loaded.columns[0] == "date_key"


def test_csv_float_precision_four_decimal_places(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    write_tables(tables, cfg, report, output_dir=tmp_path)
    with (tmp_path / "fct_engagement.csv").open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        engagement_idx = header.index("engagement_score")
        for row in reader:
            cell = row[engagement_idx]
            if not cell:
                continue
            try:
                float(cell)
            except ValueError:
                pytest.fail(f"non-numeric cell in engagement_score: {cell!r}")
            # Strip sign + leading digits + decimal point → must be 4 fractional digits.
            if "." in cell:
                frac = cell.split(".", 1)[1]
                assert len(frac) == 4, f"expected 4dp float, got {cell!r}"


def test_csv_nan_renders_as_empty_string(saas_cfg, tmp_path):
    # Build a synthetic table with an explicit NaN in a metric column.
    rows = pd.DataFrame({
        "date_key": [202301, 202302],
        "company_id": ["c-001", "c-001"],
        "engagement_score": [0.5, np.nan],
        "feature_adoption": [0.3, 0.6],
    })
    write_single_table("fct_engagement", rows, tmp_path, config=saas_cfg)
    text = (tmp_path / "fct_engagement.csv").read_text(encoding="utf-8").splitlines()
    header = text[0].replace('"', "").split(",")
    eng_idx = header.index("engagement_score")
    row2 = text[2].split(",")
    # QUOTE_NONNUMERIC wraps the empty na_rep in quotes; both "" and bare
    # empty parse back to empty on read_csv. "nan"/"NaN" is the real failure.
    assert row2[eng_idx] in ("", '""'), (
        f"expected empty cell for NaN, got {row2[eng_idx]!r}"
    )
    # And round-tripping through read_csv must restore the null.
    reloaded = pd.read_csv(tmp_path / "fct_engagement.csv")
    assert pd.isna(reloaded.loc[1, "engagement_score"])


def test_csv_integer_column_has_no_dot_zero_suffix(tmp_path):
    # Build a minimal int-column scenario: poisson counts with one NaN.
    from plotsim.config import Column, Table
    tbl = Table(
        name="fct_counts",
        type="fact",
        grain="per_entity_per_period",
        columns=[
            Column(name="date_key", dtype="int", source="fk:dim_date.date_key"),
            Column(name="entity_id", dtype="id", source="fk:dim_company.company_id"),
            Column(name="tickets", dtype="int", source="metric:support_tickets"),
        ],
        primary_key=["date_key", "entity_id"],
    )

    class _StubConfig:
        tables = [tbl]

    df = pd.DataFrame({
        "date_key": [202301, 202302, 202303],
        "entity_id": ["c-001", "c-001", "c-001"],
        "tickets": [5, float("nan"), 12],  # promoted to float by NaN
    })
    write_single_table("fct_counts", df, tmp_path, config=_StubConfig())

    raw = (tmp_path / "fct_counts.csv").read_text(encoding="utf-8")
    assert "5.0" not in raw and "12.0" not in raw, (
        f"integer column leaked .0 suffix:\n{raw}"
    )
    assert ",5," in raw or raw.find(",5,") != -1 or ",5\n" in raw or '"5"' in raw


def test_csv_non_numeric_cells_are_quoted(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    write_tables(tables, cfg, report, output_dir=tmp_path)
    raw = (tmp_path / "dim_company.csv").read_text(encoding="utf-8")
    # At least one quoted string cell should be present (company names are strings).
    assert '"' in raw


# --- Column ordering ---------------------------------------------------------


def test_column_order_pk_then_fk_then_others(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    write_tables(tables, cfg, report, output_dir=tmp_path)
    loaded = pd.read_csv(tmp_path / "fct_engagement.csv")
    cols = list(loaded.columns)
    # PK comes first. fct_engagement's composite PK is [date_key, company_id];
    # both are FKs too (so they just lead the ordering).
    assert cols[0] in ("date_key", "company_id")
    # The metric columns must come after the PK/FK pair.
    metric_col_positions = [cols.index("engagement_score"), cols.index("feature_adoption")]
    max_pk_fk_idx = max(cols.index("date_key"), cols.index("company_id"))
    assert min(metric_col_positions) > max_pk_fk_idx


def test_column_order_stage_column_appended_last(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    # `stage` is added by assign_stages onto one fact table; it's not in config.
    for name, df in tables.items():
        if "stage" in df.columns:
            write_tables(tables, cfg, report, output_dir=tmp_path)
            loaded = pd.read_csv(tmp_path / f"{name}.csv")
            assert list(loaded.columns)[-1] == "stage", (
                f"expected 'stage' last in {name}, got {list(loaded.columns)}"
            )
            return
    pytest.skip("no table carries a stage column under the current config")


# --- Config round-trip + determinism -----------------------------------------


def test_config_yaml_is_reloadable(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    write_tables(tables, cfg, report, output_dir=tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        reloaded = load_config(tmp_path / CONFIG_FILENAME)
    # Same seed and same entities → same generation plan.
    assert reloaded.seed == cfg.seed
    assert [e.name for e in reloaded.entities] == [e.name for e in cfg.entities]
    assert [m.name for m in reloaded.metrics] == [m.name for m in cfg.metrics]


def test_roundtrip_generation_is_deterministic(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    write_tables(tables, cfg, report, output_dir=first_dir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        reloaded = load_config(first_dir / CONFIG_FILENAME)
    tables_again = generate_tables(reloaded, _rng(reloaded.seed))
    report_again = validate_tables(reloaded, tables_again)
    write_tables(tables_again, reloaded, report_again, output_dir=second_dir)

    for name in tables:
        a = (first_dir / f"{name}.csv").read_bytes()
        b = (second_dir / f"{name}.csv").read_bytes()
        assert a == b, f"non-deterministic CSV output for {name}"


# --- Validation report formatting --------------------------------------------


def test_validation_report_header_has_counts_and_status(saas_bundle, tmp_path):
    cfg, tables, report = saas_bundle
    write_tables(tables, cfg, report, output_dir=tmp_path)
    text = (tmp_path / REPORT_FILENAME).read_text(encoding="utf-8")
    assert "Plotsim Validation Report" in text
    assert "Errors:" in text and "Warnings:" in text
    assert ("Status: VALID" in text) or ("Status: INVALID" in text)


def test_validation_report_lists_each_issue(tmp_path):
    report = ValidationReport(issues=(
        ValidationIssue(
            check="fk_integrity",
            severity="error",
            table="fct_engagement",
            message="orphan FK value",
            details={"column": "company_id", "orphan_count": 3},
        ),
        ValidationIssue(
            check="null_policy",
            severity="warning",
            table="dim_company",
            message="nulls detected",
        ),
    ))
    write_validation_report(report, tmp_path)
    text = (tmp_path / REPORT_FILENAME).read_text(encoding="utf-8")
    assert "[ERROR]" in text
    assert "[WARN " in text
    assert "fk_integrity" in text
    assert "null_policy" in text
    assert "orphan_count: 3" in text


def test_validation_report_clean_note_when_no_issues(tmp_path):
    report = ValidationReport(issues=())
    write_validation_report(report, tmp_path)
    text = (tmp_path / REPORT_FILENAME).read_text(encoding="utf-8")
    assert "Status: VALID" in text
    assert "All checks passed" in text


# --- Directory handling ------------------------------------------------------


def test_write_single_table_creates_missing_directory(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = write_single_table("demo", df, target)
    assert out == target / "demo.csv"
    assert out.exists()


def test_write_config_copy_produces_valid_yaml(saas_cfg, tmp_path):
    path = write_config_copy(saas_cfg, tmp_path)
    assert path == tmp_path / CONFIG_FILENAME
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        reloaded = load_config(path)
    assert dump_config(reloaded) == dump_config(saas_cfg)


# --- FIX-08 / SF-2: path-traversal guard -------------------------------------


def test_base_dir_none_allows_any_path(saas_bundle, tmp_path):
    """FIX-08: with base_dir=None the CLI contract is preserved — arbitrary
    (absolute) paths are accepted."""
    cfg, tables, report = saas_bundle
    target = tmp_path / "freely_chosen" / "deep"
    out = write_tables(tables, cfg, report, output_dir=target, base_dir=None)
    assert out == target
    assert any(target.glob("*.csv"))


def test_base_dir_set_allows_subdirectory(saas_bundle, tmp_path):
    """FIX-08: a relative subpath under base_dir resolves cleanly."""
    cfg, tables, report = saas_bundle
    sandbox = tmp_path / "sandbox"
    out = write_tables(
        tables, cfg, report,
        output_dir="runs/first", base_dir=sandbox,
    )
    assert out == (sandbox / "runs" / "first").resolve()
    assert any(out.glob("*.csv"))


def test_base_dir_rejects_parent_traversal(saas_bundle, tmp_path):
    """FIX-08: ``..`` escaping base_dir raises ValueError, nothing is written."""
    cfg, tables, report = saas_bundle
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    sibling = tmp_path / "sibling"
    with pytest.raises(ValueError, match="escapes base_dir"):
        write_tables(
            tables, cfg, report,
            output_dir="../sibling", base_dir=sandbox,
        )
    # No CSVs leaked into the sibling directory.
    assert not sibling.exists() or not any(sibling.glob("*.csv"))


def test_base_dir_rejects_absolute_path_override(saas_bundle, tmp_path):
    """FIX-08: an absolute output_dir is rejected when base_dir is set."""
    cfg, tables, report = saas_bundle
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    rogue = tmp_path / "rogue"
    with pytest.raises(ValueError, match="absolute path"):
        write_tables(
            tables, cfg, report,
            output_dir=str(rogue), base_dir=sandbox,
        )


def test_base_dir_allows_nested_subdirectory(saas_bundle, tmp_path):
    """FIX-08: deep nested relative paths inside base_dir are permitted."""
    cfg, tables, report = saas_bundle
    sandbox = tmp_path / "sandbox"
    out = write_tables(
        tables, cfg, report,
        output_dir="2026/04/run-01", base_dir=sandbox,
    )
    assert out == (sandbox / "2026" / "04" / "run-01").resolve()
    assert any(out.glob("*.csv"))
