"""0.6-M14b — Log-file writer tests.

Covers the full mission's acceptance criteria:

  * ``Table.log_format = None`` (default) — no ``.log`` file written
    for that event table; pre-M14b output byte-identical.
  * ``Table.log_format = "<template>"`` — exactly one ``.log`` file
    written for that event table; one row of the event table → one
    line in the log; placeholders resolve to column values.
  * ``Table.log_filename`` overrides the default ``<name>.log``
    filename; defaults to ``<table_name>.log`` when omitted.
  * Format-string typo (placeholder names a column that doesn't
    exist) raises ``ValueError`` naming the table and the available
    columns — fail-loud, not silent garbage.
  * Event table CSV/Parquet still emitted alongside the log.
  * Determinism: same input + same template → byte-identical log.
  * Validators: ``log_format`` / ``log_filename`` rejected on non-
    event tables (mirror M9c ``cdc`` validator pattern).
  * ``log_filename`` set without ``log_format`` rejected (filename
    alone produces no output).
  * Path sandbox: ``log_filename`` containing ``..`` or absolute
    path rejected at write time.
  * Builder ``EventInput.log_format`` plumbs through to
    ``Table.log_format``.
  * Schema export contains the two new fields on Table.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from plotsim import (
    PlotsimConfig,
    SurrogateKeyWarning,
    create,
    generate_tables_with_state,
    load_config,
    write_tables,
)
from plotsim.log_writer import write_event_logs


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "plotsim" / "configs"
SAAS_YAML = CONFIGS_DIR / "sample_saas.yaml"


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def saas_cfg() -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return load_config(SAAS_YAML)


def _saas_yaml_dict() -> dict:
    with SAAS_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_cfg_from_dict(d: dict) -> PlotsimConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(**d)


def _saas_with_log_format(
    *, evt_login_format: str | None = None, evt_login_filename: str | None = None
) -> PlotsimConfig:
    """SAAS sample with log_format set on `evt_login` (and optionally
    a custom filename). Other tables unchanged."""
    d = _saas_yaml_dict()
    for tbl in d["tables"]:
        if tbl["name"] == "evt_login":
            if evt_login_format is not None:
                tbl["log_format"] = evt_login_format
            if evt_login_filename is not None:
                tbl["log_filename"] = evt_login_filename
    return _build_cfg_from_dict(d)


# --- 1. Table model field ---------------------------------------------------


def test_table_log_format_default_none(saas_cfg):
    for tbl in saas_cfg.tables:
        assert tbl.log_format is None
        assert tbl.log_filename is None


def test_table_log_format_round_trip_yaml():
    cfg = _saas_with_log_format(evt_login_format="{event_ts} login {user_id}")
    evt = next(t for t in cfg.tables if t.name == "evt_login")
    assert evt.log_format == "{event_ts} login {user_id}"
    assert evt.log_filename is None


# --- 2. Validators ----------------------------------------------------------


def test_log_format_rejected_on_fact_table():
    d = _saas_yaml_dict()
    fct = next(t for t in d["tables"] if t["name"] == "fct_revenue")
    fct["log_format"] = "{date_key} {company_id} {mrr}"
    with pytest.raises(ValidationError, match="log_format/log_filename"):
        _build_cfg_from_dict(d)


def test_log_format_rejected_on_dim_table():
    d = _saas_yaml_dict()
    dim = next(t for t in d["tables"] if t["name"] == "dim_company")
    dim["log_format"] = "{company_id} {company_name}"
    with pytest.raises(ValidationError, match="log_format/log_filename"):
        _build_cfg_from_dict(d)


def test_log_filename_rejected_without_log_format():
    d = _saas_yaml_dict()
    evt = next(t for t in d["tables"] if t["name"] == "evt_login")
    evt["log_filename"] = "events.log"
    # log_format omitted on purpose → meaningless filename → reject.
    with pytest.raises(ValidationError, match="log_format is None"):
        _build_cfg_from_dict(d)


def test_log_filename_with_log_format_accepted():
    cfg = _saas_with_log_format(
        evt_login_format="{event_ts} {user_id}",
        evt_login_filename="logins.log",
    )
    evt = next(t for t in cfg.tables if t.name == "evt_login")
    assert evt.log_filename == "logins.log"


# --- 3. Pure write_event_logs function --------------------------------------


def test_write_event_logs_no_op_when_no_log_format(tmp_path, saas_cfg):
    rng = np.random.default_rng(saas_cfg.seed)
    tables, _ = generate_tables_with_state(saas_cfg, rng)
    written = write_event_logs(tables, saas_cfg, tmp_path)
    assert written == []
    assert list(tmp_path.glob("*.log")) == []


def test_write_event_logs_one_file_per_configured_event(tmp_path):
    cfg = _saas_with_log_format(evt_login_format="{event_ts} {user_id}")
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    written = write_event_logs(tables, cfg, tmp_path)
    assert len(written) == 1
    assert written[0].name == "evt_login.log"
    assert written[0].exists()


def test_log_line_count_matches_event_row_count(tmp_path):
    cfg = _saas_with_log_format(evt_login_format="{event_ts} {user_id}")
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    write_event_logs(tables, cfg, tmp_path)
    log_text = (tmp_path / "evt_login.log").read_text(encoding="utf-8")
    log_lines = log_text.splitlines()
    assert len(log_lines) == len(tables["evt_login"]), (
        f"log line count ({len(log_lines)}) must equal event-table "
        f"row count ({len(tables['evt_login'])})"
    )


def test_log_placeholders_resolve_to_column_values(tmp_path):
    cfg = _saas_with_log_format(evt_login_format="ts={event_ts} user={user_id} co={company_id}")
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    write_event_logs(tables, cfg, tmp_path)
    log_lines = (tmp_path / "evt_login.log").read_text(encoding="utf-8").splitlines()
    df = tables["evt_login"]
    # Spot-check the first row: every placeholder should resolve to
    # the actual value in the source DataFrame.
    first_row = df.iloc[0]
    expected = (
        f"ts={first_row['event_ts']} user={first_row['user_id']} co={first_row['company_id']}"
    )
    assert log_lines[0] == expected


def test_log_unknown_placeholder_raises_with_helpful_message(tmp_path):
    cfg = _saas_with_log_format(evt_login_format="{nonexistent_column}")
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    with pytest.raises(ValueError, match="nonexistent_column"):
        write_event_logs(tables, cfg, tmp_path)


def test_log_filename_override_used(tmp_path):
    cfg = _saas_with_log_format(
        evt_login_format="{event_ts} {user_id}",
        evt_login_filename="custom_logins.txt",
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    write_event_logs(tables, cfg, tmp_path)
    assert (tmp_path / "custom_logins.txt").exists()
    assert not (tmp_path / "evt_login.log").exists()


def test_log_filename_traversal_rejected(tmp_path):
    cfg = _saas_with_log_format(
        evt_login_format="{event_ts}",
        evt_login_filename="../escape.log",
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    with pytest.raises(ValueError, match="resolves outside output_dir"):
        write_event_logs(tables, cfg, tmp_path)


# --- 4. write_tables wiring -------------------------------------------------


def test_write_tables_off_produces_no_log_files(tmp_path, saas_cfg):
    rng = np.random.default_rng(saas_cfg.seed)
    tables, _ = generate_tables_with_state(saas_cfg, rng)
    write_tables(tables, saas_cfg, output_dir=tmp_path)
    assert (
        list(tmp_path.glob("*.log")) == []
    ), "no event tables have log_format set → zero .log files expected"


def test_write_tables_emits_log_alongside_event_csv(tmp_path):
    cfg = _saas_with_log_format(evt_login_format="{event_ts} {user_id}")
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    write_tables(tables, cfg, output_dir=tmp_path)
    # Both CSV and log present — log writer is additive.
    assert (tmp_path / "evt_login.csv").exists()
    assert (tmp_path / "evt_login.log").exists()
    # Other event table without log_format → CSV only, no log.
    assert (tmp_path / "evt_churn.csv").exists()
    assert not (tmp_path / "evt_churn.log").exists()


def test_write_tables_log_alongside_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    cfg = _saas_with_log_format(evt_login_format="{event_ts} {user_id}")
    cfg = cfg.model_copy(
        update={
            "output": cfg.output.model_copy(
                update={"format": "parquet", "directory": str(tmp_path)}
            )
        }
    )
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    write_tables(tables, cfg, output_dir=tmp_path)
    assert (tmp_path / "evt_login.parquet").exists()
    assert (tmp_path / "evt_login.log").exists()


# --- 5. Determinism ---------------------------------------------------------


def test_log_file_deterministic(tmp_path):
    cfg = _saas_with_log_format(evt_login_format="{event_ts} {user_id} {company_id}")
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for target in (a, b):
        rng = np.random.default_rng(cfg.seed)
        tables, _ = generate_tables_with_state(cfg, rng)
        write_tables(tables, cfg, output_dir=target)
    assert (a / "evt_login.log").read_bytes() == (
        b / "evt_login.log"
    ).read_bytes(), "log file must be byte-identical across runs"


# --- 6. Builder passthrough -------------------------------------------------


def _builder_kwargs(**overrides):
    """Bare-minimum ``create()`` kwargs with one event for the
    log-format passthrough check."""
    base = {
        "about": "test domain",
        "unit": "company",
        "window": {"start": "2023-01", "end": "2023-12", "every": "monthly"},
        "metrics": [
            {"name": "engagement", "type": "score", "polarity": "positive"},
        ],
        "segments": [
            {"name": "alpha", "count": 5, "archetype": "growth"},
        ],
        "events": [
            {
                "name": "evt_signin",
                "trigger": "proportional",
                "driver": "engagement",
                "scale": 2.0,
                "log_format": "{event_id} signin {event_ts}",
                "columns": [
                    {"name": "event_id", "type": "id"},
                    {"name": "date_key", "type": "ref.dim_date"},
                    {"name": "event_ts", "type": "timestamp"},
                ],
            },
        ],
    }
    base.update(overrides)
    return base


def test_builder_log_format_passthrough():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(**_builder_kwargs())
    evt = next(t for t in cfg.tables if t.name == "evt_signin")
    assert evt.log_format == "{event_id} signin {event_ts}"
    assert evt.log_filename is None


def test_builder_log_filename_passthrough():
    kwargs = _builder_kwargs()
    kwargs["events"][0]["log_filename"] = "signins.log"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(**kwargs)
    evt = next(t for t in cfg.tables if t.name == "evt_signin")
    assert evt.log_filename == "signins.log"


# --- 7. Schema export -------------------------------------------------------


def test_schema_json_includes_log_format_fields():
    schema_path = ROOT / "plotsim-schema.json"
    if not schema_path.exists():
        pytest.skip("plotsim-schema.json not generated")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    table_schema = schema["$defs"]["Table"]
    assert "log_format" in table_schema["properties"]
    assert "log_filename" in table_schema["properties"]


# --- 8. Empty event table edge case -----------------------------------------


def test_empty_event_table_produces_empty_log(tmp_path):
    """An event table that generates zero rows (e.g. a threshold
    that nothing crosses) should still produce a log file (so a
    downstream consumer doesn't have to special-case its absence)
    — just an empty file."""
    cfg = _saas_with_log_format(evt_login_format="{event_ts}")
    rng = np.random.default_rng(cfg.seed)
    tables, _ = generate_tables_with_state(cfg, rng)
    # Force evt_login to be empty by clearing the DataFrame post-gen.
    tables["evt_login"] = tables["evt_login"].iloc[0:0]
    written = write_event_logs(tables, cfg, tmp_path)
    assert len(written) == 1
    log_path = written[0]
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8") == ""
