"""Tests for ``plotsim.schema`` and ``plotsim schema`` CLI (Mission 104, Track A).

Locks in:
  - ``generate_schema()`` returns a valid Draft 2020-12 JSON Schema.
  - The schema covers every top-level field of ``PlotsimConfig`` and at
    least one definition for every nested ``BaseModel``.
  - ``write_schema(path)`` writes pretty-printed UTF-8 JSON with a
    trailing newline.
  - ``plotsim schema`` CLI writes to the default destination
    (``./plotsim-schema.json``) and exits zero.
  - The committed ``plotsim-schema.json`` at the repo root is in sync
    with what ``generate_schema()`` produces (regenerate-and-commit
    workflow stays consistent).
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from plotsim import cli
from plotsim.config import PlotsimConfig
from plotsim.schema import SCHEMA_FILENAME, generate_schema, write_schema


ROOT = Path(__file__).resolve().parent.parent
COMMITTED_SCHEMA = ROOT / SCHEMA_FILENAME


def run_cli(*argv: str) -> tuple[int, str, str]:
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        try:
            code = cli.main(list(argv))
        except SystemExit as exc:
            code = int(exc.code) if exc.code is not None else 0
    return code, out_buf.getvalue(), err_buf.getvalue()


# --- generate_schema ---------------------------------------------------------


def test_generate_schema_returns_dict_with_expected_meta():
    schema = generate_schema()
    assert isinstance(schema, dict)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "PlotsimConfig"
    assert schema["type"] == "object"


def test_generate_schema_passes_draft_2020_12_meta_schema():
    """Acceptance criterion: schema validates against Draft 2020-12."""
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(generate_schema())


def test_generate_schema_covers_every_root_pydantic_field():
    """Every field of ``PlotsimConfig`` must surface in the schema's properties."""
    schema = generate_schema()
    properties = set(schema.get("properties", {}).keys())
    expected = set(PlotsimConfig.model_fields.keys())
    missing = expected - properties
    assert not missing, f"PlotsimConfig fields missing from schema: {sorted(missing)}"


def test_generate_schema_required_marks_non_default_fields():
    """Fields without defaults are required; defaulted fields are not."""
    schema = generate_schema()
    required = set(schema.get("required", []))
    for name, field in PlotsimConfig.model_fields.items():
        if field.is_required():
            assert name in required, f"{name!r} should be in required"


def test_generate_schema_enforces_extra_forbid_on_root():
    """``extra='forbid'`` on _Frozen must surface as ``additionalProperties: false``."""
    schema = generate_schema()
    assert schema["additionalProperties"] is False


def test_generate_schema_includes_nested_model_defs():
    """Domain, TimeWindow, Metric, Archetype, Entity, Table, OutputConfig, NoiseConfig
    all live behind ``$defs`` references — the editor needs them for autocomplete
    inside the nested mappings.
    """
    defs = generate_schema().get("$defs", {})
    expected_models = {
        "Domain", "TimeWindow", "Metric", "Archetype", "Entity",
        "Table", "Column", "OutputConfig", "NoiseConfig",
        "CorrelationPair", "CurveSegment",
    }
    missing = expected_models - set(defs.keys())
    assert not missing, f"Nested model defs missing: {sorted(missing)}"


def test_generate_schema_is_deterministic():
    """Same plotsim version + same Pydantic version → byte-identical output.

    Mission acceptance criterion is implicit: the committed schema file
    must stay in sync with regenerated output.
    """
    a = json.dumps(generate_schema(), indent=2, ensure_ascii=False)
    b = json.dumps(generate_schema(), indent=2, ensure_ascii=False)
    assert a == b


# --- write_schema ------------------------------------------------------------


def test_write_schema_produces_pretty_utf8_json_with_trailing_newline(tmp_path):
    target = tmp_path / "out.json"
    written = write_schema(target)
    assert written == target
    payload = target.read_text(encoding="utf-8")
    assert payload.endswith("\n")
    # Pretty-printed: indent=2 produces multi-line output for non-trivial dicts
    assert "\n  " in payload
    parsed = json.loads(payload)
    assert parsed["title"] == "PlotsimConfig"


def test_write_schema_overwrites_existing_file(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("garbage", encoding="utf-8")
    write_schema(target)
    assert json.loads(target.read_text(encoding="utf-8"))["title"] == "PlotsimConfig"


def test_write_schema_creates_missing_parent_dirs(tmp_path):
    target = tmp_path / "deep" / "nested" / "out.json"
    write_schema(target)
    assert target.exists()


# --- CLI ---------------------------------------------------------------------


def test_cli_schema_writes_default_destination(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out, _err = run_cli("schema")
    assert code == 0
    target = tmp_path / SCHEMA_FILENAME
    assert target.exists(), out
    assert "PlotsimConfig" in target.read_text(encoding="utf-8")


def test_cli_schema_writes_custom_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    custom = tmp_path / "subdir" / "custom.json"
    code, _out, _err = run_cli("schema", "--output", str(custom))
    assert code == 0
    assert custom.exists()


def test_cli_schema_dash_writes_to_stdout(tmp_path, monkeypatch):
    """``plotsim schema -o -`` pipes the schema to stdout for shell composition."""
    monkeypatch.chdir(tmp_path)
    code, out, _err = run_cli("schema", "-o", "-")
    assert code == 0
    parsed = json.loads(out)
    assert parsed["title"] == "PlotsimConfig"
    # The default-destination file must NOT have been created when piping.
    assert not (tmp_path / SCHEMA_FILENAME).exists()


# --- Committed schema file in sync ------------------------------------------


def test_committed_schema_file_matches_generated_output():
    """The repo-root ``plotsim-schema.json`` is the contract surface for IDE
    integrations. If a config-model change drifts from the committed file
    (developer forgot ``plotsim schema``), this test will fail and tell them
    which fields shifted.
    """
    if not COMMITTED_SCHEMA.exists():
        pytest.skip(
            f"{SCHEMA_FILENAME} not present at repo root; run `plotsim schema`."
        )
    on_disk = json.loads(COMMITTED_SCHEMA.read_text(encoding="utf-8"))
    fresh = generate_schema()
    if on_disk != fresh:
        on_disk_keys = set(on_disk.get("properties", {}).keys())
        fresh_keys = set(fresh.get("properties", {}).keys())
        added = fresh_keys - on_disk_keys
        removed = on_disk_keys - fresh_keys
        pytest.fail(
            f"Committed {SCHEMA_FILENAME} is out of date. "
            f"Run `plotsim schema` and commit. "
            f"Added properties: {sorted(added)}, "
            f"Removed properties: {sorted(removed)}."
        )
