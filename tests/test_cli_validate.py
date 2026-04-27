"""Tests for ``plotsim validate`` / ``plotsim validate --config-only`` (Mission 104, Track B).

The acceptance contract:
  - both invocations load and validate the config without running the engine
  - runtime stays under 500 ms on every bundled template
  - all four classes of load-time validators fire on bad input:
      * extra="forbid" rejection of unknown keys
      * PSD correlation matrix check
      * distribution parameter range validation (Pydantic Field bounds)
      * FK reference integrity (column source naming an unknown table)
  - no output files are created on disk
  - invalid configs produce identical error messages with or without the flag
"""
from __future__ import annotations

import io
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
import yaml

from plotsim import cli


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "plotsim" / "configs"
ALL_TEMPLATES = ("saas", "hr", "ecommerce", "education", "healthcare")


def run_cli(*argv: str) -> tuple[int, str, str]:
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        try:
            code = cli.main(list(argv))
        except SystemExit as exc:
            code = int(exc.code) if exc.code is not None else 0
    return code, out_buf.getvalue(), err_buf.getvalue()


def _template_path(name: str) -> Path:
    return CONFIGS_DIR / f"sample_{name}.yaml"


def _load_template_dict(name: str) -> dict:
    return yaml.safe_load(_template_path(name).read_text(encoding="utf-8"))


def _write_yaml(target: Path, payload: dict) -> Path:
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


# --- Happy path on every bundled template -----------------------------------


@pytest.mark.parametrize("name", ALL_TEMPLATES)
def test_validate_config_only_accepts_every_bundled_template(name):
    code, out, _err = run_cli("validate", "--config-only", str(_template_path(name)))
    assert code == 0, out
    assert out.startswith("VALID:")


@pytest.mark.parametrize("name", ALL_TEMPLATES)
def test_validate_no_flag_accepts_every_bundled_template(name):
    """Bare ``plotsim validate`` keeps backward-compatible default behavior."""
    code, out, _err = run_cli("validate", str(_template_path(name)))
    assert code == 0, out
    assert out.startswith("VALID:")


# --- Runtime budget ---------------------------------------------------------


@pytest.mark.parametrize("name", ALL_TEMPLATES)
def test_validate_config_only_runtime_under_500ms(name):
    """Acceptance criterion: <500ms wall-clock on each bundled template.

    First invocation pays imports / Pydantic schema-build cost, so the
    measurement is taken on a second run after a warmup.
    """
    run_cli("validate", "--config-only", str(_template_path(name)))  # warmup
    start = time.perf_counter()
    code, _out, _err = run_cli("validate", "--config-only", str(_template_path(name)))
    elapsed = time.perf_counter() - start
    assert code == 0
    assert elapsed < 0.5, f"{name}: validate took {elapsed*1000:.1f} ms"


# --- No filesystem side effects ---------------------------------------------


def test_validate_config_only_writes_no_files(tmp_path, monkeypatch):
    """Run from a clean cwd; assert no files exist after validation."""
    monkeypatch.chdir(tmp_path)
    code, _out, _err = run_cli(
        "validate", "--config-only", str(_template_path("saas"))
    )
    assert code == 0
    leftovers = list(tmp_path.iterdir())
    assert leftovers == [], f"validate left files: {leftovers}"


# --- Each load-time validator surfaces ---------------------------------------


def test_validate_config_only_rejects_extra_field(tmp_path):
    """``extra='forbid'`` must surface as a Pydantic ValidationError."""
    payload = _load_template_dict("saas")
    payload["zzz_unknown_top_level_key"] = "bogus"
    cfg_path = _write_yaml(tmp_path / "broken.yaml", payload)

    code, out, _err = run_cli("validate", "--config-only", str(cfg_path))
    assert code == 1
    assert "INVALID" in out
    # Pydantic v2 phrasing for extra-forbidden fields
    assert "extra" in out.lower() or "zzz_unknown_top_level_key" in out


def test_validate_config_only_accepts_non_psd_correlation_after_projection(tmp_path):
    """Non-PD correlation matrix is auto-corrected via Higham projection at load (M111).

    Pre-M111 (FIX-F04) this triggered a hard ValueError → ``code == 1`` and
    ``INVALID`` in output. Under M111 the load-time validator projects the
    matrix to nearest-PD via Higham, emits a UserWarning listing the
    adjusted pairs, and the CLI accepts the config.
    """
    payload = _load_template_dict("saas")
    # Triangle of mutually negative-1.0 correlations: classic non-PSD —
    # mathematically impossible (transitivity violation). Higham
    # projects to a valid correlation matrix; the CLI accepts the config.
    metrics = [m["name"] for m in payload["metrics"]][:3]
    payload["correlations"] = [
        {"metric_a": metrics[0], "metric_b": metrics[1], "coefficient": -1.0},
        {"metric_a": metrics[1], "metric_b": metrics[2], "coefficient": -1.0},
        {"metric_a": metrics[0], "metric_b": metrics[2], "coefficient": -1.0},
    ]
    cfg_path = _write_yaml(tmp_path / "non_psd.yaml", payload)

    code, out, _err = run_cli("validate", "--config-only", str(cfg_path))
    assert code == 0
    assert "VALID" in out


def test_validate_config_only_rejects_distribution_param_out_of_range(tmp_path):
    """Field-level Pydantic bounds on distribution-related fields fire at load.

    Picked the ``CorrelationPair.coefficient`` field (``ge=-1.0, le=1.0``) — a
    coefficient of 1.5 is clearly out of bounds and exercises the
    Pydantic ``Field(...)`` constraint path. (PSD covers cross-pair
    consistency; this covers per-field ranges.)
    """
    payload = _load_template_dict("saas")
    metrics = [m["name"] for m in payload["metrics"]]
    payload["correlations"] = [
        {"metric_a": metrics[0], "metric_b": metrics[1], "coefficient": 1.5},
    ]
    cfg_path = _write_yaml(tmp_path / "bad_coeff.yaml", payload)

    code, out, _err = run_cli("validate", "--config-only", str(cfg_path))
    assert code == 1
    assert "INVALID" in out


def test_validate_config_only_rejects_unknown_fk_target(tmp_path):
    """A column whose source names a non-existent table fails cross-ref check."""
    payload = _load_template_dict("saas")
    # Swap one FK to point at a table that doesn't exist.
    for table in payload["tables"]:
        for col in table["columns"]:
            if isinstance(col.get("source"), str) and col["source"].startswith("fk:"):
                col["source"] = "fk:nonexistent_table.bogus_col"
                break
        else:
            continue
        break
    cfg_path = _write_yaml(tmp_path / "bad_fk.yaml", payload)

    code, out, _err = run_cli("validate", "--config-only", str(cfg_path))
    assert code == 1
    assert "INVALID" in out
    assert "nonexistent_table" in out or "unknown table" in out.lower()


# --- Same error messages with and without --config-only ---------------------


def test_validate_flag_and_default_produce_identical_messages_on_invalid(tmp_path):
    """Invalid config: ``plotsim validate`` and ``... --config-only`` agree."""
    payload = _load_template_dict("saas")
    payload["unknown_field"] = "bogus"
    cfg_path = _write_yaml(tmp_path / "broken.yaml", payload)

    code_a, out_a, _ = run_cli("validate", str(cfg_path))
    code_b, out_b, _ = run_cli("validate", "--config-only", str(cfg_path))
    assert code_a == code_b == 1
    assert out_a == out_b


def test_validate_flag_and_default_produce_identical_messages_on_valid(tmp_path):
    """Valid config: messages are identical (both print ``VALID: <path>``)."""
    code_a, out_a, _ = run_cli("validate", str(_template_path("saas")))
    code_b, out_b, _ = run_cli("validate", "--config-only", str(_template_path("saas")))
    assert code_a == code_b == 0
    assert out_a == out_b
