"""F5 regression — validation_report.txt determinism (M102).

Pre-fix: ``output._format_report`` injected ``datetime.now()`` into the
``Generated:`` header line. Two invocations of ``write_tables`` with the
same ``(config, seed)`` produced byte-different ``validation_report.txt``
files because the wall-clock advanced between them. CSV output was already
deterministic; the validation report alone broke the project's
"same config + same seed → byte-identical output" invariant.

Post-fix: ``write_tables`` accepts an optional ``generated_at: datetime``
parameter. When omitted (the library default), ``_format_report`` renders
a deterministic identifier — a 16-character SHA-256 prefix of the config
dump, threaded through automatically by ``write_tables`` from its
``config`` argument. CLI's ``cmd_run`` passes
``generated_at=datetime.now()`` so operators still see the wall-clock
stamp.
"""

from __future__ import annotations

import datetime as _dt
import time
import warnings
from pathlib import Path

import numpy as np

from plotsim import generate_tables, load_config, write_tables
from plotsim.config import SurrogateKeyWarning


ROOT = Path(__file__).resolve().parent.parent
SAAS_YAML = ROOT / "plotsim" / "configs" / "sample_saas.yaml"
REPORT_FILENAME = "validation_report.txt"


def _gen_and_write(yaml_path: Path, out_dir: Path, generated_at=None) -> Path:
    """Generate + write. Conditionally passes ``generated_at`` so the
    pre-fix API path (``write_tables`` without the kwarg) is exercised
    when the caller doesn't explicitly need a wall-clock stamp — this is
    what makes the byte-identical regression below fail pre-fix on the
    actual bug (timestamp drift), rather than failing with a TypeError
    on the missing kwarg.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        cfg = load_config(yaml_path)
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)
    kwargs = {}
    if generated_at is not None:
        kwargs["generated_at"] = generated_at
    write_tables(tables, cfg, output_dir=out_dir, **kwargs)
    return out_dir / REPORT_FILENAME


def test_validation_report_is_byte_identical_across_runs(tmp_path):
    """Library default — two write_tables runs with the same (config, seed)
    must produce byte-identical validation_report.txt.

    Pre-fix: the ``Generated:`` line had a wall-clock timestamp from
    ``datetime.now()``; even back-to-back runs differed by at least one
    second. Sleep added between the two runs to make the pre-fix failure
    reliable on any clock resolution.

    Post-fix: ``Generated:`` renders the deterministic config-fingerprint;
    no time-of-day component leaks into the file.
    """
    a = _gen_and_write(SAAS_YAML, tmp_path / "run_a")
    time.sleep(1.05)  # pad past whole-second clock resolution
    b = _gen_and_write(SAAS_YAML, tmp_path / "run_b")

    text_a = a.read_text(encoding="utf-8")
    text_b = b.read_text(encoding="utf-8")
    assert text_a == text_b, (
        "F5 regression: validation_report.txt is not byte-identical across "
        "two runs of the same (config, seed). The Generated: header is "
        "leaking a wall-clock timestamp.\n"
        f"--- run_a ---\n{text_a}\n--- run_b ---\n{text_b}"
    )


def test_validation_report_includes_config_fingerprint_by_default(tmp_path):
    """The deterministic Generated: line should expose the config
    fingerprint so two reports for *different* configs are still
    distinguishable. Locks the documented contract.
    """
    a = _gen_and_write(SAAS_YAML, tmp_path / "run_a")
    text_a = a.read_text(encoding="utf-8")
    assert "Generated:" in text_a
    assert "deterministic" in text_a, (
        f"F5 regression: deterministic Generated: line missing. "
        f"Header was:\n{text_a.splitlines()[:5]}"
    )
    assert "config-sha256[:16]=" in text_a, (
        f"F5 regression: config fingerprint missing from deterministic "
        f"Generated: line. Header was:\n{text_a.splitlines()[:5]}"
    )


def test_validation_report_uses_explicit_generated_at_when_provided(tmp_path):
    """When the caller passes ``generated_at`` (the CLI path), the
    timestamp must appear in the Generated: line — proves the override
    point still works for operator-facing tooling.
    """
    when = _dt.datetime(2024, 6, 15, 12, 30, 45)
    a = _gen_and_write(SAAS_YAML, tmp_path / "run_a", generated_at=when)
    text_a = a.read_text(encoding="utf-8")
    assert "Generated: 2024-06-15T12:30:45" in text_a, (
        f"F5 regression: explicit generated_at not rendered in header.\n"
        f"Header was:\n{text_a.splitlines()[:5]}"
    )
