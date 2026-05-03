"""Generic parameter-sweep driver for M103 fidelity sweeps.

Each sweep is a list of cells (dicts of input parameters) plus two callables:
``build_config(cell) -> PlotsimConfig`` and ``measure(cell, tables) -> dict``.
The runner walks the cells, generates tables in-memory (no disk write — the
fidelity sweeps don't need persisted CSVs, only the measurements), invokes
``measure`` on the resulting tables dict, and writes one row per cell to a
result CSV.

Invariants enforced here so each claim sweep doesn't reimplement them:
- Intermediate flush every ``flush_every`` rows (default 100). A crash mid-
  sweep loses at most the last 100 cells, not the whole run.
- Progress + per-cell timing logged to stderr; result CSV is the only stdout.
- Each cell row carries every input parameter alongside its measurements, so a
  reader with only the CSV can rerun any cell. The CSV's column ordering is
  ``cell_keys + measurement_keys`` — both alphabetised within each group, so
  the schema is stable across reruns.
- Memory: tables dict is dropped after ``measure`` returns. A 1000+ cell sweep
  stays bounded at one cell's tables in memory at a time.

Used directly from claim driver scripts (claim1_correlation.py etc) which sit
alongside this file. Also imported by tests/test_fidelity_smoke.py to assert
the headline finding of each claim reproduces under the smoke-test budget.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from plotsim import generate_tables
from plotsim.config import PlotsimConfig

# Type aliases for the two driver callables every sweep supplies.
BuildConfig = Callable[[dict[str, Any]], PlotsimConfig]
Measure = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def run_sweep(
    cells: Iterable[dict[str, Any]],
    build_config: BuildConfig,
    measure: Measure,
    output_csv: Path,
    *,
    flush_every: int = 100,
    progress_label: str = "sweep",
) -> int:
    """Drive ``build_config`` + ``generate_tables`` + ``measure`` over cells.

    Returns the number of rows written. On the first cell, derives the result
    CSV's column schema from ``set(cell.keys()) | set(measurement.keys())``;
    subsequent cells must produce the same union or the row is skipped with
    a stderr warning (mismatched schema in mid-sweep is a bug, but losing the
    whole CSV to it is worse than losing the offending row).
    """
    cells_list = list(cells)
    if not cells_list:
        raise ValueError("run_sweep called with no cells")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    fieldnames: list[str] | None = None
    buffered: list[dict[str, Any]] = []
    t_start = time.monotonic()

    def _append_buffered() -> None:
        if not buffered or fieldnames is None:
            return
        with output_csv.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            for r in buffered:
                writer.writerow(r)

    for i, cell in enumerate(cells_list):
        cell_t0 = time.monotonic()
        cfg = build_config(cell)
        rng = np.random.default_rng(cfg.seed)
        tables = generate_tables(cfg, rng)
        measurement = measure(cell, tables)
        del tables  # release memory before next cell

        merged = {**cell, **measurement}
        if fieldnames is None:
            cell_keys = sorted(cell.keys())
            meas_keys = sorted(k for k in measurement if k not in cell)
            fieldnames = cell_keys + meas_keys
            with output_csv.open("w", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

        if set(merged.keys()) != set(fieldnames):
            extra = set(merged.keys()) - set(fieldnames)
            missing = set(fieldnames) - set(merged.keys())
            sys.stderr.write(
                f"[{progress_label}] cell {i} schema drift "
                f"(extra={sorted(extra)}, missing={sorted(missing)}); "
                f"skipping row\n"
            )
            continue

        buffered.append({k: merged.get(k) for k in fieldnames})
        rows_written += 1

        elapsed_cell = time.monotonic() - cell_t0
        if (i + 1) % flush_every == 0 or i == len(cells_list) - 1:
            _append_buffered()
            buffered = []
            elapsed_total = time.monotonic() - t_start
            sys.stderr.write(
                f"[{progress_label}] {rows_written}/{len(cells_list)} "
                f"({elapsed_total:.1f}s, last cell {elapsed_cell:.2f}s)\n"
            )
            sys.stderr.flush()

    if buffered:
        _append_buffered()

    return rows_written


__all__ = ["run_sweep", "BuildConfig", "Measure"]
