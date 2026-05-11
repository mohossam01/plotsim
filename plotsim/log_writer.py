"""plotsim.log_writer — structured log-file companion writer.

What it does:
    Walks the event tables in a generated dataset; for each event
    table whose config sets ``Table.log_format``, formats every row
    through the template via Python's ``str.format`` against the
    row's column dict and writes one ``.log`` file alongside the
    CSV/Parquet output.

    Pure post-process layer — consumes the same dict ``write_tables``
    is about to write. No engine logic, no RNG, no FK resolution.

When it runs:
    Only when at least one event ``Table`` in the config has a
    non-None ``log_format``. Configs without log formats no-op
    entirely so pre-M14b output is byte-identical.

Format string:
    Python ``str.format`` template. Placeholders must match column
    names on the event table. Examples:

      ``"{event_ts} [INFO] user={user_id} company={company_id}"``
      ``'{"ts": "{event_ts}", "evt_id": "{event_id}", "user": "{user_id}"}'``
      ``"{date_key} {event_id} {company_id} login"``

    Missing placeholders raise ``KeyError`` at format time, which
    surfaces as a wrapped ``ValueError`` naming the bad placeholder
    and the available column set — fail-loud so a typo doesn't
    silently produce empty fields.

Filename:
    ``Table.log_filename`` if set, otherwise ``<table_name>.log``.
    Always ``.log`` extension implicitly appended? No — the user
    owns the filename when they set it explicitly; we don't second-
    guess them. The default carries ``.log`` so casual users get
    the expected extension.

Determinism:
    Row order is preserved from the source DataFrame; same input
    DataFrame + same format string → byte-identical log file
    across runs.

Input:
    ``tables`` (dict[str, pd.DataFrame]) — the same dict
    ``write_tables`` is about to write.
    ``config`` (PlotsimConfig) — for table metadata + format strings.
    ``output_dir`` (Path) — target directory; resolved by the caller
    (``write_tables``) so the sandbox check has already run.

Output:
    Side effect: one ``.log`` file per event table with
    ``log_format`` configured. Returns the list of paths written so
    callers can include them in any post-write reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from plotsim.config import PlotsimConfig, Table


_LOG_DEFAULT_EXTENSION = ".log"
_LOG_LINE_TERMINATOR = "\n"
_LOG_ENCODING = "utf-8"


def _event_tables_with_log_format(config: PlotsimConfig) -> list[Table]:
    """Return the event tables that have ``log_format`` configured.

    Order follows ``config.tables`` so output ordering is stable
    across runs (matches the order the regular table writers see).
    """
    return [tbl for tbl in config.tables if tbl.type == "event" and tbl.log_format is not None]


def _format_one_row(
    template: str,
    row: pd.Series,
    table_name: str,
    available_cols: list[str],
) -> str:
    """Format a single row via ``template.format(**row.to_dict())``.

    Wraps ``KeyError`` from missing placeholders with a clearer
    message that names the table and the available columns. Numeric
    NaN cells are formatted as the string ``"NaN"`` (Python's
    ``str.format`` default) — unsurprising and easy to grep for in
    log post-processing.
    """
    try:
        return template.format(**row.to_dict())
    except KeyError as exc:
        bad_key = exc.args[0] if exc.args else "<unknown>"
        raise ValueError(
            f"log_writer: event table {table_name!r} format string "
            f"references placeholder {{{bad_key}}} which is not a "
            f"column on the table. Available columns: "
            f"{sorted(available_cols)}"
        ) from exc


def _resolve_log_path(output_dir: Path, table: Table) -> Path:
    """Compute the on-disk path for an event-table log.

    Uses ``Table.log_filename`` verbatim when set; otherwise defaults
    to ``<table_name>.log``. SEC-02 sandbox check (path resolves
    inside ``output_dir``) mirrors ``write_single_table`` — the
    ``log_filename`` field is user-controlled string input and could
    embed ``..`` or absolute paths.
    """
    filename = table.log_filename or f"{table.name}{_LOG_DEFAULT_EXTENSION}"
    path = output_dir / filename
    resolved_dir = output_dir.resolve()
    if path.resolve().parent != resolved_dir:
        raise ValueError(
            f"log_writer: log_filename {filename!r} on table "
            f"{table.name!r} resolves outside output_dir "
            f"{str(output_dir)!r}; filenames must be SQL-safe "
            f"identifiers (no path separators, no ..)"
        )
    return path


def write_event_logs(
    tables: dict[str, pd.DataFrame],
    config: PlotsimConfig,
    output_dir: Path,
) -> list[Path]:
    """Write one ``.log`` file per event table that has ``log_format``.

    Returns the list of paths written. Empty list when no event table
    has a log format configured — no I/O, no allocation.

    Per-row formatting uses Python's ``str.format``: placeholders
    in the template (``{column_name}``) are resolved against each
    row's column values. Unknown placeholders raise ``ValueError``
    with the table name and available columns, so typos surface
    immediately instead of producing silent garbage.

    Filename precedence: ``Table.log_filename`` if set, else
    ``<table_name>.log``. Path is sandboxed to ``output_dir``
    (rejects absolute paths and ``..`` traversal) the same way
    ``write_single_table`` sandboxes table names.

    Determinism: row order follows the source DataFrame; same input
    + same template → byte-identical file every run.
    """
    eligible = _event_tables_with_log_format(config)
    if not eligible:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for tbl in eligible:
        df = tables.get(tbl.name)
        if df is None:
            # Event table declared with log_format but absent from
            # the tables dict — programmatic caller passed a partial
            # dict. Skip rather than raise so the helper composes
            # with partial pipelines.
            continue
        # ``log_format`` is non-None for every entry in ``eligible``
        # (filtered above), but mypy's narrowing can't see across
        # the helper — pull it explicitly so the type is ``str``.
        template: Optional[str] = tbl.log_format
        if template is None:  # defensive — already filtered
            continue
        path = _resolve_log_path(output_dir, tbl)
        cols = list(df.columns)
        lines = [_format_one_row(template, row, tbl.name, cols) for _, row in df.iterrows()]
        # Pin LF: match the rest of the writer's CSV behavior so
        # cross-platform diffs are byte-identical. Trailing newline
        # so the file ends cleanly per POSIX text-file convention.
        body = _LOG_LINE_TERMINATOR.join(lines)
        if lines:
            body += _LOG_LINE_TERMINATOR
        path.write_text(body, encoding=_LOG_ENCODING)
        written.append(path)
    return written
