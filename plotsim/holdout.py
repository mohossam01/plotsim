"""plotsim.holdout — temporal train/holdout split for fact tables (M109).

What it does:
    Slices every per-entity-per-period fact table along the period axis
    into a training set and a holdout set:

      * **Training**  — rows whose period index lies in
        ``[0, n_periods - holdout_periods)``.
      * **Holdout**   — rows whose period index lies in
        ``[n_periods - holdout_periods, n_periods)``.

    The split is strictly temporal — random shuffling on time-series
    data is a leakage pattern (training rows from later periods would
    teach the model to peek at the future), so the cutoff is a hard
    boundary on ``period_index`` derived from ``dim_date``.

    Only fact tables with ``grain == 'per_entity_per_period'`` are
    split. Dim tables (per_entity, per_period, per_reference) and
    bridge tables hold static / associative rows that don't have a
    per-row period index in the same sense — splitting them would
    duplicate state across both files. Event tables hold variable
    per-period grain rows; their date_key column is still in scope of
    the cutoff conceptually, but the M109 acceptance criteria
    explicitly excludes them — variable-grain rows don't have a
    period-indexed grain compatible with temporal splitting (a
    multi-row event firing on the cutoff period would land in only
    one half).

Architectural rules:
    * No import of ``plotsim.tables`` — receives DataFrames as
      arguments, same pattern as ``plotsim.entity_features`` and
      ``plotsim.manifest``.
    * Pure: same ``(config, tables)`` produces byte-identical splits
      every call. No filesystem touch, no RNG, no clock.
    * The writer in ``plotsim.output`` is the sole production caller.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from plotsim.config import (
    FKSource,
    PlotsimConfig,
    Table,
    parse_source,
)


def cutoff_period_index(config: PlotsimConfig) -> int:
    """Return the period index that separates training from holdout.

    ``cutoff = n_periods - holdout_periods``. Rows with
    ``period_index < cutoff`` are training; rows with
    ``period_index >= cutoff`` are holdout. Independent of any
    DataFrame — derived purely from ``time_window.period_count()`` and
    ``holdout.holdout_periods``. The load-time validator has already
    guaranteed ``cutoff >= holdout.min_training_periods >= 1`` when
    ``holdout.enabled`` is True, so the value is always positive in
    practice.
    """
    return config.time_window.period_count() - config.holdout.holdout_periods


def _date_fk_column(tbl: Table) -> Optional[str]:
    """FK column on ``tbl`` that targets ``dim_date`` (or None)."""
    for col in tbl.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource) and parsed.table == "dim_date":
            return col.name
    return None


def _period_index_by_date_key(dim_date: pd.DataFrame) -> dict:
    """Map ``date_key -> period_index`` from the date spine.

    ``dim_date`` row order is period order (period_index 0 is the first
    row, etc.). The dim-date builder writes both ``date_key`` and
    ``period_index`` columns; using the explicit ``period_index``
    column avoids re-deriving the position from the iteration index
    and survives any future re-ordering of the spine.
    """
    return {int(row.date_key): int(row.period_index) for row in dim_date.itertuples(index=False)}


def _splittable_fact_tables(
    config: PlotsimConfig,
    tables: dict[str, pd.DataFrame],
) -> list[Table]:
    """Return the per_entity_per_period fact tables that have data.

    Only fact tables with the composite grain are eligible (per the
    mission's "Only fact tables with per_entity_per_period grain are
    split" rule). Tables that the engine emitted as empty DataFrames
    are skipped — splitting an empty frame is a no-op, but keeping
    them out of the result keeps the ``write_tables`` loop free of
    zero-byte ``_train`` / ``_holdout`` files.
    """
    eligible: list[Table] = []
    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        if tbl.grain != "per_entity_per_period":
            continue
        df = tables.get(tbl.name)
        if df is None or df.empty:
            continue
        eligible.append(tbl)
    return eligible


def split_fact_tables(
    config: PlotsimConfig,
    tables: dict[str, pd.DataFrame],
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Split every eligible fact table into ``(train_df, holdout_df)``.

    Pre-conditions enforced upstream:

      * ``config.holdout.enabled == True`` — callers guarded by the
        same flag used to gate the writer.
      * ``dim_date`` is non-empty (covered by the date-spine validator
        on every successful generation).

    Returns a dict keyed by fact-table name; each value is the pair
    ``(train_df, holdout_df)``. Tables not present in the result are
    not eligible to be split (dims, bridges, events, empty facts).
    Both halves preserve the original DataFrame's index and column
    order; ``.reset_index(drop=True)`` is intentionally NOT applied so
    downstream code that joined a column back by index isn't broken
    by the split.

    Sanity invariants the caller can rely on:

      * ``len(train) + len(holdout) == len(original)`` — every row
        lands in exactly one half (no loss, no duplication).
      * Training rows correspond to ``period_index < cutoff``; holdout
        rows to ``period_index >= cutoff``. Rows whose date_key
        doesn't resolve through ``dim_date`` would land in neither
        half — that case is impossible after a successful date-spine
        validation, but the function never silently drops a row, so
        such rows would surface as a row-count mismatch.
    """
    if not config.holdout.enabled:
        return {}

    dim_date = tables.get("dim_date")
    if dim_date is None or dim_date.empty:
        raise ValueError(
            "holdout: 'dim_date' is missing or empty; cannot derive "
            "period indices for the train/holdout cutoff"
        )

    period_index_by_date_key = _period_index_by_date_key(dim_date)
    cutoff = cutoff_period_index(config)

    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for tbl in _splittable_fact_tables(config, tables):
        df = tables[tbl.name]
        date_col = _date_fk_column(tbl)
        if date_col is None or date_col not in df.columns:
            # Fact table without a dim_date FK — outside the engine's
            # standard fact-table contract, but we won't guess. Skip
            # rather than emit a malformed split.
            continue
        period_indices = df[date_col].map(period_index_by_date_key)
        train_mask = period_indices < cutoff
        holdout_mask = period_indices >= cutoff
        train_df = df.loc[train_mask].copy(deep=False)
        holdout_df = df.loc[holdout_mask].copy(deep=False)
        splits[tbl.name] = (train_df, holdout_df)
    return splits


__all__ = [
    "cutoff_period_index",
    "split_fact_tables",
]
