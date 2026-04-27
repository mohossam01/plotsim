"""plotsim.quality — post-generation data-quality corruption layer (M107).

What it does:
    Takes the dict of DataFrames produced by ``plotsim.tables.generate_tables``
    and applies the configured ``QualityConfig.quality_issues`` to it,
    returning a corrupted dict plus a per-issue ground-truth record. The
    layer is *additive over generation*: it never re-derives metric values,
    never reads trajectories, and never touches FK or period columns. The
    inputs are fully constructed tables; the outputs preserve every
    ground-truth label the manifest builder needs to recover what was
    corrupted.

    Five issue types are supported. Each is applied independently with its
    own seeded RNG (``base_seed + seed_offset``) so reordering issues in
    the config never perturbs a different issue's affected row set:

      * ``null_injection`` — set ``rate`` of cells in each target column
        to null (NaN for numeric, ``None`` for string/object).
      * ``duplicate_rows`` — insert exact copies of ``rate`` of randomly
        chosen rows back at random positions in the same table.
      * ``type_mismatch`` — convert ``rate`` of values in each target
        column to the wrong type (numerics rendered as strings, strings
        cast to integer codes, etc.). The column's pandas dtype is
        promoted to ``object`` so mixed-type cells coexist.
      * ``late_arrival`` — append a new ``_arrival_period`` column. For
        ``rate`` of rows the column carries ``original period +
        random(1, 5)``; unaffected rows carry null. The original period
        column is unchanged.
      * ``schema_drift`` — for ``rate`` of rows in each target column,
        copy the value into a new ``{column}_v2`` column and null the
        original at those rows. Unaffected rows retain the original and
        get null in ``_v2``.

Architectural constraints (mission spec):
    1. Pure: no filesystem access, no logging, no time / wall-clock reads.
       DataFrames in, DataFrames out.
    2. Deterministic: same ``(tables, config, base_seed)`` → byte-identical
       corrupted output and identical ``QualityInjection`` records.
    3. FK and period columns are protected — the validator in
       ``PlotsimConfig`` rejects them at load, and the ``"*"`` sentinel
       expansion here excludes them defensively.
    4. The clean tables passed in are NOT mutated in place — a deep copy
       is taken before any corruption. Manifest construction reads the
       clean copy; the corrupted dict is what callers write to disk.

Input:
    ``tables`` (dict[str, pd.DataFrame]), ``PlotsimConfig`` (for table
    schema lookup and protected-column resolution), an integer
    ``base_seed`` (typically ``config.seed``).

Output:
    ``(corrupted_tables, ground_truth)`` — the corrupted dict and a list
    of ``manifest.QualityInjection`` records ready to attach to the
    manifest's ``quality_injections`` field.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from plotsim.config import (
    FKSource,
    PlotsimConfig,
    QualityIssue,
    Table,
    parse_source,
)
from plotsim.manifest import QualityInjection


_PROTECTED_COL_NAMES = frozenset({
    "date_key", "period", "period_index", "period_label",
})


def _protected_columns(tbl: Table) -> set[str]:
    """Return the set of columns the quality layer must never corrupt.

    Mirrors the validator at config-load: every FK column plus the
    well-known temporal column names. PK columns are also protected
    here (validator allows them in explicit lists, but injecting nulls
    into a PK breaks downstream uniqueness / FK validation, and the
    ``"*"`` sentinel below should never expand to one).
    """
    protected = set(_PROTECTED_COL_NAMES)
    for col in tbl.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource):
            protected.add(col.name)
    for pk_col in tbl.primary_key_cols:
        protected.add(pk_col)
    return protected


def _resolve_target_columns(
    tbl: Table,
    df: pd.DataFrame,
    requested: list[str],
) -> list[str]:
    """Expand the ``target_columns`` request to concrete column names.

    ``"*"`` (the sentinel) → every column on the DataFrame that isn't
    protected. Explicit lists pass through after the validator's
    membership / protection checks at load time. Either way, columns
    not present on the actual DataFrame are silently dropped — a
    builder that omitted a column shouldn't break the corruption.
    """
    protected = _protected_columns(tbl)
    if requested == ["*"]:
        return [c for c in df.columns if c not in protected]
    return [c for c in requested if c in df.columns and c not in protected]


def _select_row_indices(
    n_rows: int, rate: float, rng: np.random.Generator,
) -> np.ndarray:
    """Pick ``floor(rate * n_rows)`` distinct row indices in [0, n_rows).

    ``floor`` rather than ``round`` so a rate of 0 always selects zero
    rows (round would still be zero, but floor matches the user's
    intuition that ``rate=0`` is "no corruption"). Empty selections
    return a zero-length int64 array so downstream ``np.fromiter`` /
    DataFrame slicing works without branching.
    """
    if n_rows == 0 or rate <= 0.0:
        return np.empty(0, dtype=np.int64)
    n_pick = int(np.floor(rate * n_rows))
    if n_pick == 0:
        return np.empty(0, dtype=np.int64)
    return np.sort(rng.choice(n_rows, size=n_pick, replace=False)).astype(np.int64)


def _is_numeric_dtype(series: pd.Series) -> bool:
    """True if the series' dtype is one numpy / pandas treats as numeric.

    Used to pick NaN vs None as the null sentinel, and to drive the
    type_mismatch direction (numeric → string when the column was
    numeric, otherwise the inverse).
    """
    return pd.api.types.is_numeric_dtype(series.dtype)


def _apply_null_injection(
    df: pd.DataFrame,
    cols: list[str],
    rate: float,
    rng: np.random.Generator,
    issue_idx: int,
    table_name: str,
) -> tuple[pd.DataFrame, list[QualityInjection]]:
    """Set a fraction of cells in each target column to null.

    Each target column gets its OWN draw from the shared per-issue rng,
    so the affected row sets typically differ across columns. Null
    sentinels follow the column's dtype: numeric → NaN, anything else
    → ``None``. The DataFrame is mutated in place on a copy the caller
    already took, so the returned ``df`` is the same object passed in.
    """
    out = df
    records: list[QualityInjection] = []
    for col in cols:
        idxs = _select_row_indices(len(out), rate, rng)
        if len(idxs) == 0:
            continue
        clean = out[col].iloc[idxs].tolist()
        if _is_numeric_dtype(out[col]):
            out[col] = out[col].astype("float64")
            out.loc[out.index[idxs], col] = np.nan
        else:
            out[col] = out[col].astype(object)
            out.loc[out.index[idxs], col] = None
        records.append(QualityInjection(
            issue_index=issue_idx,
            issue_type="null_injection",
            table=table_name,
            column=col,
            row_indices=[int(i) for i in idxs.tolist()],
            clean_values=list(clean),
        ))
    return out, records


def _apply_duplicate_rows(
    df: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
    issue_idx: int,
    table_name: str,
) -> tuple[pd.DataFrame, list[QualityInjection]]:
    """Insert exact copies of selected rows at random positions.

    The row indices recorded in the ground truth are the indices of the
    *source* rows in the pre-duplication DataFrame; the duplicated
    rows themselves land at indices that are determined by the
    insertion order. The total row count grows by ``len(idxs)``.
    Empty input or ``rate=0`` is a no-op.
    """
    out = df
    n_orig = len(out)
    if n_orig == 0 or rate <= 0.0:
        return out, []
    src_idxs = _select_row_indices(n_orig, rate, rng)
    if len(src_idxs) == 0:
        return out, []
    duplicated = out.iloc[src_idxs].copy()
    # Random positions to insert (within the post-insertion frame).
    insert_at = rng.integers(0, n_orig + 1, size=len(src_idxs))
    insert_at = np.sort(insert_at)

    # Insert one at a time in ascending position to keep positions stable.
    new_frame_rows: list[pd.DataFrame] = []
    cursor = 0
    dup_iter = iter(range(len(duplicated)))
    insert_positions = list(insert_at.tolist())
    for pos in insert_positions:
        if pos > cursor:
            new_frame_rows.append(out.iloc[cursor:pos])
            cursor = pos
        try:
            d_idx = next(dup_iter)
        except StopIteration:
            break
        new_frame_rows.append(duplicated.iloc[d_idx:d_idx + 1])
    if cursor < n_orig:
        new_frame_rows.append(out.iloc[cursor:])
    if new_frame_rows:
        out = pd.concat(new_frame_rows, ignore_index=True)
    else:
        out = pd.concat([out, duplicated], ignore_index=True)

    records = [QualityInjection(
        issue_index=issue_idx,
        issue_type="duplicate_rows",
        table=table_name,
        column="_rows",
        row_indices=[int(i) for i in src_idxs.tolist()],
        clean_values=[],
    )]
    return out, records


def _apply_type_mismatch(
    df: pd.DataFrame,
    cols: list[str],
    rate: float,
    rng: np.random.Generator,
    issue_idx: int,
    table_name: str,
) -> tuple[pd.DataFrame, list[QualityInjection]]:
    """Convert a fraction of cell values in each target column to wrong types.

    Numeric → ``str(value)`` (the column dtype is promoted to ``object``
    so mixed-type cells coexist). Non-numeric → integer hash code derived
    from ``hash(value) % 1000`` (deterministic per Python session because
    the corruption is operating on already-built strings, not user
    input — the hash randomization that breaks dict ordering doesn't
    affect numeric output of this conversion as long as the same
    interpreter invocation produces the same hash).
    """
    out = df
    records: list[QualityInjection] = []
    for col in cols:
        idxs = _select_row_indices(len(out), rate, rng)
        if len(idxs) == 0:
            continue
        clean = out[col].iloc[idxs].tolist()
        out[col] = out[col].astype(object)
        if all(_is_numeric_dtype(pd.Series([v])) for v in clean if v is not None and not (isinstance(v, float) and np.isnan(v))):
            corrupted_vals = [
                None if v is None or (isinstance(v, float) and np.isnan(v))
                else str(v) for v in clean
            ]
        else:
            corrupted_vals = [
                None if v is None
                else (abs(hash(str(v))) % 1000) for v in clean
            ]
        for offset, ridx in enumerate(idxs.tolist()):
            out.loc[out.index[ridx], col] = corrupted_vals[offset]
        records.append(QualityInjection(
            issue_index=issue_idx,
            issue_type="type_mismatch",
            table=table_name,
            column=col,
            row_indices=[int(i) for i in idxs.tolist()],
            clean_values=list(clean),
        ))
    return out, records


def _find_period_column(tbl: Table, df: pd.DataFrame) -> str | None:
    """Locate the per-row period anchor for a fact/event table.

    Picks the first FK that points at ``dim_date`` (the conventional
    ``date_key``); falls back to the literal column name ``date_key``
    when the FK can't be resolved (programmatically-built tables that
    skipped FK declarations still tend to carry ``date_key``).
    """
    for col in tbl.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource) and parsed.table == "dim_date":
            if col.name in df.columns:
                return col.name
    return "date_key" if "date_key" in df.columns else None


def _apply_late_arrival(
    df: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
    issue_idx: int,
    table_name: str,
    tbl: Table,
) -> tuple[pd.DataFrame, list[QualityInjection]]:
    """Add an ``_arrival_period`` column to mark rows arriving late.

    For affected rows: ``arrival_period = period_value + uniform(1, 5)``.
    For unaffected rows: null in the new column. The original period
    column is untouched, so existing temporal joins still work. When
    no period column can be located the issue is a no-op (records
    nothing).
    """
    out = df
    period_col = _find_period_column(tbl, out)
    if period_col is None or len(out) == 0:
        return out, []
    idxs = _select_row_indices(len(out), rate, rng)
    if len(idxs) == 0:
        out["_arrival_period"] = None
        return out, []
    offsets = rng.integers(1, 6, size=len(idxs))
    arrival = pd.Series([None] * len(out), index=out.index, dtype=object)
    for k, ridx in enumerate(idxs.tolist()):
        period_val = out[period_col].iloc[ridx]
        try:
            arrival.iloc[ridx] = int(period_val) + int(offsets[k])
        except (TypeError, ValueError):
            arrival.iloc[ridx] = None
    out["_arrival_period"] = arrival.values
    records = [QualityInjection(
        issue_index=issue_idx,
        issue_type="late_arrival",
        table=table_name,
        column="_arrival_period",
        row_indices=[int(i) for i in idxs.tolist()],
        clean_values=[],
    )]
    return out, records


def _apply_schema_drift(
    df: pd.DataFrame,
    cols: list[str],
    rate: float,
    rng: np.random.Generator,
    issue_idx: int,
    table_name: str,
) -> tuple[pd.DataFrame, list[QualityInjection]]:
    """Move values to a new ``{col}_v2`` column on a fraction of rows.

    Affected rows: the original column is set to null; the new
    ``{col}_v2`` column gets the value. Unaffected rows: the original
    column keeps its value; ``_v2`` is null. The pre-corruption
    cell value is recorded in ``clean_values`` for full reversibility.
    """
    out = df
    records: list[QualityInjection] = []
    for col in cols:
        idxs = _select_row_indices(len(out), rate, rng)
        v2_name = f"{col}_v2"
        v2_series = pd.Series([None] * len(out), index=out.index, dtype=object)
        clean = out[col].iloc[idxs].tolist() if len(idxs) > 0 else []
        for k, ridx in enumerate(idxs.tolist()):
            v2_series.iloc[ridx] = clean[k]
        out[v2_name] = v2_series.values
        if len(idxs) > 0:
            out[col] = out[col].astype(object)
            for ridx in idxs.tolist():
                out.loc[out.index[ridx], col] = None
        if len(idxs) > 0:
            records.append(QualityInjection(
                issue_index=issue_idx,
                issue_type="schema_drift",
                table=table_name,
                column=col,
                row_indices=[int(i) for i in idxs.tolist()],
                clean_values=list(clean),
            ))
    return out, records


def apply_issues(
    tables: dict[str, pd.DataFrame],
    config: PlotsimConfig,
    base_seed: int,
) -> tuple[dict[str, pd.DataFrame], list[QualityInjection]]:
    """Apply every configured ``QualityIssue`` to ``tables``.

    Returns a *new* dict containing deep copies of every modified table,
    plus the per-issue ``QualityInjection`` ground-truth list. Tables
    not targeted by any issue are passed through by reference — the
    deep-copy cost only lands on modified tables.

    Issue order follows ``config.quality.quality_issues`` order; within
    one issue, columns follow the resolved target_columns order.
    Determinism: each issue draws from
    ``np.random.default_rng(base_seed + issue.seed_offset)``, so two
    issues with the same offset DO see correlated draws — that's a
    user choice. Distinct offsets keep them independent.

    Empty ``quality_issues`` short-circuits to ``(tables, [])`` without
    copying.
    """
    if not config.quality.quality_issues:
        return tables, []

    out: dict[str, pd.DataFrame] = dict(tables)
    ground_truth: list[QualityInjection] = []

    table_by_name = {t.name: t for t in config.tables}

    for issue_idx, issue in enumerate(config.quality.quality_issues):
        rng = np.random.default_rng(int(base_seed) + int(issue.seed_offset))
        target_table = issue.target_table
        if target_table not in out:
            continue
        if target_table not in table_by_name:
            continue
        df_clean = out[target_table]
        df = df_clean.copy(deep=True).reset_index(drop=True)
        tbl = table_by_name[target_table]
        cols = _resolve_target_columns(tbl, df, list(issue.target_columns))

        records: list[QualityInjection] = []
        if issue.type == "null_injection":
            df, records = _apply_null_injection(
                df, cols, issue.rate, rng, issue_idx, target_table,
            )
        elif issue.type == "duplicate_rows":
            df, records = _apply_duplicate_rows(
                df, issue.rate, rng, issue_idx, target_table,
            )
        elif issue.type == "type_mismatch":
            df, records = _apply_type_mismatch(
                df, cols, issue.rate, rng, issue_idx, target_table,
            )
        elif issue.type == "late_arrival":
            df, records = _apply_late_arrival(
                df, issue.rate, rng, issue_idx, target_table, tbl,
            )
        elif issue.type == "schema_drift":
            df, records = _apply_schema_drift(
                df, cols, issue.rate, rng, issue_idx, target_table,
            )
        else:
            raise ValueError(
                f"quality.apply_issues: unknown issue type {issue.type!r} "
                f"on issue {issue_idx}; the validator should reject this "
                f"at load time"
            )

        out[target_table] = df
        ground_truth.extend(records)

    return out, ground_truth


__all__ = [
    "apply_issues",
]
