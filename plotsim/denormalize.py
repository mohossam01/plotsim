"""plotsim.denormalize — wide-table companion writer.

What it does:
    Joins every fact table with the dim tables it FKs into, producing
    one wide DataFrame per fact (``<fct_name>_wide``). Pure post-process
    layer — consumes the dict ``write_tables`` is about to write and
    returns a parallel dict of the wide companions. No engine logic,
    no RNG, no filesystem.

When it runs:
    Only when ``OutputConfig.denormalized: true``. Off by default so
    pre-M14a output is byte-identical.

Join semantics:
    For each fact table:
      * Walk its columns; every column with source ``fk:<dim>.<col>``
        triggers a left join onto ``tables[<dim>]`` keyed on
        ``fact[<fk_col>] == dim[<col>]``.
      * The dim's join-key column is dropped post-join (it duplicates
        the fact's FK column).
      * Remaining dim columns are renamed ``<dim_table_name>__<col>``
        so two dims that share a column name (``name``,
        ``created_at``) don't collide.
      * SCD2 dims are filtered to ``is_current == True`` rows BEFORE
        the join, so each entity contributes exactly one current-state
        row. SCD2 audit columns (``dim_row_id`` / ``valid_from`` /
        ``valid_to`` / ``is_current``) are dropped from the joined
        output because they're internals of the dim's history layer
        and a wide table is the as-of-now view.
      * ``dim_date`` is joined like any other dim. Its
        ``period_label`` / ``year`` / ``quarter`` / ``month`` cols
        ride along with the ``dim_date__`` prefix so a downstream
        analyst can group by period attributes without hand-rolling
        the join.

Multi-fact handling:
    Each fact is denormalized independently. No cross-fact joins —
    those would multiply rows. The result is one ``<fact>_wide``
    frame per fact in ``config.tables``.

Input:
    ``tables`` (dict[str, pd.DataFrame]) — the same dict ``write_tables``
    is about to write (post-CDC, post-quality if those features ran).
    ``config`` (PlotsimConfig) — for table metadata + FK parsing.

Output:
    ``dict[str, pd.DataFrame]`` keyed by ``<fct_name>_wide``. Empty
    dict when no fact tables in config (no-op).
"""

from __future__ import annotations

import pandas as pd

from plotsim.config import (
    FKSource,
    PlotsimConfig,
    Table,
    parse_source,
)


_SCD2_AUDIT_COLUMNS: frozenset[str] = frozenset(
    {"dim_row_id", "valid_from", "valid_to", "is_current"}
)

_WIDE_SUFFIX = "_wide"


def _table_by_name(config: PlotsimConfig, name: str) -> Table | None:
    for tbl in config.tables:
        if tbl.name == name:
            return tbl
    return None


def _fact_fk_columns(tbl: Table) -> list[tuple[str, FKSource]]:
    """Return every (fact_column_name, FKSource) pair on a fact table.

    Order follows the table's column declaration so the resulting wide
    frame has predictable column ordering: fact PKs/FKs/metrics in
    config order first, then each dim's columns in the order the FKs
    appear on the fact.
    """
    pairs: list[tuple[str, FKSource]] = []
    for col in tbl.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource):
            pairs.append((col.name, parsed))
    return pairs


def _prepare_dim_for_join(
    dim_df: pd.DataFrame,
    dim_table: Table,
    join_key: str,
) -> pd.DataFrame:
    """Return the dim slice ready to merge: SCD2 filter + audit-col drop.

    SCD2 dims have the audit columns ``is_current`` / ``valid_from`` /
    ``valid_to`` / ``dim_row_id`` and multiple rows per entity. Filter
    to the current row per entity (``is_current == True``) so each
    entity contributes exactly one row to the wide output, then drop
    all four audit columns so they don't pollute the join.

    Non-SCD2 dims pass through unchanged. Detection is structural:
    the presence of ``is_current`` column is the marker. (This matches
    how ``_expand_scd_dim`` materialises SCD2 dims — every SCD2 dim
    has the four audit columns; no non-SCD2 dim has ``is_current``.)
    """
    out = dim_df
    if "is_current" in dim_df.columns:
        # Coerce in case pandas widened to object/bool with NA — the
        # SCD expander writes plain Python bools, but defensive cast
        # keeps the filter correct under any upstream coercion.
        mask = dim_df["is_current"].astype(bool)
        out = dim_df.loc[mask].copy(deep=False)
        drop_cols = [c for c in _SCD2_AUDIT_COLUMNS if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

    # Defensive: if the same join_key appears more than once after the
    # current-state filter, the merge would multiply rows. SCD2's
    # ``is_current`` invariant prevents this for SCD2 dims, but a
    # malformed dim (duplicate PK rows in a static dim) would surface
    # as a silent row explosion. Raise loudly instead.
    if join_key in out.columns:
        dup_count = int(out[join_key].duplicated().sum())
        if dup_count > 0:
            raise ValueError(
                f"denormalize: dim {dim_table.name!r} has {dup_count} "
                f"duplicate value(s) on join key {join_key!r} after the "
                f"SCD2 current-state filter; cannot left-join cleanly. "
                f"This indicates a malformed dim (duplicate PK rows on a "
                f"static dim, or an SCD2 dim missing the ``is_current`` "
                f"invariant)."
            )
    return out


def _merge_one_fk(
    wide: pd.DataFrame,
    dim_df: pd.DataFrame,
    dim_table: Table,
    fact_fk_col: str,
    dim_join_col: str,
) -> pd.DataFrame:
    """Left-merge ``dim_df`` onto ``wide`` on (fact_fk_col, dim_join_col).

    Drops the dim-side join key (duplicates the fact FK), prefixes
    every remaining dim column with ``<dim_table_name>__`` so two
    dims that share a column name don't collide.
    """
    prepared = _prepare_dim_for_join(dim_df, dim_table, dim_join_col)
    if dim_join_col not in prepared.columns:
        # A dim that doesn't carry the referenced column is a config
        # error caught at validation time; defensive guard here keeps
        # the helper robust against bypass paths (programmatic dicts
        # bypassing config validation).
        return wide

    rename_map = {c: f"{dim_table.name}__{c}" for c in prepared.columns if c != dim_join_col}
    renamed = prepared.rename(columns=rename_map)

    merged = wide.merge(
        renamed,
        how="left",
        left_on=fact_fk_col,
        right_on=dim_join_col,
    )
    # The right-side join key is dropped because it duplicates the
    # fact's FK column. When the fact's FK column has the same name
    # as the dim's PK (the common case — ``company_id`` on both
    # sides), pandas merges them into one column already and there's
    # nothing to drop. When they differ in name, drop the dim side.
    if dim_join_col in merged.columns and dim_join_col != fact_fk_col:
        merged = merged.drop(columns=[dim_join_col])
    return merged


def denormalize_fact_tables(
    tables: dict[str, pd.DataFrame],
    config: PlotsimConfig,
) -> dict[str, pd.DataFrame]:
    """Produce one wide DataFrame per fact table.

    For each fact in ``config.tables``: walk its FK columns, left-merge
    each referenced dim onto the fact, return the resulting frame
    keyed ``<fct_name>_wide``. Tables that aren't fact tables are
    skipped. Facts referenced in config but absent from ``tables``
    (e.g. a programmatic caller passed a partial dict) are skipped.

    Returns an empty dict when no fact tables exist or none of the
    config's facts are present in ``tables``.
    """
    out: dict[str, pd.DataFrame] = {}
    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        fact_df = tables.get(tbl.name)
        if fact_df is None:
            continue

        wide = fact_df.copy(deep=False)
        for fact_fk_col, fk_source in _fact_fk_columns(tbl):
            dim_df = tables.get(fk_source.table)
            if dim_df is None:
                # Dim referenced by FK but not in the table dict — skip
                # this FK rather than raising, so partial dicts don't
                # explode. Validation at config-load time has already
                # confirmed the FK target exists in the schema.
                continue
            dim_table = _table_by_name(config, fk_source.table)
            if dim_table is None:
                continue
            wide = _merge_one_fk(
                wide,
                dim_df,
                dim_table,
                fact_fk_col=fact_fk_col,
                dim_join_col=fk_source.column,
            )

        out[f"{tbl.name}{_WIDE_SUFFIX}"] = wide

    return out
