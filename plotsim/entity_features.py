"""plotsim.entity_features — flat per-entity feature table builder (M108).

What it does:
    Aggregates the temporal fact tables produced by ``plotsim.tables`` into
    a single one-row-per-entity DataFrame suitable for downstream tabular
    ML or notebook-level joining. For every numeric metric the engine
    landed in a fact table, six aggregate columns are emitted:

      * ``{metric}_mean``         — np.nanmean over the entity's series
      * ``{metric}_std``          — np.nanstd (population, ddof=0)
      * ``{metric}_slope``        — slope of a degree-1 polyfit of value
                                    over period index (NaN-aware)
      * ``{metric}_first``        — value at the entity's earliest period
      * ``{metric}_last``         — value at the entity's latest period
      * ``{metric}_peak_period``  — period index where the value is max

    Optional ground-truth columns (``include_labels=true``):

      * ``archetype``                  — from ``config.entities[i].archetype``
      * ``final_trajectory_position``  — from the largest-period
                                         ``trajectory_samples`` entry per
                                         entity in the manifest. NaN for
                                         entities outside the manifest's
                                         ``trajectory_sample_rate`` subset.

    The module is pure — same ``(config, tables, manifest)`` produces a
    byte-identical DataFrame every call. No filesystem touch, no RNG, no
    clock. The writer in ``plotsim.output`` is the only caller in
    production paths.

Architectural rules:
    * No import of ``plotsim.tables`` — receives DataFrames as arguments,
      same pattern as ``plotsim.manifest``.
    * Bridge tables are NEVER aggregated — bridges are associative, not
      temporal, so their metrics don't fit the per-period reduction
      schema this module produces.
    * Mutual exclusion with ``quality.quality_issues`` is enforced at
      config load (see ``plotsim.validation.validate_entity_features_config``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from plotsim.config import (
    FKSource,
    MetricSource,
    PKSource,
    PlotsimConfig,
    Table,
    parse_source,
)
from plotsim.manifest import ManifestSchema


ENTITY_FEATURES_BASENAME = "_entity_features"

_AGGREGATE_SUFFIXES = ("mean", "std", "slope", "first", "last", "peak_period")


# --- Helpers -----------------------------------------------------------------


def _primary_per_entity_dim(config: PlotsimConfig) -> Optional[Table]:
    """Return the first ``grain='per_entity'`` dim table.

    Bundled templates each declare exactly one such dim (``dim_company``,
    ``dim_employee``, ``dim_student``, ...) so "first in config order"
    matches operator intent. Configs that genuinely model multiple
    per_entity dims would still return a deterministic choice — the
    first one — and the operator can override the selection by ordering
    the dim list to put the desired anchor first.
    """
    for tbl in config.tables:
        if tbl.type == "dim" and tbl.grain == "per_entity":
            return tbl
    return None


def _pk_column(tbl: Table) -> Optional[str]:
    """First column on ``tbl`` whose source is ``pk``."""
    for col in tbl.columns:
        if isinstance(parse_source(col.source), PKSource):
            return col.name
    return None


def _entity_fk_column(tbl: Table, primary_dim_name: str) -> Optional[str]:
    """FK column on ``tbl`` that targets ``primary_dim_name``."""
    for col in tbl.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource) and parsed.table == primary_dim_name:
            return col.name
    return None


def _date_fk_column(tbl: Table) -> Optional[str]:
    """FK column on ``tbl`` that targets ``dim_date``."""
    for col in tbl.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource) and parsed.table == "dim_date":
            return col.name
    return None


def _resolve_target_metrics(config: PlotsimConfig) -> list[str]:
    """Return the metric names to aggregate, preserving config.metrics order.

    When ``entity_features.metrics`` is empty (default), every metric on
    ``config.metrics`` that has at least one ``metric:<name>`` column on
    a fact table participates. When a non-empty list is configured, the
    load-time validator has already verified each name resolves to a
    numeric fact column; we re-filter here to preserve config order.

    M109: when ``holdout.enabled`` is true and ``holdout.target_metric``
    is set, that metric is filtered out so its six aggregate columns
    never reach the entity-features output. The exclusion is the
    leakage-prevention rule for downstream ML — a model that gets
    ``revenue_mean`` / ``revenue_last`` as features would trivially
    "predict" the held-out ``revenue`` periods.
    """
    fact_metric_names: set[str] = set()
    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if isinstance(parsed, MetricSource) and col.dtype in ("int", "float"):
                fact_metric_names.add(parsed.metric)

    excluded: set[str] = set()
    if config.holdout.enabled and config.holdout.target_metric is not None:
        excluded.add(config.holdout.target_metric)

    requested = list(config.entity_features.metrics)
    if not requested:
        return [
            m.name for m in config.metrics
            if m.name in fact_metric_names and m.name not in excluded
        ]
    requested_set = set(requested)
    return [
        m.name for m in config.metrics
        if m.name in requested_set and m.name not in excluded
    ]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Degree-1 polyfit slope of ``y`` against ``x``, NaN-aware.

    Drops NaN cells from ``y`` before fitting — under the engine's MCAR
    noise (``noise.mcar_rate``) random cells become NaN and a naive
    ``polyfit`` would propagate the NaN through every coefficient.
    Returns NaN when fewer than two non-NaN observations remain (one
    point underdetermines the line) or when every retained ``x`` value
    is identical (zero variance in the predictor).
    """
    mask = ~np.isnan(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if float(np.std(xv)) == 0.0:
        return float("nan")
    coeffs = np.polyfit(xv, yv, 1)
    return float(coeffs[0])


def _peak_period(periods: np.ndarray, values: np.ndarray) -> float:
    """Period index where ``values`` reaches its NaN-aware maximum.

    Returned as float so an all-NaN series can render NaN in the same
    column without forcing a separate sentinel. Ties on the maximum
    take the earliest period (numpy's ``nanargmax`` semantics).
    """
    mask = ~np.isnan(values)
    if int(mask.sum()) == 0:
        return float("nan")
    idx = int(np.nanargmax(values))
    return float(periods[idx])


def _aggregate_series(
    periods: np.ndarray, values: np.ndarray,
) -> dict[str, float]:
    """Six-statistic reduction of one entity's metric series.

    Inputs must be pre-sorted by period (the caller always sorts before
    calling). All six values are returned even when ``values`` is empty
    or all-NaN — every entity row in the output frame keeps the same
    column set so the writer doesn't need a per-row schema.
    """
    if len(values) == 0:
        nan = float("nan")
        return {suffix: nan for suffix in _AGGREGATE_SUFFIXES}
    valid_mask = ~np.isnan(values)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        nan = float("nan")
        return {suffix: nan for suffix in _AGGREGATE_SUFFIXES}
    mean_v = float(np.nanmean(values))
    std_v = float(np.nanstd(values, ddof=0))
    slope_v = _slope(periods.astype(float), values)
    first_v = float(values[0]) if not np.isnan(values[0]) else float("nan")
    last_v = float(values[-1]) if not np.isnan(values[-1]) else float("nan")
    peak_v = _peak_period(periods, values)
    return {
        "mean": mean_v,
        "std": std_v,
        "slope": slope_v,
        "first": first_v,
        "last": last_v,
        "peak_period": peak_v,
    }


def _final_position_by_entity(
    manifest: ManifestSchema,
) -> dict[str, float]:
    """Map ``entity_name -> last sampled trajectory position``.

    Walks ``manifest.trajectory_samples`` once and keeps the entry with
    the largest ``period_index`` per entity. Entities not in the
    manifest's sampled subset (under ``trajectory_sample_rate < 1.0``)
    are absent from the returned dict; callers fall back to NaN.
    """
    best: dict[str, tuple[int, float]] = {}
    for sample in manifest.trajectory_samples:
        cur = best.get(sample.entity)
        if cur is None or sample.period_index > cur[0]:
            best[sample.entity] = (sample.period_index, sample.position)
    return {name: pos for name, (_, pos) in best.items()}


def _entity_pk_values_in_config_order(
    primary_dim: Table,
    primary_dim_df: pd.DataFrame,
    pk_col: str,
    n_entities: int,
) -> list:
    """Return the per_entity dim's PK values in ``config.entities`` order.

    SCD-expanded per_entity dims hold multiple versioned rows per
    entity; ``drop_duplicates(keep="first")`` collapses them back to
    one row per entity in the order ``expand_scd_dims`` produced —
    which is ``config.entities`` order. Non-SCD dims are already 1:1
    with config order. Either way, the returned list aligns positionally
    with ``config.entities``.
    """
    deduped = primary_dim_df.drop_duplicates(subset=[pk_col], keep="first")
    pks = deduped[pk_col].tolist()
    if len(pks) != n_entities:
        raise ValueError(
            f"entity_features: per_entity dim {primary_dim.name!r} has "
            f"{len(pks)} unique PK value(s) but config has {n_entities} "
            f"entities; the dim must be 1:1 with config.entities for "
            f"per-entity aggregation"
        )
    return pks


# --- Public API --------------------------------------------------------------


def build_entity_features(
    config: PlotsimConfig,
    tables: dict[str, pd.DataFrame],
    manifest: ManifestSchema,
) -> pd.DataFrame:
    """Build the flat per-entity feature DataFrame.

    Pre-conditions enforced upstream by
    ``validate_entity_features_config`` at config load:

      * ``config.entity_features.enabled == True``
      * ``config.manifest.include == True``
      * ``config.quality.quality_issues == []``
      * Every ``config.entity_features.metrics`` entry, if specified,
        resolves to a numeric metric on a fact table.

    The function is pure: same inputs → same DataFrame. The output
    column order is fully determined by config order, so two runs at
    the same seed produce a byte-identical CSV / Parquet file.
    """
    primary_dim = _primary_per_entity_dim(config)
    if primary_dim is None:
        raise ValueError(
            "entity_features: config has no per_entity dim table; "
            "entity-level aggregation requires exactly one anchor dim "
            "with grain='per_entity'"
        )
    pk_col = _pk_column(primary_dim)
    if pk_col is None:
        raise ValueError(
            f"entity_features: per_entity dim {primary_dim.name!r} has no "
            f"PK column; one column must declare source 'pk' so per-entity "
            f"rows can be keyed deterministically"
        )

    primary_dim_df = tables.get(primary_dim.name)
    if primary_dim_df is None or primary_dim_df.empty:
        raise ValueError(
            f"entity_features: per_entity dim {primary_dim.name!r} was not "
            f"generated or is empty; nothing to aggregate against"
        )

    entity_pks = _entity_pk_values_in_config_order(
        primary_dim, primary_dim_df, pk_col, len(config.entities),
    )
    pk_to_entity_name = {
        pk: entity.name for pk, entity in zip(entity_pks, config.entities)
    }

    dim_date = tables.get("dim_date")
    if dim_date is None or dim_date.empty:
        raise ValueError(
            "entity_features: 'dim_date' is missing or empty; period "
            "indices cannot be derived without the date spine"
        )
    period_index_by_date_key: dict = {
        dk: idx for idx, dk in enumerate(dim_date["date_key"].tolist())
    }

    # M109: when holdout is enabled, restrict aggregation to the
    # training window so the per-entity feature row never sees a value
    # from a held-out period. The load-time validator has guaranteed
    # ``cutoff >= holdout.min_training_periods >= 1`` whenever
    # ``holdout.enabled`` is True, so the training set is always
    # non-empty — but a small training window can still produce NaN
    # ``slope`` values when fewer than two non-NaN cells survive MCAR
    # noise; that's the same NaN-aware path ``_aggregate_series``
    # handles for the unsplit case.
    training_cutoff: Optional[int] = None
    if config.holdout.enabled:
        # Local import keeps ``entity_features`` from depending on
        # ``holdout`` at module-load time (only the function call site
        # imports it, mirroring the lazy-import pattern between
        # ``config`` and ``validation``).
        from plotsim.holdout import cutoff_period_index

        training_cutoff = cutoff_period_index(config)

    target_metrics = _resolve_target_metrics(config)

    # Pre-seed every (entity, metric, suffix) cell with NaN so the
    # output has a rectangular shape regardless of which fact tables
    # produced rows for which entities. ``aggregates[pk]`` holds one
    # dict of column-name -> value per entity.
    aggregates: dict = {pk: {} for pk in entity_pks}
    metric_seen: dict[str, bool] = {m: False for m in target_metrics}

    for fact_tbl in (t for t in config.tables if t.type == "fact"):
        df = tables.get(fact_tbl.name)
        if df is None or df.empty:
            continue
        entity_fk_col = _entity_fk_column(fact_tbl, primary_dim.name)
        date_fk_col = _date_fk_column(fact_tbl)
        if entity_fk_col is None or date_fk_col is None:
            continue
        metric_cols: list[tuple[str, str]] = []
        for col in fact_tbl.columns:
            parsed = parse_source(col.source)
            if not isinstance(parsed, MetricSource):
                continue
            if col.dtype not in ("int", "float"):
                continue
            if parsed.metric not in metric_seen:
                continue
            if metric_seen[parsed.metric]:
                # First fact table that exposes this metric wins the
                # aggregate columns; later fact tables that re-emit the
                # same metric column are skipped to avoid silent
                # last-write-wins ambiguity.
                continue
            metric_cols.append((col.name, parsed.metric))
        if not metric_cols:
            continue

        for col_name, metric_name in metric_cols:
            metric_seen[metric_name] = True

        # Group once per fact table — every metric column on the table
        # shares the same (entity_fk, date_fk) layout.
        for pk, group in df.groupby(entity_fk_col, sort=False):
            if pk not in aggregates:
                continue
            # Build the (period_index, original_row_position) pairs in
            # one pass so the metric-value lookup below can index back
            # by row position even when ``training_cutoff`` drops some
            # rows from the entity's window.
            row_periods: list[tuple[int, int]] = []
            for row_pos, dk in enumerate(group[date_fk_col].tolist()):
                if dk not in period_index_by_date_key:
                    continue
                period_idx = period_index_by_date_key[dk]
                if (
                    training_cutoff is not None
                    and period_idx >= training_cutoff
                ):
                    continue
                row_periods.append((period_idx, row_pos))
            if len(row_periods) == 0:
                continue
            row_periods.sort(key=lambda pair: pair[0])
            periods_sorted = np.array(
                [p for p, _ in row_periods], dtype=np.int64,
            )
            row_positions = np.array(
                [r for _, r in row_periods], dtype=np.int64,
            )
            for col_name, metric_name in metric_cols:
                # ``na_value=np.nan`` is required: poisson metrics ride on
                # ``pd.Int64Dtype`` (the nullable integer extension dtype)
                # so ``pd.NA`` cells under MCAR noise won't go through a
                # plain ``to_numpy(dtype=float)`` without an explicit NaN
                # bridge.
                raw_values = pd.to_numeric(
                    group[col_name], errors="coerce",
                ).to_numpy(dtype=float, na_value=np.nan)
                values_sorted = raw_values[row_positions]
                stats = _aggregate_series(periods_sorted, values_sorted)
                for suffix in _AGGREGATE_SUFFIXES:
                    aggregates[pk][f"{metric_name}_{suffix}"] = stats[suffix]

    final_pos_by_name = _final_position_by_entity(manifest)
    archetype_by_name = {e.name: e.archetype for e in config.entities}

    column_order: list[str] = [pk_col]
    for metric_name in target_metrics:
        for suffix in _AGGREGATE_SUFFIXES:
            column_order.append(f"{metric_name}_{suffix}")
    if config.entity_features.include_labels:
        column_order.append("archetype")
        column_order.append("final_trajectory_position")

    rows: list[dict] = []
    nan = float("nan")
    for pk in entity_pks:
        row: dict = {pk_col: pk}
        for metric_name in target_metrics:
            for suffix in _AGGREGATE_SUFFIXES:
                key = f"{metric_name}_{suffix}"
                row[key] = aggregates[pk].get(key, nan)
        if config.entity_features.include_labels:
            entity_name = pk_to_entity_name[pk]
            row["archetype"] = archetype_by_name[entity_name]
            row["final_trajectory_position"] = final_pos_by_name.get(
                entity_name, nan,
            )
        rows.append(row)

    return pd.DataFrame(rows, columns=column_order)


__all__ = [
    "ENTITY_FEATURES_BASENAME",
    "build_entity_features",
]
