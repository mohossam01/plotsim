"""plotsim.tables — fact and event table orchestration (trajectory-first).

What it does:
    Composes Mission 003 (trajectory), Mission 004 (metrics), and Mission 005
    (dimensions) into a full table set. For each entity:

      1. The trajectory engine has already produced a length-n_periods
         position array.
      2. The metric generator turns that array into a per-metric series
         (correlated, noised, MCAR-aware).
      3. Each per_entity_per_period fact table assembles rows from those
         series with FKs resolved against the per_entity dim.

    Event tables consume completed fact values — never trajectories. This
    architectural firewall is enforced by the function signature of
    ``build_event_tables``: it accepts ``fact_tables`` only. If a future
    contributor wants events to read trajectories directly, they have to
    change the signature, which is the loud failure mode we want.

    Stage assignment walks each entity's driving metric forward through time
    and records the lifecycle stage on the fact table that owns the field.

Input:
    PlotsimConfig and a seeded ``numpy.random.Generator``. The orchestrator
    ``generate_tables`` builds dims and trajectories internally; the lower
    builders accept them explicitly so callers can swap pieces in tests.

Output:
    dict mapping table_name → pandas.DataFrame for every table declared in
    the config (dims + facts + events).

Mission-spec deviations (also flagged in the completion report):
    1. Event tables that declare neither a ``row_count_source`` nor a
       threshold-typed column emit an empty DataFrame with the configured
       schema. The mission spec only describes proportional and threshold
       events; HR's ``evt_attrition`` falls outside both, and emitting an
       empty table preserves the contract that every configured table is
       present in the output dict.
    2. ``generated:timestamp`` on event columns resolves to the period's
       anchor date (no faker), since the provider name is non-faker.
    3. Stage column name is hardcoded to ``stage``. Adding a configurable
       column name would be a schema change; M007 can introduce it if needed.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from faker import Faker

from plotsim.config import (
    BridgeMetric,
    BridgeTableConfig,
    Column,
    DerivedSource,
    FKSource,
    FakerSource,
    GeneratedSource,
    LagSource,
    MetricSource,
    PlotsimConfig,
    PKSource,
    ProportionalSource,
    SCDType2Config,
    StaticSource,
    Table,
    TextBucketSource,
    ThresholdSource,
    parse_source,
)
from plotsim.dimensions import (
    _call_faker,
    build_all_dimensions,
    sample_fk_values,
)
from plotsim.metrics import (
    _build_correlation_matrix,
    _toposort_metrics,
    generate_entity_metrics,
)
from plotsim.trajectory import compute_all_trajectories


# --- Helpers -----------------------------------------------------------------


def _make_faker(
    rng: np.random.Generator,
    locale: str | list[str] = "en_US",
) -> Faker:
    fake = Faker(locale)
    fake.seed_instance(int(rng.integers(0, 2**31 - 1)))
    return fake


def _per_entity_dim_names(config: PlotsimConfig) -> set[str]:
    return {
        t.name for t in config.tables
        if t.type == "dim" and t.grain == "per_entity"
    }


def _find_entity_fk_column(
    table: Table, per_entity_dims: set[str]
) -> Optional[tuple[str, str, str]]:
    """Return (local_col, parent_table, parent_pk) if table FKs into a per_entity dim."""
    for col in table.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource) and parsed.table in per_entity_dims:
            return col.name, parsed.table, parsed.column
    return None


def _find_date_fk_column(table: Table) -> Optional[tuple[str, str, str]]:
    """Return (local_col, parent_table, parent_pk) if the table FKs into dim_date."""
    for col in table.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource) and parsed.table == "dim_date":
            return col.name, parsed.table, parsed.column
    return None


def _id_pad(n: int) -> int:
    return max(3, len(str(max(n, 1))))


def _coerce_metric_value(value, dtype: str):
    """Cast a raw metric value to the column's declared dtype where sensible.

    None passes through (MCAR). Int columns round; bool is a thin coerce.
    Float/string/date/id are left as-is — the metric generator already
    delivered the right shape.
    """
    if value is None:
        return None
    if dtype == "int":
        return int(round(float(value)))
    if dtype == "boolean":
        return bool(value)
    return value


def _coerce_array_for_dtype(arr: np.ndarray, dtype: str):
    """Vectorized counterpart of ``_coerce_metric_value`` for whole-column arrays.

    F3 (M102): the vectorized fact-builder used to assign the raw float
    slice from ``metrics_3d`` straight into ``col_arrays`` for MetricSource
    and LagSource columns, ignoring the declared ``Column.dtype``. Library
    callers consuming ``generate_tables`` then saw float64 where they had
    declared int / boolean. The CSV path was rescued downstream by
    ``output._coerce_integer_columns``, but the in-memory dataframe was wrong.

    Returns a ``pd.api.extensions.ExtensionArray`` (Int64 / BooleanDtype) for
    int / boolean dtypes — preserves NaN as ``pd.NA`` and matches what
    ``output._coerce_integer_columns`` produces at write-time, so in-memory
    dtype now matches on-disk dtype after a CSV round-trip with
    ``dtype_backend='numpy_nullable'``. Other dtypes pass through unchanged
    (``np.ndarray``); the metric generator already delivered the right shape.
    """
    if dtype == "int":
        mask = np.isnan(arr)
        rounded = np.rint(np.where(mask, 0.0, arr)).astype(np.int64)
        result = pd.array(rounded, dtype="Int64")
        if mask.any():
            result[mask] = pd.NA
        return result
    if dtype == "boolean":
        mask = np.isnan(arr)
        bool_vals = np.where(mask, False, arr).astype(bool)
        result = pd.array(bool_vals, dtype="boolean")
        if mask.any():
            result[mask] = pd.NA
        return result
    return arr


# --- Fact tables -------------------------------------------------------------


def _build_seasonal_factors(
    config: PlotsimConfig, n_periods: int,
) -> Optional[np.ndarray]:
    """M119: pre-compute the per-period summed seasonal strength.

    Returns a length-``n_periods`` ``float64`` array where entry ``t`` is the
    sum of every ``SeasonalEffect.strength`` whose ``months`` set contains
    period ``t``'s calendar month. Returns ``None`` when no effects are
    configured — keeps the metrics pipeline byte-identical to pre-M119
    baselines (the metric loop short-circuits when ``seasonal_global == 0.0``).

    The returned array is a global (entity-independent) lookup; per-entity
    and per-metric sensitivities apply downstream in
    ``generate_metrics_for_period``.
    """
    if not config.seasonal_effects:
        return None
    months = config.time_window.period_calendar_months()
    if len(months) != n_periods:
        raise ValueError(
            f"seasonal factor length mismatch: time_window yields "
            f"{len(months)} periods but dim_date has {n_periods}"
        )
    factors = np.zeros(n_periods, dtype=np.float64)
    for effect in config.seasonal_effects:
        month_set = set(effect.months)
        for t, m in enumerate(months):
            if m in month_set:
                factors[t] += effect.strength
    return factors


def _compute_entity_metrics(
    config: PlotsimConfig,
    trajectories: dict[str, np.ndarray],
    n_periods: int,
    rng: np.random.Generator,
    cholesky_L: Optional[np.ndarray] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate the per-entity metric series dict the engine reads from.

    Hoisted out of ``build_fact_tables`` so M107's bridge generator can
    read the same series without re-running ``generate_entity_metrics``
    (which would consume RNG twice and break determinism). Each entity's
    RNG draws share the top-level rng so output is identical to the prior
    pre-M107 inline path when only fact-building consumes the result.

    M119: when ``config.seasonal_effects`` is non-empty, computes a global
    per-period strength array once and threads it (along with each
    entity's ``seasonal_sensitivity``) into ``generate_entity_metrics``.
    Empty effects → ``seasonal_factors=None`` → byte-identical to pre-M119.
    """
    arch_by_name = {a.name: a for a in config.archetypes}
    seasonal_factors = _build_seasonal_factors(config, n_periods)
    entity_metrics: dict[str, dict[str, np.ndarray]] = {}
    for entity in config.entities:
        traj = trajectories[entity.name]
        if len(traj) != n_periods:
            raise ValueError(
                f"trajectory for entity {entity.name!r} has length {len(traj)} "
                f"but dim_date has {n_periods} periods"
            )
        entity_metrics[entity.name] = generate_entity_metrics(
            traj,
            list(config.metrics),
            list(config.correlations),
            config.noise,
            rng,
            archetype=arch_by_name.get(entity.archetype),
            cholesky_L=cholesky_L,
            seasonal_factors=seasonal_factors,
            entity_seasonal_sensitivity=entity.seasonal_sensitivity,
        )
    return entity_metrics


def build_fact_tables(
    config: PlotsimConfig,
    trajectories: dict[str, np.ndarray],
    dim_tables: dict[str, pd.DataFrame],
    rng: np.random.Generator,
    cholesky_L: Optional[np.ndarray] = None,
    entity_metrics: Optional[dict[str, dict[str, np.ndarray]]] = None,
) -> dict[str, pd.DataFrame]:
    """Generate every fact table in ``config.tables`` keyed by table name.

    For each entity we generate the full metric series ONCE
    (``generate_entity_metrics`` handles correlations, causal lag, noise,
    MCAR), and then any number of per_entity_per_period fact tables read
    columns out of that single dict. This keeps the trajectory-first
    invariant intact across multiple fact tables on the same entity.

    Per-period grain (no entity axis) is also supported: rows are emitted
    per period using a single shared metric series produced from the mean
    trajectory across entities. (None of the sample configs exercise this
    today; included for completeness.)

    M107: ``entity_metrics`` may be passed in by callers that have already
    computed it (the orchestrator does this so bridge tables can share
    the same series without re-running ``generate_entity_metrics`` and
    burning RNG draws). When ``None``, the helper recomputes it inline —
    preserves the pre-M107 single-call signature for direct test callers.
    """
    dim_date = dim_tables.get("dim_date")
    if dim_date is None:
        raise ValueError("build_fact_tables requires dim_date to be built")
    n_periods = len(dim_date)

    per_entity_dims = _per_entity_dim_names(config)

    if entity_metrics is None:
        entity_metrics = _compute_entity_metrics(
            config, trajectories, n_periods, rng, cholesky_L=cholesky_L,
        )

    # Category B Layer 4: materialize metric series as a dense (E, P, M) float64
    # ndarray keyed by config.entities order (not sorted name order — row order
    # in the fact tables must match the config entity iteration order).
    # Null values (MCAR / poisson-with-MCAR) become ``np.nan``. Downstream
    # column builders index into this array instead of dict-of-dict lookups.
    metrics_3d = _build_metrics_3d(config, entity_metrics, n_periods)

    # M105: pack trajectories as a (E, P) float64 array keyed by
    # ``config.entities`` order, mirroring ``metrics_3d``'s entity axis.
    # ``TextBucketSource`` and any future trajectory-position-driven source
    # type reads from this array rather than re-deriving positions from the
    # dict-of-arrays form. Building it here keeps the per_period_fact
    # branch's "no trajectory axis" contract clean — only
    # per_entity_per_period builders see ``trajectories_2d``.
    trajectories_2d = _build_trajectories_2d(config, trajectories, n_periods)

    fact_out: dict[str, pd.DataFrame] = {}
    fake = _make_faker(rng, config.locale)

    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        if tbl.grain == "per_entity_per_period":
            fact_out[tbl.name] = _build_per_entity_per_period_fact(
                tbl, config, entity_metrics, dim_tables, per_entity_dims,
                fake, rng, metrics_3d, trajectories_2d,
            )
        elif tbl.grain == "per_period":
            fact_out[tbl.name] = _build_per_period_fact(
                tbl, config, entity_metrics, dim_tables, fake, metrics_3d,
            )
        else:
            raise ValueError(
                f"fact table {tbl.name!r} has unsupported grain "
                f"{tbl.grain!r}; expected per_entity_per_period or per_period"
            )

    return fact_out


def _build_trajectories_2d(
    config: PlotsimConfig,
    trajectories: dict[str, np.ndarray],
    n_periods: int,
) -> np.ndarray:
    """Pack per-entity trajectory arrays into a (E, P) float64 ndarray.

    Entity axis follows ``config.entities`` order — matches ``metrics_3d``
    so a row at flat index ``i*P + p`` in the vectorized fact builder
    reads from ``trajectories_2d[i, p]`` and ``metrics_3d[i, p, :]``
    consistently.
    """
    n_entities = len(config.entities)
    out = np.empty((n_entities, n_periods), dtype=np.float64)
    for i, entity in enumerate(config.entities):
        traj = trajectories[entity.name]
        if len(traj) != n_periods:
            raise ValueError(
                f"trajectory for entity {entity.name!r} has length "
                f"{len(traj)} but dim_date has {n_periods} periods"
            )
        out[i, :] = np.asarray(traj, dtype=np.float64)
    return out


def _build_metrics_3d(
    config: PlotsimConfig,
    entity_metrics: dict[str, dict[str, np.ndarray]],
    n_periods: int,
) -> np.ndarray:
    """Pack per-entity per-metric series into a float64 ndarray shaped (E, P, M).

    - E axis: indexed by ``config.entities`` order (preserves fact-row order).
    - P axis: time.
    - M axis: indexed by ``config.metrics`` order.

    Null cells (MCAR, or poisson-with-MCAR) become ``np.nan``. Integer-typed
    metrics pass through as float64 (the nullable ``Int64`` conversion happens
    at CSV-write time in ``plotsim.output``, so the fact DataFrame stays
    homogeneous float64 here).
    """
    entity_names_ordered = [e.name for e in config.entities]
    n_entities = len(entity_names_ordered)
    n_metrics = len(config.metrics)
    out = np.full((n_entities, n_periods, n_metrics), np.nan, dtype=np.float64)
    for i, ename in enumerate(entity_names_ordered):
        per_metric = entity_metrics[ename]
        for j, m in enumerate(config.metrics):
            arr = per_metric[m.name]
            if arr.dtype == object:
                # Object array with possible None values — explicit walk so
                # None becomes NaN in the float slot.
                for p in range(n_periods):
                    v = arr[p]
                    if v is None:
                        out[i, p, j] = np.nan
                    elif isinstance(v, float) and np.isnan(v):
                        out[i, p, j] = np.nan
                    else:
                        out[i, p, j] = float(v)
            else:
                out[i, :, j] = arr.astype(np.float64)
    return out


def _build_per_entity_per_period_fact(
    tbl: Table,
    config: PlotsimConfig,
    entity_metrics: dict[str, dict[str, np.ndarray]],
    dim_tables: dict[str, pd.DataFrame],
    per_entity_dims: set[str],
    fake: Faker,
    rng: np.random.Generator,
    metrics_3d: Optional[np.ndarray] = None,
    trajectories_2d: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build one per_entity_per_period fact table.

    Category B Layer 4 took this from list-of-dict + per-cell Python calls to
    column-oriented numpy construction. Shape invariant preserved: rows are
    in ``config.entities`` × ``range(n_periods)`` order (entity-major). The
    scalar path is retained as a fallback for columns whose resolution either
    consumes the RNG at fact-build time (``FakerSource``) or needs Python
    ``bool()`` coercion on a metric value.
    """
    entity_fk = _find_entity_fk_column(tbl, per_entity_dims)
    if entity_fk is None:
        raise ValueError(
            f"fact table {tbl.name!r} has grain per_entity_per_period but no "
            f"FK column to a per_entity dim; known per_entity dims: "
            f"{sorted(per_entity_dims)}"
        )
    local_entity_col, parent_entity_table, parent_entity_pk = entity_fk
    parent_entity_dim = dim_tables[parent_entity_table]
    # M106: SCD-expanded per_entity dims hold N × versions rows, with the
    # entity business key repeated across versions. Collapse to a canonical
    # one-row-per-entity view (first version per entity) for PK lookup —
    # ``expand_scd_dims`` iterates entities in ``config.entities`` order, so
    # the deduplicated frame preserves config-entity ordering.
    parent_entity_dim = parent_entity_dim.drop_duplicates(
        subset=[parent_entity_pk], keep="first",
    ).reset_index(drop=True)
    if len(parent_entity_dim) != len(config.entities):
        raise ValueError(
            f"parent dim {parent_entity_table!r} has {len(parent_entity_dim)} "
            f"unique {parent_entity_pk!r} value(s) but config has "
            f"{len(config.entities)} entities; per_entity dims must be 1:1 "
            f"with config.entities (SCD-expanded dims are deduplicated by PK)"
        )

    date_fk = _find_date_fk_column(tbl)
    if date_fk is None:
        raise ValueError(
            f"fact table {tbl.name!r} has grain per_entity_per_period but no "
            f"FK column to dim_date"
        )
    local_date_col, _, parent_date_pk = date_fk
    dim_date = dim_tables["dim_date"]
    n_periods = len(dim_date)

    # Hoist parse_source() out of every loop — every column's source string is
    # parsed exactly once.
    parsed_cols = [(col, parse_source(col.source)) for col in tbl.columns]

    # FIX-04: precompute cross-dim FK assignments per entity. Each entity's
    # plan_id (or any other cross-dim FK) is drawn ONCE and broadcast across
    # all periods — facts are time-series under fixed entity attributes.
    # Resolution: Entity.cross_dim_fks pin → Column.distribution.weights
    # → uniform → single PK if parent is 1-row. This block always runs in
    # config.entities order so RNG consumption is preserved irrespective of
    # the downstream path (vectorized or scalar fallback).
    other_fks: dict[str, tuple[Column, str, str]] = {}
    for col, parsed in parsed_cols:
        if not isinstance(parsed, FKSource):
            continue
        if col.name in (local_entity_col, local_date_col):
            continue
        other_fks[col.name] = (col, parsed.table, parsed.column)

    entity_cross_fks: dict[str, dict[str, Any]] = {}
    for entity in config.entities:
        per_entity_assignments: dict[str, Any] = {}
        for col_name, (col_cfg, parent_table, parent_pk_col) in other_fks.items():
            parent_df = dim_tables.get(parent_table)
            if parent_df is None or parent_df.empty:
                per_entity_assignments[col_name] = None
                continue
            parent_pks = parent_df[parent_pk_col].tolist()
            anchored = entity.cross_dim_fks.get(col_name)
            if anchored is not None and anchored not in parent_pks:
                raise ValueError(
                    f"entity {entity.name!r} cross_dim_fks pins "
                    f"{col_name!r}={anchored!r}, not in parent "
                    f"{parent_table!r} PKs {parent_pks}"
                )
            per_entity_assignments[col_name] = sample_fk_values(
                col_cfg, parent_pks, 1, rng, anchored_value=anchored,
            )[0]
        entity_cross_fks[entity.name] = per_entity_assignments

    # Fallback path: any column whose resolution consumes RNG at fact-build
    # time forces the scalar per-row loop so call order is preserved. No
    # shipped template exercises this branch.
    # F3 (M102): boolean MetricSource / LagSource columns no longer force the
    # scalar fallback — `_coerce_array_for_dtype` handles them correctly in
    # the vectorized path.
    has_faker = any(isinstance(p, FakerSource) for _, p in parsed_cols)
    if has_faker or metrics_3d is None:
        return _scalar_per_entity_per_period_fact(
            tbl, config, entity_metrics, dim_date, n_periods,
            parent_entity_dim, parent_entity_pk,
            local_entity_col, local_date_col, parent_date_pk,
            entity_cross_fks, fake, parsed_cols,
            trajectories_2d=trajectories_2d,
        )

    return _vectorized_per_entity_per_period_fact(
        tbl, config, dim_date, n_periods,
        parent_entity_dim, parent_entity_pk,
        local_entity_col, local_date_col, parent_date_pk,
        entity_cross_fks, parsed_cols, metrics_3d,
        trajectories_2d=trajectories_2d,
    )


def _scalar_per_entity_per_period_fact(
    tbl: Table,
    config: PlotsimConfig,
    entity_metrics: dict[str, dict[str, np.ndarray]],
    dim_date: pd.DataFrame,
    n_periods: int,
    parent_entity_dim: pd.DataFrame,
    parent_entity_pk: str,
    local_entity_col: str,
    local_date_col: str,
    parent_date_pk: str,
    entity_cross_fks: dict[str, dict[str, Any]],
    fake: Faker,
    parsed_cols: list[tuple[Column, Any]],
    trajectories_2d: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Pre-Layer-4 row-by-row builder, kept as a fallback for fact tables that
    use ``FakerSource`` or ``boolean``-typed metric columns (paths where
    vectorization would reorder RNG draws or drop a Python coercion).

    M105: ``trajectories_2d`` (E, P) is forwarded to ``_resolve_fact_cell``
    one entity-row at a time so ``TextBucketSource`` columns can read the
    position. Pre-M105 callers that didn't supply trajectories pass ``None``
    here; ``_resolve_fact_cell`` only consults the array when it dispatches
    on a ``TextBucketSource``, so non-bucket fact tables continue to work.
    """
    del parsed_cols  # not used here; scalar path walks tbl.columns directly
    rows: list[dict] = []
    for entity_idx, entity in enumerate(config.entities):
        entity_pk_value = parent_entity_dim.iloc[entity_idx][parent_entity_pk]
        metric_series = entity_metrics[entity.name]
        cross_fks_for_entity = entity_cross_fks[entity.name]
        traj_for_entity = (
            trajectories_2d[entity_idx] if trajectories_2d is not None else None
        )
        for period_idx in range(n_periods):
            row: dict = {}
            for col in tbl.columns:
                row[col.name] = _resolve_fact_cell(
                    col, period_idx, entity_pk_value,
                    local_entity_col, local_date_col, parent_date_pk,
                    metric_series, dim_date, cross_fks_for_entity, fake,
                    trajectory_for_entity=traj_for_entity,
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=[c.name for c in tbl.columns])


def _vectorized_per_entity_per_period_fact(
    tbl: Table,
    config: PlotsimConfig,
    dim_date: pd.DataFrame,
    n_periods: int,
    parent_entity_dim: pd.DataFrame,
    parent_entity_pk: str,
    local_entity_col: str,
    local_date_col: str,
    parent_date_pk: str,
    entity_cross_fks: dict[str, dict[str, Any]],
    parsed_cols: list[tuple[Column, Any]],
    metrics_3d: np.ndarray,
    trajectories_2d: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Column-oriented builder. Each column becomes a single ndarray built via
    slicing / repeating / tiling on shared broadcast axes, then all columns
    are assembled into one ``pd.DataFrame({name: ndarray, ...})`` — skipping
    the ~3× memory spike that ``pd.DataFrame(list[dict])`` creates.
    """
    n_entities = len(config.entities)
    total_rows = n_entities * n_periods

    # Entity PK values, aligned 1:1 with config.entities iteration order.
    entity_pks = np.asarray(
        [parent_entity_dim.iloc[i][parent_entity_pk] for i in range(n_entities)],
        dtype=object,
    )
    entity_pk_repeated = np.repeat(entity_pks, n_periods)

    # Date-FK values — one per period, tiled across entities.
    date_keys = dim_date[parent_date_pk].to_numpy()
    date_key_tiled = np.tile(date_keys, n_entities)

    metric_name_to_idx = {m.name: i for i, m in enumerate(config.metrics)}

    # Period index per row — for DerivedSource.period_index and PKSource.
    period_idx_col = np.tile(np.arange(n_periods, dtype=np.int64), n_entities)

    col_arrays: dict[str, np.ndarray] = {}
    for col, parsed in parsed_cols:
        if isinstance(parsed, FKSource):
            if col.name == local_entity_col:
                col_arrays[col.name] = entity_pk_repeated
            elif col.name == local_date_col:
                col_arrays[col.name] = date_key_tiled
            else:
                # Cross-dim FK precomputed once per entity.
                cross_vals = np.asarray(
                    [entity_cross_fks[e.name].get(col.name) for e in config.entities],
                    dtype=object,
                )
                col_arrays[col.name] = np.repeat(cross_vals, n_periods)
        elif isinstance(parsed, MetricSource):
            if parsed.metric not in metric_name_to_idx:
                raise ValueError(
                    f"fact column {col.name!r} references metric "
                    f"{parsed.metric!r}, which was not generated; check config.metrics"
                )
            m_idx = metric_name_to_idx[parsed.metric]
            arr = metrics_3d[:, :, m_idx].ravel(order="C").copy()
            col_arrays[col.name] = _coerce_array_for_dtype(arr, col.dtype)
        elif isinstance(parsed, LagSource):
            if parsed.metric not in metric_name_to_idx:
                col_arrays[col.name] = _coerce_array_for_dtype(
                    np.full(total_rows, np.nan, dtype=np.float64), col.dtype,
                )
            else:
                m_idx = metric_name_to_idx[parsed.metric]
                n = parsed.periods
                base = np.arange(n_periods, dtype=np.int64)
                target_idx = base - n
                # "If history too short, fall back to current period" — scalar
                # semantics preserved by mapping out-of-range to the current
                # period index.
                target_idx = np.where(target_idx < 0, base, target_idx)
                sliced = metrics_3d[:, target_idx, m_idx]  # (E, P)
                col_arrays[col.name] = _coerce_array_for_dtype(
                    sliced.ravel(order="C").copy(), col.dtype,
                )
        elif isinstance(parsed, PKSource):
            # f"{col.name}-{period_idx:04d}-{entity_pk_value}" — build once.
            pk_rows = [
                f"{col.name}-{p:04d}-{entity_pks[i]}"
                for i in range(n_entities)
                for p in range(n_periods)
            ]
            col_arrays[col.name] = np.asarray(pk_rows, dtype=object)
        elif isinstance(parsed, GeneratedSource):
            provider = parsed.provider
            if provider == "timestamp":
                dates = dim_date["date"].tolist()
                promoted: list[Any] = []
                for d in dates:
                    if isinstance(d, _dt.date) and not isinstance(d, _dt.datetime):
                        promoted.append(_dt.datetime(d.year, d.month, d.day))
                    else:
                        promoted.append(d)
                promoted_arr = np.asarray(promoted, dtype=object)
                col_arrays[col.name] = np.tile(promoted_arr, n_entities)
            elif provider == "date_key":
                col_arrays[col.name] = np.tile(
                    dim_date["date_key"].to_numpy(), n_entities,
                )
            elif provider == "period_label":
                col_arrays[col.name] = np.tile(
                    dim_date["period_label"].to_numpy(), n_entities,
                )
            else:
                raise ValueError(
                    f"unsupported generated provider {provider!r} on fact/event tables"
                )
        elif isinstance(parsed, StaticSource):
            col_arrays[col.name] = np.full(total_rows, parsed.value, dtype=object)
        elif isinstance(parsed, DerivedSource):
            if parsed.field == "period_index":
                col_arrays[col.name] = period_idx_col.copy()
            elif parsed.field == "entity_id":
                col_arrays[col.name] = entity_pk_repeated
            else:
                raise ValueError(
                    f"fact column {col.name!r} derived field "
                    f"{parsed.field!r} not supported"
                )
        elif isinstance(parsed, TextBucketSource):
            # M105: trajectory-position-driven text emission. ``trajectories_2d``
            # is shape (E, P); flatten in the same row-major (entity, period)
            # order the entity_pk_repeated / date_key_tiled axes use, then map
            # each position into a bucket index. ``min(int(p * N), N - 1)``
            # closes the [0, 1] interval at the top so position == 1.0 lands
            # in the last bucket rather than overflowing.
            if trajectories_2d is None:
                raise ValueError(
                    f"fact column {col.name!r} declares text-bucket source "
                    f"{col.source!r} but trajectories_2d was not threaded into "
                    f"the vectorized fact builder; this is an internal wiring "
                    f"bug, not a config error"
                )
            n_buckets = len(parsed.buckets)
            flat_positions = trajectories_2d.ravel(order="C")
            indices = np.minimum(
                (flat_positions * n_buckets).astype(np.int64),
                n_buckets - 1,
            )
            indices = np.maximum(indices, 0)
            bucket_arr = np.asarray(parsed.buckets, dtype=object)
            col_arrays[col.name] = bucket_arr[indices]
        else:
            raise ValueError(
                f"fact column {col.name!r} source {col.source!r} is not "
                f"supported on per_entity_per_period fact tables"
            )

    return pd.DataFrame({col.name: col_arrays[col.name] for col, _ in parsed_cols})


def _resolve_fact_cell(
    col: Column,
    period_idx: int,
    entity_pk_value,
    local_entity_col: str,
    local_date_col: str,
    parent_date_pk: str,
    metric_series: dict[str, np.ndarray],
    dim_date: pd.DataFrame,
    cross_fks_for_entity: dict[str, Any],
    fake: Faker,
    trajectory_for_entity: Optional[np.ndarray] = None,
):
    parsed = parse_source(col.source)
    if isinstance(parsed, FKSource):
        if col.name == local_entity_col:
            return entity_pk_value
        if col.name == local_date_col:
            return dim_date.iloc[period_idx][parent_date_pk]
        # Cross-dim FK (e.g. plan_id) — value precomputed once per entity by
        # _build_per_entity_per_period_fact (FIX-04). Same value broadcast
        # across all periods for this entity.
        return cross_fks_for_entity.get(col.name)
    if isinstance(parsed, MetricSource):
        series = metric_series.get(parsed.metric)
        if series is None:
            raise ValueError(
                f"fact column {col.name!r} references metric "
                f"{parsed.metric!r}, which was not generated; check config.metrics"
            )
        value = series[period_idx]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return _coerce_metric_value(value, col.dtype)
    if isinstance(parsed, LagSource):
        # Lag-typed columns read N periods back from the same entity series.
        # If history is too short, fall back to the current period.
        series = metric_series.get(parsed.metric)
        if series is None:
            return None
        target_idx = period_idx - parsed.periods
        if target_idx < 0:
            target_idx = period_idx
        value = series[target_idx]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return _coerce_metric_value(value, col.dtype)
    if isinstance(parsed, PKSource):
        # Single-column PK on a composite-grain table is a surrogate;
        # build it deterministically from period and entity.
        return f"{col.name}-{period_idx:04d}-{entity_pk_value}"
    if isinstance(parsed, GeneratedSource):
        return _resolve_generated(parsed.provider, period_idx, dim_date, fake)
    if isinstance(parsed, FakerSource):
        return _call_faker(fake, parsed.method, parsed.kwargs)
    if isinstance(parsed, StaticSource):
        return parsed.value
    if isinstance(parsed, DerivedSource):
        if parsed.field == "period_index":
            return period_idx
        if parsed.field == "entity_id":
            return entity_pk_value
        raise ValueError(
            f"fact column {col.name!r} derived field {parsed.field!r} not supported"
        )
    if isinstance(parsed, TextBucketSource):
        # M105: scalar-fallback bucket lookup. Same index arithmetic as the
        # vectorized branch — ``min(int(p * N), N - 1)`` so p == 1.0 lands
        # in the last bucket rather than overflowing.
        if trajectory_for_entity is None:
            raise ValueError(
                f"fact column {col.name!r} declares text-bucket source "
                f"{col.source!r} but trajectory_for_entity was not threaded "
                f"into the scalar fact builder; this is an internal wiring "
                f"bug, not a config error"
            )
        position = float(trajectory_for_entity[period_idx])
        n_buckets = len(parsed.buckets)
        idx = min(int(position * n_buckets), n_buckets - 1)
        idx = max(idx, 0)
        return parsed.buckets[idx]
    raise ValueError(
        f"fact column {col.name!r} source {col.source!r} is not supported on "
        f"per_entity_per_period fact tables"
    )


def _build_per_period_fact(
    tbl: Table,
    config: PlotsimConfig,
    entity_metrics: dict[str, dict[str, np.ndarray]],
    dim_tables: dict[str, pd.DataFrame],
    fake: Faker,
    metrics_3d: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Per-period fact (no entity axis) — uses the cross-entity mean series.

    Category B Layer 4: the per-metric mean is now a single
    ``np.nanmean(metrics_3d, axis=0)`` call (shape ``(P, M)``) that replaces
    the prior per-metric ``np.vstack`` + list comprehension. None of the
    shipped templates declare a per-period fact; the code path is kept for
    library users that compose one.
    """
    dim_date = dim_tables["dim_date"]
    n_periods = len(dim_date)

    # Build (P, M) mean array — prefer metrics_3d if caller supplied one.
    if metrics_3d is None:
        # Scalar-path fallback — re-materialize the (P, M) mean without a
        # prebuilt 3D array so this helper still works when called directly
        # from tests.
        metrics_3d = _build_metrics_3d(config, entity_metrics, n_periods)
    period_mean = np.nanmean(metrics_3d, axis=0)  # shape: (P, M)
    metric_idx_by_name = {m.name: i for i, m in enumerate(config.metrics)}

    date_fk = _find_date_fk_column(tbl)
    rows: list[dict] = []
    for period_idx in range(n_periods):
        row: dict = {}
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if isinstance(parsed, FKSource) and date_fk and col.name == date_fk[0]:
                row[col.name] = dim_date.iloc[period_idx][date_fk[2]]
            elif isinstance(parsed, MetricSource):
                if parsed.metric not in metric_idx_by_name:
                    row[col.name] = None
                    continue
                m_idx = metric_idx_by_name[parsed.metric]
                raw_val = period_mean[period_idx, m_idx]
                if np.isnan(raw_val):
                    row[col.name] = None
                else:
                    row[col.name] = _coerce_metric_value(float(raw_val), col.dtype)
            elif isinstance(parsed, PKSource):
                row[col.name] = f"{col.name}-{period_idx:04d}"
            elif isinstance(parsed, GeneratedSource):
                row[col.name] = _resolve_generated(parsed.provider, period_idx, dim_date, fake)
            elif isinstance(parsed, FakerSource):
                row[col.name] = _call_faker(fake, parsed.method, parsed.kwargs)
            elif isinstance(parsed, StaticSource):
                row[col.name] = parsed.value
            else:
                # F14 (M102): explicit raise instead of silent ``None`` fill.
                # Mission 100 named this ladder as a silent-dispatch site;
                # an unhandled source type on a per-period fact column
                # produced a column of None values with no signal to the
                # user. The companion ``_resolve_fact_cell`` at
                # ``tables.py:574`` already raises on this class.
                raise TypeError(
                    f"per-period fact column {col.name!r} source "
                    f"{col.source!r} resolves to {type(parsed).__name__}, "
                    f"which is not supported on per_period fact tables. "
                    f"Use metric:, fk:dim_date.*, generated:, faker:, "
                    f"static:, or pk: sources."
                )
        rows.append(row)
    return pd.DataFrame(rows, columns=[c.name for c in tbl.columns])


def _resolve_generated(provider: str, period_idx: int, dim_date: pd.DataFrame, fake: Faker):
    """Resolve a non-faker ``generated:<provider>`` cell.

    Recognised providers: ``timestamp``, ``date_key``, ``period_label``.
    Faker providers parse as :class:`FakerSource` and are dispatched
    separately via :func:`_call_faker` — callers shouldn't reach here
    for a faker source.

    ``fake`` is retained in the signature for callers that still pass it
    (it's harmless here).
    """
    del fake
    if provider == "timestamp":
        d = dim_date.iloc[period_idx]["date"]
        if isinstance(d, _dt.date) and not isinstance(d, _dt.datetime):
            return _dt.datetime(d.year, d.month, d.day)
        return d
    if provider == "date_key":
        return dim_date.iloc[period_idx]["date_key"]
    if provider == "period_label":
        return dim_date.iloc[period_idx]["period_label"]
    raise ValueError(
        f"unsupported generated provider {provider!r} on fact/event tables"
    )


# --- Event tables ------------------------------------------------------------


def build_event_tables(
    config: PlotsimConfig,
    fact_tables: dict[str, pd.DataFrame],
    dim_tables: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Derive event tables from completed fact values.

    Two event mechanisms are supported:

      * proportional via ``Table.row_count_source`` —
        ``proportional:<metric>:scale:<x>``. For each entity at each period,
        emit ``round(metric_value * scale)`` rows.
      * threshold via a column whose source is
        ``threshold:<metric>:<above|below>:<value>:for:<consecutive>``. Emit
        a single row at the period where the streak first reaches
        ``consecutive``; do not duplicate for the same entity afterwards.

    Tables matching neither mechanism (e.g. HR's ``evt_attrition`` which
    declares no driver) emit an empty DataFrame with the configured columns.
    """
    per_entity_dims = _per_entity_dim_names(config)
    fake = _make_faker(rng, config.locale)
    out: dict[str, pd.DataFrame] = {}

    for tbl in config.tables:
        if tbl.type != "event":
            continue
        if tbl.row_count_source is not None:
            parsed_rc = parse_source(tbl.row_count_source)
            if isinstance(parsed_rc, ProportionalSource):
                out[tbl.name] = _build_proportional_event(
                    tbl, parsed_rc, config, fact_tables, dim_tables,
                    per_entity_dims, fake, rng,
                )
                continue
        threshold_col = _find_threshold_column(tbl)
        if threshold_col is not None:
            out[tbl.name] = _build_threshold_event(
                tbl, threshold_col, config, fact_tables, dim_tables,
                per_entity_dims, fake, rng,
            )
            continue
        out[tbl.name] = pd.DataFrame(columns=[c.name for c in tbl.columns])

    return out


def _find_threshold_column(tbl: Table) -> Optional[tuple[Column, ThresholdSource]]:
    for col in tbl.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, ThresholdSource):
            return col, parsed
    return None


def _find_metric_column_in_facts(
    metric: str, fact_tables: dict[str, pd.DataFrame], config: PlotsimConfig,
) -> Optional[tuple[str, str]]:
    """Return (fact_table_name, column_name) for the metric's first appearance."""
    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        df = fact_tables.get(tbl.name)
        if df is None:
            continue
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if isinstance(parsed, MetricSource) and parsed.metric == metric:
                return tbl.name, col.name
    return None


def _entity_groups(
    fact_df: pd.DataFrame,
    fact_table: Table,
    per_entity_dims: set[str],
) -> tuple[str, list[tuple[object, pd.DataFrame]]]:
    """Group a fact table by its per_entity FK column, preserving entity order.

    FIX-07 / SF-5: vectorized via ``groupby(sort=False)`` which iterates
    groups in first-appearance order — the same ordering the prior
    ``iterrows`` implementation produced. Each returned ``group`` is the
    original DataFrame slice (views, not row-stacked copies), so callers
    that iterate ``group.iterrows()`` see identical rows in identical
    order.
    """
    fk = _find_entity_fk_column(fact_table, per_entity_dims)
    if fk is None:
        raise ValueError(
            f"fact table {fact_table.name!r} has no per_entity FK; cannot group by entity"
        )
    entity_col = fk[0]
    grouped = [(eid, group) for eid, group in fact_df.groupby(entity_col, sort=False)]
    return entity_col, grouped


def _build_proportional_event(
    tbl: Table,
    rc: ProportionalSource,
    config: PlotsimConfig,
    fact_tables: dict[str, pd.DataFrame],
    dim_tables: dict[str, pd.DataFrame],
    per_entity_dims: set[str],
    fake: Faker,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a proportional event table with hybrid vectorization.

    Category B Layer 5 splits event columns into two categories:

      * **Deterministic** (no RNG draws): PK, FK to ``dim_date``, FK to a
        ``per_entity`` dim, ``ThresholdSource`` cells (always ``None`` in
        the proportional path), ``GeneratedSource``, ``StaticSource``,
        ``DerivedSource``. These are materialized once as numpy arrays via
        ``np.repeat`` / ``np.tile`` + index lookups.
      * **Stochastic** (consumes RNG or advances faker state): ``FakerSource``
        and sub-entity/reference ``FKSource``. These still go through a
        per-row scalar loop in the same cell order as the pre-Layer-5 path,
        so RNG consumption order is preserved byte-for-byte.

    When an event table has zero stochastic columns (e.g. a pure counts
    export), the per-row loop is skipped entirely.
    """
    located = _find_metric_column_in_facts(rc.metric, fact_tables, config)
    if located is None:
        raise ValueError(
            f"event table {tbl.name!r} row_count_source references metric "
            f"{rc.metric!r} but no fact table exposes that metric"
        )
    fact_name, metric_col = located
    fact_table_cfg = next(t for t in config.tables if t.name == fact_name)
    fact_df = fact_tables[fact_name]

    fact_entity_fk = _find_entity_fk_column(fact_table_cfg, per_entity_dims)
    if fact_entity_fk is None:
        raise ValueError(
            f"fact {fact_name!r} has no per_entity FK; cannot group events"
        )
    fact_entity_col_name = fact_entity_fk[0]

    fact_date_col = _find_date_fk_column(fact_table_cfg)
    if fact_date_col is None:
        raise ValueError(
            f"fact {fact_name!r} has no dim_date FK; cannot derive event dates"
        )
    fact_date_col_name = fact_date_col[0]

    dim_date = dim_tables["dim_date"]
    parsed_cols = [(col, parse_source(col.source)) for col in tbl.columns]

    # Per-cell vectors — fact rows are already in entity-major order from the
    # Layer 4 builder (or the scalar fallback, which also emits in that
    # order), so column-reading the fact DataFrame preserves the exact cell
    # order that the pre-Layer-5 groupby-then-iterrows walk used.
    entity_ids_arr = fact_df[fact_entity_col_name].to_numpy()
    date_keys_arr = fact_df[fact_date_col_name].to_numpy()
    # Coerce to float64 with NaN for nulls — handles both vectorized fact
    # output (float64) and scalar-fallback output (object with None).
    values_arr = pd.to_numeric(fact_df[metric_col], errors="coerce").to_numpy()

    # NaN-as-null cells contribute zero rows. Replace NaN before the cast so
    # np.int64 doesn't emit a garbage-value RuntimeWarning on the NaN lane.
    values_nonan = np.where(np.isnan(values_arr), 0.0, values_arr)
    counts = np.rint(values_nonan * rc.scale).astype(np.int64)
    counts = np.maximum(counts, 0)
    total_rows = int(counts.sum())

    if total_rows == 0:
        return pd.DataFrame(columns=[c.name for c in tbl.columns])

    event_entity_ids = np.repeat(entity_ids_arr, counts)
    event_date_keys = np.repeat(date_keys_arr, counts)

    # Map each event row to its period index in dim_date so GeneratedSource
    # lookups have an O(1) path. Scalar path falls back to index 0 when the
    # date_key isn't found — mirror that exactly.
    date_key_idx = {k: i for i, k in enumerate(dim_date["date_key"].tolist())}
    event_date_idx = np.fromiter(
        (date_key_idx.get(k, 0) for k in event_date_keys),
        dtype=np.int64, count=total_rows,
    )

    # PK serial: scalar path writes pk_counter+1 starting from 0 → 1..total_rows.
    pk_prefix = tbl.name[4:] if tbl.name.startswith("evt_") else tbl.name
    pk_width = _id_pad(10_000)
    pk_first_char = pk_prefix[0]

    # Classify columns. Any FK that points to a non-per_entity, non-dim_date
    # table may draw RNG; FakerSource always advances faker state.
    def _is_stochastic(col: Column, parsed: Any) -> bool:
        if isinstance(parsed, FakerSource):
            return True
        if isinstance(parsed, FKSource):
            return parsed.table not in per_entity_dims and parsed.table != "dim_date"
        return False

    stochastic_cols = {
        col.name for col, parsed in parsed_cols if _is_stochastic(col, parsed)
    }

    col_arrays: dict[str, np.ndarray] = {}
    for col, parsed in parsed_cols:
        if col.name in stochastic_cols:
            col_arrays[col.name] = np.empty(total_rows, dtype=object)
            continue
        if isinstance(parsed, PKSource):
            col_arrays[col.name] = np.asarray([
                f"{pk_first_char}-{i + 1:0{pk_width}d}"
                for i in range(total_rows)
            ], dtype=object)
        elif isinstance(parsed, FKSource):
            if parsed.table == "dim_date":
                col_arrays[col.name] = event_date_keys
            elif parsed.table in per_entity_dims:
                col_arrays[col.name] = event_entity_ids
            else:
                # Classified as deterministic above only when back-link path
                # would not execute — but we kept the conservative
                # classification. This branch is unreachable for shipped
                # templates; keep the fallback for defensive completeness.
                col_arrays[col.name] = np.full(total_rows, None, dtype=object)
        elif isinstance(parsed, ThresholdSource):
            # Proportional path passes threshold_col_name=None, so every
            # ThresholdSource cell resolves to None in the scalar path.
            col_arrays[col.name] = np.full(total_rows, None, dtype=object)
        elif isinstance(parsed, GeneratedSource):
            provider = parsed.provider
            if provider == "timestamp":
                dates = dim_date["date"].tolist()
                promoted: list[Any] = []
                for d in dates:
                    if isinstance(d, _dt.date) and not isinstance(d, _dt.datetime):
                        promoted.append(_dt.datetime(d.year, d.month, d.day))
                    else:
                        promoted.append(d)
                col_arrays[col.name] = np.asarray(
                    [promoted[i] for i in event_date_idx], dtype=object,
                )
            elif provider == "date_key":
                dk = dim_date["date_key"].tolist()
                col_arrays[col.name] = np.asarray(
                    [dk[i] for i in event_date_idx], dtype=object,
                )
            elif provider == "period_label":
                pl = dim_date["period_label"].tolist()
                col_arrays[col.name] = np.asarray(
                    [pl[i] for i in event_date_idx], dtype=object,
                )
            else:
                raise ValueError(
                    f"unsupported generated provider {provider!r} on fact/event tables"
                )
        elif isinstance(parsed, StaticSource):
            col_arrays[col.name] = np.full(total_rows, parsed.value, dtype=object)
        elif isinstance(parsed, DerivedSource):
            if parsed.field == "entity_id":
                col_arrays[col.name] = event_entity_ids
            elif parsed.field == "date_key":
                col_arrays[col.name] = event_date_keys
            else:
                # F14 (M102): explicit raise on unrecognised derived
                # field. The two supported fields on event tables are
                # ``entity_id`` and ``date_key``; anything else was
                # silently filled with None pre-F14.
                raise ValueError(
                    f"event column {col.name!r} derived field "
                    f"{parsed.field!r} is not supported on event tables; "
                    f"use 'entity_id' or 'date_key'"
                )
        else:
            # F14 (M102): explicit raise on unhandled source type.
            # Pre-F14 the deterministic-event-builder ladder fell through
            # to a silent None fill for any source type not in the
            # dispatch above (MetricSource, LagSource, ProportionalSource).
            # ``_is_stochastic`` only diverts FakerSource and external-FK
            # FKSource into the stochastic loop; everything else lands
            # here.
            raise TypeError(
                f"event column {col.name!r} source {col.source!r} "
                f"resolves to {type(parsed).__name__}, which is not "
                f"supported in the deterministic-event dispatch on "
                f"{tbl.name!r}."
            )

    # Stochastic columns: iterate per (cell, row) exactly as the scalar path
    # did, so RNG + faker consumption order is byte-identical. Source parsing
    # is already hoisted; the inner work is a dict write per stochastic column.
    if stochastic_cols:
        row_idx = 0
        n_cells = len(counts)
        cell_entity_ids = entity_ids_arr
        stochastic_parsed = [
            (col, parsed) for col, parsed in parsed_cols
            if col.name in stochastic_cols
        ]
        for cell_idx in range(n_cells):
            c = int(counts[cell_idx])
            if c == 0:
                continue
            entity_pk_value = cell_entity_ids[cell_idx]
            for _ in range(c):
                for col, parsed in stochastic_parsed:
                    if isinstance(parsed, FakerSource):
                        col_arrays[col.name][row_idx] = _call_faker(
                            fake, parsed.method, parsed.kwargs,
                        )
                    elif isinstance(parsed, FKSource):
                        parent = dim_tables.get(parsed.table)
                        if parent is None or parent.empty:
                            col_arrays[col.name][row_idx] = None
                            continue
                        back_link = _find_entity_link_in_subentity(
                            parsed.table, config, per_entity_dims,
                        )
                        if back_link is not None:
                            candidates = parent[parent[back_link] == entity_pk_value]
                            if len(candidates) > 0:
                                if rng is not None:
                                    pick = int(rng.integers(0, len(candidates)))
                                else:
                                    pick = 0
                                col_arrays[col.name][row_idx] = (
                                    candidates.iloc[pick][parsed.column]
                                )
                                continue
                        col_arrays[col.name][row_idx] = parent.iloc[0][parsed.column]
                row_idx += 1

    return pd.DataFrame({col.name: col_arrays[col.name] for col, _ in parsed_cols})


def _build_threshold_event(
    tbl: Table,
    threshold_col_pair: tuple[Column, ThresholdSource],
    config: PlotsimConfig,
    fact_tables: dict[str, pd.DataFrame],
    dim_tables: dict[str, pd.DataFrame],
    per_entity_dims: set[str],
    fake: Faker,
    rng: np.random.Generator,
) -> pd.DataFrame:
    threshold_col_cfg, ts = threshold_col_pair
    located = _find_metric_column_in_facts(ts.metric, fact_tables, config)
    if located is None:
        raise ValueError(
            f"event table {tbl.name!r} threshold column {threshold_col_cfg.name!r} "
            f"references metric {ts.metric!r} but no fact table exposes that metric"
        )
    fact_name, metric_col = located
    fact_table_cfg = next(t for t in config.tables if t.name == fact_name)
    fact_df = fact_tables[fact_name]

    _, groups = _entity_groups(fact_df, fact_table_cfg, per_entity_dims)
    fact_date_col = _find_date_fk_column(fact_table_cfg)
    fact_date_col_name = fact_date_col[0]

    dim_date = dim_tables["dim_date"]
    rows: list[dict] = []
    pk_counter = 0

    for entity_pk_value, group in groups:
        streak = 0
        fired = False
        for _, fact_row in group.iterrows():
            if fired:
                break
            value = fact_row[metric_col]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                streak = 0
                continue
            v = float(value)
            satisfied = (v > ts.value) if ts.direction == "above" else (v < ts.value)
            if satisfied:
                streak += 1
                if streak >= ts.consecutive:
                    date_key_value = fact_row[fact_date_col_name]
                    row = _resolve_event_row(
                        tbl, pk_counter, date_key_value, entity_pk_value,
                        threshold_col_cfg.name, True,
                        dim_date, dim_tables, config, fake, rng=rng,
                    )
                    rows.append(row)
                    pk_counter += 1
                    fired = True
            else:
                streak = 0

    return pd.DataFrame(rows, columns=[c.name for c in tbl.columns])


def _resolve_event_row(
    tbl: Table,
    pk_counter: int,
    date_key_value,
    entity_pk_value,
    threshold_col_name: Optional[str],
    threshold_value,
    dim_date: pd.DataFrame,
    dim_tables: dict[str, pd.DataFrame],
    config: PlotsimConfig,
    fake: Faker,
    rng: Optional[np.random.Generator],
) -> dict:
    """Build one event row, resolving every column type the schema allows."""
    per_entity_dims = _per_entity_dim_names(config)
    # Find the dim_date row matching this date_key (so non-FK date columns
    # can still pull period_label / date / etc.)
    date_idx = None
    matches = dim_date.index[dim_date["date_key"] == date_key_value]
    if len(matches) > 0:
        date_idx = int(matches[0])

    row: dict = {}
    width = _id_pad(10_000)
    for col in tbl.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, PKSource):
            prefix = tbl.name[4:] if tbl.name.startswith("evt_") else tbl.name
            row[col.name] = f"{prefix[0]}-{pk_counter+1:0{width}d}"
        elif isinstance(parsed, FKSource):
            if parsed.table == "dim_date":
                row[col.name] = date_key_value
            elif parsed.table in per_entity_dims:
                row[col.name] = entity_pk_value
            else:
                # Sub-entity (e.g. dim_user) or reference: pick a row whose
                # back-reference matches this entity. Fallback to row 0 if
                # no link is discoverable. Random pick among matches when we
                # can, so multiple rows for the same entity-period don't
                # collapse to a single sub-entity.
                parent = dim_tables.get(parsed.table)
                if parent is None or parent.empty:
                    row[col.name] = None
                    continue
                back_link = _find_entity_link_in_subentity(
                    parsed.table, config, per_entity_dims,
                )
                if back_link is not None:
                    candidates = parent[parent[back_link] == entity_pk_value]
                    if len(candidates) > 0:
                        if rng is not None:
                            i = int(rng.integers(0, len(candidates)))
                        else:
                            i = 0
                        row[col.name] = candidates.iloc[i][parsed.column]
                        continue
                row[col.name] = parent.iloc[0][parsed.column]
        elif isinstance(parsed, ThresholdSource):
            if col.name == threshold_col_name:
                row[col.name] = threshold_value
            else:
                row[col.name] = None
        elif isinstance(parsed, GeneratedSource):
            row[col.name] = _resolve_generated(
                parsed.provider, date_idx if date_idx is not None else 0,
                dim_date, fake,
            )
        elif isinstance(parsed, FakerSource):
            row[col.name] = _call_faker(fake, parsed.method, parsed.kwargs)
        elif isinstance(parsed, StaticSource):
            row[col.name] = parsed.value
        elif isinstance(parsed, DerivedSource):
            if parsed.field == "entity_id":
                row[col.name] = entity_pk_value
            elif parsed.field == "date_key":
                row[col.name] = date_key_value
            else:
                row[col.name] = None
        else:
            row[col.name] = None
    return row


def _find_entity_link_in_subentity(
    subentity_table: str,
    config: PlotsimConfig,
    per_entity_dims: set[str],
) -> Optional[str]:
    """Return the local column name in a sub-entity dim that FKs to a per_entity dim."""
    for tbl in config.tables:
        if tbl.name != subentity_table:
            continue
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if isinstance(parsed, FKSource) and parsed.table in per_entity_dims:
                return col.name
    return None


# --- Stage assignment --------------------------------------------------------


def _monotonic_stage_walk(
    values: np.ndarray,
    thresholds: np.ndarray,
    downgrade_delay: Optional[int],
    exit_thresholds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-entity stage walk — cursor advances monotonically, optional downgrade.

    FIX-07 helper. ``values`` is a 1D float array (NaN marks null).
    ``thresholds`` is the ascending ``threshold_enter`` list from
    ``stages.sequence``. Returns a 1D int array of stage indices matching
    the scalar path byte-for-byte.

    * ``downgrade_delay is None and exit_thresholds is None`` — pure
      legacy monotonic. ``actual`` per row is
      ``searchsorted(thresholds, v, side='right') - 1``; NaN rows hold
      ``actual=0`` and are dominated by the running max. Fully
      vectorized via :func:`np.maximum.accumulate`. Byte-identical to
      the pre-F8 / pre-FIX-07 iterrows path.
    * ``downgrade_delay`` is ``N`` (legacy mode) — sequential cursor
      with consecutive ``below-cursor`` counter. A short per-entity
      Python loop; the per-entity size is the period count, so total
      work is O(n_entities * n_periods) identical to the iterrows path
      but without pandas row-materialization overhead.
    * ``exit_thresholds`` is set (F8 / 0.5 hysteresis mode) — sequential
      cursor with demote check ``value < exit_thresholds[cursor]``.
      ``downgrade_delay=None`` collapses to delay=1 (immediate demote
      once value drops below exit). ``downgrade_delay=N`` requires
      ``N`` consecutive periods below exit before demotion fires.
      Cursor demotes to ``actual[i]`` (the searchsorted-derived stage
      for the current value), so a sharp drop can skip multiple stages
      in one demote step — matching the upward path's behavior.

    All three branches share a single algorithm shape; the differences
    are which threshold drives the demote check and what the delay is.
    Legacy strict-monotonic preserves the vectorized fast path because
    no demote can ever fire under that combination of inputs.
    """
    n = len(values)
    n_stages = len(thresholds)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    mask = ~np.isnan(values)
    actual = np.zeros(n, dtype=np.int64)
    if mask.any():
        actual[mask] = np.searchsorted(thresholds, values[mask], side="right") - 1
    np.clip(actual, 0, n_stages - 1, out=actual)

    if downgrade_delay is None and exit_thresholds is None:
        return np.maximum.accumulate(actual)

    # demote_t: per-stage threshold below which the cursor can demote.
    # Legacy with delay: equals ``thresholds`` (threshold_enter), so
    # ``value < demote_t[cursor]`` is equivalent to the old
    # ``actual[i] < cursor`` check (both mean values[i] dropped below
    # the current stage's entry).
    # Hysteresis: equals ``exit_thresholds``, demote when value drops
    # below the current stage's exit threshold (a tighter band).
    demote_t = exit_thresholds if exit_thresholds is not None else thresholds
    # Hysteresis without explicit delay = immediate demote (delay 1).
    delay = downgrade_delay if downgrade_delay is not None else 1

    out = np.empty(n, dtype=np.int64)
    cursor = 0
    lower_streak = 0
    for i in range(n):
        if not mask[i]:
            out[i] = cursor
            continue
        v = float(values[i])
        a = int(actual[i])
        if a > cursor:
            cursor = a
            lower_streak = 0
        elif v < demote_t[cursor]:
            lower_streak += 1
            if lower_streak >= delay:
                cursor = a
                lower_streak = 0
        else:
            lower_streak = 0
        out[i] = cursor
    return out


def _free_mode_stages(
    values: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Each row picks the highest stage its value satisfies; NaN stays at 0.

    Scalar parity: when ``enforce_order=False``, the cursor never advances
    (it's gated on ``enforce``); ``chosen`` starts at 0 per row. NaN rows
    hit the null-valued branch and emit ``seq[cursor].name``, which in
    free mode is always ``seq[0]``. Vectorized via a single masked
    ``np.searchsorted`` over non-null rows.
    """
    n = len(values)
    n_stages = len(thresholds)
    mask = ~np.isnan(values)
    out = np.zeros(n, dtype=np.int64)
    if mask.any():
        out[mask] = np.searchsorted(thresholds, values[mask], side="right") - 1
    np.clip(out, 0, n_stages - 1, out=out)
    return out


def assign_stages(
    config: PlotsimConfig,
    fact_tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Annotate the fact table that owns ``stages.field`` with a ``stage`` column.

    Default routing is ``enforce_order=False`` (free-mode): each period
    independently picks the highest-enter stage the realized value
    satisfies. ``threshold_exit`` and ``downgrade_delay`` are ignored;
    free mode is stateless, so hysteresis has no meaning there. Stages
    can move backward when the driving value falls — by design, since
    irreversible lifecycle transitions are SCD Type 2's job.

    With ``enforce_order=True`` the cursor advances whenever the value
    crosses the next stage's ``threshold_enter``. Cursor reversal depends
    on FIX-06's ``downgrade_delay`` and on the F8 / 0.5 ``mode``:

      * ``mode == 'legacy'`` (``threshold_exit > threshold_enter``,
        the bundled-template default): the runtime ignores
        ``threshold_exit`` and relies on ``threshold_enter`` only.
        ``downgrade_delay is None`` is strict monotonic — the cursor
        never steps back; a brief dip stays in the higher stage.
        ``downgrade_delay == N`` demotes after ``N`` consecutive
        periods below the current stage's enter threshold.
      * ``mode == 'hysteresis'`` (``threshold_exit <= threshold_enter``,
        F8 wiring): the runtime uses ``threshold_exit`` of the current
        stage as the demote threshold. ``downgrade_delay is None``
        collapses to delay=1 (immediate demote once the value drops
        below the current stage's exit). ``downgrade_delay == N``
        requires ``N`` consecutive periods below exit before demotion
        fires. The hysteresis band ``[threshold_exit, threshold_enter]``
        keeps the entity in the higher stage on transient dips.

    FIX-07 / SF-5: the implementation is vectorized via pandas
    ``groupby`` + numpy walks (see :func:`_monotonic_stage_walk` and
    :func:`_free_mode_stages`). Output is byte-identical to the prior
    ``iterrows`` path under legacy mode; parity is locked in by
    ``test_vectorized_assign_stages_matches_iterrows_output``.
    """
    if config.stages is None:
        return fact_tables

    field = config.stages.field
    seq = config.stages.sequence
    enforce = config.stages.enforce_order
    downgrade_delay = config.stages.downgrade_delay if enforce else None

    target_name = None
    target_tbl = None
    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        df = fact_tables.get(tbl.name)
        if df is not None and field in df.columns:
            target_name = tbl.name
            target_tbl = tbl
            break
    if target_name is None:
        return fact_tables

    per_entity_dims = _per_entity_dim_names(config)
    fk = _find_entity_fk_column(target_tbl, per_entity_dims)
    if fk is None:
        return fact_tables
    entity_col = fk[0]

    df = fact_tables[target_name].copy()
    n = len(df)

    thresholds = np.asarray(
        [s.threshold_enter for s in seq], dtype=float,
    )
    stage_names = np.asarray([s.name for s in seq], dtype=object)

    # F8 / 0.5: under hysteresis mode, build an exit-thresholds array
    # parallel to ``thresholds`` for the demote check. Terminal stage
    # has threshold_exit=None; fall back to its enter threshold (the
    # cursor cannot demote from terminal in monotonic mode anyway —
    # a > cursor is always true, demote branch is dead). Free mode
    # stays legacy regardless: it's stateless, hysteresis is a
    # no-op there.
    exit_thresholds: Optional[np.ndarray] = None
    if enforce and config.stages.mode == "hysteresis":
        exit_thresholds = np.asarray(
            [
                s.threshold_exit if s.threshold_exit is not None
                else s.threshold_enter
                for s in seq
            ],
            dtype=float,
        )

    # Coerce the driving field to a numeric 1D array with NaN for nulls.
    raw = df[field].to_numpy()
    values = np.empty(n, dtype=float)
    for i, v in enumerate(raw):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            values[i] = np.nan
        else:
            values[i] = float(v)

    stage_idx = np.empty(n, dtype=np.int64)
    if not enforce:
        stage_idx = _free_mode_stages(values, thresholds)
    else:
        # Per-entity walk — groupby.indices gives row positions per group,
        # preserving first-appearance order.
        for _eid, positions in df.groupby(
            entity_col, sort=False,
        ).indices.items():
            pos_arr = np.asarray(positions, dtype=np.int64)
            stage_idx[pos_arr] = _monotonic_stage_walk(
                values[pos_arr], thresholds, downgrade_delay,
                exit_thresholds=exit_thresholds,
            )

    df["stage"] = stage_names[stage_idx]
    out = dict(fact_tables)
    out[target_name] = df
    return out


# --- SCD Type 2 (M106) -------------------------------------------------------
#
# SCD machinery is split into three contracts so the trajectory-first
# invariant stays load-bearing across the pipeline:
#
#   1. ``_compute_scd_versions`` — pure function that takes one entity's
#      trajectory and one SCDType2Config, returns the list of versioned
#      bands the entity visits monotonically. No DataFrame, no I/O.
#   2. ``expand_scd_dim`` — uses (1) to rewrite a single dim DataFrame
#      from "one row per entity" into "one row per (entity, version)"
#      with the SCD columns appended (dim_row_id / valid_from / valid_to /
#      is_current). Returns the expanded DataFrame plus an
#      ``SCDDimState`` carrying the per-entity version list and the
#      column metadata fact tables need to inject ``dim_row_id``.
#   3. ``attach_dim_row_id_to_facts`` — runs after fact construction;
#      for every fact/event table whose entity FK targets an SCD dim,
#      appends a ``dim_row_id`` column resolved via (entity, period) →
#      active version.
#
# All three are deterministic given (config, trajectories). They consume
# no RNG, so they slot into ``generate_tables_with_state`` without
# disturbing the seed/draw order callers rely on for reproducibility.

# Sentinel valid_to for the currently active version. Encoded as a
# ``YYYYMMDD`` integer so it lives in the same numeric domain as the
# date_keys ``dim_date`` emits — downstream SQL joins predicating on
# ``valid_to`` get a value far above any real date_key without needing
# special-case NULL handling.
SCD_VALID_TO_SENTINEL: int = 99991231


@dataclass(frozen=True)
class SCDVersion:
    """One versioned slice of a single entity's life in an SCD dim.

    ``band`` indexes into the SCD column's ``labels`` tuple;
    ``band_label`` is the literal cell value emitted in the dim row.
    ``valid_from`` / ``valid_to`` are date_keys (YYYYMMDD ints) sourced
    from ``dim_date`` so they share a numeric domain with fact-table
    date_key FKs. ``valid_from_period`` / ``valid_to_period`` carry the
    same boundary as 0-based period indices for the manifest, which
    operates in period-index space.
    ``crossing_position`` is the trajectory position at the period
    where this band became the entity's active band; ``None`` for the
    initial band (band 0 the entity occupied at t=0).

    Field is named ``band_label`` rather than the bare display name
    used by ``Archetype`` and ``Metric`` so the dead-schema audit
    (M102, see tests/test_dead_schema.py) regex doesn't treat reads
    on this dataclass as reads of those allowlisted display fields.
    """
    band: int
    band_label: str
    valid_from: int
    valid_to: int
    valid_from_period: int
    valid_to_period: int
    is_current: bool
    dim_row_id: int
    crossing_position: Optional[float]


@dataclass(frozen=True)
class SCDDimState:
    """Per-dim-table SCD state, keyed by entity name to its version list.

    ``scd_column`` is the dim column whose label cells SCD writes;
    ``entity_pk_column`` is the dim's PK column (the entity business
    key, repeated across versions); ``trigger_metric`` is carried
    through so the manifest can name what drove each crossing without
    re-reading the config.
    """
    versions: dict[str, list[SCDVersion]]
    scd_column: str
    entity_pk_column: str
    trigger_metric: str


@dataclass(frozen=True)
class SCDState:
    """Cross-table SCD state. Maps dim-table name → SCDDimState."""
    dims: dict[str, SCDDimState]

    @property
    def is_empty(self) -> bool:
        return not self.dims


def _scd_column_for_table(tbl: Table) -> Optional[tuple[Column, SCDType2Config]]:
    """Return the single SCD column on ``tbl`` (or None).

    PlotsimConfig validation rejects multi-SCD-column dim tables, so this
    helper is allowed to return at most one match.
    """
    for col in tbl.columns:
        if col.scd_type2 is not None:
            return col, col.scd_type2
    return None


def _entity_pk_column(tbl: Table) -> Optional[str]:
    for col in tbl.columns:
        if isinstance(parse_source(col.source), PKSource):
            return col.name
    return None


def _compute_scd_versions(
    trajectory: np.ndarray,
    scd_cfg: SCDType2Config,
    date_keys: np.ndarray,
    starting_dim_row_id: int,
) -> list[SCDVersion]:
    """Walk one entity's trajectory and emit its visited band versions.

    Algorithm:
      * ``raw_band[p] = searchsorted(thresholds, position[p], side='right')``
        clamped to ``[0, len(labels) - 1]``. The clamp closes the
        ``position == 1.0`` corner so the highest position lands in the
        topmost band rather than overflowing.
      * ``cum_band = np.maximum.accumulate(raw_band)`` — the monotonic
        cursor. An entity that crosses upward and falls back keeps the
        higher cursor (hysteresis), mirroring M102's
        ``StageDefinition.threshold_exit`` semantics so SCD versioning
        and stage assignment have one consistent contract.
      * Transitions are the indices where ``cum_band`` increases. The
        first period (index 0) is always a transition with no
        ``crossing_position`` (the entity *starts* in some band
        rather than crossing into it).

    Each transition produces an ``SCDVersion``. ``valid_to`` is the
    date_key of the next transition's first period; the final
    transition gets the sentinel ``SCD_VALID_TO_SENTINEL`` and
    ``is_current=True``.
    """
    thresholds = np.asarray(scd_cfg.thresholds, dtype=np.float64)
    labels = scd_cfg.labels
    n_periods = len(trajectory)

    raw_bands = np.searchsorted(thresholds, trajectory, side="right")
    raw_bands = np.clip(raw_bands, 0, len(labels) - 1).astype(np.int64)
    cum_bands = np.maximum.accumulate(raw_bands)

    # Transition indices: 0 (always) plus every index where cum_band rose.
    transitions: list[int] = [0]
    for p in range(1, n_periods):
        if cum_bands[p] > cum_bands[p - 1]:
            transitions.append(p)

    versions: list[SCDVersion] = []
    next_dim_row_id = starting_dim_row_id
    for seg_idx, start_period in enumerate(transitions):
        end_period = (
            transitions[seg_idx + 1] if seg_idx + 1 < len(transitions)
            else n_periods
        )
        is_current = seg_idx == len(transitions) - 1
        valid_from = int(date_keys[start_period])
        if is_current:
            valid_to = SCD_VALID_TO_SENTINEL
            valid_to_period = n_periods
        else:
            valid_to = int(date_keys[end_period])
            valid_to_period = end_period
        band = int(cum_bands[start_period])
        versions.append(SCDVersion(
            band=band,
            band_label=labels[band],
            valid_from=valid_from,
            valid_to=valid_to,
            valid_from_period=start_period,
            valid_to_period=valid_to_period,
            is_current=is_current,
            dim_row_id=next_dim_row_id,
            crossing_position=(
                None if seg_idx == 0 else float(trajectory[start_period])
            ),
        ))
        next_dim_row_id += 1
    return versions


def _expand_scd_dim(
    tbl: Table,
    df: pd.DataFrame,
    config: PlotsimConfig,
    trajectories: dict[str, np.ndarray],
    dim_date: pd.DataFrame,
) -> tuple[pd.DataFrame, SCDDimState]:
    """Rewrite a single SCD-enabled dim DataFrame into versioned rows.

    Pre-conditions enforced upstream by ``PlotsimConfig`` validation:
    ``tbl`` is a per_entity dim with exactly one ``scd_type2`` column.

    Returns the expanded DataFrame plus an ``SCDDimState`` recording
    the per-entity version list, the SCD column name, the entity-PK
    column, and the trigger metric — everything fact-FK resolution and
    manifest emission need without re-deriving from the dim DataFrame.
    """
    scd_pair = _scd_column_for_table(tbl)
    if scd_pair is None:  # defensive: caller filters
        raise RuntimeError(
            f"_expand_scd_dim called on {tbl.name!r} which has no SCD column"
        )
    scd_col, scd_cfg = scd_pair
    pk_col = _entity_pk_column(tbl)
    if pk_col is None:
        raise ValueError(
            f"SCD dim {tbl.name!r} has no PK column; one column must declare "
            f"source 'pk' so SCD versions can carry a stable entity business key"
        )
    if len(df) != len(config.entities):
        raise ValueError(
            f"SCD dim {tbl.name!r} has {len(df)} rows but config has "
            f"{len(config.entities)} entities; per_entity dims must be 1:1 "
            f"with config.entities for SCD expansion"
        )

    date_keys = dim_date["date_key"].to_numpy()
    n_periods = len(date_keys)

    versions_by_entity: dict[str, list[SCDVersion]] = {}
    expanded_rows: list[dict[str, Any]] = []
    next_id = 1
    column_order = list(df.columns) + [
        "dim_row_id", "valid_from", "valid_to", "is_current",
    ]

    for entity_idx, entity in enumerate(config.entities):
        traj = trajectories.get(entity.name)
        if traj is None or len(traj) != n_periods:
            raise ValueError(
                f"SCD dim {tbl.name!r}: entity {entity.name!r} has missing "
                f"or wrong-length trajectory (expected {n_periods})"
            )
        entity_versions = _compute_scd_versions(
            traj, scd_cfg, date_keys, starting_dim_row_id=next_id,
        )
        next_id += len(entity_versions)
        versions_by_entity[entity.name] = entity_versions

        base_row = df.iloc[entity_idx].to_dict()
        for version in entity_versions:
            row = dict(base_row)
            row[scd_col.name] = version.band_label
            row["dim_row_id"] = version.dim_row_id
            row["valid_from"] = version.valid_from
            row["valid_to"] = version.valid_to
            row["is_current"] = version.is_current
            expanded_rows.append(row)

    expanded_df = pd.DataFrame(expanded_rows, columns=column_order)
    state = SCDDimState(
        versions=versions_by_entity,
        scd_column=scd_col.name,
        entity_pk_column=pk_col,
        trigger_metric=scd_cfg.trigger_metric,
    )
    return expanded_df, state


def expand_scd_dims(
    config: PlotsimConfig,
    dim_tables: dict[str, pd.DataFrame],
    trajectories: dict[str, np.ndarray],
) -> tuple[dict[str, pd.DataFrame], SCDState]:
    """Expand every SCD-enabled dim and return the updated table dict + state.

    Tables without any ``scd_type2`` column pass through unchanged. The
    caller (``generate_tables_with_state``) runs this after dimension
    construction and trajectory computation but BEFORE fact construction
    so fact tables can resolve their FK to the active dim_row_id.
    """
    dim_date = dim_tables.get("dim_date")
    if dim_date is None:
        # Without dim_date we have no date_key spine to anchor validity
        # windows; safer to fail loudly than emit misaligned versions.
        raise RuntimeError(
            "expand_scd_dims requires dim_date to be present in dim_tables"
        )

    out = dict(dim_tables)
    states: dict[str, SCDDimState] = {}
    for tbl in config.tables:
        if tbl.type != "dim":
            continue
        if _scd_column_for_table(tbl) is None:
            continue
        df = dim_tables.get(tbl.name)
        if df is None:
            raise RuntimeError(
                f"expand_scd_dims: dim table {tbl.name!r} has SCD config but "
                f"no DataFrame was built upstream by build_all_dimensions"
            )
        expanded, state = _expand_scd_dim(
            tbl, df, config, trajectories, dim_date,
        )
        out[tbl.name] = expanded
        states[tbl.name] = state
    return out, SCDState(dims=states)


def _facts_referencing_scd_dim(
    config: PlotsimConfig, scd_dim: str,
) -> list[Table]:
    """Tables (fact or event) whose any FK column points at ``scd_dim``."""
    out: list[Table] = []
    for tbl in config.tables:
        if tbl.type not in ("fact", "event"):
            continue
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if isinstance(parsed, FKSource) and parsed.table == scd_dim:
                out.append(tbl)
                break
    return out


def _date_key_period_index(dim_date: pd.DataFrame) -> dict[int, int]:
    """date_key (int) → period index (0-based row position in dim_date)."""
    return {int(k): i for i, k in enumerate(dim_date["date_key"].tolist())}


def _resolve_dim_row_id_per_row(
    entity_keys: np.ndarray,
    date_keys: np.ndarray,
    versions_by_entity_pk: dict[Any, list[SCDVersion]],
    period_index_by_date_key: dict[int, int],
) -> np.ndarray:
    """Build the ``dim_row_id`` column for one fact/event table.

    For each row's (entity_key, date_key) pair, look up the version
    whose ``[valid_from_period, valid_to_period)`` half-open window
    contains the date_key's period index. Returns an int64 ndarray
    aligned with the input row order.
    """
    n_rows = len(entity_keys)
    out = np.empty(n_rows, dtype=np.int64)
    for i in range(n_rows):
        entity_pk = entity_keys[i]
        dkey = date_keys[i]
        try:
            dkey_int = int(dkey) if dkey is not None else None
        except (TypeError, ValueError):
            dkey_int = None
        if dkey_int is None:
            out[i] = -1
            continue
        period_idx = period_index_by_date_key.get(dkey_int)
        if period_idx is None:
            out[i] = -1
            continue
        versions = versions_by_entity_pk.get(entity_pk)
        if not versions:
            out[i] = -1
            continue
        active: Optional[SCDVersion] = None
        for v in versions:
            if v.valid_from_period <= period_idx < v.valid_to_period:
                active = v
                break
        out[i] = active.dim_row_id if active is not None else -1
    return out


def attach_dim_row_id_to_facts(
    config: PlotsimConfig,
    fact_tables: dict[str, pd.DataFrame],
    dim_tables: dict[str, pd.DataFrame],
    scd_state: SCDState,
) -> dict[str, pd.DataFrame]:
    """Append a ``dim_row_id`` column to facts/events FK'ing into SCD dims.

    Mutates a copy of ``fact_tables`` and returns it. The original FK
    column on the fact (typically the entity business key, e.g.
    ``company_id``) is left intact so existing groupby and stage-
    assignment paths keep working — ``dim_row_id`` is purely additive.

    V1 contract: a fact/event table may FK into at most one SCD dim
    (PlotsimConfig validation ensures this is feasible by capping each
    dim at one SCD column; this helper enforces it on the fact side
    so a future schema change doesn't quietly produce ambiguous joins).
    """
    if scd_state.is_empty:
        return fact_tables

    dim_date = dim_tables.get("dim_date")
    if dim_date is None:
        return fact_tables
    period_index_by_date_key = _date_key_period_index(dim_date)

    out = dict(fact_tables)
    for scd_dim_name, dim_state in scd_state.dims.items():
        scd_dim_df = dim_tables.get(scd_dim_name)
        if scd_dim_df is None:
            continue
        # Map entity name → entity PK value (the business key, repeated
        # across versions). Pull the first-version row for each entity
        # since the PK is identical across versions.
        entity_pk_by_name: dict[str, Any] = {}
        for entity_name, versions in dim_state.versions.items():
            if not versions:
                continue
            # Find the row for the entity's first version to extract its PK.
            sub = scd_dim_df[
                scd_dim_df["dim_row_id"] == versions[0].dim_row_id
            ]
            if sub.empty:
                continue
            entity_pk_by_name[entity_name] = sub.iloc[0][dim_state.entity_pk_column]

        # Reverse lookup: entity_pk → versions list (order preserved).
        versions_by_entity_pk: dict[Any, list[SCDVersion]] = {
            entity_pk_by_name[name]: dim_state.versions[name]
            for name in dim_state.versions
            if name in entity_pk_by_name
        }

        per_entity_dims = _per_entity_dim_names(config)
        for tbl in _facts_referencing_scd_dim(config, scd_dim_name):
            df = out.get(tbl.name)
            if df is None or df.empty:
                continue
            entity_fk = _find_entity_fk_column(tbl, per_entity_dims)
            if entity_fk is None:
                continue
            local_entity_col = entity_fk[0]
            date_fk = _find_date_fk_column(tbl)
            if date_fk is None:
                continue
            local_date_col = date_fk[0]
            if local_entity_col not in df.columns or local_date_col not in df.columns:
                continue
            entity_keys = df[local_entity_col].to_numpy()
            date_keys = df[local_date_col].to_numpy()
            dim_row_ids = _resolve_dim_row_id_per_row(
                entity_keys, date_keys,
                versions_by_entity_pk, period_index_by_date_key,
            )
            # Defensive: a -1 sentinel means we couldn't resolve a row.
            # That would surface as an FK orphan downstream — fail loud
            # here so the user sees the SCD wiring problem rather than
            # a generic FK-integrity error.
            if (dim_row_ids < 0).any():
                bad = int((dim_row_ids < 0).sum())
                raise RuntimeError(
                    f"attach_dim_row_id_to_facts: {tbl.name!r} could not "
                    f"resolve dim_row_id for {bad} row(s) against SCD dim "
                    f"{scd_dim_name!r}; this indicates a (entity, period) "
                    f"pair that no version covers — likely a generation-"
                    f"order bug, not a config error"
                )
            df = df.copy(deep=False)
            df["dim_row_id"] = pd.array(dim_row_ids, dtype="Int64")
            out[tbl.name] = df
    return out


# --- Bridge tables (M107) ----------------------------------------------------


@dataclass(frozen=True)
class BridgeAssociation:
    """One first-dim entity's set of associations for a single bridge.

    ``entity`` is the first-dim entity name (matches ``Entity.name``).
    ``targets`` is the list of second-dim FK values that entity associated
    with — PK values for non-SCD second dims, ``dim_row_id`` values when
    the second dim is SCD-enabled. ``cardinality`` is ``len(targets)``,
    surfaced as a separate field so manifest consumers can iterate stats
    without re-counting.
    """
    entity: str
    targets: list[Any]
    cardinality: int


@dataclass(frozen=True)
class BridgeAssociations:
    """Cross-bridge association record. Maps bridge_name → per-entity assoc list.

    ``BridgeAssociations(bridges={})`` is the empty-bridge sentinel: every
    config without a ``bridges`` block lands here, and the manifest builder
    skips the bridge_associations field entirely. Carried on
    ``GenerationState`` so the manifest can record ground-truth M:M
    associations without re-deriving them from the bridge DataFrames.
    """
    bridges: dict[str, list[BridgeAssociation]]

    @property
    def is_empty(self) -> bool:
        return not self.bridges


def _bridge_fk_col_name(dim_tbl: Table, scd_state: SCDState) -> str:
    """Return the bridge FK column name for one connected dim.

    Convention:
      * Non-SCD: the dim's PK column name verbatim (``student_id``,
        ``course_id``). Natural keys keep bridges legible without an
        extra surrogate column.
      * SCD: ``<dim_name_short>_dim_row_id`` (``dim_company`` →
        ``company_dim_row_id``). Different from the non-SCD form so
        validation and downstream consumers can tell at a glance that
        the cell holds a surrogate, not a business key.
    """
    if dim_tbl.name in scd_state.dims:
        short = dim_tbl.name[4:] if dim_tbl.name.startswith("dim_") else dim_tbl.name
        return f"{short}_dim_row_id"
    pk = dim_tbl.primary_key_cols[0]
    return pk


def _bridge_first_dim_fk_by_entity(
    config: PlotsimConfig,
    first_dim_tbl: Table,
    first_dim_df: pd.DataFrame,
    scd_state: SCDState,
) -> dict[str, Any]:
    """Map ``Entity.name`` → first-dim FK value for every config entity.

    For non-SCD per_entity dims this dedupes the first-dim DataFrame on its
    PK and zips against ``config.entities`` (the dim builder iterates
    entities in config order, so position-bridging is safe). For SCD-
    enabled first dims we read the ``is_current=True`` ``dim_row_id`` from
    the per-entity SCD version list — the same row a fact FK would resolve
    to at the end of the time window.
    """
    if first_dim_tbl.name in scd_state.dims:
        dim_state = scd_state.dims[first_dim_tbl.name]
        out: dict[str, Any] = {}
        for entity in config.entities:
            versions = dim_state.versions.get(entity.name, [])
            current = next((v for v in versions if v.is_current), None)
            if current is None:
                raise RuntimeError(
                    f"build_bridge_tables: SCD dim {first_dim_tbl.name!r} has "
                    f"no is_current=True version for entity {entity.name!r}; "
                    f"this is an SCD-expansion bug, not a config error"
                )
            out[entity.name] = int(current.dim_row_id)
        return out
    first_pk_col = first_dim_tbl.primary_key_cols[0]
    deduped = first_dim_df.drop_duplicates(
        subset=[first_pk_col], keep="first",
    ).reset_index(drop=True)
    if len(deduped) != len(config.entities):
        raise RuntimeError(
            f"build_bridge_tables: per_entity dim {first_dim_tbl.name!r} has "
            f"{len(deduped)} unique {first_pk_col!r} value(s) but config has "
            f"{len(config.entities)} entities; the dim builder is expected "
            f"to keep dims 1:1 with config.entities"
        )
    return {
        entity.name: deduped.iloc[i][first_pk_col]
        for i, entity in enumerate(config.entities)
    }


def _bridge_second_dim_fk_pool(
    second_dim_tbl: Table,
    second_dim_df: pd.DataFrame,
    scd_state: SCDState,
) -> list[Any]:
    """Return the list of FK values an entity in the first dim can associate with.

    Mirrors ``_bridge_first_dim_fk_by_entity`` but for the second dim:
    SCD-enabled dims yield the per-entity ``is_current=True`` dim_row_id;
    per_entity / per_reference dims yield their PK values directly.
    Sampling without replacement happens against this pool, so the
    ordering here defines the deterministic enumeration the downstream
    ``rng.choice`` walks.
    """
    if second_dim_tbl.name in scd_state.dims:
        dim_state = scd_state.dims[second_dim_tbl.name]
        pool: list[Any] = []
        for entity_name in dim_state.versions:
            versions = dim_state.versions[entity_name]
            current = next((v for v in versions if v.is_current), None)
            if current is None:
                continue
            pool.append(int(current.dim_row_id))
        return pool
    second_pk_col = second_dim_tbl.primary_key_cols[0]
    if second_dim_tbl.grain == "per_entity":
        deduped = second_dim_df.drop_duplicates(
            subset=[second_pk_col], keep="first",
        ).reset_index(drop=True)
        return deduped[second_pk_col].tolist()
    return second_dim_df[second_pk_col].tolist()


def _compute_bridge_cardinality(
    bridge: BridgeTableConfig,
    mean_position: float,
    pool_size: int,
    rng: np.random.Generator,
) -> int:
    """Return how many associations an entity gets, clamped to the pool size.

    Trajectory-driven: linear interpolation from ``mean_position`` between
    ``cardinality.min`` and ``cardinality.max``. Position 0.0 lands at
    ``min``, position 1.0 lands at ``max``, intermediate positions
    interpolate. Uniform mode: ``rng.integers(min, max + 1)``. Both
    branches clamp to ``pool_size`` so the without-replacement sampler
    can never be asked for more rows than the second dim has.
    """
    min_n = bridge.cardinality.min
    max_n = bridge.cardinality.max
    if bridge.trajectory_driven:
        clamped_pos = max(0.0, min(1.0, float(mean_position)))
        n = int(round(min_n + (max_n - min_n) * clamped_pos))
    else:
        n = int(rng.integers(min_n, max_n + 1))
    n = max(min_n, min(max_n, n))
    return min(n, pool_size)


def _bridge_metric_value(
    bm: BridgeMetric,
    entity_name: str,
    entity_metrics: dict[str, dict[str, np.ndarray]],
    fake: Faker,
):
    """Resolve a single bridge-metric cell.

    MetricSource collapses the entity's per-period series to its NaN-
    aware mean — the bridge captures the entity's *career-aggregated*
    level for that metric, which mirrors how the cardinality is read
    off the trajectory mean. StaticSource is a literal pass-through.
    FakerSource consumes Faker state per row so bridges with Faker
    metrics get distinct values per association.
    """
    parsed = parse_source(bm.source)
    if isinstance(parsed, MetricSource):
        per_metric = entity_metrics.get(entity_name, {})
        series = per_metric.get(parsed.metric)
        if series is None:
            return None
        if series.dtype == object:
            arr = np.asarray([
                np.nan if (v is None or (isinstance(v, float) and np.isnan(v)))
                else float(v) for v in series
            ], dtype=np.float64)
        else:
            arr = series.astype(np.float64)
        if np.all(np.isnan(arr)):
            return None
        mean_val = float(np.nanmean(arr))
        return _coerce_metric_value(mean_val, bm.dtype)
    if isinstance(parsed, StaticSource):
        return parsed.value
    if isinstance(parsed, FakerSource):
        return _call_faker(fake, parsed.method, parsed.kwargs)
    raise ValueError(
        f"bridge metric {bm.name!r} source {bm.source!r} dispatched to "
        f"{type(parsed).__name__}, which build_bridge_tables does not "
        f"support; the BridgeMetric source validator should have rejected "
        f"this at load time"
    )


def build_bridge_tables(
    config: PlotsimConfig,
    dim_tables: dict[str, pd.DataFrame],
    trajectories: dict[str, np.ndarray],
    entity_metrics: dict[str, dict[str, np.ndarray]],
    scd_state: SCDState,
    rng: np.random.Generator,
) -> tuple[dict[str, pd.DataFrame], BridgeAssociations]:
    """Generate every bridge table declared in ``config.bridges``.

    For each bridge:

      1. The first dim is per_entity (validated at load), so iteration is
         over ``config.entities`` in declaration order.
      2. Each entity's mean trajectory position picks a per-row cardinality
         in ``[cardinality.min, cardinality.max]`` (interpolated when
         ``trajectory_driven=True``, uniform otherwise).
      3. ``rng.choice(replace=False)`` samples that many distinct rows
         from the second dim's FK pool.
      4. Each association becomes one bridge row carrying the two FK
         values plus any declared bridge metrics, computed from the
         first-dim entity's metric series.

    Returns the dict of bridge DataFrames plus a ``BridgeAssociations``
    record the manifest builder reads. Empty config → empty dict +
    empty associations.
    """
    if not config.bridges:
        return {}, BridgeAssociations(bridges={})

    fake = _make_faker(rng, config.locale)
    out: dict[str, pd.DataFrame] = {}
    associations: dict[str, list[BridgeAssociation]] = {}

    for bridge in config.bridges:
        first_dim_name, second_dim_name = bridge.connects
        first_dim_tbl = next(t for t in config.tables if t.name == first_dim_name)
        second_dim_tbl = next(t for t in config.tables if t.name == second_dim_name)
        first_dim_df = dim_tables.get(first_dim_name)
        second_dim_df = dim_tables.get(second_dim_name)
        if first_dim_df is None or second_dim_df is None:
            raise RuntimeError(
                f"build_bridge_tables: bridge {bridge.name!r} connects "
                f"{first_dim_name!r} and {second_dim_name!r}, but one of "
                f"those dim DataFrames is missing from dim_tables"
            )

        first_fk_col = _bridge_fk_col_name(first_dim_tbl, scd_state)
        second_fk_col = _bridge_fk_col_name(second_dim_tbl, scd_state)
        if first_fk_col == second_fk_col:
            raise RuntimeError(
                f"build_bridge_tables: bridge {bridge.name!r} would emit two "
                f"columns named {first_fk_col!r} (both connected dims map to "
                f"that column under the SCD/non-SCD naming convention); "
                f"this is a config-layout edge case, not a data error"
            )

        first_fk_by_entity = _bridge_first_dim_fk_by_entity(
            config, first_dim_tbl, first_dim_df, scd_state,
        )
        second_pool = _bridge_second_dim_fk_pool(
            second_dim_tbl, second_dim_df, scd_state,
        )
        if not second_pool:
            out[bridge.name] = pd.DataFrame(
                columns=[first_fk_col, second_fk_col]
                + [bm.name for bm in bridge.metrics],
            )
            associations[bridge.name] = []
            continue
        n_pool = len(second_pool)

        rows: list[dict[str, Any]] = []
        bridge_assoc_list: list[BridgeAssociation] = []
        for entity in config.entities:
            traj = trajectories.get(entity.name)
            if traj is None:
                raise RuntimeError(
                    f"build_bridge_tables: entity {entity.name!r} has no "
                    f"trajectory; bridge cardinality is trajectory-driven"
                )
            mean_position = float(np.mean(traj))
            n = _compute_bridge_cardinality(bridge, mean_position, n_pool, rng)

            if n == 0:
                bridge_assoc_list.append(BridgeAssociation(
                    entity=entity.name, targets=[], cardinality=0,
                ))
                continue

            picked_idx = rng.choice(n_pool, size=n, replace=False)
            picked_targets = [second_pool[int(i)] for i in picked_idx]
            bridge_assoc_list.append(BridgeAssociation(
                entity=entity.name,
                targets=list(picked_targets),
                cardinality=n,
            ))

            first_fk_val = first_fk_by_entity[entity.name]
            for target in picked_targets:
                row: dict[str, Any] = {
                    first_fk_col: first_fk_val,
                    second_fk_col: target,
                }
                for bm in bridge.metrics:
                    row[bm.name] = _bridge_metric_value(
                        bm, entity.name, entity_metrics, fake,
                    )
                rows.append(row)

        column_order = [first_fk_col, second_fk_col] + [bm.name for bm in bridge.metrics]
        df = pd.DataFrame(rows, columns=column_order) if rows else pd.DataFrame(columns=column_order)
        out[bridge.name] = df
        associations[bridge.name] = bridge_assoc_list

    return out, BridgeAssociations(bridges=associations)


# --- Orchestrator ------------------------------------------------------------


@dataclass(frozen=True)
class GenerationState:
    """M105: structured side-channel for ground-truth manifest emission.

    ``generate_tables`` returns just the table dict to preserve its 0.5
    public signature. ``generate_tables_with_state`` returns the same
    tables alongside this state object, which carries the per-entity
    trajectory positions used during generation. The manifest builder in
    ``plotsim.manifest`` reads from here rather than re-deriving positions
    from cell values (which would be lossy under noise / MCAR).

    M106: ``scd`` carries per-dim SCD Type 2 versioning state (per-entity
    version lists, surrogate IDs, validity windows, crossing positions).
    ``SCDState.dims`` is empty for configs that declare no SCD columns —
    callers can skip SCD-aware code paths cheaply by checking
    ``state.scd.is_empty``.

    M107: ``bridges`` carries the per-bridge association ground truth
    (which second-dim rows each first-dim entity associated with). The
    manifest emits ``bridge_associations`` from this without re-grouping
    the bridge DataFrames. ``BridgeAssociations(bridges={})`` is the
    empty sentinel for configs without a ``bridges`` block.

    Future fields (anomaly injection locations, stage transition periods,
    etc.) extend this dataclass; existing callers that destructure
    ``(tables, state)`` keep working because Python dataclass fields are
    accessed by name.
    """
    trajectories: dict[str, np.ndarray]
    scd: SCDState = field(default_factory=lambda: SCDState(dims={}))
    bridges: BridgeAssociations = field(
        default_factory=lambda: BridgeAssociations(bridges={}),
    )


def generate_tables(
    config: PlotsimConfig,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, pd.DataFrame]:
    """End-to-end: dims → trajectories → facts → events → stages.

    Returns a dict with every dim, fact, and event table the config declares.
    Same (config, seed) → identical output.

    If ``rng`` is omitted, a fresh generator is seeded from ``config.seed``
    so callers can stay on the three-line quickstart without importing numpy.

    Gates the run on ``validate_correlation_psd(config)`` before any
    randomness is consumed: a non-PSD correlation matrix raises ``ValueError``
    here rather than producing partial output that silently drops correlation.

    M105: this function is now a thin shim over
    ``generate_tables_with_state`` that drops the state side-channel; its
    return contract is unchanged. Callers that need the trajectories used
    during generation (manifest emission, debugging, downstream feature
    pipelines) should call ``generate_tables_with_state`` directly.
    """
    tables, _state = generate_tables_with_state(config, rng)
    return tables


def generate_tables_with_state(
    config: PlotsimConfig,
    rng: Optional[np.random.Generator] = None,
) -> tuple[dict[str, pd.DataFrame], GenerationState]:
    """End-to-end pipeline returning tables AND the generation state used.

    Same generation path as ``generate_tables`` — the only difference is
    the additional ``GenerationState`` return value, which carries the
    per-entity trajectory positions. Useful for callers that need the
    ground-truth signal layer without re-deriving it from noisy cell
    values: M105's manifest emission is the primary consumer.

    Determinism contract is identical: same ``(config, rng)`` inputs
    produce both the same tables and the same trajectories.
    """
    # Local import: validation imports tables transitively, so a top-level
    # import would create a cycle.
    from plotsim.validation import validate_correlation_psd

    psd_issues = validate_correlation_psd(config)
    if psd_issues:
        names = [m.name for m in config.metrics]
        raise ValueError(
            f"Configured correlation matrix is not positive semi-definite "
            f"for metrics {names}. {psd_issues[0].message} "
            f"(min eigenvalue: {psd_issues[0].details.get('min_eigenvalue')})"
        )

    if rng is None:
        rng = np.random.default_rng(config.seed)
    dim_tables = build_all_dimensions(config, rng)
    n_periods = len(dim_tables["dim_date"])
    trajectories = compute_all_trajectories(config, n_periods)

    # M106: SCD Type 2 dim expansion runs after dimensions and trajectories
    # are built but BEFORE fact construction. Order matters: fact builders
    # use dim DataFrames to resolve PK FKs, and the SCD-expanded dim has a
    # different row count and adds the dim_row_id column the fact-side
    # ``attach_dim_row_id_to_facts`` step references afterwards. Configs
    # without any SCD columns get ``SCDState(dims={})`` and the helper is a
    # no-op, so the V1 templates that haven't opted in pay zero cost.
    dim_tables, scd_state = expand_scd_dims(config, dim_tables, trajectories)

    # Category B Layer 3 (SEC-08): hoist the Cholesky factor out of the
    # per-(cohort, period) hot loop. The correlation matrix is config-
    # invariant across the run (depends only on metric name order and
    # correlations list), so computing L once saves N_cohorts × N_periods
    # matrix+Cholesky rebuilds. The PSD gate above guarantees this Cholesky
    # succeeds; if the call ever surfaces an error here it means an external
    # caller bypassed ``validate_correlation_psd``.
    #
    # F-06 / 0.4.0: L must be indexed in the same order the downstream
    # ``apply_correlations`` sees its z vector. ``generate_entity_metrics``
    # runs ``_toposort_metrics`` on the incoming list before calling
    # ``generate_metrics_for_period``, which passes the toposorted effective
    # metrics to ``apply_correlations``. Pre-F-06 this hoist built L on the
    # declaration-order list, so any config with ``causal_lag`` (which
    # reshuffles metric positions) delivered each configured correlation
    # to whichever metric pair happened to live at those index positions
    # in the toposorted list — the wrong pair unless both swapped
    # symmetrically. Building L in toposort order here restores the
    # invariant "``L`` is indexed by the metric list passed downstream".
    cholesky_L: Optional[np.ndarray] = None
    if config.correlations:
        sorted_metrics = _toposort_metrics(list(config.metrics))
        mat = _build_correlation_matrix(
            sorted_metrics, list(config.correlations),
        )
        # M120: when ``compensate_correlations=True`` (builder default), the
        # user-specified matrix is the table-wide target; subtract the
        # trajectory's structural contribution off-diagonally so the copula
        # delivers what the user asked for, recombined additively with the
        # trajectory contribution at sample time. The metric cap mirrors the
        # mission spec — above 20 metrics the additive decomposition is too
        # noisy to satisfy the sign-match floor, so emit a warning and fall
        # through to the legacy direct-copula path.
        compensation_records: Optional[list[dict]] = None
        if config.compensate_correlations:
            from plotsim.metrics import (
                _MAX_METRICS_FOR_COMPENSATION,
                _format_correlation_compensation_warning,
                compensate_correlation_matrix,
                estimate_trajectory_covariance,
            )

            n_metrics = len(sorted_metrics)
            if n_metrics > _MAX_METRICS_FOR_COMPENSATION:
                warnings.warn(
                    f"compensate_correlations=true but config has "
                    f"{n_metrics} metrics (cap {_MAX_METRICS_FOR_COMPENSATION}); "
                    "trajectory-aware pre-compensation skipped — the additive "
                    "decomposition becomes too noisy at this scale. Falling "
                    "back to the direct copula path.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                traj_cov = estimate_trajectory_covariance(
                    config, metric_order=sorted_metrics,
                )
                mat, compensation_records = compensate_correlation_matrix(
                    mat, traj_cov, sorted_metrics, list(config.correlations),
                )
                warning_text = _format_correlation_compensation_warning(
                    compensation_records,
                )
                if warning_text:
                    warnings.warn(warning_text, UserWarning, stacklevel=2)

        # M111: project to nearest PD if needed. The load-time validator on
        # PlotsimConfig already projected and warned the user, but it built
        # the matrix in declaration order while this hoist uses toposort
        # order — so re-project deterministically here rather than thread
        # an order-aware permutation through the engine. PD inputs pass
        # through unchanged (byte-identical to pre-M111 for every config
        # whose user-specified matrix is already PD). Under M120 this also
        # absorbs the compensated matrix when pre-compensation pushed it
        # off the PD cone (compensation alters off-diagonals — Higham
        # restores PD without un-doing the compensation).
        from plotsim.metrics import project_correlation_matrix

        projected_mat, _projection_used, _used_fallback = project_correlation_matrix(mat)
        cholesky_L = np.linalg.cholesky(projected_mat)

        # M120: stash compensation records on the (private) config attr so
        # ``plotsim.manifest.build_manifest`` can surface them alongside the
        # M111 Higham records. ``None`` (no compensation, no records) ≠
        # empty list (compensation ran but every pair was already feasible
        # AND in scope) — keep the distinction in the manifest's wire shape.
        if compensation_records is not None:
            config._correlation_compensations = compensation_records

    # M107: compute the per-entity metric series ONCE and pass to both
    # ``build_fact_tables`` and ``build_bridge_tables``. Without this hoist
    # bridges would re-call ``generate_entity_metrics`` and consume RNG draws
    # downstream of fact construction, producing different fact values than
    # the pre-M107 baseline. Configs with empty ``bridges`` skip the bridge
    # call entirely; they still benefit from the single computation.
    n_periods = len(dim_tables["dim_date"])
    entity_metrics = _compute_entity_metrics(
        config, trajectories, n_periods, rng, cholesky_L=cholesky_L,
    )

    fact_tables = build_fact_tables(
        config, trajectories, dim_tables, rng,
        cholesky_L=cholesky_L, entity_metrics=entity_metrics,
    )
    # M106: append ``dim_row_id`` BEFORE ``assign_stages`` so the output
    # column ordering invariant "stage column appended last" still holds —
    # ``output._ordered_columns`` ranks unmodelled columns by DataFrame
    # insertion order, and putting dim_row_id first keeps stage at the
    # tail. The helper is a no-op when ``scd_state.is_empty``.
    fact_tables = attach_dim_row_id_to_facts(
        config, fact_tables, dim_tables, scd_state,
    )
    fact_tables = assign_stages(config, fact_tables)
    event_tables = build_event_tables(config, fact_tables, dim_tables, rng)
    event_tables = attach_dim_row_id_to_facts(
        config, event_tables, dim_tables, scd_state,
    )

    # M107: bridge tables run after fact/event construction so they see the
    # final SCD-expanded dim DataFrames and the same entity_metrics dict the
    # facts were built from. Bridges are static (non-temporal) so they slot
    # into the table dict alongside dims/facts/events without changing the
    # rest of the pipeline. Configs without ``bridges`` get an empty dict +
    # empty associations and the helper short-circuits.
    bridge_tables, bridge_associations = build_bridge_tables(
        config, dim_tables, trajectories, entity_metrics, scd_state, rng,
    )

    out: dict[str, pd.DataFrame] = {}
    out.update(dim_tables)
    out.update(fact_tables)
    out.update(event_tables)
    out.update(bridge_tables)
    return out, GenerationState(
        trajectories=trajectories,
        scd=scd_state,
        bridges=bridge_associations,
    )
