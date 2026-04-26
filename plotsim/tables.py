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
from typing import Any, Optional

import numpy as np
import pandas as pd
from faker import Faker

from plotsim.config import (
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
    StaticSource,
    Table,
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


def build_fact_tables(
    config: PlotsimConfig,
    trajectories: dict[str, np.ndarray],
    dim_tables: dict[str, pd.DataFrame],
    rng: np.random.Generator,
    cholesky_L: Optional[np.ndarray] = None,
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
    """
    dim_date = dim_tables.get("dim_date")
    if dim_date is None:
        raise ValueError("build_fact_tables requires dim_date to be built")
    n_periods = len(dim_date)

    per_entity_dims = _per_entity_dim_names(config)

    # Generate metric series per entity once. Each entity's RNG draws share
    # the top-level rng, so determinism is preserved across the whole run.
    arch_by_name = {a.name: a for a in config.archetypes}
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
        )

    # Category B Layer 4: materialize metric series as a dense (E, P, M) float64
    # ndarray keyed by config.entities order (not sorted name order — row order
    # in the fact tables must match the config entity iteration order).
    # Null values (MCAR / poisson-with-MCAR) become ``np.nan``. Downstream
    # column builders index into this array instead of dict-of-dict lookups.
    metrics_3d = _build_metrics_3d(config, entity_metrics, n_periods)

    fact_out: dict[str, pd.DataFrame] = {}
    fake = _make_faker(rng, config.locale)

    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        if tbl.grain == "per_entity_per_period":
            fact_out[tbl.name] = _build_per_entity_per_period_fact(
                tbl, config, entity_metrics, dim_tables, per_entity_dims,
                fake, rng, metrics_3d,
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
    if len(parent_entity_dim) != len(config.entities):
        raise ValueError(
            f"parent dim {parent_entity_table!r} has {len(parent_entity_dim)} "
            f"rows but config has {len(config.entities)} entities; "
            f"per_entity dims must be 1:1 with config.entities"
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
        )

    return _vectorized_per_entity_per_period_fact(
        tbl, config, dim_date, n_periods,
        parent_entity_dim, parent_entity_pk,
        local_entity_col, local_date_col, parent_date_pk,
        entity_cross_fks, parsed_cols, metrics_3d,
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
) -> pd.DataFrame:
    """Pre-Layer-4 row-by-row builder, kept as a fallback for fact tables that
    use ``FakerSource`` or ``boolean``-typed metric columns (paths where
    vectorization would reorder RNG draws or drop a Python coercion)."""
    del parsed_cols  # not used here; scalar path walks tbl.columns directly
    rows: list[dict] = []
    for entity_idx, entity in enumerate(config.entities):
        entity_pk_value = parent_entity_dim.iloc[entity_idx][parent_entity_pk]
        metric_series = entity_metrics[entity.name]
        cross_fks_for_entity = entity_cross_fks[entity.name]
        for period_idx in range(n_periods):
            row: dict = {}
            for col in tbl.columns:
                row[col.name] = _resolve_fact_cell(
                    col, period_idx, entity_pk_value,
                    local_entity_col, local_date_col, parent_date_pk,
                    metric_series, dim_date, cross_fks_for_entity, fake,
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
                row[col.name] = None
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
                col_arrays[col.name] = np.full(total_rows, None, dtype=object)
        else:
            col_arrays[col.name] = np.full(total_rows, None, dtype=object)

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

    With ``enforce_order=False``, both ``downgrade_delay`` and
    ``threshold_exit`` are ignored and each period chooses the
    highest-enter stage that the value satisfies. Free mode is
    stateless, so hysteresis has no meaning there.

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


# --- Orchestrator ------------------------------------------------------------


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
        cholesky_L = np.linalg.cholesky(mat)

    fact_tables = build_fact_tables(
        config, trajectories, dim_tables, rng, cholesky_L=cholesky_L,
    )
    fact_tables = assign_stages(config, fact_tables)
    event_tables = build_event_tables(config, fact_tables, dim_tables, rng)

    out: dict[str, pd.DataFrame] = {}
    out.update(dim_tables)
    out.update(fact_tables)
    out.update(event_tables)
    return out
