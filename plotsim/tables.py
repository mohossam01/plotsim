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

Notes on the public surface:
    1. Event tables that declare neither a ``row_count_source`` nor a
       threshold-typed column emit an empty DataFrame with the configured
       schema. The contract is "every configured table appears in the
       output dict" — emitting an empty table for an HR-style
       ``evt_attrition`` (no driver) preserves that contract.
    2. ``generated:timestamp`` on event columns resolves to the period's
       anchor date (no faker), since the provider name is non-faker.
    3. Stage column name is hardcoded to ``stage``. Adding a configurable
       column name would be a schema change.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from faker import Faker

from plotsim._column_dispatch import (
    COLUMN_DISPATCH,
    BuilderKind,
)
from plotsim._faker import _make_faker
from plotsim.config import (
    BridgeMetric,
    BridgeTableConfig,
    Column,
    DerivedSource,
    Entity,
    FKSource,
    FakerSource,
    GeneratedSource,
    LagSource,
    MetricSource,
    NarrativeSource,
    NestedSource,
    PlotsimConfig,
    PKSource,
    PoolSource,
    ProportionalSource,
    RangeSource,
    SCDType2Config,
    StaticSource,
    Table,
    TextBucketSource,
    ThresholdSource,
    parse_source,
)
from plotsim.dimensions import (
    _call_faker,
    _coerce_static,
    build_all_dimensions,
    sample_fk_values,
)
from plotsim.metrics import (
    _VECTORIZED_AUTO_THRESHOLD,
    _build_correlation_matrix,
    _toposort_metrics,
    generate_archetype_batch,
    generate_entity_metrics,
)
from plotsim.trajectory import compute_all_trajectories


# --- Helpers -----------------------------------------------------------------


def _per_entity_dim_names(config: PlotsimConfig) -> set[str]:
    return {t.name for t in config.tables if t.type == "dim" and t.grain == "per_entity"}


def _fact_topo_order(config: PlotsimConfig) -> list[str]:
    """0.6-M18: topological order over fact-table dependencies.

    Two edge types contribute to the graph:

      * ``per_parent_row`` child → parent fact (``parent_table`` field).
      * Any fact column with source ``fk:fct_<other>.<col>`` → referenced
        fact (sibling-fact reference, 0.6-M18 Fix 3).

    Returns fact names in a build order that guarantees every fact's
    dependencies are materialized before the fact itself. Stability:
    ties broken by ``config.tables`` declaration order. Raises on
    cycles — load-time validators already reject these but the runtime
    sort double-checks so a downstream caller can't accidentally feed
    an invalid config.
    """
    fact_names = [t.name for t in config.tables if t.type == "fact"]
    fact_name_set = set(fact_names)
    deps: dict[str, set[str]] = {name: set() for name in fact_names}
    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        if tbl.grain == "per_parent_row" and tbl.parent_table:
            deps[tbl.name].add(tbl.parent_table)
        for col in tbl.columns:
            try:
                parsed = parse_source(col.source)
            except ValueError:
                continue
            if isinstance(parsed, FKSource) and parsed.table in fact_name_set:
                deps[tbl.name].add(parsed.table)

    in_degree = {name: len(deps[name]) for name in fact_names}
    downstream: dict[str, set[str]] = {name: set() for name in fact_names}
    for name, ds in deps.items():
        for d in ds:
            downstream[d].add(name)

    queue: list[str] = [name for name in fact_names if in_degree[name] == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        # Preserve config.tables declaration order when multiple nodes
        # become ready at the same step.
        for ds_name in fact_names:
            if ds_name not in downstream[node]:
                continue
            in_degree[ds_name] -= 1
            if in_degree[ds_name] == 0:
                queue.append(ds_name)

    if len(order) != len(fact_names):
        unresolved = [n for n in fact_names if n not in order]
        raise ValueError(
            f"fact dependency cycle detected; could not resolve "
            f"build order for {unresolved!r}. Check parent_table and "
            f"fk:fct_* column sources."
        )
    return order


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


# Event PK widths are pre-sized for up to 10,000 rows so the same zero-pad
# convention applies regardless of the actual emitted row count for a given
# (entity, period). At most this overpads narrow tables by a few characters;
# the alternative is two PK passes (count rows, then re-format), which costs
# more than the spare digit. Used by ``_build_proportional_event`` and
# ``_resolve_event_row``.
_EVENT_PK_WIDTH_HINT: int = 10_000


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

    Returns a ``pd.api.extensions.ExtensionArray`` (Int64 / BooleanDtype) for
    int / boolean dtypes — preserves NaN as ``pd.NA`` and matches what
    ``output._coerce_integer_columns`` produces at write-time, so the
    in-memory dtype matches the on-disk dtype after a CSV round-trip with
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
    config: PlotsimConfig,
    n_periods: int,
) -> Optional[np.ndarray]:
    """Pre-compute the per-period summed seasonal strength.

    Returns a length-``n_periods`` ``float64`` array where entry ``t`` is the
    sum of every ``SeasonalEffect.strength`` whose ``months`` set contains
    period ``t``'s calendar month. Returns ``None`` when no effects are
    configured — the metrics pipeline short-circuits when
    ``seasonal_global == 0.0``.

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


def _resolve_generation_mode(config: PlotsimConfig) -> str:
    """Resolve ``"auto"`` to ``"serial"`` or ``"vectorized"`` from archetype batch size.

    Returns ``"serial"`` or ``"vectorized"`` — the two concrete modes
    ``_compute_entity_metrics`` dispatches against. ``"auto"`` selects
    ``vectorized`` when **the largest single-archetype entity group**
    crosses ``_VECTORIZED_AUTO_THRESHOLD`` (50). The vectorized path
    works archetype-batch-by-archetype-batch — its per-cell savings
    only amortize the per-batch numpy setup cost when at least one
    batch is large. A 60-entity, 12-archetype config (avg group size
    5) used to flip to vectorized on the old ``len(config.entities)``
    threshold and pay overhead with no win; the per-archetype variant
    keeps it on serial.

    Per-entity-override entities are excluded from the batch and run
    serial regardless of mode (handled inside the vectorized path),
    so they do not affect the size used for selection — the count is
    of the entity rows themselves.
    """
    mode = config.generation_mode
    if mode != "auto":
        return mode
    if not config.entities:
        return "serial"
    counts: dict[str, int] = {}
    for ent in config.entities:
        counts[ent.archetype] = counts.get(ent.archetype, 0) + ent.size
    largest_group = max(counts.values())
    return "vectorized" if largest_group >= _VECTORIZED_AUTO_THRESHOLD else "serial"


def _compute_entity_metrics(
    config: PlotsimConfig,
    trajectories: dict[str, np.ndarray],
    n_periods: int,
    rng: np.random.Generator,
    cholesky_L: Optional[np.ndarray] = None,
    cholesky_by_period: Optional[list[np.ndarray]] = None,
    bypass_counter: Optional[dict[str, int]] = None,  # noqa: ARG001 — M127b backward compat
) -> dict[str, dict[str, np.ndarray]]:
    """Generate the per-entity metric series dict the engine reads from.

    Hoisted out of ``build_fact_tables`` so the bridge generator can read
    the same series without re-running ``generate_entity_metrics`` (which
    would consume RNG twice and break determinism). Each entity's RNG draws
    share the top-level rng.

    When ``config.seasonal_effects`` is non-empty, computes a global
    per-period strength array once and threads it (along with each
    entity's ``seasonal_sensitivity``) into ``generate_entity_metrics``.
    Empty effects → ``seasonal_factors=None`` and the seasonal step is a
    no-op.

    Dispatches on ``config.generation_mode``. ``serial`` runs the per-entity
    loop. ``vectorized`` groups entities by archetype and calls
    ``generate_archetype_batch`` for each archetype's no-override subset;
    overridden entities still run the per-entity serial path inside the
    same dispatch so their exact behavior is preserved. ``auto`` resolves
    via ``_resolve_generation_mode``. The two paths consume RNG in
    different orders → cross-mode cell values are not byte-identical
    even at the same seed.
    """
    arch_by_name = {a.name: a for a in config.archetypes}
    seasonal_factors = _build_seasonal_factors(config, n_periods)
    mode = _resolve_generation_mode(config)
    if mode == "serial":
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
                cholesky_by_period=cholesky_by_period,
                seasonal_factors=seasonal_factors,
                entity_seasonal_sensitivity=entity.seasonal_sensitivity,
                treatment_lift_log_odds=entity.treatment_lift_log_odds,
                treatment_start_period=entity.treatment_start_period,
                treatment_target_metric=entity.treatment_target_metric,
            )
        return entity_metrics

    # Vectorized: archetype-batched.
    #
    # 1. Validate trajectory shapes for every entity (matches the serial
    #    path's contract — fail-fast on a mismatch is preferable to
    #    confusing downstream array-shape errors).
    for entity in config.entities:
        traj = trajectories[entity.name]
        if len(traj) != n_periods:
            raise ValueError(
                f"trajectory for entity {entity.name!r} has length {len(traj)} "
                f"but dim_date has {n_periods} periods"
            )

    # 2. Group entities by archetype — preserve the order in which each
    #    archetype first appears in ``config.entities`` so the RNG draw
    #    order is stable across runs. Within each group, preserve
    #    declaration order. Standard (no overrides) entities go to the
    #    batch; overridden entities run the serial path AFTER the batch
    #    closes, so all batch RNG draws happen first in a fixed order.
    archetype_order: list[str] = []
    standard_by_arch: dict[str, list] = {}
    overridden_by_arch: dict[str, list] = {}
    for entity in config.entities:
        if entity.archetype not in standard_by_arch:
            standard_by_arch[entity.archetype] = []
            overridden_by_arch[entity.archetype] = []
            archetype_order.append(entity.archetype)
        # 0.6-M8a: cold-start entities also exit the batch — the
        # archetype-batched path computes one shared trajectory per
        # archetype (no per-entity start_period axis), so an entity with
        # ``start_period > 0`` would silently use the wrong trajectory.
        # Route them through the serial path (same lane as overridden
        # entities), where ``compute_all_trajectories`` already passed
        # the right per-entity trajectory via ``trajectories[entity.name]``.
        # 0.6-M8c: same routing for entities with a treatment lift —
        # the vectorized batch path doesn't apply per-entity treatment
        # shifts (the trajectory tensor and centers are shared across
        # the batch axis). Routing to serial keeps the fix simple;
        # lifting the constraint is documented in the M8a completion
        # report under [m8a/vectorized-cold-start-fallback-perf] (the
        # same tensor reshaping unlocks both cold-start and treatment
        # in the vectorized path).
        if (
            entity.overrides
            or entity.start_period > 0
            or entity.treatment_lift_log_odds is not None
        ):
            overridden_by_arch[entity.archetype].append(entity)
        else:
            standard_by_arch[entity.archetype].append(entity)

    # 3. Run batched generation per archetype, then serial fallback for
    #    overridden entities. The result dict is keyed by entity.name —
    #    downstream ``_build_metrics_3d`` reorders into config.entities
    #    order via ``entity_names_ordered``, so insertion order here
    #    doesn't affect on-disk row order.
    entity_metrics_v: dict[str, dict[str, np.ndarray]] = {}
    for arch_name in archetype_order:
        archetype = arch_by_name.get(arch_name)
        if archetype is None:
            # Defensive — config validators reject unknown archetypes
            # at load time. If this fires, the operator constructed a
            # PlotsimConfig programmatically and bypassed validation.
            raise ValueError(
                f"entity references unknown archetype {arch_name!r}; "
                "vectorized generation cannot proceed"
            )
        standard_batch = standard_by_arch[arch_name]
        if standard_batch:
            batch_result = generate_archetype_batch(
                archetype,
                standard_batch,
                list(config.metrics),
                list(config.correlations),
                config.noise,
                n_periods,
                rng,
                cholesky_L=cholesky_L,
                cholesky_by_period=cholesky_by_period,
                seasonal_factors=seasonal_factors,
                bypass_counter=bypass_counter,
            )
            entity_metrics_v.update(batch_result)
        for entity in overridden_by_arch[arch_name]:
            traj = trajectories[entity.name]
            entity_metrics_v[entity.name] = generate_entity_metrics(
                traj,
                list(config.metrics),
                list(config.correlations),
                config.noise,
                rng,
                archetype=archetype,
                cholesky_L=cholesky_L,
                cholesky_by_period=cholesky_by_period,
                seasonal_factors=seasonal_factors,
                entity_seasonal_sensitivity=entity.seasonal_sensitivity,
                treatment_lift_log_odds=entity.treatment_lift_log_odds,
                treatment_start_period=entity.treatment_start_period,
                treatment_target_metric=entity.treatment_target_metric,
            )
    return entity_metrics_v


# --- M121b: chunked fact builder seam ---------------------------------------


def iter_fact_chunks(
    config: PlotsimConfig,
    fact_tables: dict[str, pd.DataFrame],
):
    """Yield per-archetype fact-DataFrame chunks for streaming Parquet.

    The streaming Parquet writer in ``plotsim.output`` consumes this
    generator to write each archetype batch as its own Parquet row
    group, instead of materializing the full pyarrow table from the
    unified DataFrame. Downstream consumers
    (``attach_dim_row_id_to_facts``, ``assign_stages``,
    ``build_event_tables``, ``build_bridge_tables``,
    ``entity_features``) all continue to receive the unified
    DataFrames returned by ``build_fact_tables`` — the chunked
    iterator is an additive seam, not a replacement.

    Yields ``(archetype_name, dict[fact_name → DataFrame])``. For
    ``per_entity_per_period`` facts, ``archetype_name`` is the
    archetype each chunk's entities share; the DataFrame is a slice
    of the unified DataFrame keyed by the row indices that belong to
    those entities. For ``per_period`` facts (no entity axis, no
    archetype concept), the full DataFrame is yielded once under the
    sentinel name ``"__per_period__"``.

    Implementation notes:

    * Slicing uses precomputed row-index spans derived from
      ``config.entities`` order and ``n_periods``. The fact builders
      emit rows in entity-major order
      (``[e0×p1..pN, e1×p1..pN, ...]``), so each entity contributes
      a contiguous run of ``n_periods`` rows. An archetype's chunk
      may not be a single contiguous block when entities of multiple
      archetypes are interleaved in ``config.entities``; the helper
      uses ``df.iloc`` with a list of row indices in those cases,
      which is still O(rows-in-chunk) and memory-efficient.
    * When the unified DataFrame's row count doesn't equal
      ``len(config.entities) × n_periods`` (e.g., a fact table whose
      entity-FK targets a per_subentity dim with a different
      cardinality), the helper falls back to yielding the full
      DataFrame under the sentinel name. The streaming writer treats
      that as a single-row-group write — correct, just no per-archetype
      decomposition.
    * Archetype iteration order matches ``_compute_entity_metrics``
      (first appearance in ``config.entities``) so streaming and
      non-streaming Parquet write row groups in the same logical
      order. Read-back DataFrames must compare equal across modes.
    """
    n_periods = config.time_window.period_count()
    expected_rows = len(config.entities) * n_periods

    # Build per-archetype lists of entity indices in config.entities order.
    archetype_order: list[str] = []
    indices_by_arch: dict[str, list[int]] = {}
    for i, entity in enumerate(config.entities):
        if entity.archetype not in indices_by_arch:
            indices_by_arch[entity.archetype] = []
            archetype_order.append(entity.archetype)
        indices_by_arch[entity.archetype].append(i)

    # Decide which fact tables stream per-archetype. The
    # per_entity_per_period grain has the entity-major row layout the
    # slicing model assumes. per_period facts (and any with a row-count
    # mismatch) yield once under the sentinel name.
    per_arch_facts: list[str] = []
    per_period_facts: list[str] = []
    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        df = fact_tables.get(tbl.name)
        if df is None:
            continue
        if tbl.grain == "per_entity_per_period" and len(df) == expected_rows:
            per_arch_facts.append(tbl.name)
        else:
            per_period_facts.append(tbl.name)

    # First emit per-archetype chunks for the entity-period facts.
    # When an archetype's entities are contiguous in ``config.entities``
    # (the production case — builder segments group by archetype, and
    # engine-direct configs with mixed archetypes are rare), the
    # archetype's row indices form a single contiguous slice and the
    # chunk is a pandas view instead of a fancy-indexed copy. This
    # matters for memory: a deep copy per chunk adds O(chunk-size)
    # Python heap on top of the unified DataFrame, swamping the
    # streaming Parquet pyarrow-buffer win on small/medium configs.
    # The non-contiguous fall-back (interleaved archetypes) does
    # allocate a copy via ``df.iloc[indices]`` — unavoidable for
    # discontiguous row sets.
    for arch in archetype_order:
        ent_indices = indices_by_arch[arch]
        is_contiguous = len(ent_indices) > 0 and ent_indices[-1] - ent_indices[0] + 1 == len(
            ent_indices
        )
        chunk: dict[str, pd.DataFrame] = {}
        for fact_name in per_arch_facts:
            df = fact_tables[fact_name]
            if is_contiguous:
                start = ent_indices[0] * n_periods
                stop = (ent_indices[-1] + 1) * n_periods
                chunk[fact_name] = df.iloc[start:stop]
            else:
                row_indices: list[int] = []
                for i in ent_indices:
                    row_indices.extend(range(i * n_periods, (i + 1) * n_periods))
                chunk[fact_name] = df.iloc[row_indices]
        if chunk:
            yield arch, chunk

    # Then emit per-period facts (and any rejected entity-period facts)
    # as a single chunk under the sentinel name.
    if per_period_facts:
        chunk = {name: fact_tables[name] for name in per_period_facts}
        yield "__per_period__", chunk


def build_fact_tables(
    config: PlotsimConfig,
    trajectories: dict[str, np.ndarray],
    dim_tables: dict[str, pd.DataFrame],
    rng: np.random.Generator,
    cholesky_L: Optional[np.ndarray] = None,
    cholesky_by_period: Optional[list[np.ndarray]] = None,
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

    ``entity_metrics`` may be passed in by callers that have already
    computed it (the orchestrator does this so bridge tables can share
    the same series without re-running ``generate_entity_metrics`` and
    burning RNG draws). When ``None``, the helper recomputes it inline.
    """
    dim_date = dim_tables.get("dim_date")
    if dim_date is None:
        raise ValueError("build_fact_tables requires dim_date to be built")
    n_periods = len(dim_date)

    per_entity_dims = _per_entity_dim_names(config)

    if entity_metrics is None:
        entity_metrics = _compute_entity_metrics(
            config,
            trajectories,
            n_periods,
            rng,
            cholesky_L=cholesky_L,
            cholesky_by_period=cholesky_by_period,
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

    # 0.6-M18: topological build order over fact dependencies. Two edge
    # types contribute: ``per_parent_row`` child → parent (parent_table)
    # and any fact column with ``fk:fct_*`` → referenced fact (sibling
    # references). Single pass through facts in topo order replaces the
    # earlier two-pass scheme and generalizes to multi-fact stars
    # (orders + line items + returns).
    fact_build_order = _fact_topo_order(config)
    tables_by_name = {t.name: t for t in config.tables}

    for fact_name in fact_build_order:
        tbl = tables_by_name[fact_name]
        if tbl.grain == "per_entity_per_period":
            df = _build_per_entity_per_period_fact(
                tbl,
                config,
                entity_metrics,
                dim_tables,
                per_entity_dims,
                fake,
                rng,
                metrics_3d,
                trajectories_2d,
            )
            fact_out[tbl.name] = _drop_cold_start_rows(df, config, n_periods)
        elif tbl.grain == "per_period":
            # Per-period (no entity axis) facts aggregate across all entities;
            # cold-start filtering at this grain would conflate "no entity yet
            # exists" with "all-aggregate-NaN" and isn't well-defined. Leave
            # the rows in place — downstream consumers see NaN-aggregated
            # cells if every entity is dormant at a period.
            fact_out[tbl.name] = _build_per_period_fact(
                tbl,
                config,
                entity_metrics,
                dim_tables,
                fake,
                metrics_3d,
            )
        elif tbl.grain == "variable":
            # 0.6-M18: variable-grain fact (parent of per_parent_row
            # children OR sibling-referenced fact). Row count is
            # trajectory-driven via ``row_count_source``. Routes through
            # ``_build_variable_grain_fact`` which reads the driver
            # metric directly from ``entity_metrics`` — no intermediate
            # driver-host fact required.
            if tbl.row_count_source is None:
                raise ValueError(
                    f"variable-grain fact {tbl.name!r} has no "
                    f"row_count_source; declare one (e.g. "
                    f"'proportional:order_volume:scale:1.5') so the "
                    f"engine can derive per-(entity, period) row counts"
                )
            parsed_rc = parse_source(tbl.row_count_source)
            if not isinstance(parsed_rc, ProportionalSource):
                raise ValueError(
                    f"variable-grain fact {tbl.name!r} row_count_source "
                    f"{tbl.row_count_source!r} resolves to "
                    f"{type(parsed_rc).__name__}; only "
                    f"ProportionalSource is supported on variable-grain "
                    f"fact tables in 0.6"
                )
            fact_out[tbl.name] = _build_variable_grain_fact(
                tbl,
                parsed_rc,
                config,
                dim_tables,
                per_entity_dims,
                entity_metrics,
                fact_out,
                fake,
                rng,
            )
        elif tbl.grain == "per_parent_row":
            fact_out[tbl.name] = _build_per_parent_row_fact(
                tbl,
                config,
                fact_out,
                dim_tables,
                per_entity_dims,
                fake,
                rng,
            )
        else:
            raise ValueError(
                f"fact table {tbl.name!r} has unsupported grain "
                f"{tbl.grain!r}; expected per_entity_per_period, "
                f"per_period, variable, or per_parent_row"
            )

    return fact_out


def _drop_cold_start_rows(
    fact_df: pd.DataFrame,
    config: PlotsimConfig,
    n_periods: int,
) -> pd.DataFrame:
    """Drop per-(entity, period) rows where ``period_index < entity.start_period``.

    0.6-M8a: cold-start entities NaN-fill their trajectory prefix; the
    metric pipeline emits ``None`` cells for those periods, but the row
    itself is still constructed (in entity-major order, ``E * P`` rows
    total). This helper takes the entity-major fact DataFrame and drops
    the cold-start prefix rows for each entity, leaving rows for every
    active period only.

    Fast-path: if every entity has ``start_period == 0`` the function
    returns the input unchanged — no allocation, no copy. This preserves
    pre-M8a behaviour byte-for-byte for existing configs.

    Row order assumption: ``_build_per_entity_per_period_fact`` emits
    rows in ``config.entities`` order, ``n_periods`` rows per entity. The
    flat index ``i * n_periods + p`` corresponds to entity ``i`` at
    period ``p``. Both vectorized and scalar paths preserve this order.
    """
    if all(e.start_period == 0 for e in config.entities):
        return fact_df
    expected_rows = len(config.entities) * n_periods
    if len(fact_df) != expected_rows:
        # Defensive: if a future per_entity_per_period builder breaks the
        # entity-major contract, surface it loudly rather than silently
        # mangling rows. Cold-start configs simply won't ship until the
        # contract is restored.
        raise ValueError(
            f"cold-start filter expected {expected_rows} rows "
            f"({len(config.entities)} entities × {n_periods} periods) "
            f"but fact DataFrame has {len(fact_df)}; the per-entity-per-period "
            f"builder broke the entity-major row order this filter assumes"
        )
    keep = np.ones(expected_rows, dtype=bool)
    for i, ent in enumerate(config.entities):
        sp = ent.start_period
        if sp > 0:
            base = i * n_periods
            keep[base : base + min(sp, n_periods)] = False
    if keep.all():
        return fact_df
    return fact_df.loc[keep].reset_index(drop=True)


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
        subset=[parent_entity_pk],
        keep="first",
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
            f"fact table {tbl.name!r} has grain per_entity_per_period but no FK column to dim_date"
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
                col_cfg,
                parent_pks,
                1,
                rng,
                anchored_value=anchored,
            )[0]
        entity_cross_fks[entity.name] = per_entity_assignments

    # Fallback path: any column whose resolution consumes RNG at fact-build
    # time forces the scalar per-row loop so call order is preserved. No
    # shipped template exercises this branch via FakerSource alone, but the
    # `narrative_reviews` template lands here via NarrativeSource.
    # F3 (M102): boolean MetricSource / LagSource columns no longer force the
    # scalar fallback — `_coerce_array_for_dtype` handles them correctly in
    # the vectorized path.
    forces_scalar = any(
        isinstance(p, (FakerSource, NarrativeSource, NestedSource)) for _, p in parsed_cols
    )
    if forces_scalar or metrics_3d is None:
        return _scalar_per_entity_per_period_fact(
            tbl,
            config,
            entity_metrics,
            dim_date,
            n_periods,
            parent_entity_dim,
            parent_entity_pk,
            local_entity_col,
            local_date_col,
            parent_date_pk,
            entity_cross_fks,
            fake,
            rng,
            parsed_cols,
            trajectories_2d=trajectories_2d,
        )

    return _vectorized_per_entity_per_period_fact(
        tbl,
        config,
        dim_date,
        n_periods,
        parent_entity_dim,
        parent_entity_pk,
        local_entity_col,
        local_date_col,
        parent_date_pk,
        entity_cross_fks,
        parsed_cols,
        metrics_3d,
        rng=rng,
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
    rng: np.random.Generator,
    parsed_cols: list[tuple[Column, Any]],
    trajectories_2d: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Row-by-row builder, kept as a fallback for fact tables that use
    ``FakerSource`` / ``NarrativeSource`` or ``boolean``-typed metric columns
    (paths where vectorization would reorder RNG draws or drop a Python
    coercion).

    ``trajectories_2d`` (E, P) is forwarded to ``_resolve_fact_cell`` one
    entity-row at a time so ``TextBucketSource`` and ``NarrativeSource``
    columns can read the position. ``rng`` and the current ``entity`` are
    threaded for the same reason: ``NarrativeSource`` consumes one RNG
    draw per slot per row and looks up the entity's archetype to pick the
    lexicon. Non-narrative / non-bucket facts can ignore the extra ctx
    keys (``_resolve_fact_cell`` only consults what its dispatched
    handler reads).
    """
    del parsed_cols  # not used here; scalar path walks tbl.columns directly
    rows: list[dict] = []
    for entity_idx, entity in enumerate(config.entities):
        entity_pk_value = parent_entity_dim.iloc[entity_idx][parent_entity_pk]
        metric_series = entity_metrics[entity.name]
        cross_fks_for_entity = entity_cross_fks[entity.name]
        traj_for_entity = trajectories_2d[entity_idx] if trajectories_2d is not None else None
        for period_idx in range(n_periods):
            row: dict = {}
            for col in tbl.columns:
                row[col.name] = _resolve_fact_cell(
                    col,
                    period_idx,
                    entity_pk_value,
                    local_entity_col,
                    local_date_col,
                    parent_date_pk,
                    metric_series,
                    dim_date,
                    cross_fks_for_entity,
                    fake,
                    trajectory_for_entity=traj_for_entity,
                    entity=entity,
                    rng=rng,
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
    rng: Optional[np.random.Generator] = None,
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
    base_ctx = {
        "config": config,
        "dim_date": dim_date,
        "n_periods": n_periods,
        "n_entities": n_entities,
        "total_rows": total_rows,
        "parent_entity_dim": parent_entity_dim,
        "parent_entity_pk": parent_entity_pk,
        "local_entity_col": local_entity_col,
        "local_date_col": local_date_col,
        "parent_date_pk": parent_date_pk,
        "entity_cross_fks": entity_cross_fks,
        "metric_name_to_idx": metric_name_to_idx,
        "metrics_3d": metrics_3d,
        "trajectories_2d": trajectories_2d,
        "entity_pks": entity_pks,
        "entity_pk_repeated": entity_pk_repeated,
        "date_key_tiled": date_key_tiled,
        "period_idx_col": period_idx_col,
        # 0.6-M19 Fix 2: RangeSource vec handler reads ``rng`` for its
        # bulk ``rng.uniform`` / ``rng.integers`` draw. Other handlers
        # ignore the key.
        "rng": rng,
    }
    for col, parsed in parsed_cols:
        ctx = dict(base_ctx)
        ctx["col"] = col
        col_arrays[col.name] = COLUMN_DISPATCH.dispatch(
            BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
            parsed,
            ctx,
        )

    return pd.DataFrame({col.name: col_arrays[col.name] for col, _ in parsed_cols})


# --- Vectorized fact-cell dispatch handlers ----------------------------------


def _fact_vec_fk(parsed: FKSource, ctx: dict):
    col = ctx["col"]
    if col.name == ctx["local_entity_col"]:
        return ctx["entity_pk_repeated"]
    if col.name == ctx["local_date_col"]:
        return ctx["date_key_tiled"]
    cross_vals = np.asarray(
        [ctx["entity_cross_fks"][e.name].get(col.name) for e in ctx["config"].entities],
        dtype=object,
    )
    return np.repeat(cross_vals, ctx["n_periods"])


def _fact_vec_metric(parsed: MetricSource, ctx: dict):
    col = ctx["col"]
    metric_name_to_idx = ctx["metric_name_to_idx"]
    if parsed.metric not in metric_name_to_idx:
        raise ValueError(
            f"fact column {col.name!r} references metric "
            f"{parsed.metric!r}, which was not generated; check config.metrics"
        )
    m_idx = metric_name_to_idx[parsed.metric]
    arr = ctx["metrics_3d"][:, :, m_idx].ravel(order="C").copy()
    return _coerce_array_for_dtype(arr, col.dtype)


def _fact_vec_lag(parsed: LagSource, ctx: dict):
    col = ctx["col"]
    metric_name_to_idx = ctx["metric_name_to_idx"]
    if parsed.metric not in metric_name_to_idx:
        return _coerce_array_for_dtype(
            np.full(ctx["total_rows"], np.nan, dtype=np.float64),
            col.dtype,
        )
    m_idx = metric_name_to_idx[parsed.metric]
    n = parsed.periods
    base = np.arange(ctx["n_periods"], dtype=np.int64)
    target_idx = base - n
    # "If history too short, fall back to current period" — scalar
    # semantics preserved by mapping out-of-range to the current period.
    target_idx = cast(np.ndarray, np.where(target_idx < 0, base, target_idx))
    sliced = ctx["metrics_3d"][:, target_idx, m_idx]  # (E, P)
    return _coerce_array_for_dtype(
        sliced.ravel(order="C").copy(),
        col.dtype,
    )


def _fact_vec_pk(parsed: PKSource, ctx: dict):
    col = ctx["col"]
    entity_pks = ctx["entity_pks"]
    n_entities = ctx["n_entities"]
    n_periods = ctx["n_periods"]
    pk_rows = [
        f"{col.name}-{p:04d}-{entity_pks[i]}" for i in range(n_entities) for p in range(n_periods)
    ]
    return np.asarray(pk_rows, dtype=object)


def _fact_vec_generated(parsed: GeneratedSource, ctx: dict):
    provider = parsed.provider
    dim_date = ctx["dim_date"]
    n_entities = ctx["n_entities"]
    if provider == "timestamp":
        dates = dim_date["date"].tolist()
        promoted: list[Any] = []
        for d in dates:
            if isinstance(d, _dt.date) and not isinstance(d, _dt.datetime):
                promoted.append(_dt.datetime(d.year, d.month, d.day))
            else:
                promoted.append(d)
        promoted_arr = np.asarray(promoted, dtype=object)
        return np.tile(promoted_arr, n_entities)
    if provider == "date_key":
        return np.tile(dim_date["date_key"].to_numpy(), n_entities)
    if provider == "period_label":
        return np.tile(dim_date["period_label"].to_numpy(), n_entities)
    raise ValueError(f"unsupported generated provider {provider!r} on fact/event tables")


def _fact_vec_static(parsed: StaticSource, ctx: dict):
    return np.full(ctx["total_rows"], parsed.value, dtype=object)


def _fact_vec_derived(parsed: DerivedSource, ctx: dict):
    col = ctx["col"]
    if parsed.field == "period_index":
        return ctx["period_idx_col"].copy()
    if parsed.field == "entity_id":
        return ctx["entity_pk_repeated"]
    raise ValueError(f"fact column {col.name!r} derived field {parsed.field!r} not supported")


def _fact_vec_range(parsed: RangeSource, ctx: dict):
    """0.6-M19 Fix 2: bulk per-row uniform draw on a vectorized fact.

    Integer columns get ``rng.integers(min, max + 1)`` (inclusive
    upper bound — matches numpy's semantics for discrete ranges);
    float columns get ``rng.uniform(min, max)`` (exclusive upper
    bound — matches numpy's continuous-range convention).
    """
    col = ctx["col"]
    rng = ctx["rng"]
    if rng is None:
        raise ValueError(
            f"fact column {col.name!r} has source {col.source!r} but no "
            f"RNG was supplied to the vectorized fact builder; range "
            f"draws require the per-table RNG"
        )
    total_rows = ctx["total_rows"]
    if col.dtype == "int":
        return rng.integers(int(parsed.min), int(parsed.max) + 1, size=total_rows)
    return rng.uniform(parsed.min, parsed.max, size=total_rows)


def _fact_vec_pool(parsed: PoolSource, ctx: dict):
    """Bulk per-row pool draw on a vectorized per_entity_per_period fact.

    Output is entity-major (matches ``entity_pk_repeated`` /
    ``date_key_tiled`` layout). One bulk ``rng.integers`` draw per
    entity sized to ``n_periods``, scattered into the contiguous
    entity block. Per-entity draws keep ordering stable when entities
    have heterogeneous pool sizes.
    """
    del parsed  # PoolSource carries only a marker name; data is on col.value_pool.
    col = ctx["col"]
    rng = ctx["rng"]
    if rng is None:
        raise ValueError(
            f"fact column {col.name!r} has source {col.source!r} but no "
            f"RNG was supplied to the vectorized fact builder; pool "
            f"draws require the per-table RNG"
        )
    if col.value_pool is None:
        raise ValueError(
            f"fact column {col.name!r} declares pool source {col.source!r} "
            f"but Column.value_pool is None; Column._pool_pairing should "
            f"have rejected this at load"
        )
    config = ctx["config"]
    n_periods = ctx["n_periods"]
    total_rows = ctx["total_rows"]
    out = np.empty(total_rows, dtype=object)
    cursor = 0
    for entity in config.entities:
        choices = col.value_pool.get(entity.name)
        if choices is None:
            raise ValueError(
                f"fact column {col.name!r} value_pool has no entry for "
                f"entity {entity.name!r}; validate_value_pool_coverage "
                f"should have caught this at load"
            )
        indices = rng.integers(0, len(choices), size=n_periods)
        for k in range(n_periods):
            out[cursor + k] = _coerce_static(choices[int(indices[k])], col.dtype)
        cursor += n_periods
    return out


def _fact_vec_text_bucket(parsed: TextBucketSource, ctx: dict):
    # M105: trajectory-position-driven text emission. ``trajectories_2d``
    # is shape (E, P); flatten in the same row-major (entity, period)
    # order the entity_pk_repeated / date_key_tiled axes use, then map
    # each position into a bucket index. ``min(int(p * N), N - 1)``
    # closes the [0, 1] interval at the top so position == 1.0 lands
    # in the last bucket rather than overflowing.
    col = ctx["col"]
    trajectories_2d = ctx["trajectories_2d"]
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
    return bucket_arr[indices]


def _fact_vec_unsupported(parsed: Any, ctx: dict):
    col = ctx["col"]
    raise ValueError(
        f"fact column {col.name!r} source {col.source!r} is not "
        f"supported on per_entity_per_period fact tables"
    )


# Vectorized per-entity-per-period fact builder dispatchers. Note the
# critical FakerSource contract — Faker columns route through the scalar
# path BEFORE this dispatcher ever runs. The caller in
# ``_build_per_entity_per_period_fact`` checks ``has_faker`` and selects
# the scalar builder when any column is FakerSource. This module-level
# registry therefore intentionally has NO FakerSource handler for the
# vectorized site: reaching the dispatcher with a FakerSource is itself
# an internal wiring bug.
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
    FKSource,
    _fact_vec_fk,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
    MetricSource,
    _fact_vec_metric,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
    LagSource,
    _fact_vec_lag,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
    PKSource,
    _fact_vec_pk,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
    GeneratedSource,
    _fact_vec_generated,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
    StaticSource,
    _fact_vec_static,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
    DerivedSource,
    _fact_vec_derived,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
    TextBucketSource,
    _fact_vec_text_bucket,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
    RangeSource,
    _fact_vec_range,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
    PoolSource,
    _fact_vec_pool,
)
COLUMN_DISPATCH.register_unsupported(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_VECTORIZED,
    _fact_vec_unsupported,
)


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
    entity: Optional[Entity] = None,
    rng: Optional[np.random.Generator] = None,
):
    """Scalar per_entity_per_period fact cell resolver.

    M127b: dispatch table moved into ``COLUMN_DISPATCH`` under the
    ``PER_ENTITY_PER_PERIOD_FACT_SCALAR`` builder kind. The handler bodies
    below are unchanged from the pre-M127b inline ladder; the registry
    just collapses the ``isinstance`` chain into a dict lookup.

    ``entity`` and ``rng`` are present for the ``NarrativeSource`` handler
    only; other handlers ignore them. Callers from inside the engine pass
    both; tests that call this directly with positional args (pre-M10) keep
    working because the new kwargs default to ``None``.
    """
    parsed = parse_source(col.source)
    ctx = {
        "col": col,
        "period_idx": period_idx,
        "entity_pk_value": entity_pk_value,
        "local_entity_col": local_entity_col,
        "local_date_col": local_date_col,
        "parent_date_pk": parent_date_pk,
        "metric_series": metric_series,
        "dim_date": dim_date,
        "cross_fks_for_entity": cross_fks_for_entity,
        "fake": fake,
        "trajectory_for_entity": trajectory_for_entity,
        "entity": entity,
        "rng": rng,
    }
    return COLUMN_DISPATCH.dispatch(
        BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
        parsed,
        ctx,
    )


def _fact_scalar_fk(parsed: FKSource, ctx: dict):
    col = ctx["col"]
    if col.name == ctx["local_entity_col"]:
        return ctx["entity_pk_value"]
    if col.name == ctx["local_date_col"]:
        return ctx["dim_date"].iloc[ctx["period_idx"]][ctx["parent_date_pk"]]
    # Cross-dim FK (e.g. plan_id) — value precomputed once per entity by
    # _build_per_entity_per_period_fact (FIX-04). Same value broadcast
    # across all periods for this entity.
    return ctx["cross_fks_for_entity"].get(col.name)


def _fact_scalar_metric(parsed: MetricSource, ctx: dict):
    col = ctx["col"]
    series = ctx["metric_series"].get(parsed.metric)
    if series is None:
        raise ValueError(
            f"fact column {col.name!r} references metric "
            f"{parsed.metric!r}, which was not generated; check config.metrics"
        )
    value = series[ctx["period_idx"]]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return _coerce_metric_value(value, col.dtype)


def _fact_scalar_lag(parsed: LagSource, ctx: dict):
    # Lag-typed columns read N periods back from the same entity series.
    # If history is too short, fall back to the current period.
    col = ctx["col"]
    period_idx = ctx["period_idx"]
    series = ctx["metric_series"].get(parsed.metric)
    if series is None:
        return None
    target_idx = period_idx - parsed.periods
    if target_idx < 0:
        target_idx = period_idx
    value = series[target_idx]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return _coerce_metric_value(value, col.dtype)


def _fact_scalar_pk(parsed: PKSource, ctx: dict):
    # Single-column PK on a composite-grain table is a surrogate;
    # build it deterministically from period and entity.
    col = ctx["col"]
    return f"{col.name}-{ctx['period_idx']:04d}-{ctx['entity_pk_value']}"


def _fact_scalar_generated(parsed: GeneratedSource, ctx: dict):
    return _resolve_generated(
        parsed.provider,
        ctx["period_idx"],
        ctx["dim_date"],
        ctx["fake"],
    )


def _fact_scalar_faker(parsed: FakerSource, ctx: dict):
    return _call_faker(ctx["fake"], parsed.method, parsed.kwargs)


def _fact_scalar_static(parsed: StaticSource, ctx: dict):
    return parsed.value


def _fact_scalar_derived(parsed: DerivedSource, ctx: dict):
    col = ctx["col"]
    if parsed.field == "period_index":
        return ctx["period_idx"]
    if parsed.field == "entity_id":
        return ctx["entity_pk_value"]
    raise ValueError(f"fact column {col.name!r} derived field {parsed.field!r} not supported")


def _fact_scalar_nested(parsed: NestedSource, ctx: dict):
    """Build one nested cell (struct → dict, array → list) on a scalar fact row.

    0.6-M14c: forced-scalar path (the vectorized branch hands off via
    ``forces_scalar`` because nested values can't ride numpy's typed
    arrays). Per-row determinism follows the seeded engine RNG, so
    same seed → byte-identical column.
    """
    from plotsim.dimensions import _generate_nested_value

    col = ctx["col"]
    rng = ctx["rng"]
    if rng is None:
        raise ValueError(
            f"fact column {col.name!r} has source 'nested' but no RNG "
            f"in scalar fact context; nested cell generation requires rng"
        )
    return _generate_nested_value(col, rng)


def _fact_scalar_narrative(parsed: NarrativeSource, ctx: dict):
    """Build one trajectory- and archetype-driven text cell.

    Resolution mirrors ``_fact_scalar_text_bucket`` for the band index,
    then layers per-slot phrase sampling on top:

      1. Read trajectory position ``p`` for this (entity, period).
      2. Compute band index via ``min(int(p * N), N - 1)`` so ``p == 1.0``
         lands in the last band.
      3. Look up the per-archetype lexicon — the lexicons dict's keys are
         validated at config-load to cover every assigned archetype, so
         a missing key here would be an internal wiring bug.
      4. For each ``{slot}`` placeholder in template scan order, draw one
         phrase index uniformly from the band's pool via the seeded RNG.
      5. ``template.format(**slot_values)`` → final cell.

    RNG-byte-parity: the per-slot draws use ``rng.integers(0, len(pool))``
    which advances the seeded engine RNG by one draw per slot per row.
    The fact builder iterates ``config.entities`` × ``range(n_periods)``
    in entity-major order; same-seed runs draw in the same order →
    byte-identical text columns.
    """
    # ``parsed.key`` is a marker key only; the lexicon + template live on
    # ``Column.narrative`` (paired field, validated at load).
    del parsed
    col = ctx["col"]
    cfg = col.narrative
    if cfg is None:
        # ``Column._narrative_pairing`` already rejects this combination at
        # config load; reaching the dispatcher with a missing config is an
        # internal wiring bug.
        raise ValueError(
            f"fact column {col.name!r} declares narrative source "
            f"{col.source!r} but Column.narrative is None; this is an "
            f"internal wiring bug, not a config error"
        )
    trajectory_for_entity = ctx["trajectory_for_entity"]
    if trajectory_for_entity is None:
        raise ValueError(
            f"fact column {col.name!r} declares narrative source "
            f"{col.source!r} but trajectory_for_entity was not threaded "
            f"into the scalar fact builder; this is an internal wiring "
            f"bug, not a config error"
        )
    entity = ctx["entity"]
    rng = ctx["rng"]
    if entity is None or rng is None:
        raise ValueError(
            f"fact column {col.name!r} narrative source needs both "
            f"`entity` and `rng` in ctx (got entity={entity!r}, "
            f"rng={'set' if rng is not None else 'None'}); this is an "
            f"internal wiring bug, not a config error"
        )
    archetype_lexicon = cfg.lexicons.get(entity.archetype)
    if archetype_lexicon is None:
        raise ValueError(
            f"fact column {col.name!r} narrative lexicon has no entry "
            f"for archetype {entity.archetype!r}; cross-config validator "
            f"`validate_narrative_columns` should have caught this at "
            f"load time — this is an internal wiring bug"
        )

    position = float(trajectory_for_entity[ctx["period_idx"]])
    n_bands = len(cfg.bands)
    band_idx = min(int(position * n_bands), n_bands - 1)
    band_idx = max(band_idx, 0)
    band_name = cfg.bands[band_idx]

    slot_values: dict[str, str] = {}
    for slot_name in cfg.template_slots():
        if slot_name in slot_values:
            # Duplicate placeholders are rejected at config load; defensive
            # skip keeps the loop O(unique slots) rather than O(template length).
            continue
        phrase_pool = archetype_lexicon[slot_name][band_name]
        choice_idx = int(rng.integers(0, len(phrase_pool)))
        slot_values[slot_name] = phrase_pool[choice_idx]
    return cfg.template.format(**slot_values)


def _fact_scalar_text_bucket(parsed: TextBucketSource, ctx: dict):
    # M105: scalar-fallback bucket lookup. Same index arithmetic as the
    # vectorized branch — ``min(int(p * N), N - 1)`` so p == 1.0 lands
    # in the last bucket rather than overflowing.
    col = ctx["col"]
    trajectory_for_entity = ctx["trajectory_for_entity"]
    if trajectory_for_entity is None:
        raise ValueError(
            f"fact column {col.name!r} declares text-bucket source "
            f"{col.source!r} but trajectory_for_entity was not threaded "
            f"into the scalar fact builder; this is an internal wiring "
            f"bug, not a config error"
        )
    position = float(trajectory_for_entity[ctx["period_idx"]])
    n_buckets = len(parsed.buckets)
    idx = min(int(position * n_buckets), n_buckets - 1)
    idx = max(idx, 0)
    return parsed.buckets[idx]


def _fact_scalar_range(parsed: RangeSource, ctx: dict):
    """0.6-M19 Fix 2: per-cell uniform draw on a scalar fact.

    Mirrors :func:`_fact_vec_range` but emits a single value per cell
    rather than a bulk array. The scalar path's RNG comes through the
    caller's per-table generator; absence is a wiring bug (every
    scalar fact builder threads ``rng`` into the ctx).
    """
    col = ctx["col"]
    rng = ctx["rng"]
    if rng is None:
        raise ValueError(
            f"fact column {col.name!r} has source {col.source!r} but no "
            f"RNG was supplied to _resolve_fact_cell; range draws "
            f"require the per-table RNG"
        )
    if col.dtype == "int":
        return int(rng.integers(int(parsed.min), int(parsed.max) + 1))
    return float(rng.uniform(parsed.min, parsed.max))


def _fact_scalar_pool(parsed: PoolSource, ctx: dict):
    """Per-cell pool draw on a scalar per_entity_per_period fact.

    Looks up the per-entity choice list on ``col.value_pool`` keyed by
    the current row's entity name, then draws one index from the
    seeded RNG. Same shape as ``_evt_row_pool`` but the entity is
    already in ctx (no PK reverse-lookup needed on the per-entity
    dim).
    """
    del parsed  # PoolSource carries only a marker name; data is on col.value_pool.
    col = ctx["col"]
    rng = ctx["rng"]
    entity = ctx["entity"]
    if rng is None or entity is None:
        raise ValueError(
            f"fact column {col.name!r} pool source needs both `entity` and "
            f"`rng` in ctx (got entity={entity!r}, "
            f"rng={'set' if rng is not None else 'None'}); this is an "
            f"internal wiring bug, not a config error"
        )
    if col.value_pool is None:
        raise ValueError(
            f"fact column {col.name!r} declares pool source {col.source!r} "
            f"but Column.value_pool is None; Column._pool_pairing should "
            f"have rejected this at load"
        )
    choices = col.value_pool.get(entity.name)
    if choices is None:
        raise ValueError(
            f"fact column {col.name!r} value_pool has no entry for entity "
            f"{entity.name!r}; validate_value_pool_coverage should have "
            f"caught this at load"
        )
    pick = int(rng.integers(0, len(choices)))
    return _coerce_static(choices[pick], col.dtype)


def _fact_scalar_unsupported(parsed: Any, ctx: dict):
    col = ctx["col"]
    raise ValueError(
        f"fact column {col.name!r} source {col.source!r} is not supported on "
        f"per_entity_per_period fact tables"
    )


# Register the scalar fact-cell resolvers with the shared dispatcher. Adding
# a new source type to per_entity_per_period fact scalar columns means
# adding one ``register(...)`` call here.
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    FKSource,
    _fact_scalar_fk,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    MetricSource,
    _fact_scalar_metric,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    LagSource,
    _fact_scalar_lag,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    PKSource,
    _fact_scalar_pk,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    GeneratedSource,
    _fact_scalar_generated,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    FakerSource,
    _fact_scalar_faker,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    StaticSource,
    _fact_scalar_static,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    DerivedSource,
    _fact_scalar_derived,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    TextBucketSource,
    _fact_scalar_text_bucket,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    NarrativeSource,
    _fact_scalar_narrative,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    NestedSource,
    _fact_scalar_nested,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    RangeSource,
    _fact_scalar_range,
)
COLUMN_DISPATCH.register(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    PoolSource,
    _fact_scalar_pool,
)
COLUMN_DISPATCH.register_unsupported(
    BuilderKind.PER_ENTITY_PER_PERIOD_FACT_SCALAR,
    _fact_scalar_unsupported,
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
        ctx = {
            "col": None,  # filled per column
            "period_idx": period_idx,
            "date_fk": date_fk,
            "dim_date": dim_date,
            "fake": fake,
            "period_mean": period_mean,
            "metric_idx_by_name": metric_idx_by_name,
        }
        for col in tbl.columns:
            parsed = parse_source(col.source)
            ctx["col"] = col
            row[col.name] = COLUMN_DISPATCH.dispatch(
                BuilderKind.PER_PERIOD_FACT,
                parsed,
                ctx,
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=[c.name for c in tbl.columns])


# --- Per-period fact dispatch handlers ---------------------------------------


def _per_period_fk(parsed: FKSource, ctx: dict):
    col = ctx["col"]
    date_fk = ctx["date_fk"]
    if date_fk and col.name == date_fk[0]:
        return ctx["dim_date"].iloc[ctx["period_idx"]][date_fk[2]]
    return _per_period_unsupported(parsed, ctx)


def _per_period_metric(parsed: MetricSource, ctx: dict):
    col = ctx["col"]
    if parsed.metric not in ctx["metric_idx_by_name"]:
        return None
    m_idx = ctx["metric_idx_by_name"][parsed.metric]
    raw_val = ctx["period_mean"][ctx["period_idx"], m_idx]
    if np.isnan(raw_val):
        return None
    return _coerce_metric_value(float(raw_val), col.dtype)


def _per_period_pk(parsed: PKSource, ctx: dict):
    col = ctx["col"]
    return f"{col.name}-{ctx['period_idx']:04d}"


def _per_period_generated(parsed: GeneratedSource, ctx: dict):
    return _resolve_generated(
        parsed.provider,
        ctx["period_idx"],
        ctx["dim_date"],
        ctx["fake"],
    )


def _per_period_faker(parsed: FakerSource, ctx: dict):
    return _call_faker(ctx["fake"], parsed.method, parsed.kwargs)


def _per_period_static(parsed: StaticSource, ctx: dict):
    return parsed.value


def _per_period_unsupported(parsed: Any, ctx: dict):
    # F14 (M102): explicit raise instead of silent ``None`` fill. Mission
    # 100 named this ladder as a silent-dispatch site; an unhandled source
    # type on a per-period fact column produced a column of None values
    # with no signal to the user.
    col = ctx["col"]
    raise TypeError(
        f"per-period fact column {col.name!r} source "
        f"{col.source!r} resolves to {type(parsed).__name__}, "
        f"which is not supported on per_period fact tables. "
        f"Use metric:, fk:dim_date.*, generated:, faker:, "
        f"static:, or pk: sources."
    )


COLUMN_DISPATCH.register(BuilderKind.PER_PERIOD_FACT, FKSource, _per_period_fk)
COLUMN_DISPATCH.register(BuilderKind.PER_PERIOD_FACT, MetricSource, _per_period_metric)
COLUMN_DISPATCH.register(BuilderKind.PER_PERIOD_FACT, PKSource, _per_period_pk)
COLUMN_DISPATCH.register(BuilderKind.PER_PERIOD_FACT, GeneratedSource, _per_period_generated)
COLUMN_DISPATCH.register(BuilderKind.PER_PERIOD_FACT, FakerSource, _per_period_faker)
COLUMN_DISPATCH.register(BuilderKind.PER_PERIOD_FACT, StaticSource, _per_period_static)
COLUMN_DISPATCH.register_unsupported(BuilderKind.PER_PERIOD_FACT, _per_period_unsupported)


def _days_in_period(anchor_date: _dt.date, granularity: str) -> int:
    """Number of days in the period that starts at ``anchor_date``.

    monthly: ``calendar.monthrange(anchor.year, anchor.month)[1]``
    weekly:  7
    daily:   1
    """
    if granularity == "monthly":
        return calendar.monthrange(anchor_date.year, anchor_date.month)[1]
    if granularity == "weekly":
        return 7
    if granularity == "daily":
        return 1
    raise ValueError(f"unknown granularity {granularity!r}")


def _within_period_timestamp(
    anchor_date: _dt.date,
    granularity: str,
    rng: np.random.Generator,
) -> _dt.datetime:
    """Draw one uniform timestamp inside the period starting at ``anchor_date``.

    Period extent by granularity:
      * monthly — 1st-of-month through end-of-month (variable days).
      * weekly  — Monday through Sunday (7 days).
      * daily   — the anchor day (24 hours).

    One rng draw per call. Callers share a single rng; downstream rng
    consumption order shifts when a table adds or removes a timestamp
    column, but is deterministic for a fixed config + seed.
    """
    seconds_in_period = _days_in_period(anchor_date, granularity) * 86400
    offset_seconds = float(rng.uniform(0.0, seconds_in_period))
    base = _dt.datetime(anchor_date.year, anchor_date.month, anchor_date.day)
    return base + _dt.timedelta(seconds=offset_seconds)


def _coerce_anchor_date(d: Any) -> _dt.date:
    """Promote a ``dim_date.date`` cell to a plain :class:`datetime.date`."""
    if isinstance(d, _dt.datetime):
        return d.date()
    return cast(_dt.date, d)


def _within_period_timestamps_for_indices(
    dim_date: pd.DataFrame,
    indices: np.ndarray,
    granularity: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized within-period timestamp draw.

    For each ``i`` in ``indices``, returns a timestamp drawn uniformly
    within the period anchored at ``dim_date.iloc[i]["date"]``. One rng
    draw per row, in ``indices`` iteration order.
    """
    dates = dim_date["date"].tolist()
    n_rows = len(indices)
    out = np.empty(n_rows, dtype=object)
    for i in range(n_rows):
        anchor = _coerce_anchor_date(dates[indices[i]])
        out[i] = _within_period_timestamp(anchor, granularity, rng)
    return out


def _resolve_generated(provider: str, period_idx: int, dim_date: pd.DataFrame, fake: Faker):
    """Resolve a non-faker ``generated:<provider>`` cell.

    Recognised providers: ``timestamp``, ``date_key``, ``period_label``.
    Faker providers parse as :class:`FakerSource` and are dispatched
    separately via :func:`_call_faker` — callers shouldn't reach here
    for a faker source.

    ``fake`` is retained in the signature for callers that still pass it
    (it's harmless here).

    Anchor-only timestamp: used by per_entity_per_period and per_period
    facts where the row's date is the period anchor by construction.
    Event and variable-grain paths route through
    :func:`_within_period_timestamp` instead.
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
    raise ValueError(f"unsupported generated provider {provider!r} on fact/event tables")


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
                    tbl,
                    parsed_rc,
                    config,
                    fact_tables,
                    dim_tables,
                    per_entity_dims,
                    fake,
                    rng,
                )
                continue
        threshold_col = _find_threshold_column(tbl)
        if threshold_col is not None:
            out[tbl.name] = _build_threshold_event(
                tbl,
                threshold_col,
                config,
                fact_tables,
                dim_tables,
                per_entity_dims,
                fake,
                rng,
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
    metric: str,
    fact_tables: dict[str, pd.DataFrame],
    config: PlotsimConfig,
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

    Vectorized via ``groupby(sort=False)`` which iterates groups in
    first-appearance order. Each returned ``group`` is the original
    DataFrame slice (views, not row-stacked copies), so callers that
    iterate ``group.iterrows()`` see identical rows in identical order.
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

    Event-table path: the driver metric must be projected onto a fact
    column. We read entity_ids / date_keys / metric values straight off
    the already-built fact DataFrame (entity-major from the Layer 4
    builder), then delegate to ``_emit_proportional_rows`` for the
    counts + column-dispatch pipeline.

    For variable-grain fact tables (0.6-M18) the same downstream
    pipeline runs but the three input arrays come from
    ``entity_metrics`` + per_entity dim PK + dim_date directly — that
    path lives in ``_build_variable_grain_fact``.
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
        raise ValueError(f"fact {fact_name!r} has no per_entity FK; cannot group events")
    fact_entity_col_name = fact_entity_fk[0]

    fact_date_col = _find_date_fk_column(fact_table_cfg)
    if fact_date_col is None:
        raise ValueError(f"fact {fact_name!r} has no dim_date FK; cannot derive event dates")
    fact_date_col_name = fact_date_col[0]

    # Per-cell vectors — fact rows are already in entity-major order from the
    # Layer 4 builder (or the scalar fallback, which also emits in that
    # order), so column-reading the fact DataFrame preserves the exact cell
    # order that the pre-Layer-5 groupby-then-iterrows walk used.
    entity_ids_arr = fact_df[fact_entity_col_name].to_numpy()
    date_keys_arr = fact_df[fact_date_col_name].to_numpy()
    # Coerce to float64 with NaN for nulls — handles both vectorized fact
    # output (float64) and scalar-fallback output (object with None).
    # M124: ``to_numpy(dtype=np.float64)`` forces a float container even when
    # the metric column is integer-typed (count drivers, poisson-distributed
    # metrics). Without the cast, ``np.isnan`` raises
    # ``TypeError: ufunc 'isnan' not supported for input types`` on int dtype.
    values_arr = pd.to_numeric(fact_df[metric_col], errors="coerce").to_numpy(dtype=np.float64)

    return _emit_proportional_rows(
        tbl,
        rc,
        entity_ids_arr,
        date_keys_arr,
        values_arr,
        config,
        dim_tables,
        fact_tables,
        per_entity_dims,
        fake,
        rng,
    )


def _emit_proportional_rows(
    tbl: Table,
    rc: ProportionalSource,
    entity_ids_arr: np.ndarray,
    date_keys_arr: np.ndarray,
    values_arr: np.ndarray,
    config: PlotsimConfig,
    dim_tables: dict[str, pd.DataFrame],
    fact_tables: dict[str, pd.DataFrame],
    per_entity_dims: set[str],
    fake: Faker,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shared row-emission pipeline for proportional-driven tables.

    Both event tables (``_build_proportional_event``) and variable-grain
    fact tables (``_build_variable_grain_fact``) funnel through this
    helper. They differ only in how the three input arrays — cell-major
    entity-IDs, cell-major date-keys, cell-major driver metric values —
    are sourced.

    Inputs (all length ``n_entities * n_periods``, entity-major order):

      * ``entity_ids_arr`` — the entity's per_entity dim PK value at each
        cell. Constant across all cells for one entity.
      * ``date_keys_arr`` — the period's ``dim_date.date_key`` at each
        cell. Cycles every ``n_periods`` entries.
      * ``values_arr`` — the driver metric value at each cell as
        ``float64``. ``NaN`` cells contribute zero rows.

    Category B Layer 5 hybrid:

      * **Deterministic** columns (PK, FK to ``dim_date``, FK to
        per_entity dim, ``ThresholdSource`` cells (always ``None`` here),
        ``GeneratedSource``, ``StaticSource``, ``DerivedSource``):
        materialized once as numpy arrays via ``np.repeat`` / ``np.tile`` +
        index lookups.
      * **Stochastic** columns (``FakerSource``, sub-entity/reference
        ``FKSource``): per-row scalar loop in the same cell order as the
        pre-Layer-5 path, so RNG + faker consumption order is byte-
        identical.

    When there are zero stochastic columns the per-row loop is skipped
    entirely.
    """
    dim_date = dim_tables["dim_date"]
    parsed_cols = [(col, parse_source(col.source)) for col in tbl.columns]

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
        dtype=np.int64,
        count=total_rows,
    )

    # PK serial: scalar path writes pk_counter+1 starting from 0 → 1..total_rows.
    # 0.6-M18: variable-grain fact tables (e.g. ``fct_orders``) route through
    # this builder too; strip the ``fct_`` prefix the same way so a fact PK
    # reads ``o-0001`` (orders) rather than ``f-0001``.
    #
    # 0.6-M19 Fix 8: route the prefix through ``config.pk_prefix_for``
    # so two tables that would otherwise collide on the same first
    # character (e.g. ``fct_orders`` and ``fct_order_items`` both →
    # ``o``) get distinguishable prefixes ("orders" / "order_items").
    pk_width = _id_pad(_EVENT_PK_WIDTH_HINT)
    pk_first_char = config.pk_prefix_for(tbl.name)

    # Classify columns. Any FK that points to a non-per_entity, non-dim_date
    # table may draw RNG; FakerSource always advances faker state;
    # 0.6-M19 Fix 1: PoolSource needs per-row entity lookup so it lands
    # in the stochastic loop too.
    def _is_stochastic(col: Column, parsed: Any) -> bool:
        if isinstance(parsed, FakerSource):
            return True
        if isinstance(parsed, FKSource):
            return parsed.table not in per_entity_dims and parsed.table != "dim_date"
        if isinstance(parsed, PoolSource):
            return True
        return False

    stochastic_cols = {col.name for col, parsed in parsed_cols if _is_stochastic(col, parsed)}

    # 0.6-M19 Fix 1: pre-compute per-column pool lookups keyed by entity
    # PK. ``value_pool`` is keyed by ``Entity.name``; the stochastic loop
    # has the entity PK, so we collapse the indirection up-front (config
    # order matches the unique-PKs-in-entity-major-order convention used
    # by every variable-grain / proportional-event caller).
    pool_by_pk_per_col: dict[str, dict[Any, list[str]]] = {}
    pool_cols = [(col, parsed) for col, parsed in parsed_cols if isinstance(parsed, PoolSource)]
    if pool_cols:
        unique_entity_pks = pd.unique(entity_ids_arr)
        if len(unique_entity_pks) != len(config.entities):
            raise RuntimeError(
                f"pool lookup on {tbl.name!r}: {len(unique_entity_pks)} "
                f"unique entity PKs vs {len(config.entities)} declared "
                f"entities; build invariant broken"
            )
        entity_name_by_pk = {
            unique_entity_pks[i]: config.entities[i].name for i in range(len(config.entities))
        }
        for col, _ in pool_cols:
            assert col.value_pool is not None  # _pool_pairing
            pool_by_pk_per_col[col.name] = {
                pk: col.value_pool[entity_name_by_pk[pk]] for pk in unique_entity_pks
            }

    # 0.6-M18 Fix 3: pre-compute per-entity index for cross-fact FK
    # columns so the stochastic loop draws in O(1) per row. Keyed by
    # column name → dict[entity_pk_value, np.ndarray of parent PK
    # values]. Empty array (or missing key) signals "no parent rows for
    # this entity"; the loop emits ``None`` for that cell.
    cross_fact_lookups: dict[str, dict[Any, np.ndarray]] = {}
    for col, parsed in parsed_cols:
        if not isinstance(parsed, FKSource):
            continue
        if parsed.table not in fact_tables:
            continue
        parent_fact = fact_tables[parsed.table]
        if parent_fact.empty:
            cross_fact_lookups[col.name] = {}
            continue
        parent_fact_tbl = next(t for t in config.tables if t.name == parsed.table)
        parent_entity_fk = _find_entity_fk_column(parent_fact_tbl, per_entity_dims)
        if parent_entity_fk is None:
            # Validator should have caught this — but be defensive.
            cross_fact_lookups[col.name] = {}
            continue
        parent_entity_col = parent_entity_fk[0]
        grouped = parent_fact.groupby(parent_entity_col, sort=False)[parsed.column].apply(list)
        cross_fact_lookups[col.name] = {k: np.asarray(v, dtype=object) for k, v in grouped.items()}

    col_arrays: dict[str, np.ndarray] = {}
    base_ctx = {
        "tbl": tbl,
        "total_rows": total_rows,
        "event_entity_ids": event_entity_ids,
        "event_date_keys": event_date_keys,
        "event_date_idx": event_date_idx,
        "dim_date": dim_date,
        "per_entity_dims": per_entity_dims,
        "pk_first_char": pk_first_char,
        "pk_width": pk_width,
        # 0.6-M19 Fix 2: RangeSource handler reads ``rng`` for the bulk
        # uniform / integers draw. Other handlers ignore the key.
        "rng": rng,
        # 0.6-M19 Fix 6: GeneratedSource timestamp handler reads
        # ``granularity`` to size the within-period draw range.
        "granularity": config.time_window.granularity,
    }
    for col, parsed in parsed_cols:
        if col.name in stochastic_cols:
            col_arrays[col.name] = np.empty(total_rows, dtype=object)
            continue
        ctx = dict(base_ctx)
        ctx["col"] = col
        col_arrays[col.name] = COLUMN_DISPATCH.dispatch(
            BuilderKind.PROPORTIONAL_EVENT,
            parsed,
            ctx,
        )

    # Stochastic columns: iterate per (cell, row) exactly as the scalar path
    # did, so RNG + faker consumption order is byte-identical. Source parsing
    # is already hoisted; the inner work is a dict write per stochastic column.
    if stochastic_cols:
        row_idx = 0
        n_cells = len(counts)
        cell_entity_ids = entity_ids_arr
        stochastic_parsed = [
            (col, parsed) for col, parsed in parsed_cols if col.name in stochastic_cols
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
                            fake,
                            parsed.method,
                            parsed.kwargs,
                        )
                    elif isinstance(parsed, PoolSource):
                        # 0.6-M19 Fix 1: per-row draw from the entity's
                        # pool. ``pool_by_pk_per_col`` was pre-computed
                        # above so the per-row work is one rng draw +
                        # one list index.
                        choices = pool_by_pk_per_col[col.name][entity_pk_value]
                        if rng is not None:
                            pick = int(rng.integers(0, len(choices)))
                        else:
                            pick = 0
                        col_arrays[col.name][row_idx] = _coerce_static(choices[pick], col.dtype)
                    elif isinstance(parsed, FKSource):
                        # 0.6-M18 Fix 3: cross-fact reference. Same-entity
                        # filtered draw from the referenced fact's PK
                        # column via pre-computed per-entity index.
                        if col.name in cross_fact_lookups:
                            candidates_arr = cross_fact_lookups[col.name].get(entity_pk_value)
                            if candidates_arr is None or len(candidates_arr) == 0:
                                col_arrays[col.name][row_idx] = None
                            else:
                                if rng is not None:
                                    pick = int(rng.integers(0, len(candidates_arr)))
                                else:
                                    pick = 0
                                col_arrays[col.name][row_idx] = candidates_arr[pick]
                            continue
                        # Dim FK (existing behavior).
                        parent = dim_tables.get(parsed.table)
                        if parent is None or parent.empty:
                            col_arrays[col.name][row_idx] = None
                            continue
                        back_link = _find_entity_link_in_subentity(
                            parsed.table,
                            config,
                            per_entity_dims,
                        )
                        if back_link is not None:
                            candidates = parent[parent[back_link] == entity_pk_value]
                            if len(candidates) > 0:
                                if rng is not None:
                                    pick = int(rng.integers(0, len(candidates)))
                                else:
                                    pick = 0
                                col_arrays[col.name][row_idx] = candidates.iloc[pick][parsed.column]
                                continue
                        # 0.6-M19 Fix 3: cross-dim FK on a variable-grain
                        # fact (e.g. dim_payment_method on fct_orders).
                        # Pre-fix this fell through to row 0, which
                        # produced a degenerate join where every fact row
                        # referenced the same dim row. Uniform draw across
                        # the parent dim mirrors the per_parent_row child
                        # builder above (~L2546).
                        if rng is not None:
                            pick = int(rng.integers(0, len(parent)))
                        else:
                            pick = 0
                        col_arrays[col.name][row_idx] = parent.iloc[pick][parsed.column]
                row_idx += 1

    return pd.DataFrame({col.name: col_arrays[col.name] for col, _ in parsed_cols})


def _build_variable_grain_fact(
    tbl: Table,
    rc: ProportionalSource,
    config: PlotsimConfig,
    dim_tables: dict[str, pd.DataFrame],
    per_entity_dims: set[str],
    entity_metrics: dict[str, dict[str, np.ndarray]],
    fact_tables: dict[str, pd.DataFrame],
    fake: Faker,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """0.6-M18: variable-grain fact table builder.

    Unlike events (which read driver values off an already-built fact),
    variable-grain facts read directly from the metric layer
    (``entity_metrics``). The minimum surface for a parent/child star is
    two facts — a variable-grain parent and a per_parent_row child — with
    no intermediate driver-host fact required.

    Cell-major input arrays:

      * ``entity_ids_arr`` — the per_entity dim's PK column repeated
        ``n_periods`` times per entity (config-entities order).
      * ``date_keys_arr`` — ``dim_date.date_key`` tiled ``n_entities``
        times.
      * ``values_arr`` — per-entity driver metric series concatenated in
        config-entities order; ``NaN`` for cold-start prefix cells.

    Cold-start entities yield zero rows for their NaN prefix cells —
    same behavior as the event-table path.
    """
    if not per_entity_dims:
        raise ValueError(
            f"variable-grain fact {tbl.name!r} requires a per_entity dim "
            f"to derive per-entity row counts; none declared"
        )
    if len(per_entity_dims) > 1:
        # Plotsim configs have exactly one per_entity dim by construction
        # (the entity body). Surface this loudly if a future change
        # breaks the invariant.
        raise RuntimeError(
            f"variable-grain fact {tbl.name!r}: expected exactly one "
            f"per_entity dim, found {sorted(per_entity_dims)}"
        )
    entity_dim_name = next(iter(per_entity_dims))
    entity_dim_tbl = next(t for t in config.tables if t.name == entity_dim_name)
    entity_dim_pk_col = entity_dim_tbl.primary_key_cols[0]
    entity_dim_df = dim_tables.get(entity_dim_name)
    if entity_dim_df is None:
        raise RuntimeError(
            f"variable-grain fact {tbl.name!r}: per_entity dim "
            f"{entity_dim_name!r} not yet materialized"
        )
    dim_date = dim_tables.get("dim_date")
    if dim_date is None:
        raise RuntimeError(f"variable-grain fact {tbl.name!r}: dim_date not yet materialized")

    n_entities = len(config.entities)
    n_periods = len(dim_date)

    # entity-major: each entity contributes n_periods cells in a row.
    # dim_entity rows are in config.entities declaration order
    # (established by dimensions.build_dim_entity). SCD2-expanded dims
    # have multiple rows per entity (one per tier band); collapse them
    # back to one PK per entity via ``drop_duplicates(keep='first')``,
    # which preserves config.entities first-occurrence ordering.
    entity_pks = entity_dim_df[entity_dim_pk_col].drop_duplicates().to_numpy()
    if len(entity_pks) != n_entities:
        raise RuntimeError(
            f"variable-grain fact {tbl.name!r}: per_entity dim resolves "
            f"to {len(entity_pks)} unique PK values but config declares "
            f"{n_entities} entities. SCD2 expansion preserves unique "
            f"natural keys; some other invariant is broken."
        )
    entity_ids_arr = np.repeat(entity_pks, n_periods)
    date_keys_arr = np.tile(dim_date["date_key"].to_numpy(), n_entities)

    # Concatenate per-entity driver series in config.entities order.
    # entity_metrics[entity_name][metric] is shape (n_periods,) with NaN
    # for cold-start prefix cells.
    per_entity_values: list[np.ndarray] = []
    for entity in config.entities:
        series = entity_metrics[entity.name][rc.metric]
        per_entity_values.append(np.asarray(series, dtype=np.float64))
    values_arr = np.concatenate(per_entity_values)

    return _emit_proportional_rows(
        tbl,
        rc,
        entity_ids_arr,
        date_keys_arr,
        values_arr,
        config,
        dim_tables,
        fact_tables,
        per_entity_dims,
        fake,
        rng,
    )


# --- Proportional event deterministic-column dispatch ------------------------


def _prop_evt_pk(parsed: PKSource, ctx: dict):
    return np.asarray(
        [f"{ctx['pk_first_char']}-{i + 1:0{ctx['pk_width']}d}" for i in range(ctx["total_rows"])],
        dtype=object,
    )


def _prop_evt_fk(parsed: FKSource, ctx: dict):
    col = ctx["col"]
    if parsed.table == "dim_date":
        return ctx["event_date_keys"]
    if parsed.table in ctx["per_entity_dims"]:
        return ctx["event_entity_ids"]
    # Cross-dim FK on a deterministic event column. ``_is_stochastic``
    # only flags FKs to non-per_entity, non-dim_date dims as stochastic,
    # so reaching this branch with a deterministic classification means
    # the column was misclassified upstream — raise rather than emit a
    # column of None.
    raise TypeError(
        f"event column {col.name!r} declared FK to "
        f"{parsed.table!r} but the deterministic path has no "
        f"resolution for this source kind"
    )


def _prop_evt_threshold(parsed: ThresholdSource, ctx: dict):
    # Proportional path passes threshold_col_name=None, so every
    # ThresholdSource cell resolves to None.
    return np.full(ctx["total_rows"], None, dtype=object)


def _prop_evt_generated(parsed: GeneratedSource, ctx: dict):
    provider = parsed.provider
    dim_date = ctx["dim_date"]
    event_date_idx = ctx["event_date_idx"]
    if provider == "timestamp":
        # 0.6-M19 Fix 6: distribute timestamps uniformly within each
        # period instead of anchoring to the period start. Event and
        # variable-grain fact rows now span the full month / week /
        # day they belong to. Per_entity_per_period and per_period
        # facts keep anchor-only behavior via :func:`_resolve_generated`.
        return _within_period_timestamps_for_indices(
            dim_date,
            event_date_idx,
            ctx["granularity"],
            ctx["rng"],
        )
    if provider == "date_key":
        dk = dim_date["date_key"].tolist()
        return np.asarray([dk[i] for i in event_date_idx], dtype=object)
    if provider == "period_label":
        pl = dim_date["period_label"].tolist()
        return np.asarray([pl[i] for i in event_date_idx], dtype=object)
    raise ValueError(f"unsupported generated provider {provider!r} on fact/event tables")


def _prop_evt_static(parsed: StaticSource, ctx: dict):
    return np.full(ctx["total_rows"], parsed.value, dtype=object)


def _prop_evt_derived(parsed: DerivedSource, ctx: dict):
    col = ctx["col"]
    if parsed.field == "entity_id":
        return ctx["event_entity_ids"]
    if parsed.field == "date_key":
        return ctx["event_date_keys"]
    # F14 (M102): explicit raise on unrecognised derived field.
    raise ValueError(
        f"event column {col.name!r} derived field "
        f"{parsed.field!r} is not supported on event tables; "
        f"use 'entity_id' or 'date_key'"
    )


def _prop_evt_range(parsed: RangeSource, ctx: dict):
    """0.6-M19 Fix 2: bulk per-row uniform draw on a proportional /
    variable-grain event row builder.

    Same shape as :func:`_fact_vec_range` but consumed by the cell-
    major proportional pipeline. Total row count is fixed at the
    point this handler runs, so a single bulk draw of size
    ``total_rows`` keeps RNG consumption proportional to the output
    size.
    """
    col = ctx["col"]
    rng = ctx["rng"]
    if rng is None:
        raise ValueError(
            f"event/fact column {col.name!r} has source {col.source!r} "
            f"but no RNG was supplied to _emit_proportional_rows; "
            f"range draws require the per-table RNG"
        )
    total_rows = ctx["total_rows"]
    if col.dtype == "int":
        return rng.integers(int(parsed.min), int(parsed.max) + 1, size=total_rows)
    return rng.uniform(parsed.min, parsed.max, size=total_rows)


def _prop_evt_unsupported(parsed: Any, ctx: dict):
    # F14 (M102): explicit raise on unhandled source type.
    col = ctx["col"]
    tbl = ctx["tbl"]
    raise TypeError(
        f"event column {col.name!r} source {col.source!r} "
        f"resolves to {type(parsed).__name__}, which is not "
        f"supported in the deterministic-event dispatch on "
        f"{tbl.name!r}."
    )


COLUMN_DISPATCH.register(BuilderKind.PROPORTIONAL_EVENT, PKSource, _prop_evt_pk)
COLUMN_DISPATCH.register(BuilderKind.PROPORTIONAL_EVENT, FKSource, _prop_evt_fk)
COLUMN_DISPATCH.register(
    BuilderKind.PROPORTIONAL_EVENT,
    ThresholdSource,
    _prop_evt_threshold,
)
COLUMN_DISPATCH.register(
    BuilderKind.PROPORTIONAL_EVENT,
    GeneratedSource,
    _prop_evt_generated,
)
COLUMN_DISPATCH.register(
    BuilderKind.PROPORTIONAL_EVENT,
    StaticSource,
    _prop_evt_static,
)
COLUMN_DISPATCH.register(
    BuilderKind.PROPORTIONAL_EVENT,
    DerivedSource,
    _prop_evt_derived,
)
COLUMN_DISPATCH.register(
    BuilderKind.PROPORTIONAL_EVENT,
    RangeSource,
    _prop_evt_range,
)
COLUMN_DISPATCH.register_unsupported(
    BuilderKind.PROPORTIONAL_EVENT,
    _prop_evt_unsupported,
)


# --- Per-parent-row child fact builder (0.6-M18) -----------------------------


def _build_per_parent_row_fact(
    tbl: Table,
    config: PlotsimConfig,
    fact_tables: dict[str, pd.DataFrame],
    dim_tables: dict[str, pd.DataFrame],
    per_entity_dims: set[str],
    fake: Faker,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a per_parent_row child fact table.

    For each row of the parent fact, draw
    ``n = rng.integers(min, max + 1)`` children (uniform over the
    configured range, independent of trajectory) and emit one row per
    child carrying:

      * an **auto-synthesized** FK column to the parent's PK. Following
        the bridge pattern (``BridgeTableConfig.connects`` synthesizes
        both FK columns at generation time), the user declares the
        relationship once via ``parent_table`` and the engine emits the
        FK column. The synthesized column's name matches the parent
        fact's PK column name verbatim (e.g. ``order_id``) and it lands
        first in column order.
      * inherited entity and period columns (the child shares its
        parent's (entity, period) coordinates);
      * dim FKs drawn per-row from the referenced dim table;
      * remaining columns (``pk``, ``static:``, ``generated:``,
        ``derived:``) resolved per the standard source dispatch.

    Trajectory-first invariant: child rows do NOT call into the
    trajectory engine. Their attribute values are independent draws from
    column-level sources. The trajectory-driven signal flows through
    the parent's row count (a growth entity has more parent rows than a
    decline entity); the child fans out uniformly from each parent row.

    RNG consumption order is deterministic and ordered by
    ``config.tables`` declaration: children counts first (one
    ``rng.integers`` call), then per-cell stochastic draws (faker,
    cross-dim FK sampling) in column-declaration order for each child
    row.
    """
    parent_name = tbl.parent_table
    assert parent_name is not None  # paired-field validator
    children_range = tbl.children_per_row
    assert children_range is not None
    mn, mx = children_range

    parent_df = fact_tables.get(parent_name)
    if parent_df is None:
        raise ValueError(
            f"per_parent_row child {tbl.name!r} parent_table="
            f"{parent_name!r} not in fact_tables; parent build "
            f"order broken"
        )

    parent_tbl = next(t for t in config.tables if t.name == parent_name)
    parent_pk_cols = parent_tbl.primary_key_cols
    if len(parent_pk_cols) != 1:
        raise ValueError(
            f"per_parent_row child {tbl.name!r} parent {parent_name!r} "
            f"has composite primary_key {parent_pk_cols!r}; M18 supports "
            f"single-column parent PKs only"
        )
    parent_pk_col = parent_pk_cols[0]
    # Auto-synthesized FK column on the child carries the parent's PK
    # column name verbatim (bridge precedent). Collision with a user-
    # declared child column is caught by the config-load validator.
    synthesized_fk_name = parent_pk_col

    parent_entity_fk = _find_entity_fk_column(parent_tbl, per_entity_dims)
    parent_entity_col = parent_entity_fk[0] if parent_entity_fk is not None else None
    parent_date_fk = _find_date_fk_column(parent_tbl)
    parent_date_col = parent_date_fk[0] if parent_date_fk is not None else None

    n_parents = len(parent_df)
    if n_parents == 0:
        return pd.DataFrame(columns=[c.name for c in tbl.columns])

    # Single RNG call for the entire counts vector — deterministic and
    # cheap. mn == mx degenerates to a constant fan-out (still one draw
    # for column-order consistency).
    counts = rng.integers(mn, mx + 1, size=n_parents).astype(np.int64)
    total_rows = int(counts.sum())
    if total_rows == 0:
        return pd.DataFrame(columns=[c.name for c in tbl.columns])

    parent_pk_repeated = np.repeat(parent_df[parent_pk_col].to_numpy(), counts)
    if parent_entity_col is not None:
        entity_repeated = np.repeat(parent_df[parent_entity_col].to_numpy(), counts)
    else:
        entity_repeated = None
    if parent_date_col is not None:
        date_repeated = np.repeat(parent_df[parent_date_col].to_numpy(), counts)
    else:
        date_repeated = None

    # Build a parallel date-index array so generated:timestamp /
    # period_label can resolve per-row even on children. dim_date might
    # not be needed (no generated:timestamp columns), so resolve lazily.
    dim_date = dim_tables.get("dim_date")
    date_idx_repeated: Optional[np.ndarray] = None
    if dim_date is not None and date_repeated is not None:
        date_key_idx = {k: i for i, k in enumerate(dim_date["date_key"].tolist())}
        date_idx_repeated = np.fromiter(
            (date_key_idx.get(k, 0) for k in date_repeated),
            dtype=np.int64,
            count=total_rows,
        )

    # 0.6-M19 Fix 8: route through ``config.pk_prefix_for`` so a
    # per_parent_row child fact whose first character collides with
    # another sequential-PK table gets a distinguishable prefix
    # (e.g. ``fct_order_items`` → ``order_items`` instead of ``o``).
    pk_first = config.pk_prefix_for(tbl.name)
    pk_width = _id_pad(_EVENT_PK_WIDTH_HINT)

    col_arrays: dict[str, np.ndarray] = {}
    parsed_cols = [(col, parse_source(col.source)) for col in tbl.columns]

    # 0.6-M18: auto-synthesize the parent FK column FIRST in column
    # order (bridge convention). User-declared columns follow in their
    # declared order. Validator at config-load time rejects an explicit
    # ``ref.fct_<parent>`` column on the child, so there's no collision
    # path here.
    col_arrays[synthesized_fk_name] = parent_pk_repeated

    for col, parsed in parsed_cols:
        if isinstance(parsed, PKSource):
            col_arrays[col.name] = np.asarray(
                [f"{pk_first}-{i + 1:0{pk_width}d}" for i in range(total_rows)],
                dtype=object,
            )
        elif isinstance(parsed, FKSource):
            if parsed.table in per_entity_dims:
                if entity_repeated is None:
                    raise ValueError(
                        f"child {tbl.name!r} column {col.name!r} fk to "
                        f"{parsed.table!r} (per_entity dim) but parent "
                        f"{parent_name!r} has no per_entity FK to inherit"
                    )
                col_arrays[col.name] = entity_repeated
            elif parsed.table == "dim_date":
                if date_repeated is None:
                    raise ValueError(
                        f"child {tbl.name!r} column {col.name!r} fk to "
                        f"dim_date but parent {parent_name!r} has no "
                        f"dim_date FK to inherit"
                    )
                col_arrays[col.name] = date_repeated
            else:
                # Cross-dim FK (e.g. dim_product, dim_payment_method).
                # Independent draws per child row.
                parent_dim = dim_tables.get(parsed.table)
                if parent_dim is None or len(parent_dim) == 0:
                    col_arrays[col.name] = np.full(total_rows, None, dtype=object)
                else:
                    candidates = parent_dim[parsed.column].to_numpy()
                    pick = rng.integers(0, len(candidates), size=total_rows)
                    col_arrays[col.name] = candidates[pick]
        elif isinstance(parsed, StaticSource):
            col_arrays[col.name] = np.full(total_rows, parsed.value, dtype=object)
        elif isinstance(parsed, DerivedSource):
            if parsed.field == "entity_id":
                if entity_repeated is None:
                    raise ValueError(
                        f"child {tbl.name!r} column {col.name!r} derived "
                        f"entity_id but parent has no per_entity FK"
                    )
                col_arrays[col.name] = entity_repeated
            elif parsed.field == "date_key":
                if date_repeated is None:
                    raise ValueError(
                        f"child {tbl.name!r} column {col.name!r} derived "
                        f"date_key but parent has no dim_date FK"
                    )
                col_arrays[col.name] = date_repeated
            else:
                raise ValueError(
                    f"child {tbl.name!r} column {col.name!r} derived "
                    f"field {parsed.field!r} is not supported on "
                    f"per_parent_row; use 'entity_id' or 'date_key'"
                )
        elif isinstance(parsed, GeneratedSource):
            provider = parsed.provider
            if provider == "date_key":
                if date_repeated is None:
                    raise ValueError(
                        f"child {tbl.name!r} column {col.name!r} "
                        f"generated:date_key but parent has no dim_date FK"
                    )
                col_arrays[col.name] = date_repeated
            elif provider == "timestamp":
                if date_idx_repeated is None or dim_date is None:
                    raise ValueError(
                        f"child {tbl.name!r} column {col.name!r} "
                        f"generated:timestamp but parent has no "
                        f"dim_date FK to derive period dates from"
                    )
                # 0.6-M19 Fix 6: distribute child timestamps uniformly
                # across each parent row's period. Per-parent-row
                # children are variable-grain per the mission spec.
                col_arrays[col.name] = _within_period_timestamps_for_indices(
                    dim_date,
                    date_idx_repeated,
                    config.time_window.granularity,
                    rng,
                )
            elif provider == "period_label":
                if date_idx_repeated is None or dim_date is None:
                    raise ValueError(
                        f"child {tbl.name!r} column {col.name!r} "
                        f"generated:period_label but parent has no "
                        f"dim_date FK"
                    )
                pl = dim_date["period_label"].tolist()
                col_arrays[col.name] = np.asarray([pl[i] for i in date_idx_repeated], dtype=object)
            else:
                raise ValueError(
                    f"child {tbl.name!r} column {col.name!r} "
                    f"generated:{provider!r} is not supported on "
                    f"per_parent_row tables"
                )
        elif isinstance(parsed, FakerSource):
            col_arrays[col.name] = np.asarray(
                [_call_faker(fake, parsed.method, parsed.kwargs) for _ in range(total_rows)],
                dtype=object,
            )
        elif isinstance(parsed, RangeSource):
            # 0.6-M19 Fix 2: bulk uniform draw, integer or float per
            # column dtype. Independent of trajectory and parent row
            # — same draw shape as the per_parent_row cross-dim FK
            # above.
            if col.dtype == "int":
                col_arrays[col.name] = rng.integers(
                    int(parsed.min), int(parsed.max) + 1, size=total_rows
                )
            else:
                col_arrays[col.name] = rng.uniform(parsed.min, parsed.max, size=total_rows)
        elif isinstance(parsed, PoolSource):
            # 0.6-M19 Fix 1: per-row pool draw on a child fact. The
            # child inherits its parent's entity, so ``entity_repeated``
            # gives the entity PK per row; map back to ``Entity.name``
            # to index ``value_pool``.
            if entity_repeated is None:
                raise ValueError(
                    f"child {tbl.name!r} column {col.name!r} declares a "
                    f"pool source but parent {parent_name!r} has no "
                    f"per_entity FK to attribute pool entries to"
                )
            assert col.value_pool is not None  # _pool_pairing
            # Build entity_name lookup from per_entity dim PK ordering.
            entity_dim_name = next(iter(per_entity_dims))
            entity_dim_df = dim_tables[entity_dim_name]
            entity_dim_tbl = next(t for t in config.tables if t.name == entity_dim_name)
            entity_pk_col = entity_dim_tbl.primary_key_cols[0]
            unique_pks = entity_dim_df[entity_pk_col].drop_duplicates().tolist()
            entity_name_by_pk = {
                unique_pks[i]: config.entities[i].name for i in range(len(config.entities))
            }
            pool_by_pk = {pk: col.value_pool[entity_name_by_pk[pk]] for pk in unique_pks}
            row_values = np.empty(total_rows, dtype=object)
            for i, entity_pk_value in enumerate(entity_repeated):
                choices = pool_by_pk[entity_pk_value]
                pool_idx = int(rng.integers(0, len(choices)))
                row_values[i] = _coerce_static(choices[pool_idx], col.dtype)
            col_arrays[col.name] = row_values
        else:
            raise TypeError(
                f"child {tbl.name!r} column {col.name!r} source "
                f"{col.source!r} resolves to {type(parsed).__name__}, "
                f"which is not supported on per_parent_row tables. "
                f"Supported: pk, fk:<dim>.<col>, "
                f"static:, derived:entity_id/date_key, "
                f"generated:date_key/timestamp/period_label, "
                f"generated:faker.<method>, range:<min>:<max>, "
                f"pool:<name>. "
                f"(The parent FK column is auto-synthesized from "
                f"parent_table; do not declare it explicitly.)"
            )

    # Synthesized parent FK column lands first (bridge convention),
    # then user-declared columns in declaration order.
    output_columns: dict[str, np.ndarray] = {synthesized_fk_name: col_arrays[synthesized_fk_name]}
    for col, _ in parsed_cols:
        output_columns[col.name] = col_arrays[col.name]
    return pd.DataFrame(output_columns)


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
    assert fact_date_col is not None  # fact tables always FK to dim_date
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
                        tbl,
                        pk_counter,
                        date_key_value,
                        entity_pk_value,
                        threshold_col_cfg.name,
                        True,
                        dim_date,
                        dim_tables,
                        config,
                        fake,
                        rng=rng,
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
    width = _id_pad(_EVENT_PK_WIDTH_HINT)
    base_ctx = {
        "tbl": tbl,
        "pk_counter": pk_counter,
        "date_key_value": date_key_value,
        "entity_pk_value": entity_pk_value,
        "threshold_col_name": threshold_col_name,
        "threshold_value": threshold_value,
        "dim_date": dim_date,
        "dim_tables": dim_tables,
        "config": config,
        "fake": fake,
        "rng": rng,
        "per_entity_dims": per_entity_dims,
        "date_idx": date_idx,
        "width": width,
    }
    for col in tbl.columns:
        parsed = parse_source(col.source)
        ctx = dict(base_ctx)
        ctx["col"] = col
        row[col.name] = COLUMN_DISPATCH.dispatch(
            BuilderKind.THRESHOLD_EVENT_ROW,
            parsed,
            ctx,
        )
    return row


# --- Threshold-event row dispatch handlers -----------------------------------


def _evt_row_pk(parsed: PKSource, ctx: dict):
    tbl = ctx["tbl"]
    width = ctx["width"]
    # 0.6-M19 Fix 8: resolve via config so same-first-char tables
    # (e.g. ``evt_login`` + ``evt_logout``) get distinguishable PKs.
    prefix = ctx["config"].pk_prefix_for(tbl.name)
    return f"{prefix}-{ctx['pk_counter'] + 1:0{width}d}"


def _evt_row_fk(parsed: FKSource, ctx: dict):
    if parsed.table == "dim_date":
        return ctx["date_key_value"]
    if parsed.table in ctx["per_entity_dims"]:
        return ctx["entity_pk_value"]
    # Sub-entity (e.g. dim_user) or reference: pick a row whose
    # back-reference matches this entity. Fallback to row 0 if
    # no link is discoverable. Random pick among matches when we
    # can, so multiple rows for the same entity-period don't
    # collapse to a single sub-entity.
    parent = ctx["dim_tables"].get(parsed.table)
    if parent is None or parent.empty:
        return None
    back_link = _find_entity_link_in_subentity(
        parsed.table,
        ctx["config"],
        ctx["per_entity_dims"],
    )
    if back_link is not None:
        candidates = parent[parent[back_link] == ctx["entity_pk_value"]]
        if len(candidates) > 0:
            rng = ctx["rng"]
            if rng is not None:
                i = int(rng.integers(0, len(candidates)))
            else:
                i = 0
            return candidates.iloc[i][parsed.column]
    return parent.iloc[0][parsed.column]


def _evt_row_threshold(parsed: ThresholdSource, ctx: dict):
    col = ctx["col"]
    if col.name == ctx["threshold_col_name"]:
        return ctx["threshold_value"]
    return None


def _evt_row_generated(parsed: GeneratedSource, ctx: dict):
    date_idx = ctx["date_idx"]
    if parsed.provider == "timestamp" and date_idx is not None:
        # 0.6-M19 Fix 6: distribute the threshold-event timestamp
        # uniformly within the period anchored at ``date_idx`` rather
        # than emitting the period anchor itself. The fact-row's date
        # is the period anchor by construction (per_entity_per_period
        # parent), so without this the firing row's event_ts would
        # always land on day 1 of the month/week.
        dim_date = ctx["dim_date"]
        rng = ctx["rng"]
        if rng is not None:
            anchor = _coerce_anchor_date(dim_date.iloc[date_idx]["date"])
            return _within_period_timestamp(
                anchor,
                ctx["config"].time_window.granularity,
                rng,
            )
    return _resolve_generated(
        parsed.provider,
        date_idx if date_idx is not None else 0,
        ctx["dim_date"],
        ctx["fake"],
    )


def _evt_row_faker(parsed: FakerSource, ctx: dict):
    return _call_faker(ctx["fake"], parsed.method, parsed.kwargs)


def _evt_row_static(parsed: StaticSource, ctx: dict):
    return parsed.value


def _evt_row_derived(parsed: DerivedSource, ctx: dict):
    if parsed.field == "entity_id":
        return ctx["entity_pk_value"]
    if parsed.field == "date_key":
        return ctx["date_key_value"]
    return None


def _evt_row_range(parsed: RangeSource, ctx: dict):
    """0.6-M19 Fix 2: per-cell uniform draw on a threshold-event row."""
    col = ctx["col"]
    rng = ctx["rng"]
    if rng is None:
        raise ValueError(
            f"event column {col.name!r} has source {col.source!r} but no "
            f"RNG was supplied to _resolve_event_row; range draws require "
            f"the per-table RNG"
        )
    if col.dtype == "int":
        return int(rng.integers(int(parsed.min), int(parsed.max) + 1))
    return float(rng.uniform(parsed.min, parsed.max))


def _evt_row_pool(parsed: PoolSource, ctx: dict):
    """0.6-M19 Fix 1: per-row pool draw on a threshold-event row.

    Threshold events emit at most one row per entity, so the per-row
    entity_name lookup is one pass over the per_entity dim. Performance
    is not a concern at this row volume.
    """
    col = ctx["col"]
    rng = ctx["rng"]
    assert col.value_pool is not None  # _pool_pairing
    entity_pk_value = ctx["entity_pk_value"]
    config = ctx["config"]
    dim_tables = ctx["dim_tables"]
    entity_name: Optional[str] = None
    for dim_name in ctx["per_entity_dims"]:
        dim_df = dim_tables.get(dim_name)
        if dim_df is None or dim_df.empty:
            continue
        dim_tbl = next(t for t in config.tables if t.name == dim_name)
        pk_col = dim_tbl.primary_key_cols[0]
        unique_pks = dim_df[pk_col].drop_duplicates().tolist()
        for i, pk in enumerate(unique_pks):
            if pk == entity_pk_value and i < len(config.entities):
                entity_name = config.entities[i].name
                break
        if entity_name is not None:
            break
    if entity_name is None or entity_name not in col.value_pool:
        return None
    choices = col.value_pool[entity_name]
    if rng is not None:
        pick = int(rng.integers(0, len(choices)))
    else:
        pick = 0
    return _coerce_static(choices[pick], col.dtype)


def _evt_row_unsupported(parsed: Any, ctx: dict):
    # The pre-M127b ladder fell through to ``None`` for any source type
    # outside the seven explicit branches; this preserves that contract.
    return None


COLUMN_DISPATCH.register(BuilderKind.THRESHOLD_EVENT_ROW, PKSource, _evt_row_pk)
COLUMN_DISPATCH.register(BuilderKind.THRESHOLD_EVENT_ROW, FKSource, _evt_row_fk)
COLUMN_DISPATCH.register(
    BuilderKind.THRESHOLD_EVENT_ROW,
    ThresholdSource,
    _evt_row_threshold,
)
COLUMN_DISPATCH.register(
    BuilderKind.THRESHOLD_EVENT_ROW,
    GeneratedSource,
    _evt_row_generated,
)
COLUMN_DISPATCH.register(
    BuilderKind.THRESHOLD_EVENT_ROW,
    FakerSource,
    _evt_row_faker,
)
COLUMN_DISPATCH.register(
    BuilderKind.THRESHOLD_EVENT_ROW,
    StaticSource,
    _evt_row_static,
)
COLUMN_DISPATCH.register(
    BuilderKind.THRESHOLD_EVENT_ROW,
    DerivedSource,
    _evt_row_derived,
)
COLUMN_DISPATCH.register(
    BuilderKind.THRESHOLD_EVENT_ROW,
    RangeSource,
    _evt_row_range,
)
COLUMN_DISPATCH.register(
    BuilderKind.THRESHOLD_EVENT_ROW,
    PoolSource,
    _evt_row_pool,
)
COLUMN_DISPATCH.register_unsupported(
    BuilderKind.THRESHOLD_EVENT_ROW,
    _evt_row_unsupported,
)


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

    ``values`` is a 1D float array (NaN marks null). ``thresholds`` is the
    ascending ``threshold_enter`` list from ``stages.sequence``. Returns a
    1D int array of stage indices.

    * ``downgrade_delay is None and exit_thresholds is None`` — pure
      legacy monotonic. ``actual`` per row is
      ``searchsorted(thresholds, v, side='right') - 1``; NaN rows hold
      ``actual=0`` and are dominated by the running max. Fully vectorized
      via :func:`np.maximum.accumulate`.
    * ``downgrade_delay`` is ``N`` (legacy mode) — sequential cursor with
      consecutive ``below-cursor`` counter. Per-entity size is the period
      count, so total work is O(n_entities * n_periods).
    * ``exit_thresholds`` is set (hysteresis mode) — sequential cursor
      with demote check ``value < exit_thresholds[cursor]``.
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
    on ``downgrade_delay`` and on the resolved ``mode``:

      * ``mode == 'legacy'`` (``threshold_exit > threshold_enter``,
        the bundled-template default): the runtime ignores
        ``threshold_exit`` and relies on ``threshold_enter`` only.
        ``downgrade_delay is None`` is strict monotonic — the cursor
        never steps back; a brief dip stays in the higher stage.
        ``downgrade_delay == N`` demotes after ``N`` consecutive
        periods below the current stage's enter threshold.
      * ``mode == 'hysteresis'`` (``threshold_exit <= threshold_enter``):
        the runtime uses ``threshold_exit`` of the current stage as the
        demote threshold. ``downgrade_delay is None`` collapses to
        delay=1 (immediate demote once the value drops below the current
        stage's exit). ``downgrade_delay == N`` requires ``N`` consecutive
        periods below exit before demotion fires. The hysteresis band
        ``[threshold_exit, threshold_enter]`` keeps the entity in the
        higher stage on transient dips.

    The implementation is vectorized via pandas ``groupby`` + numpy walks
    (see :func:`_monotonic_stage_walk` and :func:`_free_mode_stages`).
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
    if target_name is None or target_tbl is None:
        return fact_tables

    per_entity_dims = _per_entity_dim_names(config)
    fk = _find_entity_fk_column(target_tbl, per_entity_dims)
    if fk is None:
        return fact_tables
    entity_col = fk[0]

    df = fact_tables[target_name].copy()
    n = len(df)

    thresholds = np.asarray(
        [s.threshold_enter for s in seq],
        dtype=float,
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
            [s.threshold_exit if s.threshold_exit is not None else s.threshold_enter for s in seq],
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
            entity_col,
            sort=False,
        ).indices.items():
            pos_arr = np.asarray(positions, dtype=np.int64)
            stage_idx[pos_arr] = _monotonic_stage_walk(
                values[pos_arr],
                thresholds,
                downgrade_delay,
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
    (see tests/test_dead_schema.py) regex doesn't treat reads on this
    dataclass as reads of those allowlisted display fields.
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
        higher cursor (hysteresis), mirroring
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
        end_period = transitions[seg_idx + 1] if seg_idx + 1 < len(transitions) else n_periods
        is_current = seg_idx == len(transitions) - 1
        valid_from = int(date_keys[start_period])
        if is_current:
            valid_to = SCD_VALID_TO_SENTINEL
            valid_to_period = n_periods
        else:
            valid_to = int(date_keys[end_period])
            valid_to_period = end_period
        band = int(cum_bands[start_period])
        versions.append(
            SCDVersion(
                band=band,
                band_label=labels[band],
                valid_from=valid_from,
                valid_to=valid_to,
                valid_from_period=start_period,
                valid_to_period=valid_to_period,
                is_current=is_current,
                dim_row_id=next_dim_row_id,
                crossing_position=(None if seg_idx == 0 else float(trajectory[start_period])),
            )
        )
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
        raise RuntimeError(f"_expand_scd_dim called on {tbl.name!r} which has no SCD column")
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
        "dim_row_id",
        "valid_from",
        "valid_to",
        "is_current",
    ]

    for entity_idx, entity in enumerate(config.entities):
        traj = trajectories.get(entity.name)
        if traj is None or len(traj) != n_periods:
            raise ValueError(
                f"SCD dim {tbl.name!r}: entity {entity.name!r} has missing "
                f"or wrong-length trajectory (expected {n_periods})"
            )
        entity_versions = _compute_scd_versions(
            traj,
            scd_cfg,
            date_keys,
            starting_dim_row_id=next_id,
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
        raise RuntimeError("expand_scd_dims requires dim_date to be present in dim_tables")

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
            tbl,
            df,
            config,
            trajectories,
            dim_date,
        )
        out[tbl.name] = expanded
        states[tbl.name] = state
    return out, SCDState(dims=states)


def _facts_referencing_scd_dim(
    config: PlotsimConfig,
    scd_dim: str,
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
            sub = scd_dim_df[scd_dim_df["dim_row_id"] == versions[0].dim_row_id]
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
                entity_keys,
                date_keys,
                versions_by_entity_pk,
                period_index_by_date_key,
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
        subset=[first_pk_col],
        keep="first",
    ).reset_index(drop=True)
    if len(deduped) != len(config.entities):
        raise RuntimeError(
            f"build_bridge_tables: per_entity dim {first_dim_tbl.name!r} has "
            f"{len(deduped)} unique {first_pk_col!r} value(s) but config has "
            f"{len(config.entities)} entities; the dim builder is expected "
            f"to keep dims 1:1 with config.entities"
        )
    return {entity.name: deduped.iloc[i][first_pk_col] for i, entity in enumerate(config.entities)}


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
            subset=[second_pk_col],
            keep="first",
        ).reset_index(drop=True)
        deduped_list: list[Any] = deduped[second_pk_col].tolist()
        return deduped_list
    full_list: list[Any] = second_dim_df[second_pk_col].tolist()
    return full_list


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
            arr = np.asarray(
                [
                    np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
                    for v in series
                ],
                dtype=np.float64,
            )
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
            config,
            first_dim_tbl,
            first_dim_df,
            scd_state,
        )
        second_pool = _bridge_second_dim_fk_pool(
            second_dim_tbl,
            second_dim_df,
            scd_state,
        )
        if not second_pool:
            out[bridge.name] = pd.DataFrame(
                columns=[first_fk_col, second_fk_col] + [bm.name for bm in bridge.metrics],
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
                bridge_assoc_list.append(
                    BridgeAssociation(
                        entity=entity.name,
                        targets=[],
                        cardinality=0,
                    )
                )
                continue

            picked_idx = rng.choice(n_pool, size=n, replace=False)
            picked_targets = [second_pool[int(i)] for i in picked_idx]
            bridge_assoc_list.append(
                BridgeAssociation(
                    entity=entity.name,
                    targets=list(picked_targets),
                    cardinality=n,
                )
            )

            first_fk_val = first_fk_by_entity[entity.name]
            for target in picked_targets:
                row: dict[str, Any] = {
                    first_fk_col: first_fk_val,
                    second_fk_col: target,
                }
                for bm in bridge.metrics:
                    row[bm.name] = _bridge_metric_value(
                        bm,
                        entity.name,
                        entity_metrics,
                        fake,
                    )
                rows.append(row)

        column_order = [first_fk_col, second_fk_col] + [bm.name for bm in bridge.metrics]
        df = (
            pd.DataFrame(rows, columns=column_order) if rows else pd.DataFrame(columns=column_order)
        )
        out[bridge.name] = df
        associations[bridge.name] = bridge_assoc_list

    return out, BridgeAssociations(bridges=associations)


# --- Orchestrator ------------------------------------------------------------


@dataclass(frozen=True)
class GenerationState:
    """Structured side-channel for ground-truth manifest emission.

    ``generate_tables`` returns just the table dict to preserve a slim
    public signature. ``generate_tables_with_state`` returns the same
    tables alongside this state object, which carries the per-entity
    trajectory positions used during generation. The manifest builder in
    ``plotsim.manifest`` reads from here rather than re-deriving positions
    from cell values (which would be lossy under noise / MCAR).

    ``scd`` carries per-dim SCD Type 2 versioning state (per-entity
    version lists, surrogate IDs, validity windows, crossing positions).
    ``SCDState.dims`` is empty for configs that declare no SCD columns —
    callers can skip SCD-aware code paths cheaply by checking
    ``state.scd.is_empty``.

    ``bridges`` carries the per-bridge association ground truth (which
    second-dim rows each first-dim entity associated with). The manifest
    emits ``bridge_associations`` from this without re-grouping the
    bridge DataFrames. ``BridgeAssociations(bridges={})`` is the empty
    sentinel for configs without a ``bridges`` block.

    Future fields extend this dataclass; existing callers that
    destructure ``(tables, state)`` keep working because Python
    dataclass fields are accessed by name.
    """

    trajectories: dict[str, np.ndarray]
    scd: SCDState = field(default_factory=lambda: SCDState(dims={}))
    bridges: BridgeAssociations = field(
        default_factory=lambda: BridgeAssociations(bridges={}),
    )


def _date_key_to_period_label(dim_date: pd.DataFrame) -> dict[int, str]:
    """date_key (int) → ``period_label`` string from ``dim_date``.

    The ``period_label`` column carries the human-readable period
    timestamp (``"2024-01"`` for monthly, ``"2024-01-15"`` for daily).
    Used by ``_apply_cdc_audit_columns`` to populate ``_inserted_at`` /
    ``_updated_at``.
    """
    return {
        int(k): str(lbl)
        for k, lbl in zip(dim_date["date_key"].tolist(), dim_date["period_label"].tolist())
    }


def _apply_cdc_audit_columns(
    config: PlotsimConfig,
    fact_tables: dict[str, pd.DataFrame],
    dim_tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """0.6-M9c: emit ``_inserted_at`` / ``_updated_at`` / ``_op`` audit
    columns on every fact table whose ``Table.cdc`` is True.

    Resolves the ISO period string for each row via the row's
    ``date_key`` against ``dim_date``'s ``period_label`` column. Both
    timestamps initialise to that value; the ``_op`` flag initialises
    to ``"I"`` (insert). The U-flip (rows mutated by column-level
    quality issues) happens later in ``output.write_tables`` once the
    quality layer has produced its ground-truth records — this helper
    only owns the columns' presence and initial state.

    Configs without any CDC-enabled facts are a no-op (the input dict
    is returned unchanged); the ``dim_date`` lookup runs once and is
    reused across every CDC fact in the same generation.
    """
    cdc_facts = {t.name for t in config.tables if t.type == "fact" and t.cdc}
    if not cdc_facts:
        return fact_tables
    dim_date = dim_tables.get("dim_date")
    if dim_date is None or "period_label" not in dim_date.columns:
        return fact_tables
    period_label_by_dkey = _date_key_to_period_label(dim_date)
    tables_by_name = {t.name: t for t in config.tables}

    out = dict(fact_tables)
    for name, df in fact_tables.items():
        if name not in cdc_facts:
            continue
        # 0.6-M19 Fix 4: resolve the local date-FK column by source
        # target, not the literal name ``date_key``. A fact may name
        # its date-FK column ``order_date`` / ``billing_period`` /
        # whatever — what matters is that the FK target is
        # ``dim_date.date_key``. Pre-fix, any rename silently broke
        # CDC by skipping the audit-column augmentation.
        tbl = tables_by_name.get(name)
        date_fk = _find_date_fk_column(tbl) if tbl is not None else None
        date_col = date_fk[0] if date_fk is not None else "date_key"
        if date_col not in df.columns:
            # Defensive: a fact without any date FK has no period
            # anchor for the audit timestamps. Skip rather than
            # emitting nonsense — the load-time validator could be
            # tightened later, but for now we silently no-op.
            continue
        df = df.copy(deep=True)
        period_strs = [
            period_label_by_dkey.get(int(d), "") if pd.notna(d) else ""
            for d in df[date_col].tolist()
        ]
        df["_inserted_at"] = period_strs
        df["_updated_at"] = list(period_strs)
        df["_op"] = ["I"] * len(df)
        out[name] = df
    return out


def _build_phase_cholesky(
    config: PlotsimConfig,
    sorted_metrics: list,
    correlations: list,
    *,
    phase_label: str,
) -> tuple[np.ndarray, np.ndarray, Optional[list[dict]]]:
    """0.6-M11: build one Cholesky factor for a single phase's correlations.

    Encapsulates the M120 compensation + M111 Higham projection +
    decomposition that pre-M11 lived inline at the orchestrator's
    Cholesky-build site. Called once per (baseline + each configured
    phase) so each phase produces its own factor while sharing the
    matrix-assembly / compensation / projection flow.

    Returns ``(cholesky_L, projected_matrix, compensation_records)``.
    ``compensation_records`` is ``None`` when ``compensate_correlations``
    is False or when the metric cap forces the legacy direct-copula
    fallback; otherwise a list of per-pair records the manifest reads.

    ``phase_label`` is woven into the compensation-skipped UserWarning so
    multi-phase configs can identify which phase tripped the cap.
    """
    mat = _build_correlation_matrix(sorted_metrics, correlations)
    compensation_records: Optional[list[dict]] = None
    # M120: trajectory-aware pre-compensation. The metric cap mirrors the
    # mission spec — above 20 metrics the additive decomposition is too
    # noisy to satisfy the sign-match floor, so emit a warning and fall
    # through to the legacy direct-copula path.
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
                f"trajectory-aware pre-compensation skipped for {phase_label} "
                "— the additive decomposition becomes too noisy at this "
                "scale. Falling back to the direct copula path.",
                UserWarning,
                stacklevel=2,
            )
        else:
            traj_cov = estimate_trajectory_covariance(
                config,
                metric_order=sorted_metrics,
            )
            mat, compensation_records = compensate_correlation_matrix(
                mat,
                traj_cov,
                sorted_metrics,
                correlations,
            )
            warning_text = _format_correlation_compensation_warning(
                compensation_records,
            )
            if warning_text:
                # 0.6-M11: prefix multi-phase warnings with the phase label
                # so the operator can attribute compensation reports to the
                # right window. Baseline emits unchanged text for backward
                # compatibility with pre-M11 warning-parsing consumers.
                if phase_label != "baseline":
                    warning_text = f"{phase_label}: {warning_text}"
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
    return cholesky_L, projected_mat, compensation_records


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

    This function is a thin shim over ``generate_tables_with_state`` that
    drops the state side-channel; its return contract is the table dict
    alone. Callers that need the trajectories used during generation
    (manifest emission, debugging, downstream feature pipelines) should
    call ``generate_tables_with_state`` directly.
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
    values; manifest emission is the primary consumer.

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
    # invariant across each phase window (depends only on metric name
    # order and the active correlation list), so computing L once per
    # phase saves N_cohorts × N_periods matrix+Cholesky rebuilds. The
    # PSD gate above guarantees every Cholesky succeeds; if the call
    # ever surfaces an error here it means an external caller bypassed
    # ``validate_correlation_psd``.
    #
    # F-06 / 0.4.0: every L must be indexed in the same order the
    # downstream ``apply_correlations`` sees its z vector.
    # ``generate_entity_metrics`` runs ``_toposort_metrics`` on the
    # incoming list before calling ``generate_metrics_for_period``,
    # which passes the toposorted effective metrics to
    # ``apply_correlations``. Pre-F-06 this hoist built L on the
    # declaration-order list, so any config with ``causal_lag`` (which
    # reshuffles metric positions) delivered each configured correlation
    # to whichever metric pair happened to live at those index positions
    # in the toposorted list — the wrong pair unless both swapped
    # symmetrically. Building L in toposort order here restores the
    # invariant "``L`` is indexed by the metric list passed downstream".
    #
    # 0.6-M11: ``cholesky_by_period`` is a length-``n_periods`` list
    # whose entry at ``t`` is the Cholesky factor active at period ``t``
    # — the baseline factor when no phase covers ``t``, or the matching
    # phase factor otherwise. For configs without ``correlation_phases``
    # every entry is the same baseline factor (a list of references, no
    # copy), so the engine threads it identically to the pre-M11
    # single-Cholesky path: byte-identical output by construction.
    cholesky_by_period: Optional[list[np.ndarray]] = None
    cholesky_L: Optional[np.ndarray] = None  # legacy alias for direct callers
    if config.correlations:
        sorted_metrics = _toposort_metrics(list(config.metrics))
        # Baseline matrix.
        L_baseline, projected_baseline, baseline_compensation_records = _build_phase_cholesky(
            config,
            sorted_metrics,
            list(config.correlations),
            phase_label="baseline",
        )
        cholesky_L = L_baseline

        # 0.6-M5: stash the baseline projected matrix + toposorted order
        # so ``plotsim.manifest.build_manifest`` can surface
        # ``manifest.correlations`` for baseline pairs. Per-phase
        # analogues land on ``_phase_projected_correlation_matrices``.
        config._projected_correlation_matrix = projected_baseline
        config._metric_correlation_order = [m.name for m in sorted_metrics]
        if baseline_compensation_records is not None:
            config._correlation_compensations = baseline_compensation_records

        # 0.6-M11: per-phase Cholesky factors. Each phase reuses the
        # same metric set + toposort order as the baseline, so the
        # downstream z-vector index stays aligned regardless of which
        # phase is active at a given period.
        phase_choleskies: dict[int, np.ndarray] = {}
        if config.correlation_phases:
            phase_projected_matrices: dict[int, np.ndarray] = {}
            phase_compensation_records: dict[int, list[dict]] = {}
            for phase_idx, phase in enumerate(config.correlation_phases):
                if not phase.correlations:
                    # Phase declares no overrides → reuse the baseline
                    # factor for that window. (A phase block with empty
                    # ``correlations`` is structurally a no-op.)
                    phase_choleskies[phase_idx] = L_baseline
                    continue
                L_phase, projected_phase, ph_comp_records = _build_phase_cholesky(
                    config,
                    sorted_metrics,
                    list(phase.correlations),
                    phase_label=f"correlation_phases[{phase_idx}]",
                )
                phase_choleskies[phase_idx] = L_phase
                phase_projected_matrices[phase_idx] = projected_phase
                if ph_comp_records is not None:
                    phase_compensation_records[phase_idx] = ph_comp_records
            if phase_projected_matrices:
                config._phase_projected_correlation_matrices = phase_projected_matrices
            if phase_compensation_records:
                config._phase_correlation_compensations = phase_compensation_records

        # Resolve the per-period factor list. Single-phase: every
        # period points at the baseline factor (same object reference,
        # no copy). Multi-phase: pick the phase factor for periods
        # covered by a phase, baseline for any uncovered period.
        if config.correlation_phases:
            period_to_phase = config.resolve_period_to_phase()
            cholesky_by_period = [
                phase_choleskies[ph_idx] if ph_idx is not None else L_baseline
                for ph_idx in period_to_phase
            ]
        else:
            cholesky_by_period = [L_baseline] * n_periods

    # Compute the per-entity metric series ONCE and pass to both
    # ``build_fact_tables`` and ``build_bridge_tables``. Without this hoist
    # bridges would re-call ``generate_entity_metrics`` and consume RNG draws
    # downstream of fact construction, producing different fact values than
    # before. Configs with empty ``bridges`` skip the bridge call entirely;
    # they still benefit from the single computation.
    #
    # M127b: the copula flip removed the bypass machinery the manifest's
    # ``bypass_fallback_counts`` field reported on. The field is kept as
    # an empty dict for backward-compat with old manifest readers (it is
    # always populated, never ``None``); nothing populates it.
    n_periods = len(dim_tables["dim_date"])
    entity_metrics = _compute_entity_metrics(
        config,
        trajectories,
        n_periods,
        rng,
        cholesky_L=cholesky_L,
        cholesky_by_period=cholesky_by_period,
    )
    config._bypass_fallback_counts = {}

    fact_tables = build_fact_tables(
        config,
        trajectories,
        dim_tables,
        rng,
        cholesky_L=cholesky_L,
        cholesky_by_period=cholesky_by_period,
        entity_metrics=entity_metrics,
    )
    # M106: append ``dim_row_id`` BEFORE ``assign_stages`` so the output
    # column ordering invariant "stage column appended last" still holds —
    # ``output._ordered_columns`` ranks unmodelled columns by DataFrame
    # insertion order, and putting dim_row_id first keeps stage at the
    # tail. The helper is a no-op when ``scd_state.is_empty``.
    fact_tables = attach_dim_row_id_to_facts(
        config,
        fact_tables,
        dim_tables,
        scd_state,
    )
    fact_tables = assign_stages(config, fact_tables)
    # 0.6-M9c: emit CDC audit columns on every ``cdc=True`` fact table.
    # Runs AFTER ``assign_stages`` so the audit columns sit at the tail
    # of the column order (matches the existing append-at-end convention
    # for engine-added columns like ``stage`` and ``dim_row_id``).
    fact_tables = _apply_cdc_audit_columns(config, fact_tables, dim_tables)
    event_tables = build_event_tables(config, fact_tables, dim_tables, rng)
    event_tables = attach_dim_row_id_to_facts(
        config,
        event_tables,
        dim_tables,
        scd_state,
    )

    # M107: bridge tables run after fact/event construction so they see the
    # final SCD-expanded dim DataFrames and the same entity_metrics dict the
    # facts were built from. Bridges are static (non-temporal) so they slot
    # into the table dict alongside dims/facts/events without changing the
    # rest of the pipeline. Configs without ``bridges`` get an empty dict +
    # empty associations and the helper short-circuits.
    bridge_tables, bridge_associations = build_bridge_tables(
        config,
        dim_tables,
        trajectories,
        entity_metrics,
        scd_state,
        rng,
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
