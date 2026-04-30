"""plotsim.manifest — ground-truth manifest for downstream ML/feature work.

What it does:
    Builds a structured JSON sidecar (``manifest.json``) that records the
    *signal layer* of a plotsim run — the inputs an ML pipeline would want
    to predict against rather than re-derive from noisy cell values.

    For each run the manifest captures:

      * **archetype assignments** — which archetype was assigned to each
        entity. This is the latent class label downstream models
        approximate; emitting it lets a notebook compare predictions to
        ground truth without guessing at the entity → archetype mapping.
      * **trajectory samples** — for a deterministic subset of entities,
        every (period_index, position) pair the engine actually used.
        Position is the noise-free, distribution-free behavioral state in
        [0, 1] from which every metric for that entity at that period was
        derived; recovering it from the noisy fact tables is impossible
        in general.
      * **event firing periods** — for each event table, which period
        indices each entity fired at least one row in. Threshold and
        proportional events both flow through the same recording: the
        manifest reports observed firings, not the configured thresholds.
      * **seed** + **config_sha256** — reproducibility metadata. The full
        SHA-256 of the JSON config dump (deterministic, sort_keys, json
        mode) lets a downstream consumer detect that the manifest was
        produced from a different config than the one currently in their
        repo.

Design notes:
    - This module is the only place that knows the manifest's wire shape.
      ``ManifestSchema`` is the single source of truth — adding a field
      means extending the model and the build-time data flow; nothing
      else needs to change.
    - No filesystem side effects in ``build_manifest`` — that's
      ``write_manifest``'s job, called from ``plotsim.output`` so the
      "only ``output.py`` writes files" architectural rule holds.
    - JSON serialization is byte-deterministic: ``sort_keys=True``,
      ``indent=2``, ``ensure_ascii=False``, and a trailing newline.
      Every float in the trajectory samples is funneled through
      ``float(...)`` so numpy types never leak into the wire format —
      the M104 schema-export mission established that pyarrow / numpy
      types break round-trips, and the manifest is downstream of the
      same constraint.
    - Trajectory sampling is deterministic-by-construction: the entity
      subset is the first ``ceil(n * sample_rate)`` entities under
      sorted-name order. No RNG is consumed, so the manifest ordering
      stays stable independent of the seed used to generate the tables.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from plotsim.config import (
    FKSource,
    PKSource,
    PlotsimConfig,
    parse_source,
)


MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0"


class _ManifestBase(BaseModel):
    """Base for every manifest model.

    ``frozen=True`` so a built manifest is immutable (callers that need
    a mutated copy use ``model_copy(update=...)``). ``extra="forbid"``
    so a malformed manifest read off disk fails loudly during validation
    instead of silently dropping unknown fields.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")


class EntityArchetypeAssignment(_ManifestBase):
    """Single (entity, archetype) ground-truth pair."""
    entity: str
    archetype: str


class TrajectorySample(_ManifestBase):
    """One (entity, period_index, position) cell of the trajectory tape.

    ``position`` is in [0, 1]; values outside that range indicate a
    trajectory builder bug, not a manifest bug.
    """
    entity: str
    period_index: int
    position: float


class EventFiring(_ManifestBase):
    """Which period indices an entity fired rows in for one event table.

    ``period_indices`` is sorted ascending. An empty list means the
    entity never fired in this event table (kept rather than omitted so
    a downstream consumer can iterate the full entity × event-table
    matrix without fallback logic).
    """
    entity: str
    table: str
    period_indices: list[int]


class BridgeAssociationRecord(_ManifestBase):
    """M107: ground-truth M:M associations for one bridge × one entity.

    Records which second-dim entries each first-dim entity associated
    with on a given bridge. ``targets`` is the list of second-dim FK
    values (PK values for non-SCD dims, ``dim_row_id`` for SCD dims —
    matches the engine's bridge FK column semantics). ``cardinality``
    is ``len(targets)``, surfaced separately so manifest consumers can
    aggregate per-entity counts without iterating each tuple.
    """
    bridge: str
    entity: str
    targets: list[Any]
    cardinality: int


class QualityInjection(_ManifestBase):
    """M107: ground-truth record of one quality-issue corruption.

    Recorded per (issue_index, table, column) so a downstream consumer
    can recover the clean cell values for any corrupted row without
    re-running generation. ``issue_index`` is the position of the
    issue in ``config.quality.quality_issues`` so multi-issue configs
    keep their per-issue records distinguishable. ``row_indices`` are
    the integer row positions in the *output* (corrupted) DataFrame —
    the indices of the rows the corruption applied to. ``clean_values``
    is the original cell values at those rows (one entry per row in
    the same order). For ``duplicate_rows`` and ``late_arrival`` the
    column field carries a sentinel ``"_rows"`` / ``"_arrival_period"``
    name and ``clean_values`` is empty (the corruption isn't a per-cell
    edit but a row-level operation).
    """
    issue_index: int
    issue_type: str
    table: str
    column: str
    row_indices: list[int]
    clean_values: list[Any]


class SCDEvent(_ManifestBase):
    """M106: one SCD Type 2 band crossing for one entity in one dim table.

    Recorded only for *transitions* — the entity's initial band at
    period 0 is reflected in the dim table itself but does not
    generate an event (no crossing happened). Subsequent advances
    each emit one ``SCDEvent``.

    ``trigger_position`` is the trajectory position at the crossing
    period — the same scalar that drove the band change. Downstream
    consumers can join this against ``trajectory_samples`` to reconstruct
    "the entity's plan tier upgraded when its trajectory first reached
    0.42" without re-reading thresholds out of the config.
    """
    dim_table: str
    entity: str
    period_index: int
    old_label: str
    new_label: str
    old_dim_row_id: int
    new_dim_row_id: int
    trigger_metric: str
    trigger_position: float


class CorrelationAdjustment(_ManifestBase):
    """M111: ground-truth record of one Higham nearest-PD adjustment.

    Emitted in the manifest's ``correlation_adjustments`` list when the
    user-specified correlation matrix was not positive-definite and had
    to be projected. ``requested`` is the coefficient the user wrote in
    the YAML; ``achieved`` is the value at the same (i, j) cell of the
    projected matrix; ``adjustment`` is ``abs(requested - achieved)`` —
    surfaced separately so a downstream consumer can sort/filter by
    deviation magnitude without recomputing.

    Pairs whose deviation falls below the numerical noise floor
    (~1e-12) are dropped, so an empty adjustments list with
    ``correlation_adjustments=null`` and an empty list with one entry
    rounded out by tolerance never collide.
    """
    metric_a: str
    metric_b: str
    requested: float
    achieved: float
    adjustment: float


class CorrelationCompensation(_ManifestBase):
    """M120: ground-truth record of one trajectory-aware compensation.

    Emitted in the manifest's ``correlation_compensations`` list when
    ``config.compensate_correlations=True`` and at least one declared
    correlation pair was compensated. Distinct from M111's
    ``CorrelationAdjustment`` (which records "your matrix wasn't PD,
    Higham picked a nearby valid one") — this records the structural
    re-targeting the engine performed to make the user's table-wide
    correlation visible against the trajectory's covariance.

      * ``user_target`` — coefficient the user wrote in YAML
        (``connections`` for builder configs, ``correlations`` directly
        for engine-direct configs).
      * ``trajectory_contribution`` — within-archetype-weighted Pearson
        the trajectory's centers induce between this pair, before the
        copula touches anything. Range ``[-1, 1]``; sign tells the
        operator whether the trajectory amplifies or opposes the
        configured target.
      * ``compensated_target`` — pre-clamp ``user_target -
        trajectory_contribution``. May fall outside ``[-1, 1]`` when the
        trajectory contribution exceeds the user target's magnitude in
        the opposite direction, in which case the copula target is
        infeasible and the engine clamps.
      * ``achievable`` — ``compensated_target`` clamped to ``[-1, 1]``.
        Equal to ``compensated_target`` for feasible pairs; the bound
        for infeasible ones.
      * ``infeasible`` — True when ``compensated_target`` fell outside
        ``[-1, 1]``. The engine still produces valid output, but the
        realized table-wide Pearson for this pair will land at
        ``user_target ± something < |user_target|`` rather than at the
        user target exactly.
      * ``adjustment`` — ``abs(user_target - achievable)``; surfaced
        for sort/filter without recomputation.

    All declared pairs that fall in the metric set produce a record,
    feasible or not. Auto-zero off-diagonals (pairs the user didn't
    declare) are not recorded — they're implicitly feasible and don't
    change the user's contract.
    """
    metric_a: str
    metric_b: str
    user_target: float
    trajectory_contribution: float
    compensated_target: float
    achievable: float
    infeasible: bool
    adjustment: float


class HoldoutInfo(_ManifestBase):
    """M109: ground-truth record of the temporal holdout split.

    Emitted on the manifest only when ``config.holdout.enabled`` is
    True; ``None`` otherwise. Records exactly the values needed for a
    downstream consumer to reproduce the split without re-reading the
    YAML config:

      * ``target_metric`` — the metric named as the prediction target.
        Mirrors ``config.holdout.target_metric``.
      * ``holdout_periods`` — the trailing-period count reserved for
        evaluation. Mirrors ``config.holdout.holdout_periods``.
      * ``cutoff_period_index`` — the resolved boundary
        (``n_periods - holdout_periods``) so a consumer can slice the
        unsplit fact table or its own derivative on the same axis
        without recomputing ``period_count`` from ``time_window``.
    """
    target_metric: str
    holdout_periods: int
    cutoff_period_index: int


class ManifestSchema(_ManifestBase):
    """Top-level manifest payload.

    ``schema_version`` tags the wire shape; bumping it is a signal to
    downstream consumers that they need to re-read the parsing logic.
    """
    schema_version: str
    seed: int
    config_sha256: str
    archetype_assignments: list[EntityArchetypeAssignment]
    trajectory_samples: list[TrajectorySample]
    event_firings: list[EventFiring]
    scd_events: list[SCDEvent] = []
    bridge_associations: list[BridgeAssociationRecord] = []
    quality_injections: list[QualityInjection] = []
    # M109: filled by ``output.write_tables`` right before the manifest
    # is serialized when ``config.holdout.enabled`` is True. ``None``
    # for runs that don't opt into the split — backwards compatible
    # with M105–M108 manifests on disk (pydantic reads the missing
    # field as the default).
    holdout: Optional[HoldoutInfo] = None
    # M111: filled by ``build_manifest`` from
    # ``config._correlation_adjustments`` when the load-time validator had
    # to Higham-project a non-PD correlation matrix. ``None`` when the
    # user-specified matrix was already PD (the common case) or when no
    # correlations were configured. Backwards compatible with pre-M111
    # manifests (default reads as None).
    correlation_adjustments: Optional[list[CorrelationAdjustment]] = None
    # M120: filled by ``build_manifest`` from
    # ``config._correlation_compensations`` when
    # ``compensate_correlations=True`` and pre-compensation ran (at least
    # one declared pair). ``None`` for engine-direct runs that skip the
    # feature, runs whose configs have no ``correlations``, and runs
    # whose metric count exceeds ``_MAX_METRICS_FOR_COMPENSATION`` (the
    # warning-and-fall-through path). An empty list is reserved for "ran
    # but no in-scope pairs" and currently shouldn't surface — the
    # generator only sets the attr when at least one record was emitted.
    correlation_compensations: Optional[list[CorrelationCompensation]] = None


# --- Helpers -----------------------------------------------------------------


def config_sha256(config: PlotsimConfig) -> str:
    """Full SHA-256 hex of the JSON-serialized config dump.

    Mirrors ``output._config_fingerprint`` but returns the full 64-char
    hex digest instead of the 16-char prefix. The fingerprint stays
    short for human-readable validation reports; the manifest carries
    the full hash so downstream consumers can detect any config drift
    at full collision resistance.
    """
    payload = config.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_entity_subset(
    entity_names: list[str], sample_rate: float,
) -> list[str]:
    """Pick a deterministic entity subset for trajectory sampling.

    ``ceil(n * sample_rate)`` clamped to ``[1, n]`` — even at very small
    rates at least one trajectory lands. Entities are taken from sorted
    name order, so the chosen subset is stable across runs and
    independent of the RNG seed.
    """
    n = len(entity_names)
    if n == 0:
        return []
    n_sampled = max(1, int(np.ceil(n * sample_rate)))
    n_sampled = min(n_sampled, n)
    return sorted(entity_names)[:n_sampled]


def _date_key_to_period_index(dim_date: pd.DataFrame) -> dict:
    """Build a date_key → period_index lookup from dim_date.

    ``dim_date`` is the canonical period spine; row order IS period order
    (period_index 0 is the first row, etc.). Event tables carry a
    ``date_key`` FK; mapping it back through this dict gives the period
    index the manifest records.
    """
    return {
        row.date_key: idx for idx, row in enumerate(dim_date.itertuples())
    }


def _firings_for_event_table(
    table_name: str,
    event_df: pd.DataFrame,
    config: PlotsimConfig,
    tables: dict[str, pd.DataFrame],
    period_index_by_date_key: dict,
) -> list[EventFiring]:
    """Produce one EventFiring per entity for a given event table.

    Locates the event table's entity FK and date FK columns, joins each
    row's date_key back to a period index, then bridges entity FK values
    back to ``config.entities`` order via the parent dim's PK column
    (which ``build_fact_tables`` populates 1:1 with config-entity order).
    Entities that never fired get an empty list so the manifest's
    ``entity × event-table`` matrix is rectangular.
    """
    event_tbl = next((t for t in config.tables if t.name == table_name), None)
    if event_tbl is None:
        return []

    # Build a name → grain map so we can identify which FK target is the
    # per_entity dim (e.g., dim_company in saas), versus sub-entity dims
    # (dim_user, grain=variable) and reference dims (dim_plan,
    # per_reference). The bundled templates put user_id BEFORE company_id
    # on evt_login, so a "first non-dim_date FK" heuristic would route
    # firings to dim_user — which is per-row, not per-entity, and makes
    # the entity bridge nonsensical.
    dim_grain_by_name = {
        t.name: t.grain for t in config.tables if t.type == "dim"
    }

    entity_fk_col: Optional[str] = None
    parent_dim_name: Optional[str] = None
    date_fk_col: Optional[str] = None
    for col in event_tbl.columns:
        parsed = parse_source(col.source)
        if not isinstance(parsed, FKSource):
            continue
        if parsed.table == "dim_date":
            date_fk_col = col.name
        elif (
            entity_fk_col is None
            and dim_grain_by_name.get(parsed.table) == "per_entity"
        ):
            entity_fk_col = col.name
            parent_dim_name = parsed.table

    firings: list[EventFiring] = []
    if entity_fk_col is None or date_fk_col is None or parent_dim_name is None:
        # Event table without the standard (entity FK, date FK) shape —
        # emit nothing rather than guess.
        return firings

    parent_dim_df = tables.get(parent_dim_name)
    parent_dim_tbl = next(
        (t for t in config.tables if t.name == parent_dim_name), None,
    )
    if parent_dim_df is None or parent_dim_tbl is None:
        return firings

    # The PK column for the parent dim — whichever column declares ``pk``
    # as its source.
    parent_pk_col: Optional[str] = None
    for col in parent_dim_tbl.columns:
        if isinstance(parse_source(col.source), PKSource):
            parent_pk_col = col.name
            break
    if parent_pk_col is None:
        return firings

    # Parent dim row order is config.entities order (per
    # tables.build_fact_tables's contract: per_entity dims are 1:1 with
    # config.entities).
    # M106: SCD-expanded per_entity dims hold N × versions rows but the
    # entity business key is repeated across versions. Dedupe to one row
    # per PK (first-version-wins) so the bridge is config.entities-aligned
    # again — ``expand_scd_dims`` iterates entities in config order, so
    # the deduped frame preserves that ordering.
    pk_values = (
        parent_dim_df.drop_duplicates(subset=[parent_pk_col], keep="first")
        [parent_pk_col].tolist()
    )
    if len(pk_values) != len(config.entities):
        # Parent dim unique-PK count doesn't match — can't bridge by position.
        return firings
    entity_pk_by_name = {
        entity.name: pk for entity, pk in zip(config.entities, pk_values)
    }

    if event_df.empty:
        for entity in config.entities:
            firings.append(EventFiring(
                entity=entity.name, table=table_name, period_indices=[],
            ))
        return firings

    pk_to_periods: dict = {}
    for pk_value, group in event_df.groupby(entity_fk_col, sort=False):
        period_idxs = sorted({
            period_index_by_date_key[dk]
            for dk in group[date_fk_col].tolist()
            if dk in period_index_by_date_key
        })
        pk_to_periods[pk_value] = period_idxs

    for entity in config.entities:
        pk = entity_pk_by_name[entity.name]
        firings.append(EventFiring(
            entity=entity.name,
            table=table_name,
            period_indices=pk_to_periods.get(pk, []),
        ))
    return firings


def _is_event_table(name: str, config: PlotsimConfig) -> bool:
    tbl = next((t for t in config.tables if t.name == name), None)
    return tbl is not None and tbl.type == "event"


# --- Build / write -----------------------------------------------------------


def build_manifest(
    config: PlotsimConfig,
    trajectories: dict[str, np.ndarray],
    tables: dict[str, pd.DataFrame],
    sample_rate: Optional[float] = None,
    scd_state: Optional[Any] = None,
    bridge_state: Optional[Any] = None,
) -> ManifestSchema:
    """Assemble the manifest from config + generation state + tables.

    ``sample_rate`` overrides ``config.manifest.trajectory_sample_rate``
    when supplied; useful for tests that want a smaller sample without
    rewriting the YAML. Default ``None`` reads the config.

    M106: ``scd_state`` (the ``GenerationState.scd`` field) carries the
    per-entity SCD version lists ``tables.expand_scd_dims`` produced.
    When supplied, every band crossing (transition between versions) is
    recorded as an ``SCDEvent`` in the manifest. ``None`` (or an empty
    state) leaves ``manifest.scd_events`` as ``[]`` — backwards
    compatible with M105 callers that haven't been updated yet.

    M107: ``bridge_state`` (the ``GenerationState.bridges`` field)
    carries the per-bridge association lists ``tables.build_bridge_tables``
    produced. When supplied, each first-dim entity's association set on
    each bridge becomes one ``BridgeAssociationRecord``. ``None`` leaves
    ``manifest.bridge_associations`` as ``[]``.

    The function is pure and stateless — same inputs → same output. No
    RNG, no clock, no filesystem.
    """
    rate = (
        sample_rate if sample_rate is not None
        else config.manifest.trajectory_sample_rate
    )

    archetype_assignments = sorted(
        [
            EntityArchetypeAssignment(entity=e.name, archetype=e.archetype)
            for e in config.entities
        ],
        key=lambda a: a.entity,
    )

    sampled_entity_names = _sample_entity_subset(
        [e.name for e in config.entities], rate,
    )
    trajectory_samples: list[TrajectorySample] = []
    for ename in sampled_entity_names:
        traj = trajectories.get(ename)
        if traj is None:
            continue
        for p in range(len(traj)):
            trajectory_samples.append(TrajectorySample(
                entity=ename,
                period_index=p,
                position=float(traj[p]),
            ))

    dim_date = tables.get("dim_date")
    if dim_date is None:
        period_index_by_date_key: dict = {}
    else:
        period_index_by_date_key = _date_key_to_period_index(dim_date)

    event_firings: list[EventFiring] = []
    event_table_names = sorted(
        name for name in tables if _is_event_table(name, config)
    )
    for table_name in event_table_names:
        event_firings.extend(_firings_for_event_table(
            table_name, tables[table_name], config, tables,
            period_index_by_date_key,
        ))

    scd_events: list[SCDEvent] = []
    if scd_state is not None and getattr(scd_state, "dims", None):
        # Sort dim tables for stable manifest ordering across runs.
        for dim_name in sorted(scd_state.dims.keys()):
            dim_state = scd_state.dims[dim_name]
            for entity_name in sorted(dim_state.versions.keys()):
                versions = dim_state.versions[entity_name]
                # versions[0] is the entity's starting band; only later
                # entries are crossings, so iterate in pairs from index 1.
                for i in range(1, len(versions)):
                    prev = versions[i - 1]
                    curr = versions[i]
                    scd_events.append(SCDEvent(
                        dim_table=dim_name,
                        entity=entity_name,
                        period_index=int(curr.valid_from_period),
                        old_label=prev.band_label,
                        new_label=curr.band_label,
                        old_dim_row_id=int(prev.dim_row_id),
                        new_dim_row_id=int(curr.dim_row_id),
                        trigger_metric=dim_state.trigger_metric,
                        trigger_position=float(curr.crossing_position or 0.0),
                    ))

    bridge_associations: list[BridgeAssociationRecord] = []
    if bridge_state is not None and getattr(bridge_state, "bridges", None):
        # Sort bridge names for stable manifest ordering across runs.
        for bridge_name in sorted(bridge_state.bridges.keys()):
            for assoc in bridge_state.bridges[bridge_name]:
                bridge_associations.append(BridgeAssociationRecord(
                    bridge=bridge_name,
                    entity=assoc.entity,
                    targets=list(assoc.targets),
                    cardinality=int(assoc.cardinality),
                ))

    # M111: read the load-time projection record off the config's private
    # attribute. ``None`` for runs whose user-specified matrix was already
    # PD (validator never set it); a non-empty list for runs where Higham
    # had to adjust at least one pair. The PrivateAttr design keeps the
    # adjustment record out of ``model_dump`` / ``config_sha256`` so
    # YAML round-trips and the config fingerprint stay clean.
    raw_adjustments = getattr(config, "_correlation_adjustments", None)
    if raw_adjustments:
        correlation_adjustments: Optional[list[CorrelationAdjustment]] = [
            CorrelationAdjustment(**rec) for rec in raw_adjustments
        ]
    else:
        correlation_adjustments = None

    # M120: read trajectory-aware compensation records the same way M111
    # reads its Higham adjustments. The two flows are independent — a run
    # can emit one, both, or neither.
    raw_compensations = getattr(config, "_correlation_compensations", None)
    if raw_compensations:
        correlation_compensations: Optional[list[CorrelationCompensation]] = [
            CorrelationCompensation(**rec) for rec in raw_compensations
        ]
    else:
        correlation_compensations = None

    return ManifestSchema(
        schema_version=MANIFEST_SCHEMA_VERSION,
        seed=int(config.seed),
        config_sha256=config_sha256(config),
        archetype_assignments=archetype_assignments,
        trajectory_samples=trajectory_samples,
        event_firings=event_firings,
        scd_events=scd_events,
        bridge_associations=bridge_associations,
        quality_injections=[],
        correlation_adjustments=correlation_adjustments,
        correlation_compensations=correlation_compensations,
    )


def write_manifest(manifest: ManifestSchema, output_dir: Path) -> Path:
    """Serialize ``manifest`` to ``<output_dir>/manifest.json``.

    JSON is rendered with ``sort_keys=True`` and ``indent=2`` so two
    runs at the same seed produce byte-identical bytes. A trailing
    newline is appended for tooling compatibility (mirrors
    ``write_validation_report``'s convention).

    All values are funneled through Pydantic's ``model_dump(mode='json')``
    which converts numpy scalars and tuples to native Python primitives.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_FILENAME
    payload = manifest.model_dump(mode="json")
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "BridgeAssociationRecord",
    "CorrelationAdjustment",
    "CorrelationCompensation",
    "EntityArchetypeAssignment",
    "EventFiring",
    "HoldoutInfo",
    "ManifestSchema",
    "QualityInjection",
    "SCDEvent",
    "TrajectorySample",
    "build_manifest",
    "config_sha256",
    "write_manifest",
]
