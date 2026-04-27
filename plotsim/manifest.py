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

    return ManifestSchema(
        schema_version=MANIFEST_SCHEMA_VERSION,
        seed=int(config.seed),
        config_sha256=config_sha256(config),
        archetype_assignments=archetype_assignments,
        trajectory_samples=trajectory_samples,
        event_firings=event_firings,
        scd_events=scd_events,
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
    "EntityArchetypeAssignment",
    "EventFiring",
    "ManifestSchema",
    "TrajectorySample",
    "build_manifest",
    "config_sha256",
    "write_manifest",
]
