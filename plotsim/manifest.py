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
from pydantic import BaseModel, ConfigDict, Field

from plotsim.config import (
    FKSource,
    PKSource,
    PlotsimConfig,
    parse_source,
)


MANIFEST_FILENAME = "manifest.json"
# 0.6-M5: bumped 1.0 → 1.1 for the additive ``causal_graph`` /
# ``correlations`` / ``outlier_injections`` sections. All three default to
# ``[]`` or ``None`` so manifests on disk produced by 1.0 readers parse
# unchanged; 1.1 readers see the new fields populated.
# 0.6-M8a: bumped 1.1 → 1.2 for the additive per-entity
# ``EntityArchetypeAssignment.active_window`` field. ``None`` default
# means manifests on disk produced by 1.1 readers parse unchanged; 1.2
# readers see the new field populated for every entity (the default
# ``start_period=0`` produces ``ActiveWindow(start=0, end=n_periods)``,
# which is non-load-bearing but explicit).
# 0.6-M8c: bumped 1.2 → 1.3 for the additive
# ``EntityArchetypeAssignment.treatment`` field and the top-level
# ``treatment_cohorts`` list. Both default to ``None`` / ``[]`` so
# manifests on disk produced by 1.2 readers parse unchanged; 1.3
# readers see the new fields populated only when the config uses
# the M8c surface (treatment-free configs leave both empty).
# 0.6-M11: bumped 1.3 → 1.4 for the additive ``correlation_phases``
# summary list and the optional ``phase_index`` field on
# ``CorrelationAdjustment`` / ``CorrelationCompensation`` /
# ``CorrelationEntry``. ``phase_index`` defaults to ``None`` (baseline)
# so 1.3-emitted records re-read clean on a 1.4 parser; the new
# top-level list defaults to ``[]``. Configs without
# ``correlation_phases`` produce a 1.4 manifest byte-equivalent to 1.3
# modulo the schema_version string and the empty list.
# Bumped 1.4 → 1.5 for the additive ``source_entity_mappings`` list.
# Configs without ``multi_source`` produce an empty list (the default),
# so 1.4 readers parse 1.5 manifests cleanly except for the new field.
# Multi-source configs populate the list with one record per
# (entity, source, dim_table) tuple — the ground-truth answer key for
# entity-resolution exercises.
# 0.6-M18: bumped 1.5 → 1.6 for the additive ``parent_child_relations``
# list. Configs without per_parent_row child tables produce an empty
# list (the default), so 1.5 readers parse 1.6 manifests cleanly except
# for the new field. Configs with parent/child grain populate the list
# with one record per (parent_table, child_table) edge — the metadata
# downstream exercises need to enumerate header/detail pairings without
# re-scanning column sources.
# 0.6-M22: bumped 1.6 → 1.7 for the optional ``noise_config`` field.
# ``None`` by default and on every config that runs with
# ``noise.scale_with_trajectory=False`` (the historical lane), so 1.6
# readers parse 1.7 manifests cleanly. Populated only when the
# heteroscedastic-noise feature is enabled — keeps default-off runs
# byte-equivalent to pre-M22 modulo the schema version string.
# 0.6-M23: bumped 1.7 → 1.8 for ``noise_family`` and ``degrees_of_freedom``
# on ``NoiseConfigInfo``. Emission criterion widens to also cover
# non-gaussian families (``noise_family != "gaussian"``) so the manifest
# records the realized noise family whenever it diverges from the
# historical lane, not only when heteroscedastic amplitude is on.
# Default-family default-amplitude runs still emit ``noise_config=None``.
# 0.6-M24: bumped 1.8 → 1.9 for the additive ``target_metric`` field on
# both ``TreatmentAssignment`` (per-entity) and ``TreatmentCohort``
# (per-cohort). Defaults to ``None`` (trajectory-wide lift, the pre-M24
# behaviour), so configs without per-metric targeting emit a 1.9
# manifest byte-equivalent to 1.8 modulo the schema version string and
# the new field's null default. Populated when the entity's
# ``treatment_target_metric`` names a metric — the lift then applies
# only to that metric's effective-position evaluation.
# Bumped 1.9 → 1.10 for three additive sections summarizing the engine's
# realized signal layer alongside the existing trajectory tape:
#   * ``seasonal_decomposition`` — the global per-period seasonal-strength
#     array plus the per-metric / per-entity sensitivities used to scale
#     it. Defaults to empty list + empty dicts on configs without
#     seasonality, which matches the engine no-op lane.
#   * ``regression_pairs_global`` — ordinary-least-squares β + intercept
#     in both directions for every declared correlation pair, pooled
#     across entities, paired with r² and per-direction residual
#     variances. Empty list when the config declares no correlations.
#   * ``regression_pairs_by_archetype`` — the same OLS surface grouped by
#     ``Entity.archetype`` so consumers can see which archetypes carry
#     the correlation. Empty dict when correlations are absent or when
#     ``entity_metrics`` was not threaded.
# Configs without correlations or seasonality emit a 1.10 manifest
# byte-equivalent to 1.9 modulo the schema version string and the three
# new empty containers.
MANIFEST_SCHEMA_VERSION = "1.10"


class _ManifestBase(BaseModel):
    """Base for every manifest model.

    ``frozen=True`` so a built manifest is immutable (callers that need
    a mutated copy use ``model_copy(update=...)``). ``extra="forbid"``
    so a malformed manifest read off disk fails loudly during validation
    instead of silently dropping unknown fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class ActiveWindow(_ManifestBase):
    """0.6-M8a: an entity's active period range.

    ``start`` is the first period at which the entity is present (inclusive,
    matches ``Entity.start_period``). ``end`` is the exclusive upper bound
    — equal to ``n_periods`` for entities present at the end of the window.
    The pair ``(start, end)`` is closed-open: the entity has rows in the
    fact tables for every ``period_index`` in ``[start, end)``. For
    entities with the default ``start_period=0`` and the engine's normal
    "active to the end" assumption, this is ``(0, n_periods)``.
    """

    start: int
    end: int


class TreatmentAssignment(_ManifestBase):
    """0.6-M8c: an entity's treatment / control assignment.

    Four fields, all sourced from the matching ``Entity`` fields:

      * ``group`` — the cohort label (e.g. ``"treatment"`` /
        ``"control"``). Plotsim treats it as opaque metadata.
      * ``lift_log_odds`` — the known effect size for THIS entity in
        log-odds units. ``None`` for control-arm entities.
      * ``start_period`` — the absolute period index at which the lift
        kicks in. Pre-treatment periods (``period_index < start_period``)
        see the same trajectory as the control arm.
      * ``target_metric`` — M24 per-metric targeting. ``None`` means
        the lift applies trajectory-wide (every metric sees it, the
        pre-M24 default); a metric name restricts the lift to that
        metric only. Carried on both treatment and control entities
        for ground-truth symmetry — control arms have no lift to gate.

    Emitted only for entities with at least one treatment field set.
    Default-only entities (no group label, no lift, no start period,
    no target metric) get ``treatment=None`` on their
    ``EntityArchetypeAssignment`` so the M8c manifest field is
    invisible to non-A/B test datasets.
    """

    group: Optional[str]
    lift_log_odds: Optional[float]
    start_period: int
    target_metric: Optional[str] = None


class EntityArchetypeAssignment(_ManifestBase):
    """Single (entity, archetype) ground-truth pair.

    0.6-M8a: ``active_window`` carries the entity's per-(start, end)
    period range. ``None`` on manifests written by pre-1.2 readers; new
    manifests always populate it (default entities get ``(0, n_periods)``).

    0.6-M8c: ``treatment`` carries the entity's treatment assignment.
    ``None`` on manifests written by pre-1.3 readers AND on entities
    without any treatment fields set; populated for every entity with
    at least one treatment field set.
    """

    entity: str
    archetype: str
    active_window: Optional[ActiveWindow] = None
    treatment: Optional[TreatmentAssignment] = None


class TreatmentCohort(_ManifestBase):
    """0.6-M8c: aggregate ground-truth record for one treatment cohort.

    Emitted at the manifest level (one entry per distinct
    ``treatment_group`` label across all entities). Aggregates the
    per-entity lift values: a homogeneous cohort (every entity shares
    the same lift) reports that lift directly; a heterogeneous cohort
    reports the mean and flags it. Provides downstream consumers with
    a quick "what was the configured effect for this cohort" surface
    without iterating every entity.

      * ``label`` — the cohort label (matches ``Entity.treatment_group``).
      * ``n_entities`` — count of entities tagged with this label.
      * ``mean_lift_log_odds`` — average lift across the cohort. ``None``
        when no entity in the cohort has a lift set (the control arm
        contract).
      * ``start_period`` — the typical (modal) ``treatment_start_period``
        for the cohort. Most A/B tests use one start period per cohort,
        so this is the headline value; if the cohort has heterogeneous
        starts (rare, but supported), pick the most common.
      * ``target_metric`` — M24 per-metric targeting. ``None`` when
        every entity in the cohort applies the lift trajectory-wide
        (the pre-M24 default), or when no entity in the cohort
        declares a target metric. Otherwise the modal target metric
        across the cohort — heterogeneous cohorts (rare; segments
        normally map 1:1 to cohort labels and carry one
        ``TreatmentConfig.target_metric``) report their most-common
        value and downstream consumers can cross-reference per-entity
        records for outliers.
    """

    label: str
    n_entities: int
    mean_lift_log_odds: Optional[float]
    start_period: int
    target_metric: Optional[str] = None


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
    the same order). For ``duplicate_rows``, ``late_arrival``, and
    ``volume_anomaly`` the column field carries a sentinel —
    ``"_rows"`` for duplicates / volume anomalies, ``"_arrival_period"``
    for late arrivals — and ``clean_values`` is empty (the corruption
    isn't a per-cell edit but a row-level operation).
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

    0.6-M11: ``phase_index`` identifies which correlation window emitted
    this adjustment. ``None`` = baseline ``config.correlations``; an
    integer = the index into ``config.correlation_phases``. Backwards
    compatible: pre-M11 manifests have no entries with ``phase_index``
    set, and the default-None field reads cleanly on M11 parsers.
    """

    metric_a: str
    metric_b: str
    requested: float
    achieved: float
    adjustment: float
    phase_index: Optional[int] = None


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
    # 0.6-M11: which window this compensation record applies to.
    # ``None`` = baseline; integer = index into config.correlation_phases.
    phase_index: Optional[int] = None


class CausalEdge(_ManifestBase):
    """0.6-M5: one driver → target causal-lag edge from ``config.metrics``.

    Emitted once per metric whose ``causal_lag`` field is not None. The
    pair ``(driver, target)`` is the directed edge; ``lag_periods`` is
    the period offset the target reads the driver at. ``blend_weight``
    surfaces how strongly the driver overrides the target's own current
    trajectory (1.0 = full override, 0.0 = ignored). The downstream
    consumer reconstructs the run's causal DAG by reading every edge —
    no need to re-derive it from the configured connections list.
    """

    driver: str
    target: str
    lag_periods: int
    blend_weight: float


class CorrelationEntry(_ManifestBase):
    """0.6-M5: one user-declared correlation pair with its realized value.

    Emitted once per entry in ``config.correlations`` (the user's
    declared connections list). ``requested`` is the coefficient the
    user wrote in YAML; ``projected`` is the value at the matching cell
    of the matrix the engine actually drove the copula against — i.e.
    after M120 trajectory-aware compensation (when enabled) and M111
    Higham nearest-PD projection (when needed). Auto-zero off-diagonals
    (pairs the user didn't declare) are not recorded.

    Distinct from ``CorrelationAdjustment`` (which only fires when
    Higham had to project) and ``CorrelationCompensation`` (which only
    fires when compensation ran). ``CorrelationEntry`` fires on EVERY
    run that has correlations, so consumers always see the realized
    value for every declared edge.
    """

    metric_a: str
    metric_b: str
    requested: float
    projected: float
    # 0.6-M11: which window this realized-correlation entry applies to.
    # ``None`` = baseline; integer = index into config.correlation_phases.
    phase_index: Optional[int] = None


class CorrelationPhaseInfo(_ManifestBase):
    """0.6-M11: one declared phase window summarized for the manifest.

    Emitted in ``ManifestSchema.correlation_phases`` — one entry per
    ``config.correlation_phases`` declaration. Carries the window
    bounds plus the count of pairs the phase declared so a downstream
    consumer can join ``CorrelationAdjustment`` / ``CorrelationEntry``
    records (which carry ``phase_index``) back to the window they
    apply to without re-reading the source config.

    Empty list when the config did not declare any phases; default
    ``[]`` keeps pre-M11 manifests parsing unchanged.
    """

    phase_index: int
    start_period: int
    end_period: int
    n_pairs: int


class OutlierInjection(_ManifestBase):
    """0.6-M5: one cell where ``noise.outlier_rate`` fired during generation.

    Identifies a cell by ``(entity, period_index, metric)`` — the same
    coordinate space used by ``trajectory_samples`` and ``event_firings``
    so a downstream consumer can join across sections without bridging
    through table row indices. The realized cell value is intentionally
    omitted: a consumer that needs it reads the fact table directly at
    ``(entity, period)`` row, ``metric`` column.

    Emitted only for serial-mode runs (``generation_mode='serial'`` or
    ``'auto'`` resolving to serial). Vectorized mode uses
    ``_apply_noise_batch`` whose RNG consumption order differs from the
    per-cell ``apply_noise`` path the detector replays — recording
    outliers there would require a second engine path or an invasive
    instrumentation hook. Vectorized runs leave
    ``manifest.outlier_injections = None``.

    Cost-gated: detection re-runs the full metric pipeline once with an
    inline replay of ``apply_noise``, so the work scales with the cell
    count. ``manifest.outlier_injections`` is ``None`` (skipped) when
    total cells exceed the budget. ``[]`` means the detector ran and
    found no firings; a non-empty list means at least one cell had an
    outlier draw.
    """

    entity: str
    period_index: int
    metric: str


class SourceEntityMapping(_ManifestBase):
    """0.6-M13: one (entity, source, dim_table) ground-truth mapping record.

    Emitted in ``ManifestSchema.source_entity_mappings`` only when
    ``config.multi_source`` is set. One record per canonical entity per
    declared source per per_entity dim — for a 2-source config with one
    per_entity dim and 50 entities the manifest carries 100 records.

      * ``entity`` — the canonical entity name from ``config.entities[i].name``.
      * ``source`` — the declared source name (matches
        ``SourceDeclaration.name``).
      * ``dim_table`` — the canonical dim table this mapping bridges
        (``dim_<entity>``). The drifted per-source table is at
        ``dim_<entity>_<source>``.
      * ``canonical_entity_id`` — the canonical PK value on
        ``dim_<entity>``. Bridges back to the fact / event tables that
        FK off the canonical PK; those tables are untouched by drift
        (M13 is dim-only).
      * ``source_entity_id`` — the per-source PK value on
        ``dim_<entity>_<source>`` in the source's declared
        ``key_scheme``. The literal value an entity-resolution
        exercise would join on (or, more interestingly, fail to join
        on without fuzzy-matching the drifted name / attribute fields).
      * ``drifted_fields`` — canonical column names that received drift
        on this row. Empty when the row passed through untouched (the
        majority lane at low drift rates). Non-empty lists name the
        column(s) that disagree with the canonical dim — the answer
        key for "which fields will record linkage have to match on?".
    """

    entity: str
    source: str
    dim_table: str
    canonical_entity_id: str
    source_entity_id: str
    drifted_fields: list[str]


class ParentChildRelation(_ManifestBase):
    """0.6-M18: one parent-fact / child-fact pairing record.

    Emitted on the manifest only when at least one ``per_parent_row``
    child table is declared. One record per declared (parent, child)
    edge in ``config.tables`` — multi-child parents produce one record
    per child.

      * ``parent_table`` — name of the parent fact table.
      * ``child_table`` — name of the per_parent_row child fact table.
      * ``children_per_row_min`` / ``children_per_row_max`` — the
        inclusive fan-out range declared on the child
        (``Table.children_per_row``).
      * ``parent_row_count`` — actual row count of the parent fact in
        the generated output (populated post-generation).
      * ``child_row_count`` — actual row count of the child fact.
        Together with ``parent_row_count`` and the range, downstream
        consumers can verify "every parent had between min and max
        children" without re-reading the data.
    """

    parent_table: str
    child_table: str
    children_per_row_min: int
    children_per_row_max: int
    parent_row_count: int
    child_row_count: int


class NoiseConfigInfo(_ManifestBase):
    """0.6-M22 / 0.6-M23: ground-truth record of the noise model.

    Emitted on the manifest when EITHER
    ``config.noise.scale_with_trajectory=True`` (M22 heteroscedastic lane)
    OR ``config.noise.noise_family != "gaussian"`` (M23 heavy-tailed lane).
    ``None`` otherwise — default-family default-amplitude runs stay
    byte-equivalent to the historical pre-M22 manifest modulo the schema
    version string. Carries the declared ``NoiseConfig`` knobs so a
    downstream consumer can fully describe the noise model without
    re-reading the YAML config.

      * ``gaussian_sigma`` — the σ multiplier; the realized scale at a cell
        is ``gaussian_sigma * trajectory_position`` under the
        heteroscedastic lane and ``gaussian_sigma * abs(value)`` (with the
        zero-value fallback) otherwise. Used by every family as the scale
        parameter — Gaussian σ, Laplace scale, Student-t scale multiplier.
      * ``outlier_rate`` / ``mcar_rate`` — recorded for completeness so the
        manifest fully describes the noise model.
      * ``scale_with_trajectory`` — ``True`` when the heteroscedastic lane
        was engaged, ``False`` otherwise (e.g. when the record was emitted
        purely because ``noise_family`` diverged from the default).
      * ``noise_family`` — the additive-jitter distribution; one of
        ``"gaussian"``, ``"student_t"``, ``"laplace"``.
      * ``degrees_of_freedom`` — populated only when
        ``noise_family == "student_t"``; ``None`` otherwise.
    """

    gaussian_sigma: float
    outlier_rate: float
    mcar_rate: float
    scale_with_trajectory: bool
    # 0.6-M23: distribution family for the additive jitter branch.
    # Default ``"gaussian"`` keeps the field readable on pre-M23 manifests
    # on disk (they reparse cleanly with this as the inferred value).
    noise_family: str = "gaussian"
    # 0.6-M23: only populated when ``noise_family == "student_t"``.
    degrees_of_freedom: Optional[float] = None


class SeasonalDecomposition(_ManifestBase):
    """Per-run snapshot of the seasonal-strength inputs.

    Three fields describing the deterministic seasonal layer applied to
    every cell at metric-generation time:

      * ``seasonal_factors`` — the length-``n_periods`` global strength
        array produced by ``_build_seasonal_factors``. Entry ``t`` is the
        sum of every ``SeasonalEffect.strength`` whose ``months`` set
        contains period ``t``'s calendar month. Empty list when the config
        declares no seasonal effects (the engine's short-circuit lane);
        a populated list otherwise.
      * ``metric_seasonal_sensitivities`` — one entry per metric, keyed by
        ``Metric.name`` and valued by ``Metric.seasonal_sensitivity``. The
        per-metric multiplier the engine applies on top of the global
        strength. Empty dict when no seasonal effects are configured —
        the values are inert in that lane and recording them would just
        be noise.
      * ``entity_seasonal_sensitivities`` — one entry per entity, keyed by
        ``Entity.name`` and valued by ``Entity.seasonal_sensitivity``. The
        per-entity multiplier the engine applies on top of the global
        strength. Empty dict under the same condition as
        ``metric_seasonal_sensitivities``.

    A downstream consumer can reconstruct the effective seasonal lift at
    cell ``(entity, period, metric)`` exactly as the engine did:
    ``seasonal_factors[period] * metric_seasonal_sensitivities[metric] *
    entity_seasonal_sensitivities[entity]``.
    """

    seasonal_factors: list[float]
    metric_seasonal_sensitivities: dict[str, float]
    entity_seasonal_sensitivities: dict[str, float]


class RegressionPair(_ManifestBase):
    """Pair-wise OLS summary for one declared correlation edge.

    Computed pair-wise from the realized ``entity_metrics`` arrays (the
    noise-free, distribution-shaped per-cell values, pre-MCAR / outlier
    rewrites). Emitted only for pairs declared in ``config.correlations``
    — undeclared pairs are auto-zero and don't contribute. Both directions
    are surfaced so consumers don't have to re-derive one from the other.

      * ``metric_a`` / ``metric_b`` — the pair, in the order the user
        declared them. Bidirectional fields read as ``a → b`` / ``b → a``.
      * ``beta_a_to_b`` / ``intercept_a_to_b`` — OLS slope and intercept
        for the regression ``b = beta * a + intercept`` over the pooled
        ``(a, b)`` observations.
      * ``beta_b_to_a`` / ``intercept_b_to_a`` — the reverse regression
        ``a = beta * b + intercept``.
      * ``r_squared`` — direction-invariant coefficient of determination.
        Equal to ``corr(a, b) ** 2`` on the pooled observations.
      * ``residual_variance_a_to_b`` — variance of ``b - (beta_a_to_b * a
        + intercept_a_to_b)``; the unexplained-noise scale for the
        ``a → b`` direction.
      * ``residual_variance_b_to_a`` — same for the reverse direction.
      * ``n_observations`` — count of finite ``(a, b)`` pairs used. Cells
        with NaN in either metric (cold-start lead-ins, MCAR-rewritten
        values that leak into the realized series) are excluded.

    ``n_observations < 2`` produces a degenerate record (all β / intercept
    / variances are ``0.0``, ``r_squared`` is ``0.0``); downstream consumers
    should gate on the count before using the coefficients.
    """

    metric_a: str
    metric_b: str
    beta_a_to_b: float
    intercept_a_to_b: float
    beta_b_to_a: float
    intercept_b_to_a: float
    r_squared: float
    residual_variance_a_to_b: float
    residual_variance_b_to_a: float
    n_observations: int


class VariancePartition(_ManifestBase):
    """Mission 026: nested-ANOVA variance partition for one metric.

    A three-level decomposition of the realized metric series:

      * ``ss_between`` — variance attributable to the grouping axis. For
        ``scope="archetype"`` the axis is ``Entity.archetype`` (variance
        between archetype means around the grand mean). For
        ``scope="segment"`` the axis is the curve segment index within
        ``scope_name``'s archetype (variance between segment means
        around the archetype's grand mean).
      * ``ss_within_entity`` — variance between entity means within the
        same group. Captures entity-to-entity dispersion that the
        grouping axis does not explain.
      * ``ss_residual`` — within-entity variance over time, residual to
        the entity's own mean within its group. The unexplained
        time-series noise floor.

    The three sum to ``ss_total = Σ (y - grand_mean)²`` exactly modulo
    floating-point rounding (rtol ≈ 1e-10 holds in practice).
    ``fraction_*`` fields are each ``ss_* / ss_total`` (or zero when
    ``ss_total == 0``).

    Degrees of freedom follow the standard nested-ANOVA convention:

      * ``degrees_of_freedom_between = n_groups - 1``
      * ``degrees_of_freedom_within = n_cells - n_groups`` (cells =
        unique ``(group, entity)`` pairs that contributed observations)
      * ``degrees_of_freedom_residual = n_observations - n_cells``

    Cold-start entities (those with ``Entity.start_period > 0``) and
    other NaN-cell sources are excluded cell-by-cell — only finite
    observations contribute. ``cold_start_entities_excluded`` counts
    entities that had at least one NaN cell in this partition's
    domain; it surfaces "this section dropped data" to the consumer
    without forcing them to re-derive the NaN count.

    ``scope_name`` semantics:

      * ``scope="archetype"`` — single record per metric covering every
        archetype. ``scope_name`` is the literal sentinel ``"all"``.
      * ``scope="segment"`` — one record per ``(metric, archetype)``.
        ``scope_name`` is the archetype name, and ``ss_between`` /
        DOF reflect the segments within that archetype only. Segments
        are NEVER pooled across archetypes — the manifest guarantees
        the archetype-locality of every segment-scope record.
    """

    metric: str
    scope: str
    scope_name: str
    ss_between: float
    ss_within_entity: float
    ss_residual: float
    fraction_between: float
    fraction_within_entity: float
    fraction_residual: float
    degrees_of_freedom_between: int
    degrees_of_freedom_within: int
    degrees_of_freedom_residual: int
    n_observations: int
    cold_start_entities_excluded: int = 0


class GPKernelFit(_ManifestBase):
    """Mission 026: RBF Gaussian-process fit summary for one trajectory shape.

    Emitted with ``scope_type="archetype"`` (one record per declared
    archetype, fitted against the archetype's clean trajectory) plus
    additional ``scope_type="entity"`` records for any entity carrying
    a non-None ``overrides`` field (the entity's specific trajectory
    diverges from the archetype baseline and warrants its own fit).

      * ``scope_name`` — the archetype name (for archetype scope) or
        the entity name (for entity scope).
      * ``kernel_type`` — currently always ``"rbf"``. Reserved for a
        future multi-kernel extension; consumers should still gate on
        the value for forward compatibility.
      * ``hyperparameters`` — three keys when ``converged=True``:
        ``length_scale``, ``signal_variance``, ``noise_variance``. All
        three are in the natural (unstandardized) scale —
        ``length_scale`` is in units of period indices, so a value of
        ``10.0`` means the kernel correlation length spans ten periods.
        Empty dict when ``converged=False``.
      * ``log_marginal_likelihood`` — the maximized value (positive
        sign; the fitter minimizes the negative log likelihood
        internally and negates before reporting). ``None`` when
        ``converged=False``.
      * ``n_train`` — count of finite ``(period, position)`` training
        pairs used. NaN cells (cold-start prefix periods) are
        excluded.
      * ``converged`` — ``True`` when the optimizer reported success
        AND produced finite hyperparameters. ``False`` for flat
        trajectories (variance below ``1e-12``), trajectories with
        fewer than three finite training points, optimizer failures,
        and Cholesky-blow-up paths. Consumers should gate downstream
        usage on this flag rather than inspecting hyperparameters
        directly.
    """

    scope_type: str
    scope_name: str
    kernel_type: str
    hyperparameters: dict[str, float] = Field(default_factory=dict)
    log_marginal_likelihood: Optional[float] = None
    n_train: int
    converged: bool


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
    # M121b: per-archetype count of cells that triggered
    # ``_apply_correlations_batch``'s per-row scalar fallback in
    # vectorized mode. ``None`` in serial mode (the path doesn't
    # measure bypass — there's no batched copula to fall back from).
    # An empty dict means vectorized ran with zero bypass cells (the
    # production-shape case); a non-empty dict surfaces "vectorized
    # isn't faster on this config" investigations directly. Backwards
    # compatible with pre-M121b manifests via the default.
    bypass_fallback_counts: Optional[dict[str, int]] = None
    # M121b: value of ``plotsim.metrics._VECTORIZED_AUTO_THRESHOLD`` at
    # generation time. Recorded so old manifests stay reproducible if
    # the constant changes — a re-run that lands a different
    # ``_resolve_generation_mode`` decision can be detected by
    # comparing this to the current constant. Always populated;
    # default ``None`` is reserved for pre-M121b manifests on disk.
    vectorized_threshold_used: Optional[int] = None
    # 0.6-M5: the run's causal-lag DAG, derived from ``config.metrics``.
    # One ``CausalEdge`` per metric with a non-None ``causal_lag``. Empty
    # list when no metric uses ``causal_lag`` (the bundled-template
    # default for most domains). Default ``[]`` keeps backwards compat
    # with pre-0.6-M5 manifests on disk that lacked the field.
    causal_graph: list[CausalEdge] = []
    # 0.6-M5: one entry per user-declared correlation in
    # ``config.correlations``, with the realized (post-compensation,
    # post-Higham) coefficient surfaced as ``projected``. Empty list when
    # no correlations are configured. Default ``[]`` keeps backwards compat
    # with pre-0.6-M5 manifests.
    correlations: list[CorrelationEntry] = []
    # 0.6-M5: per-cell outlier injection log. ``None`` when the detector
    # was skipped (see ``OutlierInjection`` docstring for skip reasons:
    # ``noise.outlier_rate == 0``, vectorized generation, or cell budget
    # exceeded). ``[]`` when the detector ran and found no firings. A
    # non-empty list records each cell whose noise pipeline drew an
    # outlier multiplier. Default ``None`` keeps backwards compat with
    # pre-0.6-M5 manifests.
    outlier_injections: Optional[list[OutlierInjection]] = None
    # 0.6-M8c: per-cohort treatment record. One entry per distinct
    # ``Entity.treatment_group`` label appearing in the config. Empty
    # list when no entity has a treatment label (the default for
    # non-A/B-test datasets). Default ``[]`` keeps backwards compat
    # with pre-1.3 manifests.
    treatment_cohorts: list[TreatmentCohort] = []
    # 0.6-M11: per-phase window summaries. One entry per declared
    # ``config.correlation_phases`` window, carrying bounds + the
    # configured pair count. Empty list when the config has no phases
    # (single-Cholesky path). Each ``phase_index`` cross-references the
    # optional ``phase_index`` field on ``CorrelationAdjustment``,
    # ``CorrelationCompensation``, and ``CorrelationEntry``. Default
    # ``[]`` keeps pre-M11 manifests parsing unchanged.
    correlation_phases: list[CorrelationPhaseInfo] = []
    # 0.6-M13: per-(entity, source, dim_table) ground-truth mappings
    # produced by the multi-source / overlap dim emission pass. Empty
    # list when the config has no ``multi_source`` block (the default
    # lane — keeps pre-M13 manifests byte-equivalent modulo the schema
    # version bump and the empty list).
    source_entity_mappings: list[SourceEntityMapping] = []
    # 0.6-M18: one record per declared (parent, child) edge for
    # per_parent_row child tables. Empty list when the config has no
    # per_parent_row tables (the default lane — keeps pre-M18
    # manifests byte-equivalent modulo the schema version bump and
    # the empty list).
    parent_child_relations: list[ParentChildRelation] = []
    # 0.6-M22: noise-model record. ``None`` for the historical lane
    # (``noise.scale_with_trajectory=False``) so default-off runs stay
    # byte-equivalent to pre-M22 modulo the schema version bump.
    # Populated only when the heteroscedastic-noise feature is enabled,
    # so a downstream consumer can distinguish a run that opted into
    # position-scaled gaussian noise from one that didn't without
    # re-reading the config.
    noise_config: Optional[NoiseConfigInfo] = None
    # Schema 1.10: seasonal-decomposition snapshot. Always emitted —
    # configs without ``seasonal_effects`` get an empty list + empty
    # dicts (no seasonal layer was active). Configs with seasonality
    # carry the global strength array plus the per-metric / per-entity
    # multipliers so a consumer can reconstruct the effective seasonal
    # lift at any (entity, period, metric) cell without re-reading
    # the YAML config.
    seasonal_decomposition: SeasonalDecomposition = Field(
        default_factory=lambda: SeasonalDecomposition(
            seasonal_factors=[],
            metric_seasonal_sensitivities={},
            entity_seasonal_sensitivities={},
        ),
    )
    # Schema 1.10: pooled OLS summary for every declared correlation
    # pair. Empty list when no correlations are configured or when
    # ``entity_metrics`` was not threaded to ``build_manifest`` — the
    # builder degrades gracefully when called by older callers that
    # haven't been updated to pass the realized arrays. One entry per
    # ``config.correlations`` pair otherwise.
    regression_pairs_global: list[RegressionPair] = []
    # Schema 1.10: same OLS summary grouped by ``Entity.archetype``.
    # Each entry's value list mirrors ``regression_pairs_global`` but
    # restricted to entities of one archetype. Empty dict when no
    # correlations are configured, when ``entity_metrics`` was not
    # threaded, or when no archetype had enough observations to fit
    # OLS (rare; a single-entity archetype with cold-start NaN can
    # produce ``n_observations < 2``).
    regression_pairs_by_archetype: dict[str, list[RegressionPair]] = {}
    # Mission 026: nested-ANOVA variance partitions with archetype as
    # the between-group axis. One record per metric. Empty list when
    # ``entity_metrics`` was not threaded or the config declares no
    # metrics — matches the empty-section contract on no-metric runs.
    variance_partitions: list[VariancePartition] = []
    # Mission 026: nested-ANOVA variance partitions with curve segment
    # as the between-group axis, computed per archetype (never pooled
    # across archetypes). One record per ``(metric, archetype)`` pair
    # whose entities contributed at least one finite observation.
    # Empty list under the same conditions as ``variance_partitions``.
    variance_partitions_by_segment: list[VariancePartition] = []
    # Mission 026: per-archetype (and per-entity for entities with
    # ``overrides``) RBF GP kernel fits over the trajectory shape.
    # Empty list when ``config.metrics`` is empty (the section
    # piggybacks on metric presence so no-metric configs stay
    # byte-equivalent to the pre-026 lane modulo the schema version
    # string and the new empty containers).
    gp_kernel_fits: list[GPKernelFit] = []


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
    entity_names: list[str],
    sample_rate: float,
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
    return {row.date_key: idx for idx, row in enumerate(dim_date.itertuples())}


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
    dim_grain_by_name = {t.name: t.grain for t in config.tables if t.type == "dim"}

    entity_fk_col: Optional[str] = None
    parent_dim_name: Optional[str] = None
    date_fk_col: Optional[str] = None
    for col in event_tbl.columns:
        parsed = parse_source(col.source)
        if not isinstance(parsed, FKSource):
            continue
        if parsed.table == "dim_date":
            date_fk_col = col.name
        elif entity_fk_col is None and dim_grain_by_name.get(parsed.table) == "per_entity":
            entity_fk_col = col.name
            parent_dim_name = parsed.table

    firings: list[EventFiring] = []
    if entity_fk_col is None or date_fk_col is None or parent_dim_name is None:
        # Event table without the standard (entity FK, date FK) shape —
        # emit nothing rather than guess.
        return firings

    parent_dim_df = tables.get(parent_dim_name)
    parent_dim_tbl = next(
        (t for t in config.tables if t.name == parent_dim_name),
        None,
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
    pk_values = parent_dim_df.drop_duplicates(subset=[parent_pk_col], keep="first")[
        parent_pk_col
    ].tolist()
    if len(pk_values) != len(config.entities):
        # Parent dim unique-PK count doesn't match — can't bridge by position.
        return firings
    entity_pk_by_name = {entity.name: pk for entity, pk in zip(config.entities, pk_values)}

    if event_df.empty:
        for entity in config.entities:
            firings.append(
                EventFiring(
                    entity=entity.name,
                    table=table_name,
                    period_indices=[],
                )
            )
        return firings

    pk_to_periods: dict = {}
    for pk_value, group in event_df.groupby(entity_fk_col, sort=False):
        period_idxs = sorted(
            {
                period_index_by_date_key[dk]
                for dk in group[date_fk_col].tolist()
                if dk in period_index_by_date_key
            }
        )
        pk_to_periods[pk_value] = period_idxs

    for entity in config.entities:
        pk = entity_pk_by_name[entity.name]
        firings.append(
            EventFiring(
                entity=entity.name,
                table=table_name,
                period_indices=pk_to_periods.get(pk, []),
            )
        )
    return firings


def _is_event_table(name: str, config: PlotsimConfig) -> bool:
    tbl = next((t for t in config.tables if t.name == name), None)
    return tbl is not None and tbl.type == "event"


def _treatment_assignment_for(entity: Any) -> Optional[TreatmentAssignment]:
    """0.6-M8c: return a per-entity ``TreatmentAssignment`` or ``None``.

    Emits ``None`` for entities with NO treatment fields set (the
    default lane — keeps the M8c manifest field invisible to non-A/B
    test datasets). Emits a populated record otherwise — even for
    control-arm entities (``treatment_lift_log_odds=None`` but
    ``treatment_group="control"``), so the manifest carries ground
    truth for both arms of the experiment.
    """
    if (
        entity.treatment_group is None
        and entity.treatment_lift_log_odds is None
        and entity.treatment_start_period == 0
        and entity.treatment_target_metric is None
    ):
        return None
    return TreatmentAssignment(
        group=entity.treatment_group,
        lift_log_odds=entity.treatment_lift_log_odds,
        start_period=entity.treatment_start_period,
        target_metric=entity.treatment_target_metric,
    )


def _build_treatment_cohorts(entities: list) -> list[TreatmentCohort]:
    """0.6-M8c: aggregate per-entity treatment fields into per-cohort records.

    One ``TreatmentCohort`` per distinct ``treatment_group`` label.
    Entities without a label (the no-op default OR an entity with lift
    set but no label, which is debug-only) don't contribute to any
    cohort — the cohorts list reflects the user's labelled experiment
    arms, not every entity.

    For each cohort:

      * ``mean_lift_log_odds`` — average of non-None lift values across
        the cohort. ``None`` when every entity in the cohort has lift
        unset (the canonical control-arm shape: labelled but no lift).
      * ``start_period`` — modal ``treatment_start_period``. Most A/B
        tests use one start period per cohort; if the cohort has
        heterogeneous starts (rare, supported), pick the most common
        and let downstream consumers cross-reference per-entity
        records for outliers.

    Cohorts are emitted in label-sorted order so manifest output is
    deterministic across runs of the same config.
    """
    by_label: dict[str, list[Any]] = {}
    for e in entities:
        if e.treatment_group is None:
            continue
        by_label.setdefault(e.treatment_group, []).append(e)

    cohorts: list[TreatmentCohort] = []
    for label in sorted(by_label.keys()):
        members = by_label[label]
        lifts = [
            m.treatment_lift_log_odds for m in members if m.treatment_lift_log_odds is not None
        ]
        mean_lift = float(sum(lifts) / len(lifts)) if lifts else None
        # Modal start period — Counter.most_common(1) returns
        # [(value, count)]; take the value. Tie-break: the first one
        # seen (Counter preserves insertion order in 3.7+).
        from collections import Counter

        starts = Counter(m.treatment_start_period for m in members)
        modal_start = starts.most_common(1)[0][0]
        # M24: modal target_metric across the cohort. ``None`` when no
        # member declares one (the pre-M24 default — trajectory-wide
        # lift). Counted across non-None values only; if every member
        # has ``treatment_target_metric=None`` the cohort reports
        # ``None`` (trajectory-wide), matching the pre-M24 manifest
        # shape for that cohort.
        targets = Counter(
            m.treatment_target_metric for m in members if m.treatment_target_metric is not None
        )
        modal_target: Optional[str] = targets.most_common(1)[0][0] if targets else None
        cohorts.append(
            TreatmentCohort(
                label=label,
                n_entities=len(members),
                mean_lift_log_odds=mean_lift,
                start_period=modal_start,
                target_metric=modal_target,
            )
        )
    return cohorts


# --- Decomposition / regression helpers --------------------------------------


def _build_seasonal_decomposition(
    config: PlotsimConfig,
    n_periods: int,
) -> SeasonalDecomposition:
    """Snapshot the seasonal-strength inputs into a manifest record.

    Reuses ``_build_seasonal_factors`` for the global per-period strength
    array so the manifest carries the exact lookup the engine consumed.
    Configs without ``seasonal_effects`` get the empty-sentinel shape
    (empty list + empty dicts) — the engine short-circuits on those
    configs and the sensitivities are inert, so recording them would
    just be noise.
    """
    # Local import: ``plotsim.tables`` transitively imports
    # ``plotsim.manifest`` via the validation chain, so top-level would
    # introduce a cycle on cold load.
    from plotsim.tables import _build_seasonal_factors

    if not config.seasonal_effects:
        return SeasonalDecomposition(
            seasonal_factors=[],
            metric_seasonal_sensitivities={},
            entity_seasonal_sensitivities={},
        )

    factors = _build_seasonal_factors(config, n_periods)
    factor_list: list[float] = [float(x) for x in factors] if factors is not None else []
    return SeasonalDecomposition(
        seasonal_factors=factor_list,
        metric_seasonal_sensitivities={
            m.name: float(m.seasonal_sensitivity) for m in config.metrics
        },
        entity_seasonal_sensitivities={
            e.name: float(e.seasonal_sensitivity) for e in config.entities
        },
    )


def _ols_pair(
    a: np.ndarray,
    b: np.ndarray,
    metric_a: str,
    metric_b: str,
) -> RegressionPair:
    """Compute the bidirectional OLS summary for one ``(a, b)`` pair.

    Strips non-finite cells in either array before fitting; the count of
    surviving cells is the ``n_observations`` reported. ``n < 2`` short-
    circuits to a zero-filled record (no β / variance is well-defined on
    one observation) — downstream consumers should gate on the count.
    Zero-variance inputs (a degenerate column constant across every
    entity-period) also short-circuit, since β is undefined when
    ``var(x) == 0``.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    a_f = a[mask]
    b_f = b[mask]
    n_obs = int(a_f.size)
    if n_obs < 2:
        return RegressionPair(
            metric_a=metric_a,
            metric_b=metric_b,
            beta_a_to_b=0.0,
            intercept_a_to_b=0.0,
            beta_b_to_a=0.0,
            intercept_b_to_a=0.0,
            r_squared=0.0,
            residual_variance_a_to_b=0.0,
            residual_variance_b_to_a=0.0,
            n_observations=n_obs,
        )
    var_a = float(np.var(a_f))
    var_b = float(np.var(b_f))
    if var_a == 0.0 or var_b == 0.0:
        return RegressionPair(
            metric_a=metric_a,
            metric_b=metric_b,
            beta_a_to_b=0.0,
            intercept_a_to_b=float(np.mean(b_f)),
            beta_b_to_a=0.0,
            intercept_b_to_a=float(np.mean(a_f)),
            r_squared=0.0,
            residual_variance_a_to_b=float(var_b),
            residual_variance_b_to_a=float(var_a),
            n_observations=n_obs,
        )
    mean_a = float(np.mean(a_f))
    mean_b = float(np.mean(b_f))
    cov_ab = float(np.mean((a_f - mean_a) * (b_f - mean_b)))
    beta_a_to_b = cov_ab / var_a
    intercept_a_to_b = mean_b - beta_a_to_b * mean_a
    beta_b_to_a = cov_ab / var_b
    intercept_b_to_a = mean_a - beta_b_to_a * mean_b
    resid_a_to_b = b_f - (beta_a_to_b * a_f + intercept_a_to_b)
    resid_b_to_a = a_f - (beta_b_to_a * b_f + intercept_b_to_a)
    # r² is direction-invariant; equal to corr² on the same observations.
    corr = cov_ab / np.sqrt(var_a * var_b)
    r_squared = float(corr * corr)
    return RegressionPair(
        metric_a=metric_a,
        metric_b=metric_b,
        beta_a_to_b=float(beta_a_to_b),
        intercept_a_to_b=float(intercept_a_to_b),
        beta_b_to_a=float(beta_b_to_a),
        intercept_b_to_a=float(intercept_b_to_a),
        r_squared=r_squared,
        residual_variance_a_to_b=float(np.var(resid_a_to_b)),
        residual_variance_b_to_a=float(np.var(resid_b_to_a)),
        n_observations=n_obs,
    )


def _pool_metric_arrays(
    entity_metrics: dict[str, dict[str, np.ndarray]],
    entity_names: list[str],
    metric: str,
) -> np.ndarray:
    """Flatten ``entity_metrics[e][metric]`` across ``entity_names``.

    Missing entities or missing metric keys are skipped silently — the
    caller passes a deliberately-narrowed entity list (global = every
    entity, by-archetype = entities matching one archetype label), and
    cells with NaN are filtered downstream by ``_ols_pair``'s mask.
    """
    chunks: list[np.ndarray] = []
    for ename in entity_names:
        per_metric = entity_metrics.get(ename)
        if per_metric is None:
            continue
        arr = per_metric.get(metric)
        if arr is None:
            continue
        chunks.append(np.asarray(arr, dtype=np.float64))
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks)


def _build_regression_pairs(
    config: PlotsimConfig,
    entity_metrics: dict[str, dict[str, np.ndarray]],
    entity_names: list[str],
) -> list[RegressionPair]:
    """One ``RegressionPair`` per declared correlation, pooled over a subset.

    Scope is intentionally narrowed to ``config.correlations`` (D1: declared
    pairs only — avoids O(n_metrics²) bloat and matches the existing
    ``correlations`` manifest section's scope). Pairs whose pooled
    observation count is < 2 still emit a record (zero-filled β), so the
    section's entry list mirrors ``config.correlations`` 1:1 — downstream
    consumers don't have to fall back when an archetype subset is sparse.

    Output is sorted by ``(metric_a, metric_b)`` to match the
    ``correlations`` section's stable-ordering contract.
    """
    if not config.correlations or not entity_metrics or not entity_names:
        return []
    out: list[RegressionPair] = []
    for pair in config.correlations:
        a = _pool_metric_arrays(entity_metrics, entity_names, pair.metric_a)
        b = _pool_metric_arrays(entity_metrics, entity_names, pair.metric_b)
        if a.size != b.size:
            # Defensive: would indicate a metric-series-length mismatch
            # upstream. Surface as an empty record rather than crash —
            # the n_observations=0 gate handles the rest.
            n = min(a.size, b.size)
            a = a[:n]
            b = b[:n]
        out.append(_ols_pair(a, b, pair.metric_a, pair.metric_b))
    out.sort(key=lambda r: (r.metric_a, r.metric_b))
    return out


def _build_regression_pairs_by_archetype(
    config: PlotsimConfig,
    entity_metrics: dict[str, dict[str, np.ndarray]],
) -> dict[str, list[RegressionPair]]:
    """Group ``_build_regression_pairs`` output by ``Entity.archetype``.

    Archetypes whose entity subset has no matching arrays in
    ``entity_metrics`` are omitted entirely (rather than mapped to an
    empty list) — the dict reflects archetypes that actually contributed
    observations. The within-archetype pair list is sorted the same way
    as the global one.
    """
    if not config.correlations or not entity_metrics:
        return {}
    by_archetype: dict[str, list[str]] = {}
    for e in config.entities:
        by_archetype.setdefault(e.archetype, []).append(e.name)
    out: dict[str, list[RegressionPair]] = {}
    for archetype in sorted(by_archetype.keys()):
        pairs = _build_regression_pairs(
            config,
            entity_metrics,
            by_archetype[archetype],
        )
        if pairs:
            out[archetype] = pairs
    return out


# --- Variance partition / GP helpers (Mission 026) ---------------------------


def _entity_segment_membership(
    config: PlotsimConfig,
    n_periods: int,
) -> dict[str, list[int]]:
    """Per-entity ``(period_index → segment_index)`` lookup table.

    Returns one length-``n_periods`` list per entity. Entry ``p`` is the
    curve segment index that period maps to under that entity's own
    boundary computation (overrides + ``start_period`` applied), or
    ``-1`` for periods before the entity's arrival (cold-start prefix)
    and for entities whose archetype is unresolvable.

    Reuses ``trajectory._segment_boundaries`` / ``_resolve_shift`` so
    the segment definition is byte-equivalent to the trajectory the
    engine actually generated.
    """
    # Local imports: ``plotsim.trajectory`` already imports from
    # ``plotsim.config``, so importing it at module top wouldn't cycle —
    # but keeping the imports local matches the pattern M25 used for
    # ``_build_seasonal_factors``.
    from plotsim.trajectory import _resolve_shift, _segment_boundaries

    arch_by_name = {a.name: a for a in config.archetypes}
    out: dict[str, list[int]] = {}
    for entity in config.entities:
        archetype = arch_by_name.get(entity.archetype)
        membership = [-1] * n_periods
        if archetype is None:
            out[entity.name] = membership
            continue
        active_n = n_periods - entity.start_period
        if active_n < 1:
            out[entity.name] = membership
            continue
        overrides_dict = entity.overrides.model_dump() if entity.overrides is not None else None
        shift = _resolve_shift(archetype, active_n, overrides_dict)
        boundaries = _segment_boundaries(archetype, active_n, shift)
        for seg_idx in range(len(archetype.curve_segments)):
            local_start = boundaries[seg_idx]
            local_end = boundaries[seg_idx + 1]
            absolute_start = entity.start_period + local_start
            absolute_end = entity.start_period + local_end
            for p in range(absolute_start, absolute_end):
                if 0 <= p < n_periods:
                    membership[p] = seg_idx
        out[entity.name] = membership
    return out


def _three_level_anova(
    cells: list[tuple[str, str, np.ndarray]],
) -> Optional[dict[str, float]]:
    """Compute the three-level nested-ANOVA SS / DOF decomposition.

    ``cells`` is a list of ``(group_label, entity_label, values)``
    tuples. Each tuple represents one ``(group, entity)`` cell with at
    least one finite observation. The decomposition is:

        ss_total = ss_between + ss_within_entity + ss_residual

    where

      * ``ss_between`` is the variance between group means around the
        grand mean (weighted by per-group N);
      * ``ss_within_entity`` is the variance between entity means
        around their group mean (weighted by per-cell N);
      * ``ss_residual`` is the within-cell residual variance.

    Returns ``None`` when ``cells`` is empty or carries zero
    observations in total. The caller filters those out — the manifest
    section omits records with no data rather than emitting zero-only
    entries.
    """
    if not cells:
        return None
    all_values = np.concatenate([c[2] for c in cells])
    n_total = int(all_values.size)
    if n_total == 0:
        return None
    grand_mean = float(np.mean(all_values))
    ss_total = float(np.sum((all_values - grand_mean) ** 2))

    cell_n: list[int] = []
    cell_mean: list[float] = []
    cell_ss_residual: list[float] = []
    group_labels: list[str] = []
    for group_label, _entity_label, values in cells:
        n = int(values.size)
        if n == 0:
            continue
        mean = float(np.mean(values))
        cell_n.append(n)
        cell_mean.append(mean)
        cell_ss_residual.append(float(np.sum((values - mean) ** 2)))
        group_labels.append(group_label)

    if not cell_n:
        return None

    group_totals: dict[str, tuple[float, int]] = {}
    for label, n, mean in zip(group_labels, cell_n, cell_mean):
        prev_sum, prev_n = group_totals.get(label, (0.0, 0))
        group_totals[label] = (prev_sum + n * mean, prev_n + n)
    group_mean_by_label: dict[str, float] = {
        label: (s / n if n > 0 else 0.0) for label, (s, n) in group_totals.items()
    }

    ss_between = 0.0
    for label, (_, n) in group_totals.items():
        diff = group_mean_by_label[label] - grand_mean
        ss_between += float(n) * diff * diff

    ss_within_entity = 0.0
    for label, n, mean in zip(group_labels, cell_n, cell_mean):
        diff = mean - group_mean_by_label[label]
        ss_within_entity += float(n) * diff * diff

    ss_residual = float(sum(cell_ss_residual))

    n_groups = len(group_totals)
    n_cells = len(cell_n)
    df_between = max(n_groups - 1, 0)
    df_within = max(n_cells - n_groups, 0)
    df_residual = max(n_total - n_cells, 0)

    return {
        "ss_total": ss_total,
        "ss_between": ss_between,
        "ss_within_entity": ss_within_entity,
        "ss_residual": ss_residual,
        "df_between": float(df_between),
        "df_within": float(df_within),
        "df_residual": float(df_residual),
        "n_observations": float(n_total),
    }


def _make_variance_partition(
    metric: str,
    scope: str,
    scope_name: str,
    anova: dict[str, float],
    cold_start_entities_excluded: int,
) -> VariancePartition:
    """Translate an ``_three_level_anova`` result dict into a manifest record."""
    ss_total = anova["ss_total"]
    if ss_total > 0.0:
        f_between = anova["ss_between"] / ss_total
        f_within = anova["ss_within_entity"] / ss_total
        f_residual = anova["ss_residual"] / ss_total
    else:
        f_between = 0.0
        f_within = 0.0
        f_residual = 0.0
    return VariancePartition(
        metric=metric,
        scope=scope,
        scope_name=scope_name,
        ss_between=float(anova["ss_between"]),
        ss_within_entity=float(anova["ss_within_entity"]),
        ss_residual=float(anova["ss_residual"]),
        fraction_between=float(f_between),
        fraction_within_entity=float(f_within),
        fraction_residual=float(f_residual),
        degrees_of_freedom_between=int(anova["df_between"]),
        degrees_of_freedom_within=int(anova["df_within"]),
        degrees_of_freedom_residual=int(anova["df_residual"]),
        n_observations=int(anova["n_observations"]),
        cold_start_entities_excluded=int(cold_start_entities_excluded),
    )


def _build_variance_partitions_archetype(
    config: PlotsimConfig,
    entity_metrics: dict[str, dict[str, np.ndarray]],
) -> list[VariancePartition]:
    """One ``VariancePartition`` per metric with archetype as the group axis.

    Groups every finite ``(entity, period)`` observation by
    ``Entity.archetype``. Drops metrics with no finite observations
    entirely — the section omits records with no data rather than
    emitting zero-only entries (mirrors the pre-existing manifest
    convention for sparse derived sections).
    """
    if not config.metrics or not entity_metrics:
        return []
    out: list[VariancePartition] = []
    for metric in config.metrics:
        cells: list[tuple[str, str, np.ndarray]] = []
        cold_start = 0
        for entity in config.entities:
            per_metric = entity_metrics.get(entity.name)
            if per_metric is None:
                continue
            arr = per_metric.get(metric.name)
            if arr is None:
                continue
            arr64 = np.asarray(arr, dtype=np.float64)
            mask = np.isfinite(arr64)
            n_kept = int(np.sum(mask))
            if n_kept < int(arr64.size):
                cold_start += 1
            if n_kept == 0:
                continue
            cells.append((entity.archetype, entity.name, arr64[mask]))
        anova = _three_level_anova(cells)
        if anova is None:
            continue
        out.append(
            _make_variance_partition(
                metric.name,
                scope="archetype",
                scope_name="all",
                anova=anova,
                cold_start_entities_excluded=cold_start,
            )
        )
    out.sort(key=lambda v: v.metric)
    return out


def _build_variance_partitions_by_segment(
    config: PlotsimConfig,
    entity_metrics: dict[str, dict[str, np.ndarray]],
    n_periods: int,
) -> list[VariancePartition]:
    """One ``VariancePartition`` per ``(metric, archetype)`` with segment as the group axis.

    Restricted to entities of one archetype at a time — the section
    never pools observations across archetypes. The segment index is
    derived from each entity's own boundary computation
    (``_entity_segment_membership``), so cold-start entities and
    override-bearing entities contribute observations to the segment
    they actually occupied at each period, not to the archetype
    baseline segment.
    """
    if not config.metrics or not entity_metrics:
        return []
    arch_by_name = {a.name: a for a in config.archetypes}
    entities_by_archetype: dict[str, list[Any]] = {}
    for entity in config.entities:
        entities_by_archetype.setdefault(entity.archetype, []).append(entity)

    membership = _entity_segment_membership(config, n_periods)

    out: list[VariancePartition] = []
    for metric in config.metrics:
        for archetype_name in sorted(entities_by_archetype.keys()):
            if archetype_name not in arch_by_name:
                continue
            entities = entities_by_archetype[archetype_name]
            grouped: dict[tuple[int, str], list[float]] = {}
            cold_start = 0
            for entity in entities:
                per_metric = entity_metrics.get(entity.name)
                if per_metric is None:
                    continue
                arr = per_metric.get(metric.name)
                if arr is None:
                    continue
                arr64 = np.asarray(arr, dtype=np.float64)
                seg_map = membership.get(entity.name, [-1] * n_periods)
                has_nan = False
                n_periods_local = int(arr64.size)
                for p in range(n_periods_local):
                    val = arr64[p]
                    if not np.isfinite(val):
                        has_nan = True
                        continue
                    seg_idx = seg_map[p] if p < len(seg_map) else -1
                    if seg_idx < 0:
                        continue
                    grouped.setdefault((seg_idx, entity.name), []).append(float(val))
                if has_nan:
                    cold_start += 1
            if not grouped:
                continue
            cells = [
                (
                    f"segment_{seg_idx}",
                    entity_name,
                    np.asarray(values, dtype=np.float64),
                )
                for (seg_idx, entity_name), values in grouped.items()
            ]
            anova = _three_level_anova(cells)
            if anova is None:
                continue
            out.append(
                _make_variance_partition(
                    metric.name,
                    scope="segment",
                    scope_name=archetype_name,
                    anova=anova,
                    cold_start_entities_excluded=cold_start,
                )
            )
    out.sort(key=lambda v: (v.metric, v.scope_name))
    return out


def _build_gp_kernel_fits(
    config: PlotsimConfig,
    n_periods: int,
) -> list[GPKernelFit]:
    """Per-archetype (and per-override-entity) RBF GP fits.

    ``scope_type="archetype"`` for every archetype: fit against the
    archetype's clean trajectory (no overrides, no cold-start shift).
    ``scope_type="entity"`` for every entity carrying a non-None
    ``overrides`` field — those entities have trajectories that diverge
    from their archetype baseline, so a per-entity record captures the
    actual shape the engine generated.

    The piggyback on ``config.metrics`` (an empty list skips emission
    entirely) matches the byte-equivalence contract: configs with no
    metrics emit no GP records.

    Records are emitted even when the fit fails to converge — the
    ``converged=False`` path carries null hyperparameters / likelihood
    so consumers can distinguish "fit failed" from "fit not attempted."
    """
    if not config.metrics or n_periods < 1:
        return []
    # Local imports: ``plotsim.gp`` is a leaf analytical module that
    # depends on scipy; importing at module top would force scipy load
    # on every manifest build, including no-metric runs.
    from plotsim.gp import fit_rbf
    from plotsim.trajectory import compute_trajectory

    out: list[GPKernelFit] = []
    arch_by_name = {a.name: a for a in config.archetypes}
    x = np.arange(n_periods, dtype=np.float64)

    for archetype_name in sorted(arch_by_name.keys()):
        archetype = arch_by_name[archetype_name]
        traj = compute_trajectory(archetype, n_periods)
        result = fit_rbf(x, traj)
        out.append(
            GPKernelFit(
                scope_type="archetype",
                scope_name=archetype_name,
                kernel_type="rbf",
                hyperparameters=result.hyperparameters or {},
                log_marginal_likelihood=result.log_marginal_likelihood,
                n_train=result.n_train,
                converged=result.converged,
            )
        )

    for entity in config.entities:
        if entity.overrides is None:
            continue
        entity_archetype = arch_by_name.get(entity.archetype)
        if entity_archetype is None:
            continue
        overrides_dict = entity.overrides.model_dump()
        traj = compute_trajectory(
            entity_archetype,
            n_periods,
            overrides_dict,
            entity.start_period,
        )
        result = fit_rbf(x, traj)
        out.append(
            GPKernelFit(
                scope_type="entity",
                scope_name=entity.name,
                kernel_type="rbf",
                hyperparameters=result.hyperparameters or {},
                log_marginal_likelihood=result.log_marginal_likelihood,
                n_train=result.n_train,
                converged=result.converged,
            )
        )

    return out


# --- Build / write -----------------------------------------------------------


def build_manifest(
    config: PlotsimConfig,
    trajectories: dict[str, np.ndarray],
    tables: dict[str, pd.DataFrame],
    sample_rate: Optional[float] = None,
    scd_state: Optional[Any] = None,
    bridge_state: Optional[Any] = None,
    entity_metrics: Optional[dict[str, dict[str, np.ndarray]]] = None,
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

    Schema 1.10: ``entity_metrics`` (the ``GenerationState.entity_metrics``
    field) carries the per-entity, per-metric realized series the engine
    built fact tables from. When supplied, ``regression_pairs_global``
    and ``regression_pairs_by_archetype`` are populated with pair-wise
    OLS summaries for every declared correlation pair. ``None`` (older
    callers, or callers deliberately skipping the regression section)
    leaves both manifest fields at their empty defaults.

    The function is pure and stateless — same inputs → same output. No
    RNG, no clock, no filesystem.
    """
    rate = sample_rate if sample_rate is not None else config.manifest.trajectory_sample_rate

    # 0.6-M8a: derive n_periods from the trajectories dict (every trajectory
    # in the dict has the same length — this is enforced upstream in
    # ``build_fact_tables`` against ``len(dim_date)``). Empty-config edge
    # case (no entities) cannot reach this branch because the manifest is
    # only built from realized generation output.
    n_periods = len(next(iter(trajectories.values()))) if trajectories else 0

    archetype_assignments = sorted(
        [
            EntityArchetypeAssignment(
                entity=e.name,
                archetype=e.archetype,
                active_window=ActiveWindow(start=e.start_period, end=n_periods),
                treatment=_treatment_assignment_for(e),
            )
            for e in config.entities
        ],
        key=lambda a: a.entity,
    )
    treatment_cohorts = _build_treatment_cohorts(config.entities)

    sampled_entity_names = _sample_entity_subset(
        [e.name for e in config.entities],
        rate,
    )
    trajectory_samples: list[TrajectorySample] = []
    for ename in sampled_entity_names:
        traj = trajectories.get(ename)
        if traj is None:
            continue
        for p in range(len(traj)):
            trajectory_samples.append(
                TrajectorySample(
                    entity=ename,
                    period_index=p,
                    position=float(traj[p]),
                )
            )

    dim_date = tables.get("dim_date")
    if dim_date is None:
        period_index_by_date_key: dict = {}
    else:
        period_index_by_date_key = _date_key_to_period_index(dim_date)

    event_firings: list[EventFiring] = []
    event_table_names = sorted(name for name in tables if _is_event_table(name, config))
    for table_name in event_table_names:
        event_firings.extend(
            _firings_for_event_table(
                table_name,
                tables[table_name],
                config,
                tables,
                period_index_by_date_key,
            )
        )

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
                    scd_events.append(
                        SCDEvent(
                            dim_table=dim_name,
                            entity=entity_name,
                            period_index=int(curr.valid_from_period),
                            old_label=prev.band_label,
                            new_label=curr.band_label,
                            old_dim_row_id=int(prev.dim_row_id),
                            new_dim_row_id=int(curr.dim_row_id),
                            trigger_metric=dim_state.trigger_metric,
                            trigger_position=float(curr.crossing_position or 0.0),
                        )
                    )

    bridge_associations: list[BridgeAssociationRecord] = []
    if bridge_state is not None and getattr(bridge_state, "bridges", None):
        # Sort bridge names for stable manifest ordering across runs.
        for bridge_name in sorted(bridge_state.bridges.keys()):
            for assoc in bridge_state.bridges[bridge_name]:
                bridge_associations.append(
                    BridgeAssociationRecord(
                        bridge=bridge_name,
                        entity=assoc.entity,
                        targets=list(assoc.targets),
                        cardinality=int(assoc.cardinality),
                    )
                )

    # M111: read the load-time projection record off the config's private
    # attribute. ``None`` for runs whose user-specified matrix was already
    # PD (validator never set it); a non-empty list for runs where Higham
    # had to adjust at least one pair. The PrivateAttr design keeps the
    # adjustment record out of ``model_dump`` / ``config_sha256`` so
    # YAML round-trips and the config fingerprint stay clean.
    #
    # 0.6-M11: baseline records carry ``phase_index=None``; per-phase
    # records (read from ``_phase_correlation_adjustments``) carry the
    # phase index. The two streams are concatenated into one flat list
    # on the manifest with ``phase_index`` distinguishing them.
    raw_adjustments = getattr(config, "_correlation_adjustments", None)
    raw_phase_adjustments = getattr(config, "_phase_correlation_adjustments", None) or {}
    adjustments_combined: list[CorrelationAdjustment] = []
    if raw_adjustments:
        adjustments_combined.extend(
            CorrelationAdjustment(phase_index=None, **rec) for rec in raw_adjustments
        )
    for phase_idx in sorted(raw_phase_adjustments.keys()):
        for rec in raw_phase_adjustments[phase_idx]:
            adjustments_combined.append(CorrelationAdjustment(phase_index=phase_idx, **rec))
    correlation_adjustments: Optional[list[CorrelationAdjustment]] = (
        adjustments_combined if adjustments_combined else None
    )

    # M120: read trajectory-aware compensation records the same way M111
    # reads its Higham adjustments. The two flows are independent — a run
    # can emit one, both, or neither.
    #
    # 0.6-M11: same baseline + per-phase merge pattern as the adjustments.
    raw_compensations = getattr(config, "_correlation_compensations", None)
    raw_phase_compensations = getattr(config, "_phase_correlation_compensations", None) or {}
    compensations_combined: list[CorrelationCompensation] = []
    if raw_compensations:
        compensations_combined.extend(
            CorrelationCompensation(phase_index=None, **rec) for rec in raw_compensations
        )
    for phase_idx in sorted(raw_phase_compensations.keys()):
        for rec in raw_phase_compensations[phase_idx]:
            compensations_combined.append(CorrelationCompensation(phase_index=phase_idx, **rec))
    correlation_compensations: Optional[list[CorrelationCompensation]] = (
        compensations_combined if compensations_combined else None
    )

    # M121b: pull the bypass-fallback counts off the config's private
    # attr (set by ``generate_tables_with_state`` after the dispatcher
    # runs). ``None`` for serial-mode runs (the field encodes "never
    # measured"); empty dict for vectorized runs with no bypass cells
    # (production shape); populated dict for runs where pathological
    # configs forced the per-row scalar fallback.
    bypass_fallback_counts = getattr(config, "_bypass_fallback_counts", None)

    # M121b: record the auto-threshold constant at generation time so
    # old manifests stay reproducible if the constant changes in a
    # later release. Read from ``plotsim.metrics`` rather than caching
    # at config-load time — this keeps the manifest builder pure
    # without coupling to the orchestrator's state shape.
    from plotsim.metrics import _VECTORIZED_AUTO_THRESHOLD

    vectorized_threshold_used = int(_VECTORIZED_AUTO_THRESHOLD)

    # 0.6-M5: causal-lag DAG. One edge per metric with non-None
    # ``causal_lag`` — sorted by (driver, target) for byte-deterministic
    # output across runs whose metric declaration order differs.
    causal_graph: list[CausalEdge] = sorted(
        [
            CausalEdge(
                driver=m.causal_lag.driver,
                target=m.name,
                lag_periods=int(m.causal_lag.lag_periods),
                blend_weight=float(m.causal_lag.blend_weight),
            )
            for m in config.metrics
            if m.causal_lag is not None
        ],
        key=lambda e: (e.driver, e.target),
    )

    # 0.6-M5: realized correlation values. One entry per user-declared
    # connection in ``config.correlations``, with the projected
    # coefficient pulled from the matrix tables.py stashed at the
    # Cholesky-build site. Sorted by (metric_a, metric_b) for stable
    # output. Skipped when the run had no correlations (the stash
    # never ran) — empty list rather than None to mirror ``causal_graph``'s
    # contract: empty means "ran, nothing to record" / no special signal.
    #
    # 0.6-M11: extended to also emit one entry per pair per declared
    # phase, with ``phase_index`` set. Baseline entries keep
    # ``phase_index=None`` and sort first; per-phase entries follow,
    # sorted by ``(phase_index, metric_a, metric_b)`` for stable
    # output across runs whose phase declaration order differs.
    correlations: list[CorrelationEntry] = []
    projected_mat = getattr(config, "_projected_correlation_matrix", None)
    metric_order = getattr(config, "_metric_correlation_order", None)
    if projected_mat is not None and metric_order is not None and config.correlations:
        index_by_name = {name: idx for idx, name in enumerate(metric_order)}
        baseline_entries: list[CorrelationEntry] = []
        for pair in config.correlations:
            row_idx = index_by_name.get(pair.metric_a)
            col_idx = index_by_name.get(pair.metric_b)
            if row_idx is None or col_idx is None:
                # Defensive: a config with a correlations entry naming a
                # metric not in the toposort order would have failed
                # cross-reference integrity at load time. Skip rather
                # than crash if it ever surfaces.
                continue
            baseline_entries.append(
                CorrelationEntry(
                    metric_a=pair.metric_a,
                    metric_b=pair.metric_b,
                    requested=float(pair.coefficient),
                    projected=float(projected_mat[row_idx, col_idx]),
                    phase_index=None,
                )
            )
        baseline_entries.sort(key=lambda e: (e.metric_a, e.metric_b))
        correlations.extend(baseline_entries)

        # 0.6-M11: per-phase realized correlations. Same metric order as
        # the baseline (the orchestrator builds every phase against the
        # baseline-toposorted metric list), so ``index_by_name`` is reused.
        phase_projected = getattr(config, "_phase_projected_correlation_matrices", None) or {}
        for phase_idx in sorted(phase_projected.keys()):
            phase = config.correlation_phases[phase_idx]
            phase_mat = phase_projected[phase_idx]
            phase_entries: list[CorrelationEntry] = []
            for pair in phase.correlations:
                row_idx = index_by_name.get(pair.metric_a)
                col_idx = index_by_name.get(pair.metric_b)
                if row_idx is None or col_idx is None:
                    continue
                phase_entries.append(
                    CorrelationEntry(
                        metric_a=pair.metric_a,
                        metric_b=pair.metric_b,
                        requested=float(pair.coefficient),
                        projected=float(phase_mat[row_idx, col_idx]),
                        phase_index=phase_idx,
                    )
                )
            phase_entries.sort(key=lambda e: (e.metric_a, e.metric_b))
            correlations.extend(phase_entries)

    # 0.6-M11: phase window summary. One ``CorrelationPhaseInfo`` per
    # declared phase, in declaration order. Empty list when the config
    # has no phases; populates for any non-empty ``correlation_phases``.
    correlation_phases_info: list[CorrelationPhaseInfo] = [
        CorrelationPhaseInfo(
            phase_index=idx,
            start_period=ph.start_period,
            end_period=ph.end_period,
            n_pairs=len(ph.correlations),
        )
        for idx, ph in enumerate(config.correlation_phases)
    ]

    # 0.6-M13: pull the per-source mapping records off the config's private
    # attr (set by ``dimensions._emit_per_source_dims`` during the dim-build
    # pass). ``None`` for runs without ``multi_source`` → empty manifest
    # list. The records carry entity / source / dim_table / canonical_entity_id
    # / source_entity_id / drifted_fields, all already string-typed by the
    # drift module, so the SourceEntityMapping construction is trivial.
    raw_source_mappings = getattr(config, "_source_entity_mappings", None) or []
    source_entity_mappings: list[SourceEntityMapping] = [
        SourceEntityMapping(**rec) for rec in raw_source_mappings
    ]

    # 0.6-M18: per_parent_row / parent edges. One record per declared
    # (parent, child) pairing; row counts read off the realized tables
    # dict so the manifest carries actual generation output (not just
    # config metadata). Empty list when no per_parent_row table is
    # declared.
    parent_child_relations: list[ParentChildRelation] = []
    for child_tbl in config.tables:
        if child_tbl.grain != "per_parent_row":
            continue
        parent_name = child_tbl.parent_table
        if parent_name is None or child_tbl.children_per_row is None:
            continue
        mn, mx = child_tbl.children_per_row
        parent_df = tables.get(parent_name)
        child_df = tables.get(child_tbl.name)
        parent_child_relations.append(
            ParentChildRelation(
                parent_table=parent_name,
                child_table=child_tbl.name,
                children_per_row_min=int(mn),
                children_per_row_max=int(mx),
                parent_row_count=int(len(parent_df)) if parent_df is not None else 0,
                child_row_count=int(len(child_df)) if child_df is not None else 0,
            )
        )

    # 0.6-M5: outlier injection log. ``detect_outlier_injections`` returns
    # ``None`` for the three skip cases (no outlier_rate configured,
    # vectorized mode, cell budget exceeded) and a list otherwise. The
    # import is local so the manifest builder stays cheap to call when
    # ``noise.outlier_rate == 0`` (the common case) — the heavy module
    # never loads.
    from plotsim.outlier_injections import detect_outlier_injections

    outlier_injections = detect_outlier_injections(config)

    # 0.6-M22 / 0.6-M23: emit the noise-model record when EITHER the
    # heteroscedastic lane is engaged (M22) OR a non-default noise family
    # is configured (M23). Default-family default-amplitude configs leave
    # ``noise_config=None`` so the manifest stays byte-equivalent to
    # pre-M22 modulo the schema version bump.
    noise_config_info: Optional[NoiseConfigInfo] = None
    if config.noise is not None:
        heteroscedastic = getattr(config.noise, "scale_with_trajectory", False)
        family = getattr(config.noise, "noise_family", "gaussian")
        if heteroscedastic or family != "gaussian":
            noise_config_info = NoiseConfigInfo(
                gaussian_sigma=float(config.noise.gaussian_sigma),
                outlier_rate=float(config.noise.outlier_rate),
                mcar_rate=float(config.noise.mcar_rate),
                scale_with_trajectory=bool(heteroscedastic),
                noise_family=str(family),
                degrees_of_freedom=(
                    float(config.noise.degrees_of_freedom)
                    if config.noise.degrees_of_freedom is not None
                    else None
                ),
            )

    # Schema 1.10: seasonal-decomposition snapshot + per-pair OLS
    # summaries. Seasonal always emits (empty sentinel when no effects);
    # regression sections emit only when both declared correlations and
    # realized ``entity_metrics`` are present.
    seasonal_decomposition = _build_seasonal_decomposition(config, n_periods)
    if entity_metrics is not None:
        every_entity = [e.name for e in config.entities]
        regression_pairs_global = _build_regression_pairs(
            config,
            entity_metrics,
            every_entity,
        )
        regression_pairs_by_archetype = _build_regression_pairs_by_archetype(
            config,
            entity_metrics,
        )
    else:
        regression_pairs_global = []
        regression_pairs_by_archetype = {}

    # Mission 026: nested-ANOVA variance partitions (archetype-level and
    # segment-within-archetype) plus per-archetype / per-override-entity
    # GP kernel fits. All three sections piggyback on the same call-site
    # contract: variance partitions need ``entity_metrics``; GP fits
    # piggyback on ``config.metrics`` being non-empty so a no-metric
    # config emits empty sections (the byte-equivalence contract).
    if entity_metrics is not None:
        variance_partitions = _build_variance_partitions_archetype(
            config,
            entity_metrics,
        )
        variance_partitions_by_segment = _build_variance_partitions_by_segment(
            config,
            entity_metrics,
            n_periods,
        )
    else:
        variance_partitions = []
        variance_partitions_by_segment = []
    gp_kernel_fits = _build_gp_kernel_fits(config, n_periods)

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
        bypass_fallback_counts=bypass_fallback_counts,
        vectorized_threshold_used=vectorized_threshold_used,
        causal_graph=causal_graph,
        correlations=correlations,
        outlier_injections=outlier_injections,
        treatment_cohorts=treatment_cohorts,
        correlation_phases=correlation_phases_info,
        source_entity_mappings=source_entity_mappings,
        parent_child_relations=parent_child_relations,
        noise_config=noise_config_info,
        seasonal_decomposition=seasonal_decomposition,
        regression_pairs_global=regression_pairs_global,
        regression_pairs_by_archetype=regression_pairs_by_archetype,
        variance_partitions=variance_partitions,
        variance_partitions_by_segment=variance_partitions_by_segment,
        gp_kernel_fits=gp_kernel_fits,
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
    "ActiveWindow",
    "CausalEdge",
    "CorrelationAdjustment",
    "CorrelationCompensation",
    "CorrelationEntry",
    "EntityArchetypeAssignment",
    "GPKernelFit",
    "TreatmentAssignment",
    "TreatmentCohort",
    "EventFiring",
    "HoldoutInfo",
    "ManifestSchema",
    "NoiseConfigInfo",
    "OutlierInjection",
    "ParentChildRelation",
    "QualityInjection",
    "RegressionPair",
    "SCDEvent",
    "SeasonalDecomposition",
    "SourceEntityMapping",
    "TrajectorySample",
    "VariancePartition",
    "build_manifest",
    "config_sha256",
    "write_manifest",
]
