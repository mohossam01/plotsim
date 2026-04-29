"""plotsim.validation — post-generation integrity and coherence checks.

What it does:
    Runs a generated table set (the dict returned by
    ``plotsim.tables.generate_tables``) through a battery of checks and
    returns a ``ValidationReport``. Also exposes one pre-generation check
    (``validate_correlation_psd``) that fires on the config alone so a bad
    correlation matrix is caught before M004's Cholesky path falls back to
    independent samples.

    Checks:
      * correlation_psd      — configured correlation matrix is PD
      * pk_uniqueness        — single and composite PKs are unique per table
      * fk_integrity         — every FK value resolves to a parent PK
      * date_spine           — dim_date is gap-free and facts' date_keys ⊆ dim_date
      * causal_coherence     — causal_lag alignment + threshold-event coherence
      * null_policy          — metric nulls within mcar_rate's 3σ; non-metric null-free

Input:
    ``PlotsimConfig`` and a ``dict[str, pd.DataFrame]`` of generated tables.

Output:
    ``ValidationReport``: an immutable list of ``ValidationIssue`` with
    ``.ok``, ``.errors``, ``.warnings``, ``.by_check(name)`` accessors.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from plotsim.config import (
    DerivedSource,
    FKSource,
    FakerSource,
    GeneratedSource,
    LagSource,
    MetricSource,
    PKSource,
    PlotsimConfig,
    PoolSource,
    StaticSource,
    Table,
    ThresholdSource,
    parse_source,
)


def _is_scd_dim(tbl: Table) -> bool:
    """True if any column on ``tbl`` carries an SCD Type 2 config."""
    return any(c.scd_type2 is not None for c in tbl.columns)


CHECK_CORRELATION_PSD = "correlation_psd"
CHECK_PK_UNIQUENESS = "pk_uniqueness"
CHECK_FK_INTEGRITY = "fk_integrity"
CHECK_DATE_SPINE = "date_spine"
CHECK_CAUSAL_COHERENCE = "causal_coherence"
CHECK_NULL_POLICY = "null_policy"
CHECK_EMPTY_EVENT_TABLE = "empty_event_table"
CHECK_CROSS_DIM_FK_CARDINALITY = "cross_dim_fk_cardinality"
CHECK_TEMPORAL_COHERENCE = "temporal_coherence"
CHECK_SCD_INTEGRITY = "scd_integrity"
CHECK_BRIDGE_INTEGRITY = "bridge_integrity"

ALL_CHECKS: tuple[str, ...] = (
    CHECK_CORRELATION_PSD,
    CHECK_PK_UNIQUENESS,
    CHECK_FK_INTEGRITY,
    CHECK_DATE_SPINE,
    CHECK_CAUSAL_COHERENCE,
    CHECK_NULL_POLICY,
    CHECK_EMPTY_EVENT_TABLE,
    CHECK_CROSS_DIM_FK_CARDINALITY,
    CHECK_TEMPORAL_COHERENCE,
    CHECK_SCD_INTEGRITY,
    CHECK_BRIDGE_INTEGRITY,
)

# Sample size for "sample of offending values" in issue details. Keeps reports
# readable even when a whole column is broken.
_SAMPLE_LIMIT = 5


@dataclass(frozen=True)
class ValidationIssue:
    """One problem surfaced by a check. Immutable."""
    check: str
    severity: str  # "error" or "warning"
    table: Optional[str]
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    """Immutable bundle of issues with accessors for common slices."""
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    def by_check(self, name: str) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.check == name)


# --- Helpers -----------------------------------------------------------------


def _is_nullish(value: Any) -> bool:
    # F3 (M102): pandas nullable extension dtypes (Int64, BooleanDtype) carry
    # `pd.NA` for missing values, not Python None or float NaN. `pd.isna`
    # uniformly handles None, float NaN, np.datetime64('NaT'), and pd.NA.
    return bool(pd.isna(value))


def _non_null_mask(series: pd.Series) -> pd.Series:
    return series.apply(lambda v: not _is_nullish(v))


def _sample_sorted(values: list[Any], limit: int = _SAMPLE_LIMIT) -> list[Any]:
    try:
        ordered = sorted(values, key=lambda x: str(x))
    except TypeError:
        ordered = values
    return ordered[:limit]


def _per_entity_dim_names(config: PlotsimConfig) -> set[str]:
    return {t.name for t in config.tables if t.type == "dim" and t.grain == "per_entity"}


def _find_fact_for_metric(
    metric: str, config: PlotsimConfig,
) -> Optional[tuple[str, str]]:
    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if isinstance(parsed, MetricSource) and parsed.metric == metric:
                return tbl.name, col.name
    return None


def _find_entity_fk(tbl: Table, per_entity_dims: set[str]) -> Optional[str]:
    for col in tbl.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource) and parsed.table in per_entity_dims:
            return col.name
    return None


def _find_date_fk(tbl: Table) -> Optional[str]:
    for col in tbl.columns:
        parsed = parse_source(col.source)
        if isinstance(parsed, FKSource) and parsed.table == "dim_date":
            return col.name
    return None


def _numeric_series(series) -> np.ndarray:
    return np.array(
        [np.nan if _is_nullish(v) else float(v) for v in series],
        dtype=float,
    )


# --- Pre-generation: entity features config gates (M108) --------------------


def validate_entity_features_config(config: PlotsimConfig) -> list[str]:
    """Return load-time error messages for the M108 entity-features feature.

    Mirrors the pattern established by ``validate_correlation_psd`` —
    pure config check, no DataFrame inputs. Called from a Pydantic
    ``model_validator`` on ``PlotsimConfig`` so a misconfigured YAML
    fails at load instead of mid-generation.

    Returns an empty list when ``entity_features.enabled == False`` (no
    constraints apply) or when the config satisfies every gate. The
    first message in a non-empty list is the one
    ``PlotsimConfig._entity_features_gates`` raises; collecting all of
    them keeps the function reusable from contexts that want a full
    diagnostic instead of fail-fast.

    Gates (in order):
      1. ``manifest.include`` must be True. Labels read from the
         manifest payload; turning off manifest emission would leave
         ``final_trajectory_position`` permanently NaN with no signal
         to the user.
      2. ``quality.quality_issues`` must be empty. Entity features
         aggregate the pre-corruption fact tables; combining the two
         would silently mix clean and corrupted aggregates without an
         operator-visible split. Deferred to a future mission.
      3. Each name in ``entity_features.metrics`` must reference a
         metric that has at least one ``int``/``float``-typed column
         on a fact table — i.e., a numeric aggregable signal. Names
         not in ``config.metrics`` at all, or in ``config.metrics``
         but never landed on a fact column, both raise.
    """
    errors: list[str] = []
    cfg = config.entity_features
    if not cfg.enabled:
        return errors

    if not config.manifest.include:
        errors.append(
            "entity_features.enabled=true requires manifest.include=true; "
            "labels (archetype, final_trajectory_position) read from the "
            "manifest payload"
        )

    if config.quality.quality_issues:
        errors.append(
            "entity_features cannot be combined with quality_issues in this "
            "version"
        )

    if cfg.metrics:
        metric_names = {m.name for m in config.metrics}
        numeric_fact_metrics: set[str] = set()
        for tbl in config.tables:
            if tbl.type != "fact":
                continue
            for col in tbl.columns:
                parsed = parse_source(col.source)
                if (
                    isinstance(parsed, MetricSource)
                    and col.dtype in ("int", "float")
                ):
                    numeric_fact_metrics.add(parsed.metric)
        for name in cfg.metrics:
            if name not in metric_names:
                errors.append(
                    f"entity_features.metrics references unknown metric "
                    f"{name!r}; known metrics: {sorted(metric_names)}"
                )
                continue
            if name not in numeric_fact_metrics:
                errors.append(
                    f"entity_features.metrics references metric {name!r} "
                    f"which has no int/float column on any fact table; "
                    f"only numeric fact metrics can be aggregated"
                )
    return errors


# --- Pre-generation: holdout-split config gates (M109) ----------------------


def validate_holdout_config(config: PlotsimConfig) -> list[str]:
    """Return load-time error messages for the M109 holdout-split feature.

    Mirrors ``validate_entity_features_config``. Pure config check, no
    DataFrame inputs. Called from a Pydantic ``model_validator`` on
    ``PlotsimConfig`` so a misconfigured YAML fails at load instead of
    mid-generation.

    Returns an empty list when ``holdout.enabled == False`` (no
    constraints apply) or when every gate is satisfied. The first
    message in a non-empty list is the one
    ``PlotsimConfig._holdout_gates`` raises.

    Gates (in order):
      1. ``target_metric`` must be set. ``enabled=true`` without a
         declared target is meaningless — the manifest payload would
         carry no label name and downstream entity-features wouldn't
         know which columns to drop.
      2. ``holdout_periods`` must be >= 1. A zero-period holdout is a
         no-op.
      3. ``n_periods - holdout_periods >= min_training_periods``. Splits
         that leave too few training periods produce slope/std
         aggregates with pathological values; reject early.
      4. ``target_metric`` must reference an existing metric on
         ``config.metrics`` AND that metric must land on a numeric
         (int/float) column on a fact table. Threshold-only metrics
         (e.g. churn flags emitted as boolean event columns) cannot be
         training targets in this version.
      5. ``quality.quality_issues`` must be empty. Holdout slices
         operate on the clean fact tables; combining them with quality
         injection would leave the train/holdout split's semantics
         silently dependent on whether corruption was applied before
         or after the slice. Deferred to a future mission.
    """
    errors: list[str] = []
    cfg = config.holdout
    if not cfg.enabled:
        return errors

    if cfg.target_metric is None:
        errors.append(
            "holdout.enabled=true requires holdout.target_metric to be set; "
            "the metric naming the prediction target is recorded on the "
            "manifest and excluded from entity features"
        )

    if cfg.holdout_periods < 1:
        errors.append(
            f"holdout.holdout_periods must be >= 1 when holdout.enabled=true "
            f"(got {cfg.holdout_periods}); a zero-period holdout is a no-op"
        )

    n_periods = config.time_window.period_count()
    if cfg.holdout_periods >= 1:
        train_periods = n_periods - cfg.holdout_periods
        if train_periods < cfg.min_training_periods:
            errors.append(
                f"holdout split leaves {train_periods} training period(s) "
                f"(n_periods={n_periods} - holdout_periods="
                f"{cfg.holdout_periods}); minimum required by "
                f"holdout.min_training_periods is {cfg.min_training_periods}"
            )

    if cfg.target_metric is not None:
        metric_names = {m.name for m in config.metrics}
        numeric_fact_metrics: set[str] = set()
        for tbl in config.tables:
            if tbl.type != "fact":
                continue
            for col in tbl.columns:
                parsed = parse_source(col.source)
                if (
                    isinstance(parsed, MetricSource)
                    and col.dtype in ("int", "float")
                ):
                    numeric_fact_metrics.add(parsed.metric)
        if cfg.target_metric not in metric_names:
            errors.append(
                f"holdout.target_metric references unknown metric "
                f"{cfg.target_metric!r}; known metrics: "
                f"{sorted(metric_names)}"
            )
        elif cfg.target_metric not in numeric_fact_metrics:
            errors.append(
                f"holdout.target_metric references metric "
                f"{cfg.target_metric!r} which has no int/float column on "
                f"any fact table; only numeric fact metrics can serve as "
                f"prediction targets"
            )

    if config.quality.quality_issues:
        errors.append(
            "holdout cannot be combined with quality_issues in this version"
        )

    return errors


# --- Pre-generation: PoolSource entity-coverage gates (M114) ----------------


def validate_value_pool_coverage(config: PlotsimConfig) -> list[str]:
    """Return load-time error messages for M114 ``PoolSource`` columns.

    Mirrors ``validate_entity_features_config`` and
    ``validate_holdout_config``: pure config check, no DataFrame inputs.
    Called from a Pydantic ``model_validator`` on ``PlotsimConfig`` so a
    misconfigured YAML fails at load instead of mid-generation.

    Returns an empty list when no column declares a ``pool:`` source or
    when every gate is satisfied. The first message in a non-empty list
    is the one ``PlotsimConfig._value_pool_gates`` raises.

    Gates (in order):
      1. ``PoolSource`` columns are only meaningful on ``per_entity``
         dim tables in M114. Sub-entity (variable-grain) and reference
         dims have no per-entity 1:1 binding to look up against.
      2. The ``value_pool`` dict's keys must cover every ``Entity.name``
         that produces rows in this dim table. Per-entity dims emit
         exactly one row per entity, so the key set must equal the
         entity set. Missing keys → error naming each missing entity.
      3. Extra keys (entities present in ``value_pool`` but not in
         ``config.entities``) are flagged as a separate error so the
         author notices stale-after-edit pool entries.
    """
    errors: list[str] = []
    entity_names = {e.name for e in config.entities}
    per_entity_dim_names = {
        t.name for t in config.tables
        if t.type == "dim" and t.grain == "per_entity"
    }

    for tbl in config.tables:
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if not isinstance(parsed, PoolSource):
                continue
            if tbl.name not in per_entity_dim_names:
                errors.append(
                    f"table {tbl.name!r} column {col.name!r} declares a "
                    f"'pool:' source but the table is not a per_entity dim "
                    f"(type={tbl.type!r}, grain={tbl.grain!r}); pool sources "
                    f"are only supported on per_entity dim tables in this "
                    f"version"
                )
                continue
            if col.value_pool is None:
                # _pool_pairing on Column already rejects this; the guard
                # here keeps the cross-ref pass independent of column
                # validator ordering.
                continue
            pool_keys = set(col.value_pool.keys())
            missing = sorted(entity_names - pool_keys)
            extra = sorted(pool_keys - entity_names)
            if missing:
                errors.append(
                    f"table {tbl.name!r} column {col.name!r} value_pool "
                    f"is missing entries for entities {missing}; per_entity "
                    f"dim {tbl.name!r} emits one row per entity, so every "
                    f"entity must appear in value_pool"
                )
            if extra:
                errors.append(
                    f"table {tbl.name!r} column {col.name!r} value_pool "
                    f"has entries for unknown entities {extra}; remove them "
                    f"or correct the entity names in config.entities"
                )

    return errors


# --- Check 1: correlation matrix PSD ----------------------------------------


def project_correlation_or_issue(
    config: PlotsimConfig,
) -> tuple[list[ValidationIssue], Optional[list[dict]], Optional[np.ndarray]]:
    """M111 internal: PD-check + Higham projection + adjustment records.

    Single source of truth for the project-and-warn flow. Returns
    ``(issues, adjustments, projected_matrix)``:

      * ``config.correlations`` empty → ``([], None, None)``.
      * Matrix already PD → ``([], None, None)``.
      * Higham (or eigenvalue-clipping fallback) succeeds → ``([],
        records, projected)``. The projected matrix is in declaration
        order (``config.metrics`` order); callers that build in
        toposort order should re-project rather than reuse.
      * Both projection paths fail → ``([issue], None, None)``.

    Does NOT emit warnings — leaves that to the caller so the load-time
    pydantic validator fires the user-facing warning at config init
    (right stack level, fires once per config) without duplicate emits
    from later post-generation re-checks.
    """
    if not config.correlations:
        return [], None, None

    # Local imports avoid the plotsim.metrics ↔ plotsim.config ↔
    # plotsim.validation cycle that already constrains this module.
    from plotsim.metrics import (
        _build_correlation_matrix,
        _correlation_adjustment_records,
        project_correlation_matrix,
    )

    metrics = list(config.metrics)
    pairs = list(config.correlations)
    mat = _build_correlation_matrix(metrics, pairs)
    try:
        projected, projection_used, _used_fallback = project_correlation_matrix(mat)
    except RuntimeError as exc:
        eigvals = np.linalg.eigvalsh((mat + mat.T) / 2.0).tolist()
        return [ValidationIssue(
            check=CHECK_CORRELATION_PSD,
            severity="error",
            table=None,
            message=(
                "correlation matrix could not be projected to "
                f"positive-definite: {exc}"
            ),
            details={
                "metrics": [m.name for m in metrics],
                "min_eigenvalue": min(eigvals),
                "eigenvalues": eigvals,
            },
        )], None, None

    if not projection_used:
        return [], None, None

    records = _correlation_adjustment_records(mat, projected, metrics, pairs)
    return [], records, projected


def validate_correlation_psd(config: PlotsimConfig) -> list[ValidationIssue]:
    """M111: project-and-warn check.

    The pre-M111 contract was "non-PD correlation matrix → error issue
    so the engine can refuse to run" (FIX-F04 promoted that to a hard
    raise at load). Under M111, non-PD matrices are auto-corrected via
    Higham nearest-PD projection at load time; this post-generation
    check only flags the genuinely-impossible "projection itself
    failed" case.

    Returned issue list:
      * Successful projection (or matrix already PD, or no correlations
        configured) → ``[]``.
      * Both Higham and eigenvalue-clipping fallback failed → one
        ``error`` issue. Should never happen for symmetric input.

    Warning emit is owned by ``PlotsimConfig._correlation_matrix_is_psd``
    so the user sees the per-pair adjustment text at config load (once,
    at the right stack level). This wrapper does NOT re-emit; otherwise
    every post-generation validation pass would duplicate the warning.
    """
    issues, _adjustments, _projected = project_correlation_or_issue(config)
    return issues


# --- Check 2: PK uniqueness --------------------------------------------------


def validate_pk_uniqueness(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> list[ValidationIssue]:
    """Flag duplicate PK values (single-column and composite).

    M106: SCD Type 2-enabled dim tables hold multiple versioned rows per
    entity, so the declared natural PK (e.g. ``company_id``) repeats by
    design. The effective uniqueness key on those tables shifts to the
    surrogate ``dim_row_id`` the SCD expansion injects. The validator
    detects SCD dims and pivots the uniqueness check accordingly —
    natural-PK duplicates on SCD dims are expected, ``dim_row_id`` must
    be unique. Non-SCD tables go through the original path unchanged.
    """
    issues: list[ValidationIssue] = []
    for tbl in config.tables:
        df = tables.get(tbl.name)
        if df is None or df.empty:
            continue
        pk_cols = tbl.primary_key_cols
        if _is_scd_dim(tbl):
            if "dim_row_id" not in df.columns:
                issues.append(ValidationIssue(
                    check=CHECK_PK_UNIQUENESS,
                    severity="error",
                    table=tbl.name,
                    message=(
                        f"SCD dim {tbl.name!r} is missing the surrogate "
                        f"'dim_row_id' column expected after SCD expansion"
                    ),
                    details={"pk_columns": pk_cols, "actual_columns": list(df.columns)},
                ))
                continue
            dup_mask = df["dim_row_id"].duplicated(keep=False)
            if dup_mask.any():
                dup_values = df.loc[dup_mask, "dim_row_id"].unique().tolist()
                issues.append(ValidationIssue(
                    check=CHECK_PK_UNIQUENESS,
                    severity="error",
                    table=tbl.name,
                    message=(
                        f"SCD surrogate 'dim_row_id' has {len(dup_values)} "
                        f"duplicate value(s) on SCD dim {tbl.name!r}"
                    ),
                    details={
                        "pk_columns": ["dim_row_id"],
                        "duplicates_sample": _sample_sorted(dup_values),
                        "duplicate_count": int(dup_mask.sum()),
                    },
                ))
            continue
        missing = [c for c in pk_cols if c not in df.columns]
        if missing:
            issues.append(ValidationIssue(
                check=CHECK_PK_UNIQUENESS,
                severity="error",
                table=tbl.name,
                message=f"PK columns {missing} not in generated DataFrame",
                details={"pk_columns": pk_cols, "actual_columns": list(df.columns)},
            ))
            continue
        if len(pk_cols) == 1:
            col = pk_cols[0]
            dup_mask = df[col].duplicated(keep=False)
            if dup_mask.any():
                dup_values = df.loc[dup_mask, col].unique().tolist()
                issues.append(ValidationIssue(
                    check=CHECK_PK_UNIQUENESS,
                    severity="error",
                    table=tbl.name,
                    message=f"PK column {col!r} has {len(dup_values)} duplicate value(s)",
                    details={
                        "pk_columns": pk_cols,
                        "duplicates_sample": _sample_sorted(dup_values),
                        "duplicate_count": int(dup_mask.sum()),
                    },
                ))
        else:
            dup_mask = df.duplicated(subset=pk_cols, keep=False)
            if dup_mask.any():
                dup_tuples = [
                    tuple(row) for row in
                    df.loc[dup_mask, pk_cols].drop_duplicates().values.tolist()
                ]
                issues.append(ValidationIssue(
                    check=CHECK_PK_UNIQUENESS,
                    severity="error",
                    table=tbl.name,
                    message=(
                        f"composite PK {pk_cols} has {len(dup_tuples)} "
                        f"duplicate tuple(s)"
                    ),
                    details={
                        "pk_columns": pk_cols,
                        "duplicates_sample": _sample_sorted(dup_tuples),
                        "duplicate_count": int(dup_mask.sum()),
                    },
                ))
    return issues


# --- Check 3: FK integrity --------------------------------------------------


def validate_fk_integrity(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> list[ValidationIssue]:
    """Every non-null FK value must resolve to a PK in its parent table.

    Null FK values are reported as a separate warning — they indicate a
    placeholder leak from a builder, not a referential violation.
    """
    issues: list[ValidationIssue] = []
    for tbl in config.tables:
        df = tables.get(tbl.name)
        if df is None or df.empty:
            continue
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if not isinstance(parsed, FKSource):
                continue
            if col.name not in df.columns:
                issues.append(ValidationIssue(
                    check=CHECK_FK_INTEGRITY,
                    severity="error",
                    table=tbl.name,
                    message=f"FK column {col.name!r} missing from generated DataFrame",
                    details={"column": col.name, "parent": f"{parsed.table}.{parsed.column}"},
                ))
                continue
            parent_df = tables.get(parsed.table)
            if parent_df is None:
                issues.append(ValidationIssue(
                    check=CHECK_FK_INTEGRITY,
                    severity="error",
                    table=tbl.name,
                    message=(
                        f"FK column {col.name!r} references parent table "
                        f"{parsed.table!r} which was not generated"
                    ),
                    details={"column": col.name, "parent": f"{parsed.table}.{parsed.column}"},
                ))
                continue
            if parsed.column not in parent_df.columns:
                issues.append(ValidationIssue(
                    check=CHECK_FK_INTEGRITY,
                    severity="error",
                    table=tbl.name,
                    message=(
                        f"FK column {col.name!r} references column "
                        f"{parsed.column!r} missing from parent {parsed.table!r}"
                    ),
                    details={"column": col.name, "parent": f"{parsed.table}.{parsed.column}"},
                ))
                continue
            series = df[col.name]
            null_mask = ~_non_null_mask(series)
            if null_mask.any():
                issues.append(ValidationIssue(
                    check=CHECK_FK_INTEGRITY,
                    severity="warning",
                    table=tbl.name,
                    message=(
                        f"FK column {col.name!r} has {int(null_mask.sum())} "
                        f"null value(s); builders should not emit null FKs"
                    ),
                    details={
                        "column": col.name,
                        "parent": f"{parsed.table}.{parsed.column}",
                        "null_count": int(null_mask.sum()),
                    },
                ))
            parent_keys = set(parent_df[parsed.column].tolist())
            child_values = series[~null_mask].tolist()
            orphans = sorted(
                {v for v in child_values if v not in parent_keys},
                key=lambda x: str(x),
            )
            if orphans:
                issues.append(ValidationIssue(
                    check=CHECK_FK_INTEGRITY,
                    severity="error",
                    table=tbl.name,
                    message=(
                        f"FK column {col.name!r} has {len(orphans)} orphan "
                        f"value(s) not present in {parsed.table}.{parsed.column}"
                    ),
                    details={
                        "column": col.name,
                        "parent": f"{parsed.table}.{parsed.column}",
                        "orphans_sample": orphans[:_SAMPLE_LIMIT],
                        "orphan_count": len(orphans),
                    },
                ))
    return issues


# --- Check 4: date spine -----------------------------------------------------


def _expected_cadence_delta(granularity: str) -> Optional[_dt.timedelta]:
    if granularity == "daily":
        return _dt.timedelta(days=1)
    if granularity == "weekly":
        return _dt.timedelta(days=7)
    return None  # monthly handled specially — variable length


def _months_between(a: _dt.date, b: _dt.date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def validate_date_spine(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> list[ValidationIssue]:
    """dim_date is strictly monotonic, gap-free, and covers every fact's date_key."""
    issues: list[ValidationIssue] = []
    dim_date = tables.get("dim_date")
    if dim_date is None or dim_date.empty:
        issues.append(ValidationIssue(
            check=CHECK_DATE_SPINE,
            severity="error",
            table="dim_date",
            message="dim_date is missing or empty",
        ))
        return issues

    for col_name in ("date_key", "date"):
        if col_name not in dim_date.columns:
            issues.append(ValidationIssue(
                check=CHECK_DATE_SPINE,
                severity="error",
                table="dim_date",
                message=f"dim_date missing required column {col_name!r}",
            ))
            return issues

    date_keys = dim_date["date_key"].tolist()
    if len(set(date_keys)) != len(date_keys):
        issues.append(ValidationIssue(
            check=CHECK_DATE_SPINE,
            severity="error",
            table="dim_date",
            message="dim_date has duplicate date_key values",
            details={"row_count": len(date_keys), "unique": len(set(date_keys))},
        ))

    for prev, curr in zip(date_keys, date_keys[1:]):
        if curr <= prev:
            issues.append(ValidationIssue(
                check=CHECK_DATE_SPINE,
                severity="error",
                table="dim_date",
                message="dim_date.date_key is not strictly increasing",
                details={"prev": prev, "curr": curr},
            ))
            break

    dates = list(dim_date["date"])
    granularity = config.time_window.granularity
    delta = _expected_cadence_delta(granularity)
    for prev, curr in zip(dates, dates[1:]):
        if not (isinstance(prev, _dt.date) and isinstance(curr, _dt.date)):
            continue
        if granularity == "monthly":
            gap = _months_between(prev, curr)
            if gap != 1:
                issues.append(ValidationIssue(
                    check=CHECK_DATE_SPINE,
                    severity="error",
                    table="dim_date",
                    message=f"dim_date has a {gap}-month gap at {prev} → {curr}",
                    details={"prev": str(prev), "curr": str(curr), "expected_months": 1},
                ))
                break
        elif delta is not None:
            actual = curr - prev
            if actual != delta:
                issues.append(ValidationIssue(
                    check=CHECK_DATE_SPINE,
                    severity="error",
                    table="dim_date",
                    message=(
                        f"dim_date has an unexpected gap at {prev} → {curr} "
                        f"(got {actual}, expected {delta})"
                    ),
                    details={"prev": str(prev), "curr": str(curr)},
                ))
                break

    dim_date_set = set(date_keys)
    for tbl in config.tables:
        if tbl.type != "fact":
            continue
        df = tables.get(tbl.name)
        if df is None or df.empty:
            continue
        date_col = _find_date_fk(tbl)
        if date_col is None or date_col not in df.columns:
            continue
        missing = sorted(
            {v for v in df[date_col].tolist() if v not in dim_date_set and not _is_nullish(v)},
            key=lambda x: str(x),
        )
        if missing:
            issues.append(ValidationIssue(
                check=CHECK_DATE_SPINE,
                severity="error",
                table=tbl.name,
                message=(
                    f"fact table {tbl.name!r} has {len(missing)} date_key "
                    f"value(s) not present in dim_date"
                ),
                details={
                    "column": date_col,
                    "missing_sample": missing[:_SAMPLE_LIMIT],
                    "missing_count": len(missing),
                },
            ))

    return issues


# --- Check 5: causal coherence ----------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(a) | np.isnan(b))
    if int(mask.sum()) < 3:
        return None
    aa, bb = a[mask], b[mask]
    if aa.std() == 0 or bb.std() == 0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _lag_alignment_better_for_entity(
    metric_series: np.ndarray, driver_series: np.ndarray, lag: int,
) -> Optional[bool]:
    """True if the lag-shifted correlation is strong relative to the
    unshifted one.

    Historically the check was a strict ``|lagged| > |unlagged|``. That
    held under the pre-0.4.0 blend weight of 0.6 because the lagged
    metric carried a 40% same-period component that partially matched
    the driver's own same-period value, and the 60% lagged component
    reliably tipped the magnitude comparison in favour of the shifted
    series.

    Under 0.4.0's default ``blend_weight=1.0`` the lag is a pure period
    shift, which IS the cleaner semantics but also interacts poorly with
    two incidental effects on slow-varying trajectories:

      1. Iman-Conover correlation pairs induce same-period rank
         alignment between metrics. For a lagged metric that shares a
         correlation pair with a third metric that's also correlated
         with the driver (HR's absence_rate ↔ attrition_risk ↔
         engagement_index triangle), the induced same-period
         correlation can inflate ``|unlagged|``.
      2. Smooth archetypes (compound growth, exp decay, plateau) have
         ``traj[t] ≈ traj[t-1]``, so the unlagged correlation captures
         nearly the same signal as the lag-1 correlation.

    The ratio ``|lagged| / |unlagged| >= 0.5`` still catches flagrantly
    broken lags (where the shift drops the correlation magnitude toward
    zero) without flagging the IC/smooth-trajectory interaction above.

    Returns ``None`` if either correlation is undefined.
    """
    if len(metric_series) <= 2 * lag + 2:
        return None
    unlagged = _pearson(metric_series, driver_series)
    lagged = _pearson(metric_series[lag:], driver_series[:-lag])
    if unlagged is None or lagged is None:
        return None
    if abs(unlagged) < 1e-9:
        # No measurable same-period signal — any lagged correlation
        # indicates the lag is implemented. Flipping the test to
        # "abs(lagged) > 0" would false-positive on pure noise, so fall
        # back to the strict original inequality here.
        return abs(lagged) > abs(unlagged)
    return abs(lagged) >= abs(unlagged) * 0.5


def validate_causal_coherence(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> list[ValidationIssue]:
    """Two coherence checks:

    1. Every causal_lag metric's series aligns with the driver's shifted
       series better than with the unshifted driver, for a majority of
       entities.
    2. Every threshold-typed event row was emitted at a (entity, period)
       where the fact-table metric satisfied the threshold for
       ``consecutive`` consecutive periods ending at that period.
    """
    issues: list[ValidationIssue] = []
    per_entity_dims = _per_entity_dim_names(config)

    # --- 1. causal_lag alignment ---
    for m in config.metrics:
        if m.causal_lag is None:
            continue
        lag = m.causal_lag.lag_periods
        metric_fact = _find_fact_for_metric(m.name, config)
        driver_fact = _find_fact_for_metric(m.causal_lag.driver, config)
        if metric_fact is None or driver_fact is None:
            continue
        mt_name, mt_col = metric_fact
        dr_name, dr_col = driver_fact
        mt_df = tables.get(mt_name)
        dr_df = tables.get(dr_name)
        if mt_df is None or dr_df is None:
            continue
        mt_tbl = next(t for t in config.tables if t.name == mt_name)
        dr_tbl = next(t for t in config.tables if t.name == dr_name)
        mt_entity_col = _find_entity_fk(mt_tbl, per_entity_dims)
        dr_entity_col = _find_entity_fk(dr_tbl, per_entity_dims)
        mt_date_col = _find_date_fk(mt_tbl)
        dr_date_col = _find_date_fk(dr_tbl)
        if not all([mt_entity_col, dr_entity_col, mt_date_col, dr_date_col]):
            continue

        better = 0
        total = 0
        entity_ids = list(dict.fromkeys(mt_df[mt_entity_col].tolist()))
        for eid in entity_ids:
            mt_slice = mt_df[mt_df[mt_entity_col] == eid].sort_values(mt_date_col)
            dr_slice = dr_df[dr_df[dr_entity_col] == eid].sort_values(dr_date_col)
            if len(mt_slice) == 0 or len(dr_slice) == 0:
                continue
            metric_arr = _numeric_series(mt_slice[mt_col])
            driver_arr = _numeric_series(dr_slice[dr_col])
            n = min(len(metric_arr), len(driver_arr))
            result = _lag_alignment_better_for_entity(
                metric_arr[:n], driver_arr[:n], lag,
            )
            if result is None:
                continue
            total += 1
            if result:
                better += 1
        if total == 0:
            continue
        ratio = better / total
        if ratio < 0.5:
            issues.append(ValidationIssue(
                check=CHECK_CAUSAL_COHERENCE,
                severity="error",
                table=mt_name,
                message=(
                    f"metric {m.name!r} (causal_lag driver={m.causal_lag.driver!r}, "
                    f"lag={lag}) aligns with its lagged driver in {better}/{total} "
                    f"entities (ratio={ratio:.2f}); expected a majority"
                ),
                details={
                    "metric": m.name,
                    "driver": m.causal_lag.driver,
                    "lag_periods": lag,
                    "entities_total": total,
                    "entities_better": better,
                    "ratio": ratio,
                },
            ))

    # --- 2. threshold-event coherence ---
    for tbl in config.tables:
        if tbl.type != "event":
            continue
        df = tables.get(tbl.name)
        if df is None or df.empty:
            continue
        threshold_col_cfg: Optional[ThresholdSource] = None
        threshold_col_name: Optional[str] = None
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if isinstance(parsed, ThresholdSource):
                threshold_col_cfg = parsed
                threshold_col_name = col.name
                break
        if threshold_col_cfg is None or threshold_col_name is None:
            continue

        metric_fact = _find_fact_for_metric(threshold_col_cfg.metric, config)
        if metric_fact is None:
            continue
        mt_name, mt_col = metric_fact
        mt_df = tables.get(mt_name)
        mt_tbl = next(t for t in config.tables if t.name == mt_name)
        if mt_df is None:
            continue
        mt_entity_col = _find_entity_fk(mt_tbl, per_entity_dims)
        mt_date_col = _find_date_fk(mt_tbl)
        evt_entity_col = _find_entity_fk(tbl, per_entity_dims)
        evt_date_col = _find_date_fk(tbl)
        if not all([mt_entity_col, mt_date_col, evt_entity_col, evt_date_col]):
            continue

        bad_events: list[tuple[Any, Any]] = []
        for _, evt_row in df.iterrows():
            eid = evt_row[evt_entity_col]
            dkey = evt_row[evt_date_col]
            mt_slice = mt_df[mt_df[mt_entity_col] == eid].sort_values(mt_date_col).reset_index(drop=True)
            match_idx = mt_slice.index[mt_slice[mt_date_col] == dkey]
            if len(match_idx) == 0:
                bad_events.append((eid, dkey))
                continue
            end = int(match_idx[0])
            start = end - threshold_col_cfg.consecutive + 1
            if start < 0:
                bad_events.append((eid, dkey))
                continue
            window = _numeric_series(mt_slice.loc[start:end, mt_col])
            if np.isnan(window).any():
                bad_events.append((eid, dkey))
                continue
            if threshold_col_cfg.direction == "above":
                satisfied = bool(np.all(window > threshold_col_cfg.value))
            else:
                satisfied = bool(np.all(window < threshold_col_cfg.value))
            if not satisfied:
                bad_events.append((eid, dkey))

        if bad_events:
            issues.append(ValidationIssue(
                check=CHECK_CAUSAL_COHERENCE,
                severity="error",
                table=tbl.name,
                message=(
                    f"threshold event column {threshold_col_name!r} fired for "
                    f"{len(bad_events)} (entity, period) pair(s) where the "
                    f"fact-table window did not satisfy "
                    f"{threshold_col_cfg.direction} {threshold_col_cfg.value} "
                    f"for {threshold_col_cfg.consecutive} consecutive periods"
                ),
                details={
                    "column": threshold_col_name,
                    "metric": threshold_col_cfg.metric,
                    "direction": threshold_col_cfg.direction,
                    "value": threshold_col_cfg.value,
                    "consecutive": threshold_col_cfg.consecutive,
                    "bad_events_sample": _sample_sorted(bad_events),
                    "bad_event_count": len(bad_events),
                },
            ))

    return issues


# --- Check 6: null policy ----------------------------------------------------


def _null_upper_bound(n: int, p: float) -> int:
    """Binomial 3σ upper bound + 1 slack cell for small-sample jitter."""
    if p <= 0.0:
        return 0
    mean = n * p
    std = (n * p * (1.0 - p)) ** 0.5
    return int(mean + 3.0 * std) + 1


def validate_null_policy(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> list[ValidationIssue]:
    """Metric columns in fact tables: within mcar_rate's 3σ upper bound.
    Non-metric columns (FK/PK/generated/static/derived/lag) on dim and fact
    tables: zero nulls. Event tables are excluded — row counts there are
    mechanism-dependent, so the null-rate model doesn't apply.
    """
    issues: list[ValidationIssue] = []
    mcar = config.noise.mcar_rate
    for tbl in config.tables:
        if tbl.type == "event":
            continue
        df = tables.get(tbl.name)
        if df is None or df.empty:
            continue
        for col in tbl.columns:
            if col.name not in df.columns:
                continue
            parsed = parse_source(col.source)
            series = df[col.name]
            null_count = int((~_non_null_mask(series)).sum())
            if isinstance(parsed, MetricSource) and tbl.type == "fact":
                upper = _null_upper_bound(len(series), mcar)
                if null_count > upper:
                    issues.append(ValidationIssue(
                        check=CHECK_NULL_POLICY,
                        severity="error",
                        table=tbl.name,
                        message=(
                            f"metric column {col.name!r} has {null_count} null(s); "
                            f"upper bound at mcar_rate={mcar} is {upper}"
                        ),
                        details={
                            "column": col.name,
                            "null_count": null_count,
                            "mcar_rate": mcar,
                            "upper_bound": upper,
                            "row_count": len(series),
                        },
                    ))
                continue
            if isinstance(
                parsed,
                (FKSource, PKSource, GeneratedSource, FakerSource,
                 StaticSource, DerivedSource, LagSource),
            ):
                if null_count > 0:
                    issues.append(ValidationIssue(
                        check=CHECK_NULL_POLICY,
                        severity="error",
                        table=tbl.name,
                        message=(
                            f"non-metric column {col.name!r} (source "
                            f"{col.source!r}) has {null_count} null(s); "
                            f"expected zero"
                        ),
                        details={
                            "column": col.name,
                            "source": col.source,
                            "null_count": null_count,
                        },
                    ))
    return issues


# --- Check 7: empty event tables (FIX-03 / SF-9) -----------------------------


def validate_empty_event_tables(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> list[ValidationIssue]:
    """Warn when an event table emits zero rows because no driver is configured.

    ``tables.build_event_tables`` emits an empty DataFrame for any event table
    that declares neither a ``row_count_source`` (proportional driver) nor a
    column with a ``threshold:`` source. This preserves the contract that
    every configured table appears in the output dict, but it's a silent
    "you got nothing" for the user. This check surfaces it as a warning so
    the validation report and the CLI summary can flag it.

    Event tables with a configured driver that legitimately produce zero rows
    (e.g. a threshold no entity ever crossed) are NOT flagged — that's a
    valid generative outcome, not a config defect.
    """
    issues: list[ValidationIssue] = []
    for tbl in config.tables:
        if tbl.type != "event":
            continue
        df = tables.get(tbl.name)
        if df is None or len(df) > 0:
            continue
        has_row_count_source = tbl.row_count_source is not None
        has_threshold_col = any(
            isinstance(parse_source(c.source), ThresholdSource)
            for c in tbl.columns
        )
        if has_row_count_source or has_threshold_col:
            # Driver configured; zero rows is a legitimate generative outcome.
            continue
        issues.append(ValidationIssue(
            check=CHECK_EMPTY_EVENT_TABLE,
            severity="warning",
            table=tbl.name,
            message=(
                f"event table {tbl.name!r} produced 0 rows because no driver "
                f"is configured. Add a 'row_count_source: proportional:<metric>:"
                f"scale:<x>' on the table or a 'threshold:<metric>:...' column "
                f"to drive emission."
            ),
            details={"table": tbl.name, "row_count": 0},
        ))
    return issues


# --- Check 9: temporal coherence (FIX-05 / MF-2) -----------------------------


def _time_window_bounds(config: PlotsimConfig) -> tuple[_dt.date, _dt.date]:
    """Return the inclusive date bounds spanned by ``config.time_window``.

    The YAML schema stores ``start`` / ``end`` as ``YYYY-MM`` strings.
    Lower bound is the first day of the start month; upper bound is the
    last day of the end month — so a column with ``dtype: date`` that
    targets any day within the window is considered in-range at
    granularity-independent resolution.
    """
    import calendar
    start_year, start_month = (int(p) for p in config.time_window.start.split("-"))
    end_year, end_month = (int(p) for p in config.time_window.end.split("-"))
    lower = _dt.date(start_year, start_month, 1)
    last_day = calendar.monthrange(end_year, end_month)[1]
    upper = _dt.date(end_year, end_month, last_day)
    return lower, upper


def _as_date_value(value: Any) -> Optional[_dt.date]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        try:
            return _dt.date.fromisoformat(value)
        except ValueError:
            return None
    return None


def validate_temporal_coherence(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> list[ValidationIssue]:
    """Warn when a dim column with ``dtype: date`` holds values outside ``time_window``.

    FIX-05 / MF-2 regression guard: catches configs that emit dates
    from an unparameterized Faker provider (e.g. ``faker.date``) whose
    uniform 1970–2030 range silently drifts outside the configured
    window. Columns that legitimately reach outside (birth dates,
    hire dates, trial timestamps) can set ``allow_outside_window:
    true`` on the column to suppress the warning.

    Only dimension tables are checked — fact-table date-FK columns are
    already covered by ``validate_date_spine``, and event tables store
    ``date_key`` integers rather than raw dates.
    """
    issues: list[ValidationIssue] = []
    lower, upper = _time_window_bounds(config)
    for tbl in config.tables:
        if tbl.type != "dim":
            continue
        df = tables.get(tbl.name)
        if df is None or df.empty:
            continue
        for col in tbl.columns:
            if col.dtype != "date":
                continue
            if col.allow_outside_window:
                continue
            if col.name not in df.columns:
                continue
            out_of_range: list[_dt.date] = []
            for raw in df[col.name].tolist():
                d = _as_date_value(raw)
                if d is None:
                    continue
                if d < lower or d > upper:
                    out_of_range.append(d)
            if out_of_range:
                issues.append(ValidationIssue(
                    check=CHECK_TEMPORAL_COHERENCE,
                    severity="warning",
                    table=tbl.name,
                    message=(
                        f"column {col.name!r} has {len(out_of_range)} date "
                        f"value(s) outside time_window "
                        f"[{lower.isoformat()}, {upper.isoformat()}]. "
                        f"Parameterize the source (e.g. "
                        f"'generated:faker.date_between:start_date:"
                        f"{lower.isoformat()}:end_date:{upper.isoformat()}'), "
                        f"or set 'allow_outside_window: true' on the column "
                        f"if out-of-window dates are intentional."
                    ),
                    details={
                        "column": col.name,
                        "lower": lower.isoformat(),
                        "upper": upper.isoformat(),
                        "out_of_range_count": len(out_of_range),
                        "out_of_range_sample": [
                            d.isoformat() for d in
                            _sample_sorted(out_of_range)
                        ],
                    },
                ))
    return issues


# --- Check 8: cross-dim FK cardinality (FIX-04 / MF-1) -----------------------


def validate_cross_dim_fk_cardinality(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> list[ValidationIssue]:
    """Warn when a multi-row parent dim's PK collapses to a single value
    in any child column.

    Pre-FIX-04, ``dimensions._backfill_fks`` and ``tables._resolve_fact_cell``
    used ``parent.iloc[0]`` for every row, silently breaking realism the
    moment a user expanded a reference dim beyond one row. The fix replaces
    the row-0 collapse with distribution-driven sampling; this validator is
    the regression guard. Pinned values via ``Entity.cross_dim_fks`` are
    intentional and won't trigger the warning unless every entity in the
    config pinned the same value (in which case the warning is still
    accurate — variation IS missing).

    Single-row parents are skipped (no choice to make).
    """
    issues: list[ValidationIssue] = []
    for tbl in config.tables:
        df = tables.get(tbl.name)
        if df is None or df.empty:
            continue
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if not isinstance(parsed, FKSource):
                continue
            parent_df = tables.get(parsed.table)
            if parent_df is None or len(parent_df) <= 1:
                continue
            if col.name not in df.columns:
                continue
            child_unique = {
                v for v in df[col.name].tolist() if not _is_nullish(v)
            }
            if len(child_unique) <= 1:
                issues.append(ValidationIssue(
                    check=CHECK_CROSS_DIM_FK_CARDINALITY,
                    severity="warning",
                    table=tbl.name,
                    message=(
                        f"FK column {col.name!r} carries {len(child_unique)} "
                        f"unique value(s) despite parent {parsed.table!r} "
                        f"having {len(parent_df)} rows. Add 'distribution: "
                        f"uniform' or '{{weights: {{...}}}}' on the column, "
                        f"or pin per cohort via Entity.cross_dim_fks."
                    ),
                    details={
                        "column": col.name,
                        "parent": f"{parsed.table}.{parsed.column}",
                        "parent_row_count": int(len(parent_df)),
                        "child_unique_count": len(child_unique),
                        "child_unique_sample": _sample_sorted(
                            list(child_unique)
                        ),
                    },
                ))
    return issues


# --- Check 10: SCD Type 2 integrity (M106) -----------------------------------


def validate_scd_integrity(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> list[ValidationIssue]:
    """Defensive integrity check for SCD Type 2 dim and fact wiring.

    By construction (``tables.expand_scd_dims`` +
    ``tables.attach_dim_row_id_to_facts``), every SCD-enabled dim has the
    ``dim_row_id`` / ``valid_from`` / ``valid_to`` / ``is_current``
    columns and every fact/event table FK'ing into an SCD dim carries a
    matching ``dim_row_id`` column. This validator catches drift if a
    future engine change (or a programmatic table mutation downstream)
    breaks those invariants. Costs are O(rows-per-table); negligible
    next to the existing checks.

    Errors raised:
      * SCD dim missing the four expansion columns.
      * SCD dim ``is_current`` total != entity count (each entity should
        end with exactly one current version).
      * SCD dim ``valid_from <= valid_to`` violated on any row.
      * Fact/event table FK'ing to an SCD dim is missing ``dim_row_id``.
      * Fact ``dim_row_id`` value not present in the parent SCD dim's
        ``dim_row_id`` column.
    """
    issues: list[ValidationIssue] = []
    expansion_cols = ("dim_row_id", "valid_from", "valid_to", "is_current")

    scd_dims_present: dict[str, set[Any]] = {}
    for tbl in config.tables:
        if not _is_scd_dim(tbl):
            continue
        df = tables.get(tbl.name)
        if df is None or df.empty:
            continue
        missing = [c for c in expansion_cols if c not in df.columns]
        if missing:
            issues.append(ValidationIssue(
                check=CHECK_SCD_INTEGRITY,
                severity="error",
                table=tbl.name,
                message=(
                    f"SCD dim {tbl.name!r} is missing expansion column(s) "
                    f"{missing}"
                ),
                details={"missing_columns": missing},
            ))
            continue
        scd_dims_present[tbl.name] = set(df["dim_row_id"].tolist())

        n_current = int(df["is_current"].astype(bool).sum())
        n_entities = len(config.entities)
        if n_current != n_entities:
            issues.append(ValidationIssue(
                check=CHECK_SCD_INTEGRITY,
                severity="error",
                table=tbl.name,
                message=(
                    f"SCD dim {tbl.name!r} has {n_current} 'is_current=True' "
                    f"row(s) but config has {n_entities} entities; each "
                    f"entity should hold exactly one current version"
                ),
                details={
                    "is_current_count": n_current,
                    "entity_count": n_entities,
                },
            ))

        bad_window = df[df["valid_from"] > df["valid_to"]]
        if not bad_window.empty:
            issues.append(ValidationIssue(
                check=CHECK_SCD_INTEGRITY,
                severity="error",
                table=tbl.name,
                message=(
                    f"SCD dim {tbl.name!r} has {len(bad_window)} row(s) "
                    f"where valid_from > valid_to"
                ),
                details={"bad_window_count": int(len(bad_window))},
            ))

    if not scd_dims_present:
        return issues

    for tbl in config.tables:
        if tbl.type not in ("fact", "event"):
            continue
        df = tables.get(tbl.name)
        if df is None or df.empty:
            continue
        scd_parent: Optional[str] = None
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if isinstance(parsed, FKSource) and parsed.table in scd_dims_present:
                scd_parent = parsed.table
                break
        if scd_parent is None:
            continue
        if "dim_row_id" not in df.columns:
            issues.append(ValidationIssue(
                check=CHECK_SCD_INTEGRITY,
                severity="error",
                table=tbl.name,
                message=(
                    f"{tbl.type} {tbl.name!r} FKs to SCD dim {scd_parent!r} "
                    f"but is missing the 'dim_row_id' surrogate column"
                ),
                details={"scd_parent": scd_parent},
            ))
            continue
        parent_ids = scd_dims_present[scd_parent]
        child_vals = [
            int(v) for v in df["dim_row_id"].tolist() if not _is_nullish(v)
        ]
        orphans = sorted({v for v in child_vals if v not in parent_ids})
        if orphans:
            issues.append(ValidationIssue(
                check=CHECK_SCD_INTEGRITY,
                severity="error",
                table=tbl.name,
                message=(
                    f"{tbl.type} {tbl.name!r} 'dim_row_id' has "
                    f"{len(orphans)} value(s) not present in SCD dim "
                    f"{scd_parent!r}.dim_row_id"
                ),
                details={
                    "scd_parent": scd_parent,
                    "orphan_count": len(orphans),
                    "orphans_sample": orphans[:_SAMPLE_LIMIT],
                },
            ))
    return issues


# --- Check 11: bridge integrity (M107) --------------------------------------


def validate_bridge_integrity(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> list[ValidationIssue]:
    """FK integrity + cardinality + composite-PK uniqueness for M:M bridges.

    Bridge tables sit on ``config.bridges`` rather than ``config.tables``,
    so the standard FK check skips them. This validator covers the same
    ground for them: every bridge FK value must resolve to its parent
    dim's referenced column (PK for non-SCD dims, ``dim_row_id`` for
    SCD dims), the (first_fk, second_fk) tuple must be unique within
    each bridge, and per-entity association counts must lie in
    ``[cardinality.min, cardinality.max]``.

    Errors are emitted with ``check=CHECK_BRIDGE_INTEGRITY`` so the
    validation report can group bridge-specific failures separately
    from regular FK / PK errors.
    """
    issues: list[ValidationIssue] = []
    if not config.bridges:
        return issues

    scd_dim_names = {t.name for t in config.tables if _is_scd_dim(t)}

    def _fk_col_name(dim_tbl: Table) -> str:
        if dim_tbl.name in scd_dim_names:
            short = (
                dim_tbl.name[4:]
                if dim_tbl.name.startswith("dim_") else dim_tbl.name
            )
            return f"{short}_dim_row_id"
        return dim_tbl.primary_key_cols[0]

    for bridge in config.bridges:
        df = tables.get(bridge.name)
        if df is None:
            issues.append(ValidationIssue(
                check=CHECK_BRIDGE_INTEGRITY,
                severity="error",
                table=bridge.name,
                message=(
                    f"bridge {bridge.name!r} declared but no DataFrame was "
                    f"emitted by the engine"
                ),
            ))
            continue
        first_dim_name, second_dim_name = bridge.connects
        first_dim_tbl = next(t for t in config.tables if t.name == first_dim_name)
        second_dim_tbl = next(t for t in config.tables if t.name == second_dim_name)
        first_fk_col = _fk_col_name(first_dim_tbl)
        second_fk_col = _fk_col_name(second_dim_tbl)

        for col_name in (first_fk_col, second_fk_col):
            if col_name not in df.columns:
                issues.append(ValidationIssue(
                    check=CHECK_BRIDGE_INTEGRITY,
                    severity="error",
                    table=bridge.name,
                    message=(
                        f"bridge {bridge.name!r} is missing FK column "
                        f"{col_name!r}"
                    ),
                ))
        if first_fk_col not in df.columns or second_fk_col not in df.columns:
            continue

        # FK integrity: each side resolves to a value present in its parent.
        for fk_col, dim_name, dim_tbl in (
            (first_fk_col, first_dim_name, first_dim_tbl),
            (second_fk_col, second_dim_name, second_dim_tbl),
        ):
            parent_df = tables.get(dim_name)
            if parent_df is None:
                issues.append(ValidationIssue(
                    check=CHECK_BRIDGE_INTEGRITY,
                    severity="error",
                    table=bridge.name,
                    message=(
                        f"bridge {bridge.name!r} FK column {fk_col!r} "
                        f"references parent dim {dim_name!r} which was not "
                        f"generated"
                    ),
                ))
                continue
            if dim_name in scd_dim_names:
                if "dim_row_id" not in parent_df.columns:
                    continue
                parent_keys = {
                    int(v) for v in parent_df.loc[
                        parent_df["is_current"].astype(bool), "dim_row_id",
                    ].tolist()
                }
            else:
                parent_pk_col = dim_tbl.primary_key_cols[0]
                parent_keys = set(parent_df[parent_pk_col].tolist())
            child_vals = [v for v in df[fk_col].tolist() if not _is_nullish(v)]
            orphans = sorted(
                {v for v in child_vals if v not in parent_keys},
                key=lambda x: str(x),
            )
            if orphans:
                issues.append(ValidationIssue(
                    check=CHECK_BRIDGE_INTEGRITY,
                    severity="error",
                    table=bridge.name,
                    message=(
                        f"bridge {bridge.name!r} FK column {fk_col!r} has "
                        f"{len(orphans)} orphan value(s) not present in "
                        f"parent dim {dim_name!r}"
                    ),
                    details={
                        "column": fk_col,
                        "parent": dim_name,
                        "orphans_sample": orphans[:_SAMPLE_LIMIT],
                        "orphan_count": len(orphans),
                    },
                ))

        # Composite-PK uniqueness: each (first_fk, second_fk) tuple must be unique.
        if not df.empty:
            dup_mask = df.duplicated(subset=[first_fk_col, second_fk_col], keep=False)
            if dup_mask.any():
                dup_count = int(dup_mask.sum())
                issues.append(ValidationIssue(
                    check=CHECK_BRIDGE_INTEGRITY,
                    severity="error",
                    table=bridge.name,
                    message=(
                        f"bridge {bridge.name!r} has {dup_count} duplicate "
                        f"({first_fk_col}, {second_fk_col}) tuple(s); each "
                        f"M:M association must be unique"
                    ),
                    details={
                        "duplicate_count": dup_count,
                    },
                ))

            # Cardinality: per-entity association count within [min, max].
            counts = df.groupby(first_fk_col, sort=False).size()
            min_n = bridge.cardinality.min
            max_n = bridge.cardinality.max
            offenders = counts[(counts < min_n) | (counts > max_n)]
            if not offenders.empty:
                issues.append(ValidationIssue(
                    check=CHECK_BRIDGE_INTEGRITY,
                    severity="error",
                    table=bridge.name,
                    message=(
                        f"bridge {bridge.name!r}: {len(offenders)} entity "
                        f"group(s) have association counts outside "
                        f"[{min_n}, {max_n}]"
                    ),
                    details={
                        "offending_count": int(len(offenders)),
                        "min_required": min_n,
                        "max_required": max_n,
                        "sample": offenders.head(_SAMPLE_LIMIT).to_dict(),
                    },
                ))

    return issues


# --- Orchestrator ------------------------------------------------------------


def validate_tables(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> ValidationReport:
    """Run every check and return an immutable report.

    Checks are independent; order is fixed (matches ALL_CHECKS) so the issue
    list is deterministic for the same ``(config, tables)`` input.
    """
    issues: list[ValidationIssue] = []
    issues.extend(validate_correlation_psd(config))
    issues.extend(validate_pk_uniqueness(config, tables))
    issues.extend(validate_fk_integrity(config, tables))
    issues.extend(validate_date_spine(config, tables))
    issues.extend(validate_causal_coherence(config, tables))
    issues.extend(validate_null_policy(config, tables))
    issues.extend(validate_empty_event_tables(config, tables))
    issues.extend(validate_cross_dim_fk_cardinality(config, tables))
    issues.extend(validate_temporal_coherence(config, tables))
    issues.extend(validate_scd_integrity(config, tables))
    issues.extend(validate_bridge_integrity(config, tables))
    return ValidationReport(issues=tuple(issues))
