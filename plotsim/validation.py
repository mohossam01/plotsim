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
    GeneratedSource,
    LagSource,
    MetricSource,
    PlotsimConfig,
    PKSource,
    StaticSource,
    Table,
    ThresholdSource,
    parse_source,
)


CHECK_CORRELATION_PSD = "correlation_psd"
CHECK_PK_UNIQUENESS = "pk_uniqueness"
CHECK_FK_INTEGRITY = "fk_integrity"
CHECK_DATE_SPINE = "date_spine"
CHECK_CAUSAL_COHERENCE = "causal_coherence"
CHECK_NULL_POLICY = "null_policy"

ALL_CHECKS: tuple[str, ...] = (
    CHECK_CORRELATION_PSD,
    CHECK_PK_UNIQUENESS,
    CHECK_FK_INTEGRITY,
    CHECK_DATE_SPINE,
    CHECK_CAUSAL_COHERENCE,
    CHECK_NULL_POLICY,
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
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


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


# --- Check 1: correlation matrix PSD ----------------------------------------


def validate_correlation_psd(config: PlotsimConfig) -> list[ValidationIssue]:
    """Verify the configured correlation matrix is positive-definite.

    Builds the same matrix ``plotsim.metrics.apply_correlations`` uses:
    identity on the diagonal, symmetric off-diagonal from ``config.correlations``.
    A LinAlgError from ``np.linalg.cholesky`` means the matrix isn't PD and
    M004 would fall back to independent samples at runtime — caught here.
    """
    issues: list[ValidationIssue] = []
    if not config.correlations:
        return issues
    names = [m.name for m in config.metrics]
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    mat = np.eye(n)
    for pair in config.correlations:
        if pair.metric_a in idx and pair.metric_b in idx:
            i, j = idx[pair.metric_a], idx[pair.metric_b]
            mat[i, j] = pair.coefficient
            mat[j, i] = pair.coefficient
    try:
        np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvalsh(mat).tolist()
        issues.append(ValidationIssue(
            check=CHECK_CORRELATION_PSD,
            severity="error",
            table=None,
            message=(
                "correlation matrix is not positive-definite; M004 will fall "
                "back to independent samples and no correlation will be applied"
            ),
            details={
                "metrics": names,
                "min_eigenvalue": min(eigvals),
                "eigenvalues": eigvals,
            },
        ))
    return issues


# --- Check 2: PK uniqueness --------------------------------------------------


def validate_pk_uniqueness(
    config: PlotsimConfig, tables: dict[str, pd.DataFrame],
) -> list[ValidationIssue]:
    """Flag duplicate PK values (single-column and composite)."""
    issues: list[ValidationIssue] = []
    for tbl in config.tables:
        df = tables.get(tbl.name)
        if df is None or df.empty:
            continue
        pk_cols = tbl.primary_key_cols
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
    """True if corr(metric[lag:], driver[:-lag]) is stronger than corr(metric, driver).

    Returns None if either correlation is undefined.
    """
    if len(metric_series) <= 2 * lag + 2:
        return None
    unlagged = _pearson(metric_series, driver_series)
    lagged = _pearson(metric_series[lag:], driver_series[:-lag])
    if unlagged is None or lagged is None:
        return None
    return abs(lagged) > abs(unlagged)


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
                (FKSource, PKSource, GeneratedSource, StaticSource, DerivedSource, LagSource),
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
    return ValidationReport(issues=tuple(issues))
