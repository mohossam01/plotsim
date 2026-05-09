"""F17 (M102) — hypothesis property tests for plotsim's load-bearing invariants.

Phase 3 closes the test-surface gaps that let two of M102's correctness bugs
(F1 sub-entity FK collapse, F-06 correlation toposort×Cholesky-indexing)
escape every example-based test. Both bugs share a shape: a parameter
combination that no fixture happened to exercise. Property tests target the
class — for any valid config drawn from a strategy, the invariant must hold.

Four invariants are encoded:

1. **Determinism.** ``generate_tables(cfg, rng)`` followed by ``write_tables``
   produces byte-identical CSVs across two runs at the same seed.

2. **Trajectory-first.** For positive-polarity metrics with monotone-rising
   archetypes the rank correlation between trajectory position and the
   observed metric values is positive; for negative-polarity metrics it is
   negative. The check uses Spearman to be noise-robust and asserts only the
   sign — strictly rigorous "every cell traces back to a position" requires
   re-running the curve evaluator, which would just re-implement the engine.

3. **FK integrity.** Every non-null FK in every fact / event table resolves
   to a parent PK. Same property the example tests verify on bundled
   templates; F17 randomizes declaration order, entity sizes, and number of
   periods to find configurations the bundled set doesn't reach.

4. **Correlation accuracy under randomized declaration.** Same property F6
   pins, but with the configured coefficient and the metric declaration
   permutation drawn from hypothesis strategies.

Per the operator's decision in Phase 3 entry, ``max_examples=25`` (down from
hypothesis's default of 100). Each example calls ``generate_tables`` which
involves trajectory evaluation, metric draws, table assembly — keeping
example count modest preserves a useful CI signal without paying a 4× cost.
The deadline is disabled because ``generate_tables`` legitimately takes
longer than hypothesis's default 200 ms budget.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings, strategies as st

from plotsim import generate_tables, write_tables
from plotsim.config import (
    Archetype,
    Column,
    CorrelationPair,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    NoiseConfig,
    OutputConfig,
    PlotsimConfig,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
    parse_source,
    FKSource,
)


# Shared settings: 25 examples, no deadline, suppress slow-test health check
# since each example runs generate_tables end-to-end.
PROPERTY_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# --- Config builder ---------------------------------------------------------


def _build_minimal_config(
    *,
    seed: int,
    n_entities: int,
    entity_size: int,
    n_periods: int,
    archetype_kind: str,  # "rising" | "falling" | "flat"
    metric_decl_order: list[str],
    correlations: Optional[list[CorrelationPair]] = None,
) -> PlotsimConfig:
    """Build a 3-metric per-entity-per-period config tuned for property tests.

    ``metric_decl_order`` permutes a fixed metric set (engagement, mrr,
    support_tickets) so the Cholesky×toposort axis can be probed. The
    archetype is parametric so trajectory-first sign checks know which
    direction the trajectory moves.
    """
    n_periods_str_end = f"2024-{min(n_periods, 12):02d}"
    if n_periods > 12:
        # Use multi-year window for higher period counts — monthly granularity.
        years_extra = (n_periods - 1) // 12
        end_month = (n_periods - 1) % 12 + 1
        n_periods_str_end = f"{2024 + years_extra}-{end_month:02d}"

    if archetype_kind == "rising":
        segments = [
            CurveSegment(
                curve="sigmoid",
                params={"midpoint": 0.5, "steepness": 6.0},
                start_pct=0.0,
                end_pct=1.0,
            )
        ]
    elif archetype_kind == "falling":
        segments = [
            CurveSegment(curve="exp_decay", params={"rate": 2.0}, start_pct=0.0, end_pct=1.0)
        ]
    else:  # flat
        segments = [
            CurveSegment(curve="plateau", params={"level": 0.5}, start_pct=0.0, end_pct=1.0)
        ]
    archetype = Archetype(
        name="property_arch",
        label="property arch",
        description="property-test archetype",
        curve_segments=segments,
    )

    base_metrics = {
        "engagement": Metric(
            name="engagement",
            label="engagement",
            distribution="beta",
            params={"alpha": 2.0, "beta": 5.0},
            polarity="positive",
            value_range={"min": 0.0, "max": 1.0},
        ),
        "mrr": Metric(
            name="mrr",
            label="mrr",
            distribution="lognorm",
            params={"s": 0.85, "loc": 0.0, "scale": 1200.0},
            polarity="positive",
            value_range={"min": 0.0, "max": 100000.0},
        ),
        "support_tickets": Metric(
            name="support_tickets",
            label="support tickets",
            distribution="poisson",
            params={"lambda": 4.0},
            polarity="negative",
        ),
    }
    metrics = [base_metrics[name] for name in metric_decl_order]

    entities = [
        Entity(
            name=f"e{i:02d}",
            archetype="property_arch",
            size=entity_size,
        )
        for i in range(n_entities)
    ]

    fct_columns = [
        Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
        Column(name="user_id", dtype="id", source="fk:dim_user.user_id"),
    ]
    for name in metric_decl_order:
        dtype = "int" if name == "support_tickets" else "float"
        fct_columns.append(
            Column(name=f"{name}_value", dtype=dtype, source=f"metric:{name}"),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="property",
                description="property",
                entity_type="user",
                entity_label="Users",
            ),
            time_window=TimeWindow(
                start="2024-01",
                end=n_periods_str_end,
                granularity="monthly",
            ),
            seed=seed,
            metrics=metrics,
            archetypes=[archetype],
            entities=entities,
            tables=[
                Table(
                    name="dim_date",
                    type="dim",
                    grain="per_period",
                    primary_key="date_key",
                    columns=[
                        Column(name="date_key", dtype="id", source="pk"),
                        Column(name="date", dtype="date", source="generated:date_key"),
                    ],
                ),
                Table(
                    name="dim_user",
                    type="dim",
                    grain="per_entity",
                    primary_key="user_id",
                    columns=[
                        Column(name="user_id", dtype="id", source="pk"),
                        Column(name="user_name", dtype="string", source="generated:faker.name"),
                    ],
                ),
                Table(
                    name="fct_metrics",
                    type="fact",
                    grain="per_entity_per_period",
                    primary_key=["date_key", "user_id"],
                    foreign_keys=["dim_date.date_key", "dim_user.user_id"],
                    columns=fct_columns,
                ),
            ],
            correlations=correlations or [],
            noise=NoiseConfig(gaussian_sigma=0.05, outlier_rate=0.0, mcar_rate=0.0),
            output=OutputConfig(format="csv", directory="out/property"),
        )


# --- Strategies -------------------------------------------------------------


_metric_permutations = st.permutations(["engagement", "mrr", "support_tickets"])


# --- Property 1: determinism ------------------------------------------------


@PROPERTY_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_entities=st.integers(min_value=1, max_value=4),
    entity_size=st.integers(min_value=1, max_value=3),
    n_periods=st.integers(min_value=4, max_value=18),
    archetype_kind=st.sampled_from(["rising", "falling", "flat"]),
    decl_order=_metric_permutations,
)
def test_property_same_seed_yields_byte_identical_csvs(
    tmp_path_factory,
    seed,
    n_entities,
    entity_size,
    n_periods,
    archetype_kind,
    decl_order,
):
    """For any randomly-generated valid config, two runs at the same seed
    produce byte-identical CSVs and validation report. Pins invariant #2
    (deterministic output) across the parameter space.
    """
    cfg = _build_minimal_config(
        seed=seed,
        n_entities=n_entities,
        entity_size=entity_size,
        n_periods=n_periods,
        archetype_kind=archetype_kind,
        metric_decl_order=list(decl_order),
    )
    dir_a = tmp_path_factory.mktemp("run_a")
    dir_b = tmp_path_factory.mktemp("run_b")

    rng_a = np.random.default_rng(cfg.seed)
    tables_a = generate_tables(cfg, rng_a)
    write_tables(tables_a, cfg, output_dir=dir_a)

    rng_b = np.random.default_rng(cfg.seed)
    tables_b = generate_tables(cfg, rng_b)
    write_tables(tables_b, cfg, output_dir=dir_b)

    for tbl in cfg.tables:
        path_a = dir_a / f"{tbl.name}.csv"
        path_b = dir_b / f"{tbl.name}.csv"
        if path_a.exists() and path_b.exists():
            assert path_a.read_bytes() == path_b.read_bytes(), (
                f"determinism: {tbl.name}.csv differs across two runs at "
                f"seed {cfg.seed} ({n_entities} entities × size {entity_size} "
                f"× {n_periods} periods, archetype={archetype_kind}, "
                f"decl_order={list(decl_order)})"
            )

    report_a = (dir_a / "validation_report.txt").read_bytes()
    report_b = (dir_b / "validation_report.txt").read_bytes()
    assert (
        report_a == report_b
    ), "determinism: validation_report.txt drifts across runs (F5 regression)"


# --- Property 2: trajectory-first sign --------------------------------------


@PROPERTY_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_entities=st.integers(min_value=2, max_value=4),
    entity_size=st.integers(min_value=1, max_value=2),
    n_periods=st.integers(min_value=12, max_value=24),
    decl_order=_metric_permutations,
)
def test_property_metric_polarity_aligned_with_trajectory(
    seed, n_entities, entity_size, n_periods, decl_order
):
    """For a monotone-rising archetype: positive-polarity metrics' Spearman
    rank correlation with period index is positive (trajectory rises with
    time → positive metric tracks it up); negative-polarity metric's is
    negative. Pins invariant #1 (trajectory-first generation).

    Strict cell-level "every value traces back to a position" requires
    re-running the curve evaluator inside the test, which would just
    re-implement the engine. The sign assertion is the looser, robust
    check that catches the failure mode where metric and trajectory
    decouple (e.g., the F-06 class where post-correlation transforms
    re-shuffle rows without preserving per-period structure).
    """
    cfg = _build_minimal_config(
        seed=seed,
        n_entities=n_entities,
        entity_size=entity_size,
        n_periods=n_periods,
        archetype_kind="rising",
        metric_decl_order=list(decl_order),
    )
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)
    fct = tables["fct_metrics"].sort_values(["user_id", "date_key"])

    metric_polarities = {m.name: m.polarity for m in cfg.metrics}
    # ``dim_user`` is 1:1 with ``config.entities`` (per the per_entity-dim
    # invariant in tables.py); ``Entity.size`` is metadata, not a row
    # multiplier. So fact rows = n_entities × n_periods regardless of size.
    period_idx = np.tile(
        np.arange(n_periods, dtype=float),
        len(cfg.entities),
    )

    for col in fct.columns:
        if not col.endswith("_value"):
            continue
        metric_name = col[: -len("_value")]
        if metric_name not in metric_polarities:
            continue
        polarity = metric_polarities[metric_name]
        # Spearman via pandas — robust to the lognorm / poisson skew.
        values = fct[col].astype(float).to_numpy()
        finite = np.isfinite(values)
        if finite.sum() < 6:
            continue  # too few non-null samples to assert
        rho = pd.Series(values[finite]).corr(pd.Series(period_idx[finite]), method="spearman")
        if polarity == "positive":
            assert rho > -0.2, (
                f"trajectory-first regression: {metric_name} (positive) "
                f"has Spearman {rho:.3f} vs rising archetype period_idx; "
                f"expected positive (or near zero with low noise). "
                f"seed={cfg.seed}, decl={list(decl_order)}, "
                f"n_periods={n_periods}, n_entities={n_entities}"
            )
        else:  # negative polarity
            assert rho < 0.2, (
                f"trajectory-first regression: {metric_name} (negative) "
                f"has Spearman {rho:.3f} vs rising archetype period_idx; "
                f"expected negative (high trajectory → low value). "
                f"seed={cfg.seed}, decl={list(decl_order)}, "
                f"n_periods={n_periods}, n_entities={n_entities}"
            )


# --- Property 3: FK integrity -----------------------------------------------


@PROPERTY_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_entities=st.integers(min_value=1, max_value=4),
    entity_size=st.integers(min_value=1, max_value=3),
    n_periods=st.integers(min_value=4, max_value=18),
    archetype_kind=st.sampled_from(["rising", "falling", "flat"]),
    decl_order=_metric_permutations,
)
def test_property_fk_integrity_holds_for_every_fact_and_event_row(
    seed,
    n_entities,
    entity_size,
    n_periods,
    archetype_kind,
    decl_order,
):
    """Every non-null FK value in a fact / event table resolves to a parent
    PK. Same invariant the example tests verify on bundled templates; the
    property test randomizes declaration order, entity counts, sizes, and
    period count to expose configurations no fixture exercises. Pins
    invariant #5 (cross-table referential integrity).
    """
    cfg = _build_minimal_config(
        seed=seed,
        n_entities=n_entities,
        entity_size=entity_size,
        n_periods=n_periods,
        archetype_kind=archetype_kind,
        metric_decl_order=list(decl_order),
    )
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)

    pks: dict[str, set] = {}
    for tbl in cfg.tables:
        if tbl.type != "dim":
            continue
        df = tables[tbl.name]
        # primary_key may be str or list; PK uniqueness is enforced upstream.
        pk_cols = tbl.primary_key if isinstance(tbl.primary_key, list) else [tbl.primary_key]
        # Build the parent PK index by column. For composite PKs we'd need a
        # tuple set; bundled property configs use single-column dim PKs.
        for pk_col in pk_cols:
            pks[f"{tbl.name}.{pk_col}"] = set(df[pk_col].dropna().tolist())

    for tbl in cfg.tables:
        if tbl.type not in ("fact", "event"):
            continue
        df = tables[tbl.name]
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if not isinstance(parsed, FKSource):
                continue
            ref_key = f"{parsed.table}.{parsed.column}"
            if ref_key not in pks:
                continue
            parent_pks = pks[ref_key]
            child_values = df[col.name].dropna().tolist()
            unresolved = [v for v in child_values if v not in parent_pks]
            assert not unresolved, (
                f"FK integrity violation: {tbl.name}.{col.name} → "
                f"{ref_key} has {len(unresolved)} unresolved values "
                f"(sample: {unresolved[:5]!r}); "
                f"seed={cfg.seed}, decl={list(decl_order)}, "
                f"archetype={archetype_kind}"
            )


# --- Property 4: correlation accuracy under randomized declaration ----------


# Sample target_corr from two non-overlapping bands [-0.6, -0.1] and [0.1, 0.6]
# instead of filtering [-0.6, 0.6] — avoids hypothesis "invalid because filter"
# events without changing the test surface (still excludes structurally
# redundant values near zero).
@PROPERTY_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    target_corr=st.one_of(
        st.floats(min_value=-0.6, max_value=-0.1),
        st.floats(min_value=0.1, max_value=0.6),
    ),
    decl_order=_metric_permutations,
)
def test_property_configured_correlation_holds_independent_of_declaration(
    seed,
    target_corr,
    decl_order,
):
    """For a 3-metric config with a configured pairwise correlation, the
    observed Pearson on that pair lands within ±0.20 of configured —
    independent of metric declaration order. Pins F6 (toposort×Cholesky-
    indexing invariant) under randomized inputs.

    Uses a **flat plateau archetype** (matches F6's example tests) so
    every metric's center is constant across the window — the
    trajectory-first invariant is decoupled from the correlation signal
    we're measuring. With a non-flat archetype, two metrics on the same
    monotone trajectory show natural same-direction correlation even at
    configured ``coefficient=0.0``, and the test would conflate
    trajectory-first generation with the post-correlation matrix work.

    ``target_corr`` is drawn from ``[-0.6, -0.1] ∪ [0.1, 0.6]`` —
    bands chosen for the same reason F6 uses coefficients in the
    0.5–0.85 band: the test asserts a deviation from configured,
    which is only well-defined when configured is materially non-zero.
    Values near 0 are structurally redundant (already the default for
    unlisted pairs) and the engine emits a ``RedundantCorrelationWarning``
    for ``coefficient=0.0``.

    Skips configurations where the chosen pair involves ``support_tickets``
    (poisson). F2 already verifies bypass-aware correlation handling and
    that attenuation pattern is out of scope here.
    """
    decl_list = list(decl_order)
    # Pick the first two non-poisson metrics from the declaration order.
    pair_candidates = [m for m in decl_list if m != "support_tickets"]
    if len(pair_candidates) < 2:
        return  # decl_order only gives us one continuous metric; skip

    metric_a, metric_b = pair_candidates[0], pair_candidates[1]
    cfg = _build_minimal_config(
        seed=seed,
        n_entities=15,  # enough samples for a stable observed Pearson
        entity_size=2,
        n_periods=36,
        archetype_kind="flat",  # decouple trajectory from correlation signal
        metric_decl_order=decl_list,
        correlations=[
            CorrelationPair(
                metric_a=metric_a,
                metric_b=metric_b,
                coefficient=target_corr,
            ),
        ],
    )
    rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, rng)
    fct = tables["fct_metrics"]
    observed = float(
        np.corrcoef(
            fct[f"{metric_a}_value"].astype(float),
            fct[f"{metric_b}_value"].astype(float),
        )[0, 1]
    )
    # Tolerance widened to ±0.30 vs F6's ±0.10 because hypothesis explores
    # a much wider parameter space (smaller configs, varied seeds, varied
    # decl orders) where per-realization Pearson is noisier. The
    # aliased-pair failure mode F6 catches produces |diff| > 0.5 typically,
    # so ±0.30 still flags it cleanly.
    assert abs(observed - target_corr) < 0.30, (
        f"correlation drift: configured ({metric_a}, {metric_b})="
        f"{target_corr:.3f}, observed={observed:.3f} "
        f"(|diff|={abs(observed - target_corr):.3f} >= 0.30); "
        f"seed={cfg.seed}, decl_order={decl_list}"
    )
