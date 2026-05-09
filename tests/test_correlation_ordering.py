"""F6 regression — adversarial declaration vs toposort ordering (M102).

Closes the F-06 regression class. Pre-0.4.0 the Cholesky factor
``cholesky_L`` was hoisted at the top of ``generate_tables`` from the
declaration-order metric list, while ``apply_correlations`` indexed its
Gaussian-space ``z`` vector by the toposorted metric list it received
from ``generate_entity_metrics``. When a config used ``causal_lag``
(which reshuffles the toposort target away from declaration order), the
mismatch routed each configured correlation to whichever pair happened
to occupy the same ``(i, j)`` index positions in the toposorted list —
the wrong pair unless declaration and toposort coincided. The fix at
``tables.py:1431-1437`` builds ``L`` on the toposorted list, restoring
the index-axis invariant ("``L`` is indexed by the metric list passed
downstream").

This file has two tests:

* ``test_correlation_ordering_holds_when_toposort_reverses_declaration``
  — declaration ``[a, b, c, d]`` with chain ``a←b, b←c, c←d`` so
  ``_toposort_metrics`` emits ``[d, c, b, a]`` (full reversal).
  Configured correlations on every pair land within ±0.10 of the
  factor-model target. Pre-fix, off-anti-diagonal pairs (e.g.
  (a, b), (c, d)) get aliased to the wrong configured value and the
  observed median Pearson lands well outside ±0.10.

* ``test_correlation_ordering_holds_under_randomized_declaration`` —
  parametrized over five declaration permutations of the same
  4-metric chain. Adjacent declarations (where toposort happens to
  equal declaration) trivially pass; the parametrize sweep guarantees
  at least one non-trivial permutation is exercised on every run.

Both tests use a flat plateau archetype (level=0.5) so the lag chain
contributes no per-period dynamics — every metric's center is constant
across the window — isolating the test signal to the correlation-matrix
indexing question that F-06 is about.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pytest

from plotsim import generate_tables
from plotsim.config import (
    Archetype,
    CausalLag,
    Column,
    CorrelationPair,
    CurveSegment,
    Domain,
    Entity,
    Metric,
    OutputConfig,
    PlotsimConfig,
    SurrogateKeyWarning,
    Table,
    TimeWindow,
)
from plotsim.metrics import _toposort_metrics


SEEDS = list(range(8))
TOLERANCE = 0.10
LOADINGS = {"a": 0.85, "b": 0.75, "c": 0.65, "d": 0.55}


def _factor_model_pairs() -> dict[tuple[str, str], float]:
    """All 6 pairwise correlations from rank-1 loadings — PSD by construction."""
    pairs: dict[tuple[str, str], float] = {}
    names = list(LOADINGS)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs[(a, b)] = LOADINGS[a] * LOADINGS[b]
    return pairs


def _build_test_config(
    *,
    declaration_order: list[str],
    chain_pairs: list[tuple[str, str]],
    correlations: list[CorrelationPair],
    seed: int = 0,
) -> PlotsimConfig:
    """Build a 4-metric config with the requested declaration order and
    causal_lag chain. ``chain_pairs`` is a list of ``(child, driver)``
    tuples — ``("a", "b")`` means ``a.causal_lag.driver = b``.
    """
    chain = {child: driver for child, driver in chain_pairs}
    metrics = []
    for name in declaration_order:
        cl = (
            CausalLag(driver=chain[name], lag_periods=1, blend_weight=1.0)
            if name in chain
            else None
        )
        metrics.append(
            Metric(
                name=name,
                label=name,
                distribution="normal",
                params={"mu": 10.0, "sigma": 2.0},
                polarity="positive",
                causal_lag=cl,
            )
        )
    arch = Archetype(
        name="flat",
        label="flat",
        description="constant 0.5 plateau — lag chain has no dynamic effect",
        curve_segments=[
            CurveSegment(
                curve="plateau",
                params={"level": 0.5},
                start_pct=0.0,
                end_pct=1.0,
            ),
        ],
    )
    fct_columns = [
        Column(name="date_key", dtype="id", source="fk:dim_date.date_key"),
        Column(name="user_id", dtype="id", source="fk:dim_user.user_id"),
    ]
    for name in declaration_order:
        fct_columns.append(
            Column(name=name, dtype="float", source=f"metric:{name}"),
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SurrogateKeyWarning)
        return PlotsimConfig(
            domain=Domain(
                name="t",
                description="t",
                entity_type="user",
                entity_label="Users",
            ),
            time_window=TimeWindow(
                start="2024-01",
                end="2027-12",
                granularity="monthly",
            ),
            seed=seed,
            metrics=metrics,
            archetypes=[arch],
            entities=[Entity(name=f"u{i:02d}", archetype="flat", size=2) for i in range(15)],
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
            correlations=correlations,
            output=OutputConfig(format="csv", directory="out/f6"),
        )


def _observed_pearson_per_seed(
    cfg: PlotsimConfig,
    seeds: Iterable[int],
    pairs: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], list[float]]:
    observed: dict[tuple[str, str], list[float]] = {p: [] for p in pairs}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        tables = generate_tables(cfg, rng)
        fct = tables["fct_metrics"]
        for a, b in observed:
            observed[(a, b)].append(float(np.corrcoef(fct[a], fct[b])[0, 1]))
    return observed


def _correlation_pairs(targets: dict[tuple[str, str], float]) -> list[CorrelationPair]:
    return [CorrelationPair(metric_a=a, metric_b=b, coefficient=c) for (a, b), c in targets.items()]


def test_correlation_ordering_holds_when_toposort_reverses_declaration():
    """F6 — declaration ``[a, b, c, d]`` chained ``a←b←c←d`` so
    ``_toposort_metrics`` emits ``[d, c, b, a]`` (full reversal).
    Configured factor-model correlations must hold within ±0.10 on
    every pair, with particular attention to the spec's non-adjacent
    DAG pairs ``(a, c)``, ``(a, d)``, ``(b, d)``.

    Pre-F-06: cholesky_L built on declaration order, ``z`` indexed by
    toposort. The bug aliased each configured pair's coefficient to
    whichever pair sat at the same ``(i, j)`` in the toposort list.
    For 4-metric reversal, off-anti-diagonal pairs ((a,b), (a,c),
    (b,d), (c,d)) flip across the anti-diagonal and observe materially
    wrong Pearson values.
    """
    declaration_order = ["a", "b", "c", "d"]
    # child←driver: a←b means a.causal_lag.driver = b.
    chain_pairs = [("a", "b"), ("b", "c"), ("c", "d")]
    pair_targets = _factor_model_pairs()
    cfg = _build_test_config(
        declaration_order=declaration_order,
        chain_pairs=chain_pairs,
        correlations=_correlation_pairs(pair_targets),
    )

    topo_names = [m.name for m in _toposort_metrics(list(cfg.metrics))]
    assert topo_names == list(reversed(declaration_order)), (
        f"Test setup error: toposort {topo_names} did not reverse "
        f"declaration {declaration_order}"
    )

    observed = _observed_pearson_per_seed(cfg, SEEDS, pair_targets)
    for pair, target in pair_targets.items():
        median = float(np.median(observed[pair]))
        assert abs(median - target) < TOLERANCE, (
            f"F6 regression: pair {pair} observed median Pearson "
            f"{median:.3f}, configured {target:.3f} "
            f"(|diff|={abs(median - target):.3f} >= {TOLERANCE}); "
            f"per-seed {[round(v, 3) for v in observed[pair]]}"
        )


@pytest.mark.parametrize("decl_seed", list(range(5)))
def test_correlation_ordering_holds_under_randomized_declaration(decl_seed):
    """F6 — randomize declaration order against a fixed toposort target.
    Chain ``a←b←c←d`` is fixed; the input metric list permutes across
    five seeds. ``_toposort_metrics`` always emits ``[d, c, b, a]``
    regardless of declaration order, so any working implementation
    must deliver configured correlations independent of declaration.

    Pre-F-06 bug detection depends on declaration ≠ toposort. The
    parametrize sweep statistically guarantees coverage of at least
    one non-trivial permutation per run; trivial cases
    (declaration == toposort) trivially pass.
    """
    base_order = ["a", "b", "c", "d"]
    chain_pairs = [("a", "b"), ("b", "c"), ("c", "d")]

    rnd = np.random.default_rng(decl_seed * 100 + 7)
    permuted = list(rnd.permutation(base_order))

    pair_targets = _factor_model_pairs()
    cfg = _build_test_config(
        declaration_order=permuted,
        chain_pairs=chain_pairs,
        correlations=_correlation_pairs(pair_targets),
        seed=decl_seed,
    )

    # Toposort target is invariant under input permutation.
    topo_names = [m.name for m in _toposort_metrics(list(cfg.metrics))]
    assert topo_names == ["d", "c", "b", "a"], (
        f"Toposort target is not [d, c, b, a] under decl_seed={decl_seed} "
        f"(declaration={permuted}, toposort={topo_names})"
    )

    observed = _observed_pearson_per_seed(cfg, SEEDS, pair_targets)
    for pair, target in pair_targets.items():
        median = float(np.median(observed[pair]))
        assert abs(median - target) < TOLERANCE, (
            f"F6 regression (decl_seed={decl_seed}, "
            f"decl_order={permuted}): pair {pair} observed median "
            f"Pearson {median:.3f}, configured {target:.3f} "
            f"(|diff|={abs(median - target):.3f} >= {TOLERANCE}); "
            f"per-seed {[round(v, 3) for v in observed[pair]]}"
        )
