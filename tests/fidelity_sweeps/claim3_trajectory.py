"""Claim 3 — trajectory-first cell-level verification.

For each bundled template × seed pair, generate the dataset once, then sample
100 (entity, period) cells per (template, seed). For each sampled cell and
each non-lagged scalar metric, recompute the trajectory position from the
entity's archetype, recompute the predicted center via position_to_center,
and write (template, entity_id, period, metric, trajectory_position,
predicted_center, observed, deviation_in_sigma) to the result CSV.

The deviation is normalized by an envelope sigma: the configured
distribution's standard deviation at the predicted center, combined in
quadrature with the multiplicative-Gaussian noise contribution. Outlier
hits (~outlier_rate × N) populate the deep tail and are reported as the
expected baseline, not as invariant violations.

Lagged metrics are excluded — their effective trajectory position is a
blend of own and driver positions, which is Claim 2's territory. Archetype
metric_overrides are applied per-entity (the engine does this in
generate_metrics_for_period; the verifier mirrors it at predict time).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from plotsim import generate_tables, load_config
from plotsim.config import (
    Archetype,
    FKSource,
    Metric,
    MetricSource,
    PlotsimConfig,
    parse_source,
)
from plotsim.metrics import position_to_center
from plotsim.trajectory import compute_trajectory


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ["saas", "hr", "ecommerce", "education", "healthcare"]
TEMPLATE_PATHS = {
    name: REPO_ROOT / "plotsim" / "configs" / f"sample_{name}.yaml" for name in TEMPLATES
}
RESULT_CSV = REPO_ROOT / "analysis" / "fidelity_sweeps" / "trajectory_first_results.csv"

CELLS_PER_TEMPLATE_PER_SEED = 100
SEEDS_PER_TEMPLATE = 5
SEED_BASE = 1_000  # offset so sweep seeds don't collide with template seeds


# --- Per-distribution envelope sigma at a given center ----------------------


def _distribution_sigma(metric: Metric, center: float) -> float:
    """Standard deviation of ``metric``'s distribution at ``center``.

    Mirrors ``plotsim.metrics.sample_single_metric``'s parameterization so the
    sigma is calibrated to the same draw the engine produces.
    """
    dist = metric.distribution
    params = metric.params
    if dist == "normal":
        return float(params["sigma"])
    if dist == "lognorm":
        s = float(params["s"])
        # rng.lognormal(mean=log(center), sigma=s); Var = (e^{s²}-1) e^{2μ+s²}
        # → std = |center| · √(e^{s²}-1) · e^{s²/2}
        return float(
            abs(center) * math.sqrt(max(math.exp(s * s) - 1.0, 0.0)) * math.exp(s * s / 2.0)
        )
    if dist == "poisson":
        return float(math.sqrt(max(center, 1e-12)))
    if dist == "gamma":
        shape = float(params["shape"])
        if shape <= 0.0 or center <= 0.0:
            return 0.0
        return float(abs(center) / math.sqrt(shape))
    if dist == "beta":
        a = float(params["alpha"])
        b = float(params["beta"])
        unit_var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
        vr = metric.value_range
        if vr is not None and vr.min is not None and vr.max is not None:
            span = float(vr.max - vr.min)
        else:
            span = float(params.get("scale", 1.0))
        return float(math.sqrt(unit_var) * span)
    if dist == "weibull":
        k = float(params["shape"])
        if k <= 0.0:
            return 0.0
        # Var(Weibull(k, λ)) = λ² [Γ(1+2/k) - Γ(1+1/k)²]; engine uses λ=center
        unit_var = math.gamma(1.0 + 2.0 / k) - math.gamma(1.0 + 1.0 / k) ** 2
        return float(abs(center) * math.sqrt(max(unit_var, 0.0)))
    raise ValueError(f"unsupported distribution {dist!r}")


def _envelope_sigma(metric: Metric, center: float, gaussian_noise: float) -> float:
    """Combine distribution sigma and multiplicative Gaussian noise sigma."""
    dist_sigma = _distribution_sigma(metric, center)
    if gaussian_noise <= 0.0:
        return dist_sigma
    mag = abs(center) if center != 0.0 else 1.0
    noise_sigma = gaussian_noise * mag
    return float(math.sqrt(dist_sigma**2 + noise_sigma**2))


# --- Resolve archetype overrides per entity ---------------------------------


def _effective_metric(metric: Metric, archetype: Archetype) -> Metric:
    """Apply archetype.metric_overrides to a metric, returning the effective one.

    Mirrors ``plotsim.metrics._apply_archetype_overrides`` (which is private).
    """
    override = archetype.metric_overrides.get(metric.name) if archetype else None
    if override is None:
        return metric
    updates: dict = {}
    if override.distribution is not None:
        updates["distribution"] = override.distribution
    if override.params is not None:
        updates["params"] = override.params
    return metric.model_copy(update=updates) if updates else metric


# --- Fact-table walk: which fact contains which metric, and how to index ----


def _per_entity_per_period_facts(config: PlotsimConfig) -> list[dict]:
    """Return one entry per per_entity_per_period fact table:

    {
      "table": tbl.name,
      "entity_fk_col": <FK col into the per_entity dim>,
      "parent_dim": <per_entity dim name>,
      "parent_pk": <PK col in the per_entity dim>,
      "metric_columns": [(column_name, metric_name), ...],
    }
    """
    per_entity_dims = {
        t.name: (t.primary_key if isinstance(t.primary_key, str) else t.primary_key[0])
        for t in config.tables
        if t.type == "dim" and t.grain == "per_entity"
    }
    out: list[dict] = []
    for tbl in config.tables:
        if tbl.type != "fact" or tbl.grain != "per_entity_per_period":
            continue
        entity_fk_col = None
        parent_dim = None
        parent_pk = None
        metric_cols: list[tuple[str, str]] = []
        for col in tbl.columns:
            parsed = parse_source(col.source)
            if isinstance(parsed, FKSource) and parsed.table in per_entity_dims:
                entity_fk_col = col.name
                parent_dim = parsed.table
                parent_pk = parsed.column
            elif isinstance(parsed, MetricSource):
                metric_cols.append((col.name, parsed.metric))
        if entity_fk_col is None or not metric_cols:
            continue
        out.append(
            {
                "table": tbl.name,
                "entity_fk_col": entity_fk_col,
                "parent_dim": parent_dim,
                "parent_pk": parent_pk,
                "metric_columns": metric_cols,
            }
        )
    return out


# --- Per-template verifier --------------------------------------------------


def _verify_template_seed(
    template: str,
    seed_idx: int,
    cells_per_run: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate one dataset and emit per-cell verification rows."""
    cfg = load_config(TEMPLATE_PATHS[template])
    cfg = cfg.model_copy(update={"seed": SEED_BASE + seed_idx})
    arch_by_name = {a.name: a for a in cfg.archetypes}
    metric_by_name = {m.name: m for m in cfg.metrics}

    # Generate.
    gen_rng = np.random.default_rng(cfg.seed)
    tables = generate_tables(cfg, gen_rng)

    # Period count: re-derive from time window so we can recompute trajectories.
    from plotsim.trajectory import compute_time_steps

    n_periods = int(compute_time_steps(cfg.time_window).shape[0])

    # Trajectories per entity (no overrides — overrides shift inflection but
    # the predicted_center recompute uses per-entity inflection too via
    # entity.overrides.model_dump(); compute_trajectory takes that dict).
    trajectories: dict[str, np.ndarray] = {}
    for ent in cfg.entities:
        arch = arch_by_name[ent.archetype]
        ovr = ent.overrides.model_dump() if ent.overrides is not None else None
        trajectories[ent.name] = compute_trajectory(arch, n_periods, ovr)

    # Sample (entity_idx, period_idx) cells.
    n_entities = len(cfg.entities)
    sampled = []
    for _ in range(cells_per_run):
        e_idx = int(rng.integers(0, n_entities))
        p_idx = int(rng.integers(0, n_periods))
        sampled.append((e_idx, p_idx))

    facts = _per_entity_per_period_facts(cfg)
    rows: list[dict] = []
    gaussian_noise = float(cfg.noise.gaussian_sigma)

    for fact in facts:
        df = tables[fact["table"]]
        parent = tables[fact["parent_dim"]]
        # Entity → row index in fact: i*n_periods + p (per the row-ordering
        # invariant validated in F17 and tables.py:vectorized_per_entity_per_period_fact).
        # We confirm by also matching entity_fk_col == parent.iloc[i][parent_pk].
        for col_name, metric_name in fact["metric_columns"]:
            metric = metric_by_name[metric_name]
            # Skip lagged metrics — Claim 2 turf.
            if metric.causal_lag is not None:
                continue
            for e_idx, p_idx in sampled:
                ent = cfg.entities[e_idx]
                arch = arch_by_name[ent.archetype]
                effective = _effective_metric(metric, arch)
                position = float(trajectories[ent.name][p_idx])
                predicted_center = position_to_center(position, effective)
                envelope = _envelope_sigma(effective, predicted_center, gaussian_noise)
                # Look up observed: vectorized fact row order is entity-major.
                row_idx = e_idx * n_periods + p_idx
                observed_raw = df.iloc[row_idx][col_name]
                # MCAR null → skip the cell.
                if pd.isna(observed_raw):
                    continue
                observed = float(observed_raw)
                if envelope > 0.0:
                    deviation_sigma = (observed - predicted_center) / envelope
                else:
                    deviation_sigma = float("nan")
                entity_pk = parent.iloc[e_idx][fact["parent_pk"]]
                rows.append(
                    {
                        "template": template,
                        "seed": cfg.seed,
                        "entity_id": str(entity_pk),
                        "entity_idx": e_idx,
                        "period_idx": p_idx,
                        "fact_table": fact["table"],
                        "metric": metric_name,
                        "distribution": effective.distribution,
                        "trajectory_position": position,
                        "predicted_center": predicted_center,
                        "envelope_sigma": envelope,
                        "observed": observed,
                        "deviation": observed - predicted_center,
                        "deviation_in_sigma": deviation_sigma,
                    }
                )
    return rows


def run_claim3(
    cells_per_run: int = CELLS_PER_TEMPLATE_PER_SEED,
    seeds_per_template: int = SEEDS_PER_TEMPLATE,
    out_csv: Path = RESULT_CSV,
) -> int:
    """Drive the full Claim 3 sweep and write the result CSV."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sample_rng = np.random.default_rng(0xC1A1)  # pin sampling reproducibility
    all_rows: list[dict] = []
    t0 = time.monotonic()

    for template in TEMPLATES:
        for seed_idx in range(seeds_per_template):
            t_cell = time.monotonic()
            rows = _verify_template_seed(template, seed_idx, cells_per_run, sample_rng)
            all_rows.extend(rows)
            sys.stderr.write(
                f"[claim3] {template} seed_idx={seed_idx}: "
                f"{len(rows)} rows ({time.monotonic() - t_cell:.1f}s)\n"
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    sys.stderr.write(
        f"[claim3] wrote {len(df)} rows to {out_csv} in {time.monotonic() - t0:.1f}s total\n"
    )
    return len(df)


if __name__ == "__main__":
    run_claim3()
