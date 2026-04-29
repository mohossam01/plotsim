"""Shared introspection helpers for the M113 acceptance / qualifier notebooks.

Importable by both ``acceptance_test.ipynb`` and ``template_qualifier.ipynb``.
All reconstruction logic exceeding 5 lines lives here, not in cells. Tolerance
constants used by §4 (trajectory shape recovery), §5 (marginal moment recovery),
§13 (fidelity audit) are module-level with explicit provenance.

Naming convention for tolerance constants — three suffixes, semantics fixed:

* ``_PASS``    — the threshold at which the audit passes. Breach ⇒ failure.
* ``_WARN``    — stricter than ``_PASS``. Surfaces concern without failing.
* ``_OUTLIER`` — looser than ``_PASS``. Catches extreme deviations that would
                 otherwise pass quietly because ``_PASS`` is intentionally
                 relaxed for structural reasons.

For Pearson-style metrics (higher is better), thresholds are floors. For
deviation-style metrics (lower is better), thresholds are ceilings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from plotsim.config import Archetype, PlotsimConfig, load_config
from plotsim.trajectory import compute_trajectory


# --- Tolerance constants ---------------------------------------------------

# Provenance: project/research/engine-fidelity-check.md §1, non-plateau
# archetypes. Mean |Pearson| range across templates 0.640–0.908; min row 0.237
# (saas steady_grower). 0.45 floor catches genuine signal loss while leaving
# headroom for the empirical minimum. Floor — higher is better.
MONOTONIC_ARCHETYPE_PEARSON_PASS = 0.45

# Provenance: engine-fidelity-check.md §1 oscillating rows. Mean |Pearson|
# range 0.408–0.553 (retail holiday_surge, retail bargain_hunter, marketing
# deal_seeker). Pass floor 0.30 because oscillating curves with short periods
# vary faster than within-period sampling noise can follow — see flag list
# §4. Floor — higher is better.
OSCILLATING_ARCHETYPE_PEARSON_PASS = 0.30

# Stricter than the pass threshold. Surfaces oscillating-archetype signal
# weakness as a known issue without failing the audit. Floor — higher is
# better; ``_WARN > _PASS`` for floor-direction.
OSCILLATING_ARCHETYPE_PEARSON_WARN = 0.60

# Provenance: engine-fidelity-check.md §3. 23 of 31 metrics show |Δmean|
# > 10%, max +72% (marketing ad_spend). Pass ceiling 0.30 (= 30%) catches
# genuine drift while accepting structural archetype-mix shift. Ceiling —
# lower is better.
MARGINAL_MEAN_REL_PASS = 0.30

# Provenance: engine-fidelity-check.md §3. All 31 metrics show Δstd > +10%;
# median +119%. Pass ceiling 1.50 (= 150%) is intentionally relaxed because
# realized std aggregates within-period sampling variance + trajectory
# variance. Ceiling — lower is better.
MARGINAL_STD_REL_PASS = 1.50

# Looser than the pass threshold. Surfaces marketing's three scale-amplified
# outliers (impressions +2547%, AOV +516%, ad_spend +402%) as flags even
# though pass is already very loose. Ceiling — lower is better;
# ``_OUTLIER > _PASS`` for ceiling-direction.
MARGINAL_STD_REL_OUTLIER = 3.00

# Provenance: engine-fidelity-check.md §2. |Δ vs achieved| max observed 0.48
# (marketing bounce_rate↔conversion_rate). Pass ceiling 0.50 catches genuine
# correlation breakdown while accepting trajectory-shared covariance
# inflation. Ceiling — lower is better.
CORRELATION_DEVIATION_PASS = 0.50

# Stricter than the pass threshold. Surfaces deviation concerns without
# failing. Ceiling — lower is better; ``_WARN < _PASS``.
CORRELATION_DEVIATION_WARN = 0.30

# Provenance: project/state.md §M111 + M112 marketing run (max |Δ| 0.023).
# Pass ceiling 0.05 is the ceiling for *unexpected* delta growth — saas
# observes |Δ| = 0.117 on engagement↔churn_risk under M111 projection,
# which BREACHES this constant intentionally. Audit will surface the
# failure for operator decision (raise to 0.15 to accept saas, or treat
# as known issue). Ceiling — lower is better.
CORRELATION_HIGHAM_DELTA_PASS = 0.05

# Determinism contract — byte-identical CSV output across consecutive runs
# at the same seed. Ceiling 0 means ANY byte difference fails the audit.
DETERMINISM_BYTE_PASS = 0

# Theoretical bound for ``L @ L.T ≈ projected_matrix`` reconstruction. ULP-
# level precision on float64 puts this well below 1e-12 in practice.
# Ceiling — lower is better.
CHOLESKY_RECONSTRUCTION_ULP_PASS = 1e-12


# --- Archetype color mapping -----------------------------------------------

# One color per archetype name across all bundled templates. Keep the dict
# expanding rather than hashing to avoid the same color landing on two
# different archetypes in the same notebook. Anything not in this dict
# falls back to matplotlib's default color cycle.
ARCHETYPE_COLORS: dict[str, str] = {
    # saas
    "rocket_then_cliff": "#d62728",
    "steady_grower": "#2ca02c",
    "zombie_account": "#7f7f7f",
    "slow_death": "#8c564b",
    "seasonal_spiker": "#9467bd",
    "expansion_champion": "#1f77b4",
    # hr
    "fast_riser": "#1f77b4",
    "quiet_quitter": "#7f7f7f",
    "burnout_risk": "#d62728",
    "steady_performer": "#2ca02c",
    # education
    "late_bloomer": "#1f77b4",
    "burnout_trajectory": "#d62728",
    "steady_learner": "#2ca02c",
    # retail
    "steady_growth": "#2ca02c",
    "one_and_done": "#d62728",
    "holiday_surge": "#ff7f0e",
    "bargain_hunter": "#9467bd",
    # marketing
    "high_value_loyal": "#1f77b4",
    "organic_convert": "#2ca02c",
    "paid_acquisition_churn": "#d62728",
    "dormant_reactivation": "#7f7f7f",
    "deal_seeker": "#9467bd",
}


# --- Public helpers --------------------------------------------------------

# Path to the saas template, anchored at the repo root via this file's
# location. Notebooks should not hardcode absolute paths.
_REPO_ROOT = Path(__file__).resolve().parent.parent
SAAS_CONFIG_PATH = _REPO_ROOT / "plotsim" / "configs" / "sample_saas.yaml"


def load_fixed_point() -> tuple[PlotsimConfig, int]:
    """Return the saas fixed-point config + seed for the acceptance notebook.

    The acceptance notebook pins to ``sample_saas.yaml`` at seed 42 so all
    cells reproduce identical numerical output across consecutive runs.
    """
    cfg = load_config(SAAS_CONFIG_PATH)
    return cfg, 42


def setup_plot_style() -> None:
    """Apply the shared matplotlib style. Call once per notebook in §0.

    Imported lazily so the module doesn't pull matplotlib at import time —
    test suites that reach into ``_helpers.py`` for tolerance constants
    don't pay the matplotlib import cost.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["figure.figsize"] = (10, 5)
    mpl.rcParams["figure.dpi"] = 100
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.3
    mpl.rcParams["lines.linewidth"] = 1.5
    plt.rcParams.update(mpl.rcParams)


def manual_rng_replay(
    seed: int,
    n_draws: int,
    distribution: str,
    params: dict,
) -> np.ndarray:
    """Re-instantiate ``np.random.default_rng(seed)`` and consume ``n_draws``.

    Verifies engine RNG-order against an external, single-threaded replay.
    Reproduces ``sample_single_metric`` output (the pre-correlation
    ``independent_draw`` field on ``TraceResult``), NOT the post-copula
    ``correlated_draw``. The Gaussian copula transform in
    ``apply_correlations`` is deterministic given ``(independent, centers,
    cholesky_L)`` and consumes no additional randomness — it's a pure
    function. The §5 RNG-replay assertion in the acceptance notebook
    therefore checks ``independent_draw == manual_rng_replay(...)``,
    not ``correlated_draw``.

    Supported ``distribution`` strings: ``lognorm``, ``gamma``, ``poisson``,
    ``beta``, ``normal``, ``weibull``. ``params`` keys mirror
    ``sample_single_metric``'s expectations: ``s``/``scale`` for lognorm,
    ``shape``/``scale`` for gamma, ``lambda`` for poisson, ``alpha``/``beta``
    for beta, ``loc``/``sigma`` for normal, ``shape``/``scale`` for weibull.
    """
    rng = np.random.default_rng(seed)
    if distribution == "lognorm":
        s = float(params["s"])
        scale = float(params["scale"])
        mean = float(np.log(scale))
        return np.array(
            [float(rng.lognormal(mean=mean, sigma=s)) for _ in range(n_draws)]
        )
    if distribution == "gamma":
        shape = float(params["shape"])
        scale = float(params["scale"])
        return np.array(
            [float(rng.gamma(shape=shape, scale=scale)) for _ in range(n_draws)]
        )
    if distribution == "poisson":
        lam = float(params["lambda"])
        return np.array(
            [float(rng.poisson(lam=lam)) for _ in range(n_draws)]
        )
    if distribution == "beta":
        a = float(params["alpha"])
        b = float(params["beta"])
        return np.array(
            [float(rng.beta(a=a, b=b)) for _ in range(n_draws)]
        )
    if distribution == "normal":
        loc = float(params.get("loc", 0.0))
        sigma = float(params["sigma"])
        return np.array(
            [float(rng.normal(loc=loc, scale=sigma)) for _ in range(n_draws)]
        )
    if distribution == "weibull":
        shape = float(params["shape"])
        scale = float(params.get("scale", 1.0))
        return np.array(
            [float(rng.weibull(a=shape) * scale) for _ in range(n_draws)]
        )
    raise ValueError(
        f"unsupported distribution {distribution!r}; expected one of "
        "lognorm, gamma, poisson, beta, normal, weibull"
    )


def archetype_curve_eval(
    archetype: Archetype, n_periods: int,
) -> np.ndarray:
    """Return the archetype's expected trajectory at ``n_periods`` positions.

    Evaluates the archetype's ``curve_segments`` deterministically — no
    entity-side stochasticity (no inflection shift, no per-entity overrides).
    The output is the ground-truth shape that realized metric series should
    track; used by §4 trajectory shape recovery against
    ``MONOTONIC_ARCHETYPE_PEARSON_PASS`` /
    ``OSCILLATING_ARCHETYPE_PEARSON_PASS``.
    """
    return compute_trajectory(archetype, n_periods, None)


def archetype_color(name: str) -> Optional[str]:
    """Look up the consistent color assigned to an archetype name.

    Returns ``None`` when the archetype isn't in ``ARCHETYPE_COLORS`` so the
    caller can let matplotlib's default color cycle assign one.
    """
    return ARCHETYPE_COLORS.get(name)
