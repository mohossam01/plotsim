"""0.6-M23 — Heavy-tailed noise families (Student-t, Laplace).

Acceptance criteria:

  * ``noise_family="student_t"`` with low ``degrees_of_freedom`` produces
    residuals with markedly heavier tails than Gaussian — empirical
    kurtosis significantly above 3, and a KS test against the t(df) reference
    does not reject.
  * ``noise_family="laplace"`` produces residuals consistent with the
    Laplace distribution — KS test against the Laplace reference does not
    reject.
  * ``noise_family="gaussian"`` (default) produces byte-identical RNG draws
    and identical noise output as the pre-M23 code path.
  * Config-time validation: ``noise_family="student_t"`` requires
    ``degrees_of_freedom``; ``degrees_of_freedom`` is rejected when the
    family is not Student-t; ``df < 1`` is rejected.
  * Manifest records ``noise_family`` and ``degrees_of_freedom`` whenever
    the family is non-default (independent of the M22 heteroscedastic flag).
  * Builder ``NoiseInput`` mirrors the engine fields and the interpreter
    forwards them onto ``NoiseConfig`` unchanged.
  * The two new families compose with M22 ``scale_with_trajectory`` — both
    branches honor the position-scaled lane when enabled.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats as sp_stats

from plotsim import create, generate_tables
from plotsim.builder.input import NoiseInput
from plotsim.config import NoiseConfig
from plotsim.manifest import build_manifest
from plotsim.metrics import _apply_noise_batch, apply_noise


# --- Config-time validation -------------------------------------------------


def test_student_t_requires_degrees_of_freedom():
    """A config that names Student-t without supplying ``degrees_of_freedom``
    must raise at construction time with a clear message — silent fallback
    to a default df would change the realized tail thickness invisibly."""
    with pytest.raises(ValidationError) as exc:
        NoiseConfig(gaussian_sigma=0.10, noise_family="student_t")
    assert "degrees_of_freedom" in str(exc.value)


def test_degrees_of_freedom_rejected_when_family_not_student_t():
    """``degrees_of_freedom`` is a Student-t-only knob. Setting it on the
    gaussian or laplace family must raise — keeps the field from
    silently doing nothing and leaving a confusing audit trail."""
    with pytest.raises(ValidationError) as exc:
        NoiseConfig(gaussian_sigma=0.10, degrees_of_freedom=5.0)
    assert "degrees_of_freedom" in str(exc.value)

    with pytest.raises(ValidationError) as exc2:
        NoiseConfig(gaussian_sigma=0.10, noise_family="laplace", degrees_of_freedom=5.0)
    assert "degrees_of_freedom" in str(exc2.value)


def test_degrees_of_freedom_below_one_rejected():
    """df < 1 is degenerate (Student-t has no defined mean below df=1).
    Pydantic ``ge=1.0`` rejects the value with a locatable error."""
    with pytest.raises(ValidationError):
        NoiseConfig(gaussian_sigma=0.10, noise_family="student_t", degrees_of_freedom=0.5)


def test_laplace_does_not_accept_extra_params():
    """Laplace has no extra parameter beyond the shared scale. A config
    that sets only family=laplace + gaussian_sigma must construct cleanly."""
    cfg = NoiseConfig(gaussian_sigma=0.10, noise_family="laplace")
    assert cfg.noise_family == "laplace"
    assert cfg.degrees_of_freedom is None


# --- apply_noise byte-identity for the gaussian default ---------------------


def test_apply_noise_default_family_byte_identical_to_pre_m23():
    """The default ``noise_family="gaussian"`` must consume RNG bytes
    identically to the historical lane. Verified by comparing against a
    bare ``rng.normal`` call on the same seed, mirroring the M22
    byte-identity test pattern."""
    noise = NoiseConfig(gaussian_sigma=0.10)
    rng_engine = np.random.default_rng(7)
    out = apply_noise(20.0, noise, rng_engine)

    rng_ref = np.random.default_rng(7)
    expected = 20.0 + float(rng_ref.normal(loc=0.0, scale=0.10 * 20.0))
    assert out == expected


def test_apply_noise_batch_default_family_byte_identical_to_pre_m23():
    """Vectorized path: default family must match the historical batch
    lane on identical input + same RNG seed."""
    noise = NoiseConfig(gaussian_sigma=0.10)
    values = np.full(64, 30.0, dtype=np.float64)
    out = _apply_noise_batch(values, noise, np.random.default_rng(99))

    rng_ref = np.random.default_rng(99)
    mag = np.where(values != 0.0, np.abs(values), 1.0)
    expected = values + rng_ref.normal(loc=0.0, scale=0.10 * mag, size=64)
    np.testing.assert_array_equal(out, expected)


# --- Student-t draw shape ---------------------------------------------------


def test_apply_noise_student_t_uses_standard_t_draw():
    """The scalar path with ``noise_family="student_t"`` must use
    ``rng.standard_t(df)`` and multiply by the resolved scale — verified
    against a hand-computed reference using the same RNG seed."""
    noise = NoiseConfig(gaussian_sigma=0.10, noise_family="student_t", degrees_of_freedom=3.0)
    rng_engine = np.random.default_rng(13)
    out = apply_noise(50.0, noise, rng_engine)

    rng_ref = np.random.default_rng(13)
    expected = 50.0 + float(rng_ref.standard_t(3.0)) * (0.10 * 50.0)
    assert out == expected


def test_apply_noise_batch_student_t_kurtosis_significantly_above_three():
    """Empirical kurtosis of t(3) residuals must blow past Gaussian's 3.0.
    At 5_000 samples the population kurtosis (= ∞ for t(3), but the
    realized sample kurtosis lands well above any reasonable Gaussian
    band). Assertion threshold ≥ 6.0 is conservative — the theoretical
    excess kurtosis is unbounded for df=3."""
    noise = NoiseConfig(gaussian_sigma=1.0, noise_family="student_t", degrees_of_freedom=3.0)
    values = np.zeros(5_000, dtype=np.float64)
    # Use value=0 so the "fallback mag=1" lane is exercised → scale ≈ 1.0
    # for every cell. Residuals = the raw t(3) draws.
    out = _apply_noise_batch(values, noise, np.random.default_rng(2026))
    sample_kurt = float(sp_stats.kurtosis(out, fisher=False))  # Pearson def: Gaussian = 3
    assert sample_kurt > 6.0, f"kurtosis {sample_kurt} not heavy-tailed enough"


def test_apply_noise_batch_student_t_ks_does_not_reject():
    """KS test of t(df) residuals against the scipy ``t(df)`` reference
    must not reject at p > 0.01. Use a moderate df (5) for a less
    pathological tail."""
    df = 5.0
    noise = NoiseConfig(gaussian_sigma=1.0, noise_family="student_t", degrees_of_freedom=df)
    values = np.zeros(5_000, dtype=np.float64)
    out = _apply_noise_batch(values, noise, np.random.default_rng(31))
    # Residual is the draw itself (since value=0 → scale=1.0 → noise = t(df)).
    ks_stat, p_value = sp_stats.kstest(out, "t", args=(df,))
    assert p_value > 0.01, f"t({df}) KS rejected: p={p_value}, stat={ks_stat}"


# --- Laplace draw shape -----------------------------------------------------


def test_apply_noise_laplace_uses_rng_laplace():
    """The scalar path with ``noise_family="laplace"`` must use
    ``rng.laplace(loc=0.0, scale=scale)``, verified against a same-seed
    reference."""
    noise = NoiseConfig(gaussian_sigma=0.10, noise_family="laplace")
    rng_engine = np.random.default_rng(17)
    out = apply_noise(50.0, noise, rng_engine)

    rng_ref = np.random.default_rng(17)
    expected = 50.0 + float(rng_ref.laplace(loc=0.0, scale=0.10 * 50.0))
    assert out == expected


def test_apply_noise_batch_laplace_ks_does_not_reject():
    """KS test of Laplace residuals against the scipy ``laplace(scale=1)``
    reference must not reject at p > 0.01."""
    noise = NoiseConfig(gaussian_sigma=1.0, noise_family="laplace")
    values = np.zeros(5_000, dtype=np.float64)
    out = _apply_noise_batch(values, noise, np.random.default_rng(53))
    ks_stat, p_value = sp_stats.kstest(out, "laplace", args=(0.0, 1.0))
    assert p_value > 0.01, f"Laplace KS rejected: p={p_value}, stat={ks_stat}"


# --- Composition with M22 heteroscedastic flag ------------------------------


def test_student_t_composes_with_scale_with_trajectory():
    """Heavy-tailed family + heteroscedastic amplitude must compose
    orthogonally: the resolved scale is ``sigma * position`` (not
    ``sigma * abs(value)``) and the family is Student-t."""
    noise = NoiseConfig(
        gaussian_sigma=0.10,
        noise_family="student_t",
        degrees_of_freedom=4.0,
        scale_with_trajectory=True,
    )
    rng_engine = np.random.default_rng(23)
    out = apply_noise(50.0, noise, rng_engine, trajectory_position=0.7)

    rng_ref = np.random.default_rng(23)
    expected = 50.0 + float(rng_ref.standard_t(4.0)) * (0.10 * 0.7)
    assert out == expected


def test_laplace_composes_with_scale_with_trajectory():
    """Same orthogonality check for the Laplace family."""
    noise = NoiseConfig(
        gaussian_sigma=0.10,
        noise_family="laplace",
        scale_with_trajectory=True,
    )
    rng_engine = np.random.default_rng(29)
    out = apply_noise(50.0, noise, rng_engine, trajectory_position=0.4)

    rng_ref = np.random.default_rng(29)
    expected = 50.0 + float(rng_ref.laplace(loc=0.0, scale=0.10 * 0.4))
    assert out == expected


def test_student_t_position_zero_yields_zero_noise_under_heteroscedastic():
    """Under the heteroscedastic lane, position=0 collapses the scale to
    zero — a Student-t draw multiplied by zero is exactly zero, so the
    value passes through unchanged."""
    noise = NoiseConfig(
        gaussian_sigma=0.50,
        noise_family="student_t",
        degrees_of_freedom=3.0,
        scale_with_trajectory=True,
    )
    out = apply_noise(
        42.0,
        noise,
        np.random.default_rng(0),
        trajectory_position=0.0,
    )
    assert out == 42.0


# --- Builder propagation ----------------------------------------------------


def test_builder_noise_input_defaults_match_engine_defaults():
    n = NoiseInput(gaussian_sigma=0.05)
    assert n.noise_family == "gaussian"
    assert n.degrees_of_freedom is None


def test_builder_interpreter_propagates_student_t_to_engine_config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(
            about="student-t propagation",
            unit="company",
            window=("2024-01", "2024-12"),
            metrics=[
                {"name": "engagement", "type": "score", "polarity": "positive"},
            ],
            segments=[
                {"name": "alpha", "count": 5, "archetype": "growth"},
            ],
            noise={
                "gaussian_sigma": 0.05,
                "noise_family": "student_t",
                "degrees_of_freedom": 4.0,
            },
        )
    assert cfg.noise is not None
    assert cfg.noise.noise_family == "student_t"
    assert cfg.noise.degrees_of_freedom == pytest.approx(4.0)


def test_builder_interpreter_propagates_laplace_to_engine_config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(
            about="laplace propagation",
            unit="company",
            window=("2024-01", "2024-12"),
            metrics=[
                {"name": "engagement", "type": "score", "polarity": "positive"},
            ],
            segments=[
                {"name": "alpha", "count": 5, "archetype": "growth"},
            ],
            noise={"gaussian_sigma": 0.05, "noise_family": "laplace"},
        )
    assert cfg.noise is not None
    assert cfg.noise.noise_family == "laplace"
    assert cfg.noise.degrees_of_freedom is None


# --- Manifest ---------------------------------------------------------------


def _build_small_config(**noise_overrides):
    noise_payload = {"gaussian_sigma": 0.05, **noise_overrides}
    return create(
        about="manifest heavy-tail check",
        unit="company",
        window=("2024-01", "2024-12"),
        metrics=[
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {"name": "mrr", "type": "amount", "polarity": "positive", "range": [100, 50000]},
        ],
        segments=[
            {"name": "growth", "count": 5, "archetype": "growth"},
            {"name": "decline", "count": 5, "archetype": "decline"},
        ],
        noise=noise_payload,
    )


def _generate_and_manifest(cfg, seed: int = 0):
    rng = np.random.default_rng(seed)
    tables = generate_tables(cfg, rng)
    from plotsim.trajectory import compute_all_trajectories

    n_periods = len(tables["dim_date"])
    trajectories = compute_all_trajectories(cfg, n_periods)
    manifest = build_manifest(cfg, trajectories, tables)
    return tables, manifest


def test_manifest_records_student_t_family_and_df():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _build_small_config(noise_family="student_t", degrees_of_freedom=4.0)
        _tables, manifest = _generate_and_manifest(cfg)

    assert manifest.noise_config is not None
    assert manifest.noise_config.noise_family == "student_t"
    assert manifest.noise_config.degrees_of_freedom == pytest.approx(4.0)
    assert manifest.noise_config.scale_with_trajectory is False


def test_manifest_records_laplace_family():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _build_small_config(noise_family="laplace")
        _tables, manifest = _generate_and_manifest(cfg)

    assert manifest.noise_config is not None
    assert manifest.noise_config.noise_family == "laplace"
    assert manifest.noise_config.degrees_of_freedom is None


def test_manifest_omits_noise_config_when_gaussian_default_amplitude():
    """Default gaussian family + default amplitude lane → no noise_config
    record. Preserves the byte-equivalence M22 established for the
    historical lane."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _build_small_config()  # default family, default amplitude
        _tables, manifest = _generate_and_manifest(cfg)

    assert manifest.noise_config is None


def test_manifest_records_when_family_default_but_heteroscedastic_on():
    """Pre-existing M22 contract still holds: heteroscedastic lane emits
    the record even when the family is the default gaussian."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _build_small_config(scale_with_trajectory=True)
        _tables, manifest = _generate_and_manifest(cfg)

    assert manifest.noise_config is not None
    assert manifest.noise_config.noise_family == "gaussian"
    assert manifest.noise_config.scale_with_trajectory is True


# --- End-to-end engine run with the new families ---------------------------


def test_engine_run_student_t_produces_distinct_output_from_gaussian():
    """A run with ``noise_family="student_t"`` at small df must yield a
    materially different fact table than the same config under gaussian
    noise at the same seed — confirms the dispatch reaches the engine
    output, not just the noise helpers in isolation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg_gauss = _build_small_config()
        cfg_t = _build_small_config(noise_family="student_t", degrees_of_freedom=2.5)

    tables_g = generate_tables(cfg_gauss, np.random.default_rng(42))
    tables_t = generate_tables(cfg_t, np.random.default_rng(42))

    # Look at any fact table the build produced and compare a numeric column.
    fact_name = next(name for name, df in tables_g.items() if name.startswith("fct_"))
    df_g = tables_g[fact_name]
    df_t = tables_t[fact_name]
    numeric_cols = [c for c in df_g.columns if df_g[c].dtype.kind == "f"]
    assert numeric_cols, "expected at least one float metric column"
    # At least one column must differ — the noise lane was actually
    # exercised end-to-end.
    diverged = False
    for col in numeric_cols:
        if not np.allclose(df_g[col].to_numpy(), df_t[col].to_numpy(), equal_nan=True):
            diverged = True
            break
    assert diverged, "Student-t run produced identical output to Gaussian run"


def test_engine_run_default_family_byte_identical_to_pre_m23():
    """End-to-end check: a config that doesn't mention the new fields
    produces byte-identical fact tables to one that explicitly sets
    ``noise_family="gaussian"`` — proves the M23 code paths are no-ops on
    the default lane."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg_implicit = _build_small_config()
        cfg_explicit = _build_small_config(noise_family="gaussian")

    tables_a = generate_tables(cfg_implicit, np.random.default_rng(42))
    tables_b = generate_tables(cfg_explicit, np.random.default_rng(42))

    assert set(tables_a.keys()) == set(tables_b.keys())
    for name, df_a in tables_a.items():
        df_b = tables_b[name]
        for col in df_a.columns:
            np.testing.assert_array_equal(
                df_a[col].to_numpy(),
                df_b[col].to_numpy(),
                err_msg=f"column {col!r} in table {name!r} diverged on default-family lane",
            )
