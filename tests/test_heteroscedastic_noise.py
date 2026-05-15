"""0.6-M22 — Heteroscedastic noise.

Acceptance criteria:

  * Variance of realized metric values scales with trajectory position when
    enabled (variance ratio > 1 between high-position and low-position
    cohorts).
  * Default off (``NoiseConfig.scale_with_trajectory=False``) produces
    byte-identical output to the historical ``apply_noise`` /
    ``_apply_noise_batch`` lanes — confirmed by exercising the same RNG
    seeds and asserting identity against the existing magnitude-scaled path.
  * Manifest records the heteroscedastic config when enabled and omits it
    when off.
  * Builder ``NoiseInput.scale_with_trajectory`` propagates through the
    interpreter onto ``NoiseConfig`` unchanged.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from plotsim import create, generate_tables
from plotsim.builder.input import NoiseInput
from plotsim.config import NoiseConfig
from plotsim.manifest import build_manifest
from plotsim.metrics import _apply_noise_batch, apply_noise


# --- Direct apply_noise contract --------------------------------------------


def test_apply_noise_default_off_byte_identical_three_arg_call():
    """The legacy 3-arg ``apply_noise(value, noise, rng)`` shape must keep
    producing the historical magnitude-scaled result when the new field
    is absent (default ``False``)."""
    noise = NoiseConfig(gaussian_sigma=0.10)
    legacy = apply_noise(50.0, noise, np.random.default_rng(0))
    explicit_false = apply_noise(
        50.0,
        noise,
        np.random.default_rng(0),
        trajectory_position=0.5,  # value irrelevant under scale_with_trajectory=False
    )
    assert legacy == explicit_false


def test_apply_noise_default_off_rng_consumption_unchanged():
    """A default-off config must consume RNG bytes identically to before — i.e.
    one ``rng.normal`` call when ``gaussian_sigma > 0``. Verified by checking
    that two ``apply_noise`` calls in sequence draw the same pair of values
    a bare ``rng.normal`` would on the same seed (scaled by abs(value))."""
    noise = NoiseConfig(gaussian_sigma=0.10)
    rng_engine = np.random.default_rng(7)
    out_a = apply_noise(20.0, noise, rng_engine)
    out_b = apply_noise(20.0, noise, rng_engine)

    rng_ref = np.random.default_rng(7)
    expected_a = 20.0 + float(rng_ref.normal(loc=0.0, scale=0.10 * 20.0))
    expected_b = 20.0 + float(rng_ref.normal(loc=0.0, scale=0.10 * 20.0))
    assert out_a == expected_a
    assert out_b == expected_b


def test_apply_noise_heteroscedastic_uses_position_not_magnitude():
    """With the flag on, the gaussian scale is ``sigma * trajectory_position``
    regardless of the value's magnitude. Compared against a hand-computed
    reference using the same RNG seed."""
    noise = NoiseConfig(gaussian_sigma=0.10, scale_with_trajectory=True)
    rng_engine = np.random.default_rng(13)
    out = apply_noise(50.0, noise, rng_engine, trajectory_position=0.8)

    rng_ref = np.random.default_rng(13)
    expected = 50.0 + float(rng_ref.normal(loc=0.0, scale=0.10 * 0.8))
    assert out == expected


def test_apply_noise_position_zero_yields_zero_gaussian():
    """``trajectory_position=0`` under the flag must produce zero gaussian
    contribution — value passes through the gaussian branch unchanged."""
    noise = NoiseConfig(gaussian_sigma=0.50, scale_with_trajectory=True)
    out = apply_noise(
        42.0,
        noise,
        np.random.default_rng(0),
        trajectory_position=0.0,
    )
    assert out == 42.0


def test_apply_noise_flag_on_position_missing_falls_back():
    """If the flag is on but the caller forgot to thread the position
    through (``trajectory_position=None``), apply_noise should fall back
    to the legacy magnitude-scaled lane rather than failing — keeps the
    monkey-patched ``recording_apply_noise`` wrapper safe even if a
    third-party caller never adopts the new keyword."""
    noise = NoiseConfig(gaussian_sigma=0.10, scale_with_trajectory=True)
    legacy = apply_noise(50.0, NoiseConfig(gaussian_sigma=0.10), np.random.default_rng(0))
    fallback = apply_noise(50.0, noise, np.random.default_rng(0), trajectory_position=None)
    assert legacy == fallback


# --- _apply_noise_batch contract --------------------------------------------


def test_apply_noise_batch_default_off_byte_identical():
    """Vectorized path: default-off must match the historical magnitude-
    scaled batch lane on identical input + same RNG seed."""
    noise = NoiseConfig(gaussian_sigma=0.10)
    values = np.full(64, 30.0, dtype=np.float64)
    legacy = _apply_noise_batch(values, noise, np.random.default_rng(99))
    explicit = _apply_noise_batch(
        values,
        noise,
        np.random.default_rng(99),
        trajectory_position=0.5,
    )
    np.testing.assert_array_equal(legacy, explicit)


def test_apply_noise_batch_variance_ratio_high_vs_low_position():
    """The load-bearing acceptance test. ``_apply_noise_batch`` under the
    heteroscedastic lane must produce a strictly larger sample variance at
    a high trajectory position than at a low one, for the same value vector
    and same gaussian_sigma — variance ratio > 1.

    Uses 10_000 cells per cohort to keep the empirical ratio tightly
    centered on the theoretical (0.9/0.1)**2 = 81; the assertion is loose
    (> 1) so the test stays robust to RNG seed fluctuations."""
    noise = NoiseConfig(gaussian_sigma=0.20, scale_with_trajectory=True)
    n = 10_000
    values = np.full(n, 100.0, dtype=np.float64)

    high = _apply_noise_batch(values, noise, np.random.default_rng(1), trajectory_position=0.9)
    low = _apply_noise_batch(values, noise, np.random.default_rng(2), trajectory_position=0.1)

    var_high = float(np.var(high))
    var_low = float(np.var(low))
    assert var_high > var_low
    # Theoretical ratio = (0.9 / 0.1)**2 = 81; allow generous slack.
    assert var_high / var_low > 10.0


def test_apply_noise_batch_position_zero_yields_zero_gaussian():
    """Position-zero with non-zero sigma + flag-on collapses gaussian noise
    to zero across the entire batch."""
    noise = NoiseConfig(gaussian_sigma=0.50, scale_with_trajectory=True)
    values = np.array([1.0, 5.0, 100.0, -7.0], dtype=np.float64)
    out = _apply_noise_batch(values, noise, np.random.default_rng(0), trajectory_position=0.0)
    np.testing.assert_array_equal(out, values)


# --- Builder propagation ----------------------------------------------------


def test_builder_noise_input_default_off():
    """The new builder field defaults to False so existing user inputs
    construct NoiseConfigs with the historical behavior."""
    n = NoiseInput(gaussian_sigma=0.05)
    assert n.scale_with_trajectory is False


def test_builder_interpreter_propagates_flag_to_engine_config():
    """``NoiseInput.scale_with_trajectory`` must land on the engine
    ``NoiseConfig`` after ``interpret`` runs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(
            about="heteroscedastic propagation",
            unit="company",
            window=("2024-01", "2024-12"),
            metrics=[
                {"name": "engagement", "type": "score", "polarity": "positive"},
            ],
            segments=[
                {"name": "alpha", "count": 5, "archetype": "growth"},
            ],
            noise={"gaussian_sigma": 0.05, "scale_with_trajectory": True},
        )
    assert cfg.noise is not None
    assert cfg.noise.scale_with_trajectory is True
    assert cfg.noise.gaussian_sigma == pytest.approx(0.05)


def test_builder_interpreter_default_off_when_flag_omitted():
    """Omitting ``scale_with_trajectory`` keeps the engine-side default
    False — no surprise opt-in from preset shorthand either."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = create(
            about="default off",
            unit="company",
            window=("2024-01", "2024-12"),
            metrics=[
                {"name": "engagement", "type": "score", "polarity": "positive"},
            ],
            segments=[
                {"name": "alpha", "count": 5, "archetype": "growth"},
            ],
            noise="realistic",
        )
    assert cfg.noise is not None
    assert cfg.noise.scale_with_trajectory is False


# --- Manifest ---------------------------------------------------------------


def _build_small_config(*, scale_with_trajectory: bool):
    return create(
        about=f"manifest noise check (flag={scale_with_trajectory})",
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
        noise={
            "gaussian_sigma": 0.05,
            "scale_with_trajectory": scale_with_trajectory,
        },
    )


def _generate_and_manifest(cfg, seed: int = 0):
    rng = np.random.default_rng(seed)
    tables = generate_tables(cfg, rng)
    # Trajectories aren't returned by generate_tables; reconstruct them for
    # the manifest builder via the same path build_manifest expects.
    from plotsim.trajectory import compute_all_trajectories

    n_periods = len(tables["dim_date"])
    trajectories = compute_all_trajectories(cfg, n_periods)
    manifest = build_manifest(cfg, trajectories, tables)
    return tables, manifest


def test_manifest_records_noise_config_when_enabled():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _build_small_config(scale_with_trajectory=True)
        _tables, manifest = _generate_and_manifest(cfg)

    assert manifest.noise_config is not None
    assert manifest.noise_config.scale_with_trajectory is True
    assert manifest.noise_config.gaussian_sigma == pytest.approx(0.05)
    assert manifest.noise_config.outlier_rate == pytest.approx(0.0)
    assert manifest.noise_config.mcar_rate == pytest.approx(0.0)


def test_manifest_omits_noise_config_when_off():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _build_small_config(scale_with_trajectory=False)
        _tables, manifest = _generate_and_manifest(cfg)

    assert manifest.noise_config is None


def test_manifest_schema_version_pins_1_7():
    """0.6-M22 bumped the manifest schema version 1.6 → 1.7. The
    test_schema_version_bumped_to_1_7 test in tests/test_manifest.py is
    the authoritative pin; this assertion is a load-bearing reminder that
    M22 owns the bump (so a future mission that adds a manifest field
    knows to bump again)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _build_small_config(scale_with_trajectory=True)
        _tables, manifest = _generate_and_manifest(cfg)
    assert manifest.schema_version == "1.7"


# --- End-to-end byte-identity for default-off engine path -------------------


def test_engine_run_default_off_byte_identical_to_legacy_lane():
    """A full engine run with ``scale_with_trajectory=False`` (default)
    must produce exactly the same fact table as the same config
    constructed without ever mentioning the new field — proves the
    M22 code paths are no-ops on the default lane.

    Implemented by comparing two engine runs at the same seed: one with
    ``NoiseConfig(...)`` (no new field), one with
    ``NoiseConfig(..., scale_with_trajectory=False)``. Pydantic frozen
    config + identical RNG seed + identical field values must yield
    byte-identical DataFrames."""
    from plotsim.config import PlotsimConfig

    base_path = Path(__file__).resolve().parent.parent / "plotsim" / "configs" / "sample_saas.yaml"
    if not base_path.exists():
        pytest.skip(f"sample config not found at {base_path}")

    from plotsim import load_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg_a = load_config(base_path)

    # Reload the same YAML and replace its NoiseConfig with one that
    # explicitly sets the M22 field to its default value. The resulting
    # config must be functionally identical.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg_b_loaded = load_config(base_path)

    if cfg_a.noise is None:
        # No noise in the sample → can't exercise the noise path. Pick a
        # cfg with explicit noise instead.
        pytest.skip("sample config has no noise block; the byte-identity check is vacuous")

    cfg_b = cfg_b_loaded.model_copy(
        update={
            "noise": NoiseConfig(
                gaussian_sigma=cfg_a.noise.gaussian_sigma,
                outlier_rate=cfg_a.noise.outlier_rate,
                mcar_rate=cfg_a.noise.mcar_rate,
                scale_with_trajectory=False,
            )
        }
    )

    assert isinstance(cfg_a, PlotsimConfig)
    assert isinstance(cfg_b, PlotsimConfig)

    tables_a = generate_tables(cfg_a, np.random.default_rng(42))
    tables_b = generate_tables(cfg_b, np.random.default_rng(42))

    assert set(tables_a.keys()) == set(tables_b.keys())
    for name, df_a in tables_a.items():
        df_b = tables_b[name]
        # Per-column equality keeps the failure message diagnostic.
        for col in df_a.columns:
            np.testing.assert_array_equal(
                df_a[col].to_numpy(),
                df_b[col].to_numpy(),
                err_msg=f"column {col!r} in table {name!r} diverged on default-off lane",
            )
