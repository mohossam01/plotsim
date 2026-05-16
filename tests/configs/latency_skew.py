"""API service latency monitoring — Python-shaped builder template.

This is the ``create(**kwargs)`` mirror of ``latency_skew.yaml`` —
both produce identical engine configs given the same seed. Pick
whichever surface fits your workflow:

* ``latency_skew.yaml`` for config-as-data fixtures checked into git
* this file for code-shaped configs that compose with regular Python

Showcase template for explicit per-metric distributions (mission
0.6-M6). Every metric below pins its ``distribution`` +
``distribution_params`` instead of letting the interpreter auto-pick
from ``type`` + ``range``. See ``metrics-and-connections.md`` for the
six-family table and precedence rules.
"""

from plotsim import create

config = create(
    about="API service latency monitoring",
    unit="service",
    seed=2026,  # determinism
    window=("2024-01", "2024-12", "monthly"),
    # ── what we measure ─────────────────────────────────
    # One metric per family — gamma / weibull / beta / normal /
    # lognorm / poisson — to exercise the explicit-distribution
    # short-circuit on every shape the engine supports.
    metrics=[
        {
            "name": "p50_latency_ms",
            "label": "Median request latency (ms)",
            "type": "amount",
            "polarity": "negative",
            "range": [10, 800],
            "distribution": "gamma",
            "distribution_params": {"shape": 4.0},
        },
        {
            "name": "p99_latency_ms",
            "label": "99th-percentile latency (ms)",
            "type": "amount",
            "polarity": "negative",
            "range": [50, 5000],
            "distribution": "weibull",
            "distribution_params": {"shape": 1.5},
        },
        {
            "name": "error_rate",
            "label": "Request error rate",
            "type": "score",
            "polarity": "negative",
            "distribution": "beta",
            "distribution_params": {"alpha": 2.0, "beta": 25.0},
        },
        {
            "name": "cpu_utilization",
            "label": "CPU utilization",
            "type": "score",
            "polarity": "negative",
            "distribution": "normal",
            "distribution_params": {"sigma": 0.08},
        },
        {
            "name": "bytes_per_request",
            "label": "Bytes transferred per request",
            "type": "amount",
            "polarity": "positive",
            "range": [500, 500000],
            "distribution": "lognorm",
            "distribution_params": {"s": 0.85},
        },
        {
            "name": "incident_count",
            "label": "Operational incidents per period",
            "type": "count",
            "polarity": "negative",
            "distribution": "poisson",
        },
    ],
    # ── who we're simulating ────────────────────────────
    segments=[
        {
            "name": "hot_path",
            "count": 12,
            "archetype": "growth",
            "label": "User-facing endpoints — load grows over the year",
        },
        {
            "name": "batch_path",
            "count": 8,
            "archetype": "spike_then_crash",
            "label": "Nightly batch jobs — periodic large spikes",
        },
        {
            "name": "legacy_path",
            "count": 6,
            "archetype": "decline > flat @ 6",
            "label": "Deprecated services — usage tapers, then idle",
        },
    ],
)
