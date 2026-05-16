"""Lakehouse-scale device telemetry — Python-shaped builder template.

This is the ``create(**kwargs)`` mirror of ``lakehouse.yaml`` — both
produce identical engine configs given the same seed. Pick whichever
surface fits your workflow:

* ``lakehouse.yaml`` for config-as-data fixtures checked into git
* this file for code-shaped configs that compose with regular Python

Showcase template for the M7 large-scale generation surface.
Demonstrates the engine's vectorized lane on a daily-grain fact table
that lakehouse / data-warehouse exercises (partitioning, query
optimization, ELT performance) need to hit realistic row counts.

Default shape: 1,500 devices x ~1,004 daily periods ~= 1.5M fact-table
cells - under the 2M soft budget so ``plotsim run`` succeeds without
flags. To raise the cap from Python (no env vars, no CLI flags), pass
``output={"directory": "out", "format": "parquet", "cell_budget":
15_000_000}`` to ``create()`` and bump the segment counts and/or
window. ``cell_budget=0`` disables the soft gate entirely (the 50M-cell
hard ceiling still applies).
"""

from plotsim import create

config = create(
    about="Lakehouse-scale device telemetry",
    unit="device",
    seed=8675309,  # determinism
    window=("2022-01", "2024-09", "daily"),
    # ── what we measure ─────────────────────────────────
    metrics=[
        {
            "name": "cpu_load",
            "label": "CPU load (0-1)",
            "type": "score",
            "polarity": "negative",
            "distribution": "beta",
            "distribution_params": {"alpha": 2.0, "beta": 5.0},
        },
        {
            "name": "memory_pressure",
            "label": "Memory pressure (0-1)",
            "type": "score",
            "polarity": "negative",
            "distribution": "beta",
            "distribution_params": {"alpha": 2.5, "beta": 6.0},
        },
        {
            "name": "disk_throughput_mb",
            "label": "Disk throughput (MB/s)",
            "type": "amount",
            "polarity": "positive",
            "range": [1, 500],
            "distribution": "lognorm",
            "distribution_params": {"s": 0.7},
        },
        {
            "name": "error_rate",
            "label": "Per-period error rate",
            "type": "score",
            "polarity": "negative",
            "distribution": "beta",
            "distribution_params": {"alpha": 2.0, "beta": 30.0},
        },
        {
            "name": "packet_count",
            "label": "Network packets per period",
            "type": "count",
            "polarity": "positive",
            "distribution": "poisson",
        },
    ],
    # ── how metrics connect ─────────────────────────────
    connections=[
        "cpu_load related memory_pressure",
        "error_rate driven_by cpu_load",
        "disk_throughput_mb related packet_count",
    ],
    # ── who we're simulating ────────────────────────────
    # Three fleet archetypes x 500 devices each = 1,500
    # devices total - the lakehouse target shape.
    segments=[
        {
            "name": "stable_fleet",
            "count": 500,
            "archetype": "flat",
            "label": "Mature production fleet - steady baseline load",
        },
        {
            "name": "scaling_fleet",
            "count": 500,
            "archetype": "growth",
            "label": "Newly-deployed fleet - load ramps over the window",
        },
        {
            "name": "degrading_fleet",
            "count": 500,
            "archetype": "decline > spike_then_crash @ 600",
            "label": "Aging hardware - load fades, then a final failure spike",
        },
    ],
)
