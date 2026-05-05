"""Render the README trajectory-first illustration.

Generates a two-panel figure showing how every metric in a plotsim run
reads from the same archetype-driven trajectory:

  - Top panel: the underlying trajectory position (0 -> 1) over 24
    months for a single "growth" customer.
  - Bottom panel: four metrics on the same x-axis, each normalized to
    [0, 1] for shape comparison. Positive-polarity metrics
    (engagement, mrr) track the trajectory rising; negative-polarity
    metrics (support_tickets, churn_risk) track its inverse.

Output: ``docs/site/assets/trajectory-first.png`` at 1400x800 px.
The README references this asset via raw.githubusercontent.com.

Re-run after any change that affects engine output:
    python examples/render_trajectory_plot.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotsim import create
from plotsim.tables import generate_tables_with_state


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "site" / "assets" / "trajectory-first.png"

# Picked so the curve has clear early/middle/late phases at README width.
# Seed 11 / entity 1 produces the cleanest "all four metrics track the
# trajectory" visual across a 0..50 seed sweep — see the script's git
# history for the criteria. Re-tune if the engine output drifts.
SEED = 11
ENTITY_INDEX = 1


def main() -> None:
    config = create(
        about="SaaS customers",
        unit="customer",
        window=("2023-01", "2024-12", "monthly"),
        metrics=[
            {"name": "engagement", "type": "score", "polarity": "positive"},
            {"name": "mrr", "type": "amount", "polarity": "positive",
             "range": [200, 5000]},
            {"name": "support_tickets", "type": "count", "polarity": "negative"},
            {"name": "churn_risk", "type": "score", "polarity": "negative"},
        ],
        segments=[
            {"name": "growth_co", "count": 3, "archetype": "growth"},
        ],
        seed=SEED,
    )

    rng = np.random.default_rng(config.seed)
    tables, gen_state = generate_tables_with_state(config, rng)

    entity_id = sorted(gen_state.trajectories.keys())[ENTITY_INDEX]
    trajectory = gen_state.trajectories[entity_id]

    fct = tables["fct_customer"]
    customer_col = "customer_id"
    target_customer = sorted(fct[customer_col].unique())[ENTITY_INDEX]
    rows = (
        fct[fct[customer_col] == target_customer]
        .sort_values("date_key")
        .reset_index(drop=True)
    )
    period_labels = [
        f"{str(k)[:4]}-{str(k)[4:6]}" for k in rows["date_key"].tolist()
    ]

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(11, 5.6),
        gridspec_kw={"height_ratios": [1, 2]},
        sharex=True,
    )

    fig.patch.set_facecolor("white")

    # --- Top panel: the trajectory itself --------------------------------
    ax_top.plot(
        period_labels, trajectory,
        color="#1f2a44", linewidth=2.2, marker="o", markersize=4,
    )
    ax_top.fill_between(
        period_labels, 0, trajectory,
        color="#1f2a44", alpha=0.08,
    )
    ax_top.set_ylabel("Trajectory\nposition", fontsize=10)
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.set_yticks([0.0, 0.5, 1.0])
    ax_top.grid(axis="y", linestyle=":", alpha=0.4)
    ax_top.set_title(
        "One trajectory drives every metric",
        fontsize=13, fontweight="semibold", pad=12,
    )

    # --- Bottom panel: four metrics, normalized for shape comparison ----
    metric_colors = {
        "engagement":      ("#2e7dd7", "engagement (positive)"),
        "mrr":             ("#27a59b", "mrr (positive)"),
        "support_tickets": ("#e0833b", "support_tickets (negative)"),
        "churn_risk":      ("#c9466b", "churn_risk (negative)"),
    }
    for metric, (color, label) in metric_colors.items():
        values = rows[metric].to_numpy(dtype=float)
        vmin, vmax = values.min(), values.max()
        normalized = (values - vmin) / (vmax - vmin) if vmax > vmin else values * 0
        ax_bot.plot(
            period_labels, normalized,
            color=color, linewidth=1.8, marker="o", markersize=3.5,
            label=label,
        )

    ax_bot.set_ylabel("Metric value\n(min-max normalized)", fontsize=10)
    ax_bot.set_xlabel("Month", fontsize=10)
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.set_yticks([0.0, 0.5, 1.0])
    ax_bot.grid(axis="y", linestyle=":", alpha=0.4)
    ax_bot.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
    )

    # Sparser x labels — every 3 months
    for ax in (ax_top, ax_bot):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", which="both", length=0)
    visible = [lbl if i % 3 == 0 else "" for i, lbl in enumerate(period_labels)]
    ax_bot.set_xticks(range(len(period_labels)))
    ax_bot.set_xticklabels(visible, rotation=0, fontsize=9)

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=128, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
