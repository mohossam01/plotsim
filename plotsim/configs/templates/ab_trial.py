"""SaaS trial → conversion A/B test — Python-shaped builder template.

This is the ``create(**kwargs)`` mirror of ``ab_trial.yaml`` — both
produce identical engine configs given the same seed. Pick whichever
surface fits your workflow:

* ``ab_trial.yaml`` for config-as-data fixtures checked into git
* this file for code-shaped configs that compose with regular Python

Showcase template for the full M8 feature set:

  * Cold-start entities (mission 0.6-M8a) — late-arriving cohorts
    have NaN-filled trajectory prefix and zero fact rows pre-arrival.
  * Segment proportion drift (mission 0.6-M8b) — ``arrival``
    distributions shape how each cohort arrives across the time window.
  * Treatment / control (mission 0.6-M8c) — half of the trial users
    receive a "new_onboarding" treatment that lifts engagement by
    0.6 log-odds starting at period 6.

Domain narrative: a SaaS company runs a rollout experiment. Existing
customers ("legacy") are present from period 0 and receive no
treatment. New trial users arrive across the year via two channels:
an organic ramp (linear, back-loaded) and a paid-ad pulse (step at
periods 0 and 6). At period 6 the product team rolls out a redesigned
onboarding flow to half of the trial users; the other half stay on
the original. Engagement lift is statistically recoverable from the
generated data via difference-in-means on the trial cohort post
period 6.
"""

from plotsim import create
from plotsim.builder.input import (
    LinearArrival,
    StepArrival,
    StepArrivalBlock,
    TreatmentConfig,
)


config = create(
    about="SaaS trial-conversion A/B test (cold-start + drift + treatment)",
    unit="customer",
    seed=2026,
    window=("2024-01", "2024-12", "monthly"),
    metrics=[
        {
            "name": "engagement",
            "label": "Engagement score (DAU/MAU proxy)",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "trial_minutes",
            "label": "Minutes spent in product per period",
            "type": "amount",
            "polarity": "positive",
            "range": [0, 600],
        },
        {
            "name": "feature_adoption",
            "label": "Feature-adoption rate",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "support_tickets",
            "label": "Support tickets opened per period",
            "type": "count",
            "polarity": "negative",
        },
    ],
    segments=[
        # Existing customers — on-platform from period 0, no
        # arrival distribution, no treatment.
        {
            "name": "legacy",
            "count": 30,
            "archetype": "flat",
            "label": "Existing customers (no experiment exposure)",
        },
        # Organic trial signups — back-loaded linear arrival.
        # Half get the new onboarding flow at period 6.
        {
            "name": "trial_organic",
            "count": 60,
            "archetype": "growth",
            "label": "Organic trial signups",
            "arrival": LinearArrival(
                kind="linear",
                start=0,
                end=10,
                direction="increasing",
            ),
            "treatment": TreatmentConfig(
                fraction=0.5,
                lift_log_odds=0.6,
                start_period=6,
                treatment_label="new_onboarding",
                control_label="original_onboarding",
            ),
        },
        # Paid-ad trial signups — two-block step arrival.
        # Same 50/50 treatment split as organic.
        {
            "name": "trial_paid",
            "count": 40,
            "archetype": "growth",
            "label": "Paid-acquisition trial signups",
            "arrival": StepArrival(
                kind="step",
                blocks=[
                    StepArrivalBlock(period=0, fraction=0.5),
                    StepArrivalBlock(period=6, fraction=0.5),
                ],
            ),
            "treatment": TreatmentConfig(
                fraction=0.5,
                lift_log_odds=0.6,
                start_period=6,
                treatment_label="new_onboarding",
                control_label="original_onboarding",
            ),
        },
    ],
)
