"""B2B SaaS customer success — Python-shaped builder template.

This is the ``create(**kwargs)`` mirror of ``saas_template.yaml`` —
both produce identical engine configs given the same seed. Pick
whichever surface fits your workflow:

* ``saas_template.yaml`` for config-as-data fixtures checked into git
* this file for code-shaped configs that compose with regular Python

The new builder dials (``noise``, ``output``, ``locale``,
``seasonality``, custom-coefficient ``connections``) are demonstrated
inline below; comments mark the pieces that match the YAML 1-1.
"""

from plotsim import create

config = create(
    about="B2B SaaS customer success",
    unit="company",
    seed=1729,  # determinism
    noise="perfectly_clean",  # also: slightly_messy, realistic, dirty
    # output={"format": "parquet", "directory": "./out"},  # uncomment for parquet
    # locale=["en_US", "en_GB"],                # multi-locale faker mix
    window=("2023-01", "2024-12", "monthly"),
    seasonality=[
        {"months": [11, 12], "strength": 0.30},  # Q4 lift
        {"months": [6, 7, 8], "strength": -0.10},  # summer dip
    ],
    # ── what we measure ─────────────────────────────────
    metrics=[
        {
            "name": "engagement",
            "label": "Product engagement",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "mrr",
            "label": "Monthly recurring revenue",
            "type": "amount",
            "polarity": "positive",
            "range": [100, 50000],
        },
        {
            "name": "support_tickets",
            "label": "Support ticket volume",
            "type": "count",
            "polarity": "negative",
            "follows": "engagement",
            "delay": 2,
        },
        {
            "name": "feature_adoption",
            "label": "Feature adoption rate",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "churn_risk",
            "label": "Churn risk score",
            "type": "score",
            "polarity": "negative",
        },
        {
            "name": "nps",
            "label": "Net promoter score",
            "type": "index",
            "polarity": "positive",
            "range": [-100, 100],
        },
    ],
    # ── how metrics connect ─────────────────────────────
    # Mix of vocabulary words and explicit numeric coefficients —
    # both forms parse into the same correlation matrix. Numeric
    # form is for cases where you've calibrated r from real data.
    connections=[
        ("engagement", "driven_by", "mrr"),
        ("engagement", "opposes", "churn_risk"),
        ("support_tickets", "related", "churn_risk"),
        ("feature_adoption", 0.42, "mrr"),  # custom coefficient
        ("nps", 0.18, "engagement"),  # custom coefficient
    ],
    # ── who we're simulating ────────────────────────────
    segments=[
        {
            "name": "promising_client",
            "count": 20,
            "archetype": "growth > spike_then_crash > flat @ 8 @ 16",
            "label": "Strong start, lost champion at month 8, went dormant by 16",
            "attributes": {
                "industry": ["Technology", "Finance", "Healthcare"],
                "region": ["US", "EMEA"],
                "tier": "enterprise",
            },
            "baseline": {"mrr": "high", "engagement": "high", "support_tickets": "low"},
        },
        {
            "name": "steady_enterprise",
            "count": 25,
            "archetype": "growth",
            "label": "Reliable accounts, steady climb",
            "attributes": {
                "industry": ["Technology", "Finance"],
                "region": ["US", "APAC"],
                "tier": "enterprise",
            },
            "baseline": {"mrr": "high", "engagement": "high", "support_tickets": "low"},
        },
        {
            "name": "slow_churn",
            "count": 15,
            "archetype": "flat > decline @ 12",
            "label": "Coasted for a year, then quietly faded",
            "attributes": {
                "industry": ["Media", "Hospitality"],
                "region": ["EMEA"],
                "tier": "starter",
            },
            "baseline": {"mrr": "low", "engagement": "low", "support_tickets": "high"},
        },
        {
            "name": "seasonal_accounts",
            "count": 15,
            "archetype": "growth > seasonal @ 6",
            "label": "Ramped up first 6 months, settled into quarterly cycles",
            "attributes": {
                "industry": ["Retail", "Manufacturing"],
                "region": ["US"],
                "tier": "growth",
            },
            "baseline": {"mrr": "mid", "engagement": "mid", "support_tickets": "mid"},
        },
        {
            "name": "dormant",
            "count": 10,
            "archetype": "flat",
            "label": "Signed up, never activated",
            "attributes": {
                "industry": ["Education"],
                "region": ["APAC"],
                "tier": "starter",
            },
            "baseline": {"mrr": "low", "engagement": "low", "support_tickets": "low"},
        },
        {
            "name": "turnaround",
            "count": 10,
            "archetype": "decline > flat > growth @ 6 @ 14",
            "label": "Declining, hit bottom at month 6, turned around at 14",
            "attributes": {
                "industry": ["Finance", "Healthcare"],
                "region": ["US"],
                "tier": "growth",
            },
            "baseline": {"mrr": "mid", "engagement": "mid", "support_tickets": "mid"},
        },
    ],
    # ── lifecycle funnel ────────────────────────────────
    lifecycle={
        "track": "churn_risk",
        "stages": [
            ("onboarding", 0.0),
            ("active", 0.2),
            ("at_risk", 0.5),
            ("churned", 0.8),
        ],
    },
    # ── schema ──────────────────────────────────────────
    dimensions=[
        {
            "name": "dim_date",
            "per": "period",
            "columns": [
                {"name": "date_key", "type": "id"},
                {"name": "date", "type": "date"},
                {"name": "year", "type": "int"},
                {"name": "month", "type": "int"},
                {"name": "quarter", "type": "int"},
            ],
        },
        {
            "name": "dim_company",
            "per": "unit",
            "columns": [
                {"name": "company_id", "type": "id"},
                {"name": "company_name", "type": "faker.company"},
                {"name": "industry", "type": "faker.industry"},
                {"name": "founded_year", "type": "faker.year"},
                {"name": "cohort_size", "type": "segment.count"},
                {
                    "name": "plan_tier",
                    "type": "scd",
                    "tracks": "mrr",
                    "tiers": ["starter", "growth", "enterprise"],
                    "at": [0.4, 0.7],
                },
            ],
        },
        {
            "name": "dim_user",
            "per": "unit",
            "columns": [
                {"name": "user_id", "type": "id"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "user_name", "type": "faker.name"},
                {"name": "role", "type": "static.member"},
            ],
        },
        {
            "name": "dim_plan",
            "reference": True,
            "columns": [
                {"name": "plan_id", "type": "id"},
                {"name": "plan_name", "type": "static.starter"},
                {"name": "monthly_price", "type": "static.99.00"},
            ],
        },
    ],
    facts=[
        {
            "name": "fct_engagement",
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "engagement_score", "type": "metric.engagement"},
                {"name": "feature_adoption", "type": "metric.feature_adoption"},
                {
                    "name": "customer_sentiment",
                    "type": "bucket",
                    "labels": ["at_risk", "lukewarm", "satisfied", "delighted"],
                },
            ],
        },
        {
            "name": "fct_revenue",
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "plan_id", "type": "ref.dim_plan"},
                {"name": "mrr", "type": "metric.mrr"},
            ],
        },
        {
            "name": "fct_support_tickets",
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "ticket_count", "type": "metric.support_tickets"},
                {"name": "churn_risk", "type": "metric.churn_risk"},
                {"name": "nps", "type": "metric.nps"},
            ],
        },
    ],
    events=[
        {
            "name": "evt_login",
            "trigger": "proportional",
            "driver": "engagement",
            "scale": 5,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "user_id", "type": "ref.dim_user"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "event_ts", "type": "timestamp"},
            ],
        },
        {
            "name": "evt_churn",
            "trigger": "threshold",
            "metric": "churn_risk",
            "above": 0.7,
            "for_periods": 3,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "company_id", "type": "ref.dim_company"},
                {"name": "churn_reason", "type": "faker.sentence"},
                {"name": "churn_flag", "type": "flag"},
            ],
        },
    ],
)
