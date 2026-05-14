"""Retail / e-commerce customer analytics — Python builder template.

Mirror of ``retail_template.yaml``. Demonstrates:

* multi-locale faker (``locale=["en_US", "en_GB", "fr_FR"]``)
* Q4 holiday-shopping seasonality
* SCD2 ``customer_tier`` tracking ``loyalty_score``
* threshold event with ``below`` (churn fires when score crashes)
"""

from plotsim import create

config = create(
    about="Retail customer purchase and loyalty behavior",
    unit="customer",
    seed=90210,
    noise="realistic",
    # output={"format": "parquet", "directory": "./out"},  # uncomment if pyarrow installed
    locale=["en_US", "en_GB", "fr_FR"],
    window=("2023-01", "2024-12", "monthly"),
    seasonality=[
        {"months": [11, 12], "strength": 0.45},
        {"months": [7, 8], "strength": -0.15},
    ],
    metrics=[
        {
            "name": "sessions",
            "label": "Monthly site sessions",
            "type": "count",
            "polarity": "positive",
        },
        {
            "name": "cart_value",
            "label": "Average cart value",
            "type": "amount",
            "polarity": "positive",
            "range": [10, 2000],
        },
        {
            "name": "conversion_rate",
            "label": "Session-to-purchase conversion",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "return_rate",
            "label": "Purchase return rate",
            "type": "score",
            "polarity": "negative",
        },
        {
            "name": "loyalty_score",
            "label": "Customer loyalty index",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "repeat_purchase_rate",
            "label": "Repeat purchase rate",
            "type": "score",
            "polarity": "positive",
            "follows": "loyalty_score",
            "delay": 1,
        },
        {
            "name": "nps",
            "label": "Net promoter score",
            "type": "index",
            "polarity": "positive",
            "range": [-100, 100],
        },
    ],
    connections=[
        ("conversion_rate", "driven_by", "loyalty_score"),
        ("cart_value", "related", "loyalty_score"),
        ("return_rate", "opposes", "loyalty_score"),
        ("repeat_purchase_rate", "driven_by", "conversion_rate"),
        ("nps", "related", "loyalty_score"),
    ],
    segments=[
        {
            "name": "loyal_climbers",
            "count": 25,
            "archetype": "growth",
            "label": "Builds loyalty steadily across both years",
            "attributes": {
                "tier": ["gold", "platinum"],
                "channel": ["web", "mobile"],
                "churn_reason": [
                    "account_dormant",
                    "low_engagement",
                    "payment_failure",
                    "service_interruption",
                ],
            },
            "baseline": {"loyalty_score": "high", "cart_value": "high", "return_rate": "low"},
        },
        {
            "name": "holiday_shoppers",
            "count": 30,
            "archetype": "seasonal",
            "label": "Cyclical demand around holidays — Q4 surges",
            "attributes": {
                "tier": ["silver", "gold"],
                "channel": ["web", "mobile", "marketplace"],
                "churn_reason": [
                    "account_dormant",
                    "low_engagement",
                    "payment_failure",
                    "service_interruption",
                ],
            },
            "baseline": {"cart_value": "mid", "conversion_rate": "mid"},
        },
        {
            "name": "cooled_off",
            "count": 18,
            "archetype": "flat > decline @ 12",
            "label": "Active first year, gradually disengaged in year two",
            "attributes": {
                "tier": ["bronze", "silver"],
                "channel": ["marketplace"],
                "churn_reason": [
                    "account_dormant",
                    "low_engagement",
                    "payment_failure",
                    "service_interruption",
                ],
            },
            "baseline": {"loyalty_score": "low", "return_rate": "high"},
        },
        {
            "name": "one_and_done",
            "count": 15,
            "archetype": "growth > spike_then_crash > flat @ 4 @ 8",
            "label": "Tested the brand for a few months, then never returned",
            "attributes": {
                "tier": ["bronze"],
                "channel": ["web"],
                "churn_reason": [
                    "account_dormant",
                    "low_engagement",
                    "payment_failure",
                    "service_interruption",
                ],
            },
            "baseline": {"loyalty_score": "low", "cart_value": "low"},
        },
        {
            "name": "winback",
            "count": 12,
            "archetype": "decline > flat > growth @ 6 @ 14",
            "label": "Churned, then reactivated by year-two campaign",
            "attributes": {
                "tier": ["silver"],
                "channel": ["email", "web"],
                "churn_reason": [
                    "account_dormant",
                    "low_engagement",
                    "payment_failure",
                    "service_interruption",
                ],
            },
            "baseline": {"loyalty_score": "mid", "conversion_rate": "mid"},
        },
        {
            "name": "escalating_basket",
            "count": 10,
            "archetype": "accelerating",
            "label": "Compounding cart values as trust builds",
            "attributes": {
                "tier": ["gold", "platinum"],
                "channel": ["web"],
                "churn_reason": [
                    "account_dormant",
                    "low_engagement",
                    "payment_failure",
                    "service_interruption",
                ],
            },
            "baseline": {"cart_value": "high", "loyalty_score": "high"},
        },
    ],
    lifecycle={
        "track": "loyalty_score",
        "stages": [
            ("new", 0.0),
            ("casual", 0.2),
            ("regular", 0.5),
            ("loyal", 0.75),
            ("champion", 0.9),
        ],
    },
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
            "name": "dim_customer",
            "per": "unit",
            "columns": [
                {"name": "customer_id", "type": "id"},
                {"name": "customer_name", "type": "faker.name"},
                {"name": "signup_year", "type": "faker.year"},
                {"name": "cohort_size", "type": "segment.count"},
                {
                    "name": "customer_tier",
                    "type": "scd",
                    "tracks": "loyalty_score",
                    "tiers": ["browser", "casual", "loyal"],
                    "at": [0.3, 0.7],
                },
            ],
        },
        {
            "name": "dim_product_category",
            "reference": True,
            "columns": [
                {"name": "category_id", "type": "id"},
                {"name": "category_name", "type": "static.electronics,apparel,home,grocery,beauty"},
                {"name": "margin_tier", "type": "static.high,standard,standard,low,high"},
                # 0.6-M15: nested struct column (M14c) — see
                # ``retail_template.yaml`` for the Semi-Structured
                # Flattening (DE L12) exercise rationale.
                {
                    "name": "catalog_metadata",
                    "type": "struct",
                    "nested_schema": {
                        "aisle": "string",
                        "seasonality": "string",
                        "avg_basket_position": "int",
                    },
                },
            ],
        },
        {
            "name": "dim_channel",
            "reference": True,
            "columns": [
                {"name": "channel_id", "type": "id"},
                {"name": "channel_name", "type": "static.web,mobile,marketplace,email,store"},
                {
                    "name": "channel_type",
                    "type": "static.digital,digital,third_party,owned,physical",
                },
            ],
        },
        {
            "name": "dim_promotion",
            "reference": True,
            "columns": [
                {"name": "promotion_id", "type": "id"},
                {
                    "name": "promo_name",
                    "type": "static.clearance,seasonal_sale,loyalty_reward,flash_sale",
                },
                {"name": "discount_type", "type": "static.percentage,percentage,points,percentage"},
            ],
        },
    ],
    facts=[
        {
            "name": "fct_sessions",
            "metrics": ["sessions", "conversion_rate"],
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "channel_id", "type": "ref.dim_channel"},
                {"name": "session_count", "type": "metric.sessions"},
                {"name": "conversion_rate", "type": "metric.conversion_rate"},
                {
                    "name": "shopping_intent",
                    "type": "bucket",
                    "labels": ["browsing", "comparing", "purchasing", "loyal_repeat"],
                },
            ],
        },
        {
            "name": "fct_purchases",
            "metrics": ["cart_value", "return_rate", "loyalty_score", "repeat_purchase_rate"],
            # 0.6-M15: CDC fact-side (M9c) — every row carries
            # _inserted_at / _updated_at / _op audit columns. Column-
            # level quality injections (see ``quality=`` below) flip
            # _op to "U" on touched rows. See ``retail_template.yaml``.
            "cdc": True,
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "category_id", "type": "ref.dim_product_category"},
                {"name": "promotion_id", "type": "ref.dim_promotion"},
                {"name": "cart_value", "type": "metric.cart_value"},
                {"name": "return_rate", "type": "metric.return_rate"},
                {"name": "loyalty_score", "type": "metric.loyalty_score"},
                {"name": "repeat_purchase_rate", "type": "metric.repeat_purchase_rate"},
            ],
        },
        {
            "name": "fct_satisfaction",
            "metrics": ["nps"],
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "nps", "type": "metric.nps"},
            ],
        },
    ],
    events=[
        {
            "name": "evt_purchase",
            "trigger": "proportional",
            "driver": "conversion_rate",
            "scale": 6.0,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "event_ts", "type": "timestamp"},
            ],
        },
        {
            "name": "evt_churn",
            "trigger": "threshold",
            "metric": "loyalty_score",
            "below": 0.15,
            "for_periods": 4,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "reason", "type": "pool.churn_reason"},
                {"name": "churn_flag", "type": "flag"},
            ],
        },
    ],
    # 0.6-M15: data-quality issues — see ``retail_template.yaml`` for
    # the Data Quality Testing (DE L25), Data Cleaning (DE L15), and
    # Data Observability (DE L28) rationale. The volume_anomaly spike
    # at period 18 is the canonical observability scenario.
    quality=[
        {"table": "fct_purchases", "issue": "null_injection", "rate": 0.03, "column": "cart_value"},
        {
            "table": "fct_sessions",
            "issue": "volume_anomaly",
            "rate": 0.5,
            "mode": "spike",
            "period": 18,
        },
    ],
)
