"""Retail template — Python form.

Mirror of ``retail.yaml``. Omnichannel retail with SCD2 loyalty tier,
parent/child orders + line items, cross-fact FK on returns, customer
× promotion M:N bridge, geo bundle on dim_customer, multi-locale
faker output, narrative reviews, CDC audit on the orders parent fact.

Run:
    >>> from plotsim.configs.templates.retail_template import config
    >>> from plotsim import generate_tables
    >>> tables = generate_tables(config)
"""

from plotsim import create


_NAR_BLOCK = {
    "opener": {
        "low": ["I am disappointed by", "I am frustrated with", "I cannot recommend"],
        "mid": ["I keep using", "I find myself reaching for", "I keep returning to"],
        "high": ["I love", "I am thrilled with", "I keep recommending"],
    },
    "object": {
        "low": ["the broken release", "the buggy app", "the unreliable platform"],
        "mid": ["the product", "the service", "the standard offering"],
        "high": ["the polished platform", "the smooth experience", "the standout service"],
    },
    "comment": {
        "low": ["Not recommended.", "Going elsewhere.", "Disappointing."],
        "mid": ["Works as advertised.", "Fair value.", "Meets expectations."],
        "high": ["Highly recommend.", "Best in class.", "Worth every penny."],
    },
}


config = create(
    about="Omnichannel retail — customers, orders, returns, loyalty",
    unit="customer",
    seed=39131,
    noise="slightly_messy",
    locale=["en_US", "en_GB", "fr_FR", "de_DE"],
    window=("2023-01", "2024-12", "monthly"),
    seasonality=[
        {"months": [11, 12], "strength": 0.45},
        {"months": [6, 7, 8], "strength": -0.10},
        {"months": [1], "strength": -0.25},
    ],
    metrics=[
        {
            "name": "order_volume",
            "label": "Orders placed per period",
            "type": "count",
            "polarity": "positive",
        },
        {
            "name": "cart_value",
            "label": "Average cart value",
            "type": "amount",
            "polarity": "positive",
            "range": [10, 500],
        },
        {
            "name": "loyalty_score",
            "label": "Loyalty engagement",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "conversion_rate",
            "label": "Visit-to-purchase rate",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "return_rate",
            "label": "Return rate (drives evt)",
            "type": "score",
            "polarity": "negative",
        },
        {
            "name": "nps",
            "label": "Net promoter score",
            "type": "amount",
            "polarity": "positive",
            "range": [0, 100],
        },
        {
            "name": "repeat_purchase_rate",
            "label": "Repeat purchase rate",
            "type": "score",
            "polarity": "positive",
        },
    ],
    connections=[
        "order_volume driven_by loyalty_score",
        "cart_value related conversion_rate",
        "loyalty_score 0.55 repeat_purchase_rate",
        "nps 0.55 loyalty_score",
        "return_rate opposes loyalty_score",
    ],
    segments=[
        {
            "name": "loyal_repeat",
            "count": 22,
            "archetype": "accelerating",
            "label": "Compounding loyalty — high repeat rate",
            "attributes": {
                "channel": ["in_store", "web", "mobile_app"],
                "payment_method": ["credit", "debit", "mobile_wallet", "gift_card"],
                "return_reason": [
                    "damaged",
                    "wrong_size",
                    "defective",
                    "no_longer_needed",
                    "late_arrival",
                ],
                "promo_type": ["loyalty_reward", "member_discount", "free_shipping"],
            },
            "baseline": {"order_volume": "high", "loyalty_score": "high", "cart_value": "high"},
        },
        {
            "name": "holiday_shopper",
            "count": 18,
            "archetype": "seasonal",
            "label": "Cyclical holiday spender — peaks Q4 and back-to-school",
            "attributes": {
                "channel": ["in_store", "web"],
                "payment_method": ["credit", "debit", "mobile_wallet", "gift_card"],
                "return_reason": [
                    "damaged",
                    "wrong_size",
                    "defective",
                    "no_longer_needed",
                    "late_arrival",
                ],
                "promo_type": ["seasonal_sale", "doorbuster", "bogo", "free_shipping"],
            },
            "baseline": {"order_volume": "mid", "cart_value": "high"},
        },
        {
            "name": "bargain_hunter",
            "count": 22,
            "archetype": "flat",
            "label": "Steady low-value cart, promo-driven",
            "attributes": {
                "channel": ["web", "mobile_app"],
                "payment_method": ["debit", "credit", "mobile_wallet", "gift_card"],
                "return_reason": [
                    "damaged",
                    "wrong_size",
                    "defective",
                    "no_longer_needed",
                    "late_arrival",
                ],
                "promo_type": ["clearance", "doorbuster", "bogo"],
            },
            "baseline": {"order_volume": "mid", "cart_value": "low"},
        },
        {
            "name": "churning",
            "count": 16,
            "archetype": "flat > decline @ 10",
            "label": "Coasted then quietly stopped buying",
            "attributes": {
                "channel": ["web"],
                "payment_method": ["credit", "debit", "mobile_wallet"],
                "return_reason": [
                    "damaged",
                    "wrong_size",
                    "defective",
                    "no_longer_needed",
                    "late_arrival",
                ],
                "promo_type": ["loyalty_reward", "member_discount"],
            },
            "baseline": {"loyalty_score": "low", "return_rate": "high"},
        },
        {
            "name": "new_customer",
            "count": 14,
            "archetype": "flat > growth @ 4",
            "label": "Recently acquired — ramping engagement",
            "attributes": {
                "channel": ["mobile_app", "web"],
                "payment_method": ["credit", "debit", "mobile_wallet"],
                "return_reason": [
                    "damaged",
                    "wrong_size",
                    "defective",
                    "no_longer_needed",
                    "late_arrival",
                ],
                "promo_type": ["welcome_offer", "first_order_discount"],
            },
            "baseline": {"order_volume": "low", "loyalty_score": "mid"},
        },
        {
            "name": "vip",
            "count": 8,
            "archetype": "growth",
            "label": "High-value VIP cohort with rising spend",
            "attributes": {
                "channel": ["in_store", "web", "mobile_app"],
                "payment_method": ["credit", "mobile_wallet"],
                "return_reason": [
                    "damaged",
                    "wrong_size",
                    "defective",
                    "no_longer_needed",
                    "late_arrival",
                ],
                "promo_type": ["vip_exclusive", "loyalty_reward"],
            },
            "baseline": {"cart_value": "high", "nps": "high"},
        },
    ],
    lifecycle={
        "track": "loyalty_score",
        "stages": [{"browser": 0.0}, {"first_purchase": 0.15}, {"returning": 0.4}, {"loyal": 0.7}],
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
                {"name": "customer_email", "type": "faker.email"},
                {"name": "signup_year", "type": "faker.year"},
                {"name": "cohort_size", "type": "segment.count"},
                {"name": "preferred_channel", "type": "pool.channel"},
                {"name": "home_country", "type": "geo.country"},
                {"name": "home_country_code", "type": "geo.country_code"},
                {"name": "home_region", "type": "geo.region"},
                {"name": "home_city", "type": "geo.city"},
                {
                    "name": "loyalty_tier",
                    "type": "scd",
                    "tracks": "loyalty_score",
                    "tiers": ["bronze", "silver", "gold", "platinum"],
                    "at": [0.25, 0.55, 0.8],
                },
            ],
        },
        {
            "name": "dim_product",
            "reference": True,
            "columns": [
                {"name": "product_id", "type": "id"},
                {
                    "name": "product_name",
                    "type": "static.widget,gadget,gizmo,sprocket,lantern,doohickey,thingamajig,whatsit",
                },
                {
                    "name": "category",
                    "type": "static.hardware,hardware,hardware,hardware,outdoor,misc,misc,misc",
                },
                {
                    "name": "price_band",
                    "type": "static.value,value,mid,mid,mid,premium,premium,luxury",
                },
            ],
        },
        {
            "name": "dim_promotion",
            "reference": True,
            "columns": [
                {"name": "promotion_id", "type": "id"},
                {"name": "promotion_name", "type": "faker.company"},
                {
                    "name": "promo_type",
                    "type": "static.seasonal_sale,bogo,clearance,loyalty_reward,welcome_offer,doorbuster,member_discount,vip_exclusive,free_shipping,first_order_discount",
                },
                {
                    "name": "discount_band",
                    "type": "static.10pct,20pct,25pct,30pct,40pct,bogo,member_only,vip_only,free_ship,no_promo",
                },
            ],
        },
    ],
    facts=[
        {
            "name": "fct_customer_activity",
            "metrics": [
                "loyalty_score",
                "conversion_rate",
                "nps",
                "repeat_purchase_rate",
                "cart_value",
                "return_rate",
            ],
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "loyalty_score", "type": "metric.loyalty_score"},
                {"name": "conversion_rate", "type": "metric.conversion_rate"},
                {"name": "nps", "type": "metric.nps"},
                {"name": "repeat_purchase_rate", "type": "metric.repeat_purchase_rate"},
                {"name": "cart_value", "type": "metric.cart_value"},
                {"name": "return_rate", "type": "metric.return_rate"},
                {
                    "name": "review_text",
                    "type": "narrative",
                    "template": "{opener} {object}. {comment}",
                    "lexicons": {
                        "loyal_repeat": _NAR_BLOCK,
                        "holiday_shopper": _NAR_BLOCK,
                        "bargain_hunter": _NAR_BLOCK,
                        "churning": _NAR_BLOCK,
                        "new_customer": _NAR_BLOCK,
                        "vip": _NAR_BLOCK,
                    },
                },
            ],
        },
        {
            "name": "fct_orders",
            "row_count_driver": "order_volume",
            "row_count_scale": 1.2,
            "cdc": True,
            "columns": [
                {"name": "order_id", "type": "id"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "order_date", "type": "ref.dim_date"},
                {"name": "payment_method", "type": "pool.payment_method"},
                {"name": "order_channel", "type": "pool.channel"},
            ],
        },
        {
            "name": "fct_order_items",
            "parent_table": "fct_orders",
            "children_per_row": [1, 5],
            "columns": [
                {"name": "item_id", "type": "id"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "order_date", "type": "ref.dim_date"},
                {"name": "product_id", "type": "ref.dim_product"},
                {"name": "quantity", "type": "range", "range": [1, 12]},
                {"name": "unit_price", "type": "range", "range": [2.99, 499.99]},
                {"name": "discount_pct", "type": "range", "range": [0, 40]},
            ],
        },
        {
            "name": "fct_returns",
            "row_count_driver": "return_rate",
            "row_count_scale": 0.6,
            "columns": [
                {"name": "return_id", "type": "id"},
                {"name": "order_id", "type": "ref.fct_orders"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "return_date", "type": "ref.dim_date"},
                {"name": "return_reason", "type": "pool.return_reason"},
            ],
        },
    ],
    events=[
        {
            "name": "evt_session",
            "trigger": "proportional",
            "driver": "conversion_rate",
            "scale": 12.0,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "event_ts", "type": "timestamp"},
            ],
        },
        {
            "name": "evt_return",
            "trigger": "proportional",
            "driver": "return_rate",
            "scale": 2.0,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "customer_id", "type": "ref.dim_customer"},
                {"name": "severity", "type": "static.warning,investigation,fraud_review"},
                {"name": "event_ts", "type": "timestamp"},
            ],
        },
    ],
    bridges=[
        {
            "name": "bridge_customer_promotion",
            "left": "dim_customer",
            "right": "dim_promotion",
            "cardinality": [1, 4],
            "driver": "loyalty_score",
        },
    ],
    quality=[
        {
            "table": "fct_customer_activity",
            "issue": "null_injection",
            "rate": 0.03,
            "column": "conversion_rate",
        },
        {"table": "evt_session", "issue": "duplicate_rows", "rate": 0.015},
        {"table": "fct_orders", "issue": "late_arrival", "rate": 0.02},
    ],
)
