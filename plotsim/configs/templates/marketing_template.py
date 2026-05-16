"""Marketing template — Python form.

Mirror of ``marketing.yaml``. Digital marketing campaign analytics with
per-metric A/B treatment (lift targets conversion_rate specifically),
CDC on the spend fact (reconciliation restates prior periods), Q4
seasonality + summer / January dips, and three quality issues.

Run:
    >>> from plotsim.configs.templates.marketing_template import config
    >>> from plotsim import generate_tables
    >>> tables = generate_tables(config)
"""

from plotsim import create


config = create(
    about="Marketing campaign performance — spend, reach, conversion, revenue",
    unit="campaign",
    seed=80211,
    noise="slightly_messy",
    window=("2023-01", "2024-12", "monthly"),
    seasonality=[
        {"months": [11, 12], "strength": 0.45},
        {"months": [6, 7, 8], "strength": -0.10},
        {"months": [1], "strength": -0.25},
    ],
    metrics=[
        {
            "name": "ad_spend",
            "label": "Monthly paid-media spend",
            "type": "amount",
            "polarity": "positive",
            "range": [500, 50000],
        },
        {
            "name": "impressions",
            "label": "Ad impressions delivered",
            "type": "count",
            "polarity": "positive",
        },
        {
            "name": "click_through_rate",
            "label": "Ad click-through rate",
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
            "name": "bounce_rate",
            "label": "Landing-page bounce rate",
            "type": "score",
            "polarity": "negative",
        },
        {
            "name": "revenue",
            "label": "Attributed revenue",
            "type": "amount",
            "polarity": "positive",
            "range": [0, 250000],
        },
        {
            "name": "roi",
            "label": "Return on ad spend",
            "type": "index",
            "polarity": "positive",
            "range": [-1, 5],
        },
        {
            "name": "leads_generated",
            "label": "Marketing-qualified leads",
            "type": "count",
            "polarity": "positive",
            "follows": "impressions",
            "delay": 1,
        },
        {
            "name": "creative_quality",
            "label": "Creative quality score",
            "type": "score",
            "polarity": "positive",
        },
    ],
    connections=[
        "click_through_rate driven_by impressions",
        "conversion_rate driven_by click_through_rate",
        "bounce_rate opposes conversion_rate",
        "revenue 0.62 conversion_rate",
        "roi 0.48 revenue",
        "leads_generated related click_through_rate",
    ],
    segments=[
        {
            "name": "awareness_builder",
            "count": 15,
            "archetype": "growth",
            "label": "Top-of-funnel brand awareness — steady reach growth",
            "attributes": {
                "objective": ["awareness"],
                "channel": ["paid_social", "display"],
                "pause_reason": [
                    "budget_exhausted",
                    "low_conversion",
                    "creative_fatigue",
                    "audience_saturation",
                ],
                "bid_strategy": ["target_cpa", "max_clicks", "max_conversions"],
            },
            "baseline": {"impressions": "high", "ad_spend": "high", "conversion_rate": "low"},
            "treatment": {
                "fraction": 0.5,
                "lift_log_odds": 0.5,
                "start_period": 8,
                "treatment_label": "new_audience_model",
                "control_label": "incumbent_audience_model",
                "target_metric": "conversion_rate",
            },
        },
        {
            "name": "paid_burst",
            "count": 18,
            "archetype": "growth > spike_then_crash @ 12",
            "label": "Heavy paid push, then sharp budget cut after Q4",
            "attributes": {
                "objective": ["conversion"],
                "channel": ["paid_search", "paid_social"],
                "pause_reason": [
                    "budget_exhausted",
                    "low_conversion",
                    "creative_fatigue",
                    "audience_saturation",
                ],
                "bid_strategy": ["target_cpa", "max_conversions"],
            },
            "baseline": {"ad_spend": "high", "impressions": "high", "bounce_rate": "high"},
        },
        {
            "name": "seasonal_promo",
            "count": 20,
            "archetype": "seasonal",
            "label": "Cyclical holiday and seasonal-sale pushes",
            "attributes": {
                "objective": ["conversion", "awareness"],
                "channel": ["paid_search", "email", "paid_social"],
                "pause_reason": [
                    "budget_exhausted",
                    "low_conversion",
                    "creative_fatigue",
                    "audience_saturation",
                ],
                "bid_strategy": ["target_roas", "max_conversions"],
            },
            "baseline": {"ad_spend": "mid", "revenue": "mid"},
        },
        {
            "name": "delayed_breakthrough",
            "count": 12,
            "archetype": "flat > growth @ 10",
            "label": "Quiet ramp-up, breakthrough mid-campaign once creative landed",
            "attributes": {
                "objective": ["conversion"],
                "channel": ["paid_search", "display"],
                "pause_reason": [
                    "budget_exhausted",
                    "low_conversion",
                    "creative_fatigue",
                    "audience_saturation",
                ],
                "bid_strategy": ["target_cpa"],
            },
            "baseline": {"ad_spend": "mid", "conversion_rate": "mid"},
        },
        {
            "name": "viral_compound",
            "count": 10,
            "archetype": "accelerating",
            "label": "Compounding organic share — viral coefficient > 1",
            "attributes": {
                "objective": ["awareness", "engagement"],
                "channel": ["organic_social", "referral"],
                "pause_reason": [
                    "budget_exhausted",
                    "low_conversion",
                    "creative_fatigue",
                    "audience_saturation",
                ],
                "bid_strategy": ["max_clicks"],
            },
            "baseline": {"impressions": "high", "revenue": "high", "roi": "high"},
        },
        {
            "name": "end_of_life",
            "count": 10,
            "archetype": "decline",
            "label": "Sunsetting campaign — winding down spend",
            "attributes": {
                "objective": ["retention"],
                "channel": ["email", "retargeting"],
                "pause_reason": [
                    "budget_exhausted",
                    "low_conversion",
                    "creative_fatigue",
                    "audience_saturation",
                ],
                "bid_strategy": ["target_roas"],
            },
            "baseline": {"ad_spend": "low", "conversion_rate": "low"},
        },
        {
            "name": "retarget_revival",
            "count": 10,
            "archetype": "decline > flat > growth @ 6 @ 16",
            "label": "Stalled, paused, then relaunched with retargeting",
            "attributes": {
                "objective": ["retention", "conversion"],
                "channel": ["retargeting", "email"],
                "pause_reason": [
                    "budget_exhausted",
                    "low_conversion",
                    "creative_fatigue",
                    "audience_saturation",
                ],
                "bid_strategy": ["target_cpa", "max_conversions"],
            },
            "baseline": {"conversion_rate": "mid", "bounce_rate": "mid"},
        },
    ],
    lifecycle={
        "track": "conversion_rate",
        "stages": [{"launch": 0.0}, {"ramping": 0.15}, {"performing": 0.4}, {"winning": 0.7}],
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
            "name": "dim_campaign",
            "per": "unit",
            "columns": [
                {"name": "campaign_id", "type": "id"},
                {"name": "campaign_name", "type": "faker.company"},
                {"name": "launch_year", "type": "faker.year"},
                {"name": "cohort_size", "type": "segment.count"},
                {"name": "objective", "type": "pool.objective"},
                {"name": "channel", "type": "pool.channel"},
                {"name": "bid_strategy", "type": "pool.bid_strategy"},
                {
                    "name": "campaign_phase",
                    "type": "scd",
                    "tracks": "revenue",
                    "tiers": ["seed", "scale", "mature"],
                    "at": [0.3, 0.7],
                },
            ],
        },
        {
            "name": "dim_audience",
            "reference": True,
            "columns": [
                {"name": "audience_id", "type": "id"},
                {
                    "name": "audience_name",
                    "type": "static.lookalike,prospecting,retargeting,loyalty,broad,interest",
                },
                {"name": "audience_size", "type": "static.large,large,medium,small,large,medium"},
            ],
        },
        {
            "name": "dim_creative_format",
            "reference": True,
            "columns": [
                {"name": "format_id", "type": "id"},
                {
                    "name": "format_name",
                    "type": "static.video,static_image,carousel,story,native,collection",
                },
            ],
        },
    ],
    facts=[
        {
            "name": "fct_campaign_performance",
            "metrics": ["ad_spend", "impressions", "click_through_rate", "creative_quality"],
            "cdc": True,
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "campaign_id", "type": "ref.dim_campaign"},
                {"name": "audience_id", "type": "ref.dim_audience"},
                {"name": "format_id", "type": "ref.dim_creative_format"},
                {"name": "ad_spend", "type": "metric.ad_spend"},
                {"name": "impressions", "type": "metric.impressions"},
                {"name": "click_through_rate", "type": "metric.click_through_rate"},
                {"name": "creative_quality", "type": "metric.creative_quality"},
                {"name": "budget_cap", "type": "range", "range": [1000, 80000]},
                {"name": "bid_amount", "type": "range", "range": [0.10, 12.0]},
            ],
        },
        {
            "name": "fct_funnel",
            "metrics": ["conversion_rate", "bounce_rate", "leads_generated"],
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "campaign_id", "type": "ref.dim_campaign"},
                {"name": "conversion_rate", "type": "metric.conversion_rate"},
                {"name": "bounce_rate", "type": "metric.bounce_rate"},
                {"name": "leads_generated", "type": "metric.leads_generated"},
                {
                    "name": "funnel_stage",
                    "type": "bucket",
                    "labels": ["cold", "warming", "engaged", "converted"],
                },
            ],
        },
        {
            "name": "fct_revenue",
            "metrics": ["revenue", "roi"],
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "campaign_id", "type": "ref.dim_campaign"},
                {"name": "revenue", "type": "metric.revenue"},
                {"name": "roi", "type": "metric.roi"},
            ],
        },
    ],
    events=[
        {
            "name": "evt_click",
            "trigger": "proportional",
            "driver": "click_through_rate",
            "scale": 8.0,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "campaign_id", "type": "ref.dim_campaign"},
                {"name": "event_ts", "type": "timestamp"},
            ],
        },
        {
            "name": "evt_campaign_pause",
            "trigger": "threshold",
            "metric": "conversion_rate",
            "below": 0.1,
            "for": 3,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "campaign_id", "type": "ref.dim_campaign"},
                {"name": "reason", "type": "pool.pause_reason"},
                {"name": "pause_flag", "type": "flag"},
            ],
        },
    ],
    quality=[
        {
            "table": "fct_campaign_performance",
            "issue": "null_injection",
            "rate": 0.04,
            "column": "ad_spend",
        },
        {"table": "evt_click", "issue": "late_arrival", "rate": 0.03},
        {"table": "fct_funnel", "issue": "duplicate_rows", "rate": 0.01},
    ],
)
