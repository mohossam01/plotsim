"""HR / workforce analytics — Python builder template.

Mirror of ``hr_template.yaml``. Demonstrates:

* ``noise: slightly_messy`` for realistic survey jitter
* custom-coefficient connection (``compensation 0.27 performance_score``)
* ``lifecycle.enforce_order=True`` — monotonic stage walk so an
  employee who enters ``disengaging`` doesn't bounce back on a
  transient pulse-survey blip
* SCD2 ``job_level`` tracking ``performance_score``
"""

from plotsim import create

config = create(
    about="HR talent and attrition analytics",
    unit="employee",
    seed=5150,
    noise="slightly_messy",
    window=("2023-01", "2024-12", "monthly"),
    metrics=[
        {
            "name": "performance_score",
            "label": "Quarterly performance rating",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "engagement",
            "label": "Pulse engagement index",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "training_hours",
            "label": "Training hours completed",
            "type": "count",
            "polarity": "positive",
        },
        {
            "name": "absence_rate",
            "label": "Monthly absence rate",
            "type": "score",
            "polarity": "negative",
            "follows": "engagement",
            "delay": 1,
        },
        {
            "name": "attrition_risk",
            "label": "Attrition risk score",
            "type": "score",
            "polarity": "negative",
        },
        {
            "name": "compensation",
            "label": "Total monthly compensation",
            "type": "amount",
            "polarity": "positive",
            "range": [4000, 25000],
        },
    ],
    # Custom coefficient on the comp ↔ performance pair — calibrated
    # off internal performance / pay-band correlation studies.
    connections=[
        ("engagement", "driven_by", "performance_score"),
        ("engagement", "opposes", "attrition_risk"),
        ("absence_rate", "related", "attrition_risk"),
        ("compensation", 0.27, "performance_score"),
    ],
    segments=[
        {
            "name": "new_hire_ramp",
            "count": 20,
            "archetype": "flat > growth @ 6",
            "label": "Onboarding ramp, then sigmoid into full productivity",
            "attributes": {"department": ["Engineering", "Product"], "level": ["IC1", "IC2"]},
            "baseline": {"performance_score": "mid", "engagement": "mid", "training_hours": "high"},
        },
        {
            "name": "core_team",
            "count": 30,
            "archetype": "flat",
            "label": "Reliable senior contributors at sustained-high baseline",
            "attributes": {
                "department": ["Engineering", "Sales", "Operations"],
                "level": ["senior", "lead"],
            },
            "baseline": {
                "performance_score": "high",
                "engagement": "high",
                "attrition_risk": "low",
            },
        },
        {
            "name": "fast_riser",
            "count": 12,
            "archetype": "accelerating",
            "label": "Compounding performance — promotion track",
            "attributes": {"department": ["Engineering", "Product"], "level": ["senior"]},
            "baseline": {"performance_score": "high", "compensation": "high"},
        },
        {
            "name": "quiet_quitter",
            "count": 15,
            "archetype": "flat > decline @ 14",
            "label": "Coasted for a year, then quietly disengaged",
            "attributes": {
                "department": ["Sales", "Operations"],
                "level": ["IC1", "IC2", "senior"],
            },
            "baseline": {"engagement": "low", "absence_rate": "high", "attrition_risk": "high"},
        },
        {
            "name": "burnout_cohort",
            "count": 8,
            "archetype": "growth > spike_then_crash > flat @ 6 @ 14",
            "label": "Rapid early ramp, peak around month 6, crashed by 14",
            "attributes": {
                "department": ["Engineering", "Operations"],
                "level": ["senior", "lead"],
            },
            "baseline": {
                "performance_score": "high",
                "engagement": "mid",
                "attrition_risk": "high",
            },
        },
        {
            "name": "comeback",
            "count": 10,
            "archetype": "decline > flat > growth @ 6 @ 14",
            "label": "Stalled, hit bottom at month 6, recovered with new manager at 14",
            "attributes": {"department": ["Sales", "Product"], "level": ["senior"]},
            "baseline": {"performance_score": "mid", "engagement": "mid"},
        },
    ],
    # Monotonic stage walk + free-form re-entry suppressed.
    lifecycle={
        "track": "attrition_risk",
        "enforce_order": True,
        "stages": [
            ("new_hire", 0.0),
            ("established", 0.15),
            ("disengaging", 0.4),
            ("exited", 0.7),
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
            "name": "dim_employee",
            "per": "unit",
            "columns": [
                {"name": "employee_id", "type": "id"},
                {"name": "full_name", "type": "faker.name"},
                {"name": "hire_year", "type": "faker.year"},
                {"name": "cohort_size", "type": "segment.count"},
                {
                    "name": "job_level",
                    "type": "scd",
                    "tracks": "performance_score",
                    "tiers": ["ic", "senior", "lead"],
                    "at": [0.4, 0.75],
                },
            ],
        },
        {
            "name": "dim_department",
            "reference": True,
            "columns": [
                {"name": "department_id", "type": "id"},
                {"name": "department", "type": "static.engineering,sales,product,operations"},
                {"name": "cost_center", "type": "static.RnD,GTM,RnD,GnA"},
            ],
        },
        {
            "name": "dim_office",
            "reference": True,
            "columns": [
                {"name": "office_id", "type": "id"},
                {"name": "office", "type": "static.austin,berlin,singapore,remote"},
                {"name": "region", "type": "static.AMER,EMEA,APAC,GLOBAL"},
            ],
        },
    ],
    facts=[
        {
            "name": "fct_performance",
            "metrics": ["performance_score", "engagement", "training_hours"],
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "employee_id", "type": "ref.dim_employee"},
                {"name": "department_id", "type": "ref.dim_department"},
                {"name": "performance_score", "type": "metric.performance_score"},
                {"name": "engagement", "type": "metric.engagement"},
                {"name": "training_hours", "type": "metric.training_hours"},
                {
                    "name": "review_outcome",
                    "type": "bucket",
                    "labels": ["improvement_plan", "meets", "exceeds", "top_talent"],
                },
            ],
        },
        {
            "name": "fct_compensation",
            "metrics": ["compensation", "attrition_risk"],
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "employee_id", "type": "ref.dim_employee"},
                {"name": "office_id", "type": "ref.dim_office"},
                {"name": "compensation", "type": "metric.compensation"},
                {"name": "attrition_risk", "type": "metric.attrition_risk"},
            ],
        },
        {
            "name": "fct_attendance",
            "metrics": ["absence_rate"],
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "employee_id", "type": "ref.dim_employee"},
                {"name": "absence_rate", "type": "metric.absence_rate"},
            ],
        },
    ],
    events=[
        {
            "name": "evt_training_completion",
            "trigger": "proportional",
            "driver": "engagement",
            "scale": 4.0,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "employee_id", "type": "ref.dim_employee"},
                {"name": "course_name", "type": "faker.word"},
                {"name": "event_ts", "type": "timestamp"},
            ],
        },
        {
            "name": "evt_attrition",
            "trigger": "threshold",
            "metric": "attrition_risk",
            "above": 0.7,
            "for_periods": 3,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "employee_id", "type": "ref.dim_employee"},
                {"name": "reason", "type": "faker.sentence"},
                {"name": "voluntary", "type": "flag"},
            ],
        },
    ],
    # 0.6-M15: data-quality issues for Data Quality Testing (DE L25)
    # and Data Cleaning (DE L15). Manifest records every injection so
    # students can score detectors against ground truth.
    quality=[
        {
            "table": "fct_performance",
            "issue": "null_injection",
            "rate": 0.04,
            "column": "engagement",
        },
        {"table": "evt_training_completion", "issue": "late_arrival", "rate": 0.02},
    ],
)
