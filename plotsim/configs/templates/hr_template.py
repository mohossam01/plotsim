"""HR template — Python form.

Mirror of ``hr.yaml``. A workforce analytics warehouse for a mid-size
multinational: employees, departments, geographic offices (geo bundle
on dim_employee), a manager reference dim, project assignments
(M:N bridge), CDC on the compensation fact, narrative review notes,
and three quality issues across the fact + event surface.

Run:
    >>> from plotsim.configs.templates.hr_template import config
    >>> from plotsim import generate_tables
    >>> tables = generate_tables(config)
"""

from plotsim import create


_NAR_BLOCK = {
    "stem": {
        "low": [
            "Ramp slower than expected for",
            "Still finding footing on",
            "Behind on onboarding tasks for",
        ],
        "mid": ["On track in", "Settling into", "Building competence in"],
        "high": ["Exceeding ramp targets in", "Picking up momentum on", "Already contributing on"],
    },
    "assessment": {
        "low": [
            "limited delivery so far",
            "minimal independent ownership",
            "few completed milestones",
        ],
        "mid": ["steady delivery", "clear baseline ownership", "expected milestone cadence"],
        "high": ["strong early delivery", "broad ownership for tenure", "ahead-of-plan milestones"],
    },
    "action": {
        "low": [
            "Pair with mentor for the next cycle.",
            "Re-scope to smaller deliverables.",
            "Add structured check-ins.",
        ],
        "mid": [
            "Continue current ramp plan.",
            "Maintain mentor cadence.",
            "Hold present trajectory.",
        ],
        "high": [
            "Stretch with cross-team scope.",
            "Promote to mid-IC scope early.",
            "Surface for visibility projects.",
        ],
    },
}


config = create(
    about="HR talent, performance, compensation and attrition analytics",
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
    connections=[
        "engagement driven_by performance_score",
        "engagement opposes attrition_risk",
        "absence_rate related attrition_risk",
        "compensation 0.35 performance_score",
    ],
    seasonality=[
        {"months": [3, 4], "strength": 0.20},
        {"months": [11, 12], "strength": 0.15},
        {"months": [7, 8], "strength": -0.20},
    ],
    segments=[
        {
            "name": "new_hire_ramp",
            "count": 18,
            "archetype": "flat > growth @ 6",
            "label": "Onboarding ramp, then sigmoid into full productivity",
            "attributes": {
                "department": ["Engineering", "Product"],
                "role_family": ["individual_contributor"],
                "project_type": ["build", "research"],
            },
            "baseline": {"performance_score": "mid", "engagement": "mid", "training_hours": "high"},
        },
        {
            "name": "top_performer",
            "count": 22,
            "archetype": "accelerating",
            "label": "Compounding performance — promotion track",
            "attributes": {
                "department": ["Engineering", "Product", "Finance"],
                "role_family": ["individual_contributor", "manager"],
                "project_type": ["build", "growth", "research"],
            },
            "baseline": {"performance_score": "high", "compensation": "high"},
        },
        {
            "name": "core_team",
            "count": 28,
            "archetype": "flat",
            "label": "Reliable senior contributors at sustained baseline",
            "attributes": {
                "department": ["Engineering", "Sales", "Marketing", "Finance", "Legal"],
                "role_family": ["individual_contributor", "manager"],
                "project_type": ["maintain", "build"],
            },
            "baseline": {
                "performance_score": "high",
                "engagement": "high",
                "attrition_risk": "low",
            },
        },
        {
            "name": "disengaging",
            "count": 16,
            "archetype": "flat > decline @ 12",
            "label": "Coasted for a year, then quietly disengaged",
            "attributes": {
                "department": ["Operations", "Sales", "HR"],
                "role_family": ["individual_contributor"],
                "project_type": ["maintain"],
            },
            "baseline": {"engagement": "low", "absence_rate": "high", "attrition_risk": "high"},
        },
        {
            "name": "burnout_cohort",
            "count": 10,
            "archetype": "growth > spike_then_crash > flat @ 6 @ 14",
            "label": "Rapid early ramp, peak around month 6, crashed by 14",
            "attributes": {
                "department": ["Engineering", "Operations"],
                "role_family": ["manager"],
                "project_type": ["build", "growth"],
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
            "label": "Stalled, recovered with new manager at month 14",
            "attributes": {
                "department": ["Sales", "Marketing"],
                "role_family": ["individual_contributor"],
                "project_type": ["growth", "maintain"],
            },
            "baseline": {"performance_score": "mid", "engagement": "mid"},
        },
    ],
    lifecycle={
        "track": "attrition_risk",
        "enforce_order": True,
        "stages": [
            {"new_hire": 0.0},
            {"established": 0.15},
            {"disengaging": 0.4},
            {"exited": 0.7},
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
                {"name": "email", "type": "faker.email"},
                {"name": "hire_year", "type": "faker.year"},
                {"name": "cohort_size", "type": "segment.count"},
                {"name": "department", "type": "pool.department"},
                {"name": "role_family", "type": "pool.role_family"},
                {"name": "pay_band", "type": "static.B1,B2,B3,B4,B5,B6,B7"},
                {"name": "office_country", "type": "geo.country"},
                {"name": "office_country_code", "type": "geo.country_code"},
                {"name": "office_region", "type": "geo.region"},
                {"name": "office_city", "type": "geo.city"},
                {
                    "name": "job_level",
                    "type": "scd",
                    "tracks": "performance_score",
                    "tiers": ["ic", "senior", "lead", "principal"],
                    "at": [0.3, 0.6, 0.85],
                },
            ],
        },
        {
            "name": "dim_manager",
            "reference": True,
            "columns": [
                {"name": "manager_id", "type": "id"},
                {"name": "manager_name", "type": "faker.name"},
                {"name": "span_size", "type": "static.5,8,10,12,15"},
            ],
        },
        {
            "name": "dim_project",
            "reference": True,
            "columns": [
                {"name": "project_id", "type": "id"},
                {"name": "project_name", "type": "faker.company"},
                {
                    "name": "project_type",
                    "type": "static.build,growth,research,maintain,build,growth",
                },
                {"name": "budget_band", "type": "static.small,medium,medium,large,xlarge,large"},
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
                {"name": "manager_id", "type": "ref.dim_manager"},
                {"name": "performance_score", "type": "metric.performance_score"},
                {"name": "engagement", "type": "metric.engagement"},
                {"name": "training_hours", "type": "metric.training_hours"},
                {
                    "name": "review_outcome",
                    "type": "bucket",
                    "labels": ["improvement_plan", "meets", "exceeds", "top_talent"],
                },
                {
                    "name": "review_notes",
                    "type": "narrative",
                    "template": "{stem} {assessment}. {action}",
                    "lexicons": {
                        "new_hire_ramp": _NAR_BLOCK,
                        "top_performer": _NAR_BLOCK,
                        "core_team": _NAR_BLOCK,
                        "disengaging": _NAR_BLOCK,
                        "burnout_cohort": _NAR_BLOCK,
                        "comeback": _NAR_BLOCK,
                    },
                },
            ],
        },
        {
            "name": "fct_compensation",
            "metrics": ["compensation", "attrition_risk"],
            "cdc": True,
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "employee_id", "type": "ref.dim_employee"},
                {"name": "compensation", "type": "metric.compensation"},
                {"name": "attrition_risk", "type": "metric.attrition_risk"},
                {"name": "bonus_target", "type": "range", "range": [0, 0.4]},
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
            "for": 3,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "employee_id", "type": "ref.dim_employee"},
                {"name": "reason", "type": "faker.sentence"},
                {"name": "voluntary", "type": "flag"},
            ],
        },
    ],
    bridges=[
        {
            "name": "bridge_employee_project",
            "left": "dim_employee",
            "right": "dim_project",
            "cardinality": [1, 4],
            "driver": "performance_score",
        },
    ],
    quality=[
        {
            "table": "fct_performance",
            "issue": "null_injection",
            "rate": 0.04,
            "column": "engagement",
        },
        {"table": "fct_attendance", "issue": "late_arrival", "rate": 0.02},
        {"table": "evt_training_completion", "issue": "duplicate_rows", "rate": 0.01},
    ],
)
