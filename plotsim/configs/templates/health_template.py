"""Health template — Python form.

Mirror of ``health.yaml``. Clinical analytics with SCD2 risk
stratification, parent/child encounters + labs, cross-fact FK on
prescriptions, M:N patient × diagnosis bridge, geo bundle on
patients, narrative encounter notes, CDC on encounters,
per-metric treatment cohort (lift on medication_adherence
specifically), and a holdout split for readmission prediction.

Run:
    >>> from plotsim.configs.templates.health_template import config
    >>> from plotsim import generate_tables
    >>> tables = generate_tables(config)
"""

from plotsim import create


_NAR_BLOCK = {
    "stem": {
        "low": ["Patient presented with concerning", "Worsening trend in", "Acute escalation in"],
        "mid": ["Patient stable on", "Routine follow-up for", "Monitoring continues for"],
        "high": ["Patient improving on", "Strong response to therapy for", "Excellent control of"],
    },
    "assessment": {
        "low": ["uncontrolled markers", "missed dose pattern", "lab abnormalities"],
        "mid": ["stable markers", "expected variability", "consistent with baseline"],
        "high": ["controlled markers", "improving labs", "treatment goals met"],
    },
    "plan": {
        "low": [
            "Adjusting medications and reassessing in two weeks.",
            "Initiating intensive monitoring protocol.",
            "Referring to specialist.",
        ],
        "mid": [
            "Continue current plan.",
            "Routine follow-up in three months.",
            "Maintain medication regimen.",
        ],
        "high": [
            "Reduce monitoring frequency.",
            "Eligible for step-down therapy.",
            "Annual follow-up.",
        ],
    },
}


config = create(
    about="Clinical and patient analytics — encounters, labs, prescriptions, outcomes",
    unit="patient",
    seed=70021,
    noise={
        "gaussian_sigma": 0.04,
        "outlier_rate": 0.005,
        "mcar_rate": 0.0,
        "scale_with_trajectory": True,
    },
    window=("2023-01", "2024-12", "monthly"),
    seasonality=[
        {"months": [10, 11, 12, 1, 2], "strength": 0.20},
        {"months": [6, 7, 8], "strength": -0.15},
    ],
    metrics=[
        {
            "name": "encounter_volume",
            "label": "Encounters per period",
            "type": "count",
            "polarity": "positive",
        },
        {
            "name": "bp_systolic",
            "label": "Systolic blood pressure",
            "type": "amount",
            "polarity": "negative",
            "range": [90, 200],
        },
        {
            "name": "a1c",
            "label": "HbA1c percentage",
            "type": "amount",
            "polarity": "negative",
            "range": [4.0, 14.0],
        },
        {
            "name": "medication_adherence",
            "label": "Medication adherence rate",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "clinical_engagement",
            "label": "Patient engagement",
            "type": "score",
            "polarity": "positive",
        },
        {
            "name": "readmission_risk",
            "label": "30-day readmission risk",
            "type": "score",
            "polarity": "negative",
            "follows": "bp_systolic",
            "delay": 1,
        },
        {
            "name": "lab_volume",
            "label": "Lab tests ordered per period",
            "type": "count",
            "polarity": "positive",
        },
    ],
    connections=[
        "readmission_risk related bp_systolic",
        "medication_adherence opposes readmission_risk",
        "a1c -0.40 medication_adherence",
        "clinical_engagement 0.45 medication_adherence",
        "encounter_volume related lab_volume",
    ],
    segments=[
        {
            "name": "chronic_progressive",
            "count": 18,
            "archetype": "growth",
            "label": "Chronic conditions with risk increasing steadily",
            "attributes": {
                "insurance_type": ["commercial", "medicare"],
                "diagnosis_category": ["cardiovascular", "endocrine"],
                "provider_specialty": ["cardiology", "endocrinology", "primary_care"],
                "department": ["outpatient", "primary_care", "specialty"],
            },
            "baseline": {"bp_systolic": "high", "a1c": "high", "readmission_risk": "high"},
        },
        {
            "name": "recovering",
            "count": 16,
            "archetype": "decline",
            "label": "Risk dropping post-intervention",
            "attributes": {
                "insurance_type": ["commercial", "medicare", "medicaid"],
                "diagnosis_category": ["cardiovascular", "orthopedic", "surgical"],
                "provider_specialty": ["cardiology", "orthopedics", "primary_care"],
                "department": ["inpatient", "rehab", "outpatient"],
            },
            "baseline": {"readmission_risk": "high", "medication_adherence": "mid"},
            "treatment": {
                "fraction": 0.5,
                "lift_log_odds": 0.6,
                "start_period": 6,
                "treatment_label": "intervention_arm",
                "control_label": "standard_care",
                "target_metric": "medication_adherence",
            },
        },
        {
            "name": "acute_episodic",
            "count": 20,
            "archetype": "seasonal",
            "label": "Episodic acute visits — flu season cycles",
            "attributes": {
                "insurance_type": ["commercial", "medicaid", "self_pay"],
                "diagnosis_category": ["respiratory", "infectious", "acute"],
                "provider_specialty": ["primary_care", "urgent_care", "emergency"],
                "department": ["urgent_care", "emergency", "primary_care"],
            },
            "baseline": {"encounter_volume": "mid", "lab_volume": "mid"},
        },
        {
            "name": "well_managed",
            "count": 22,
            "archetype": "flat",
            "label": "Stable chronic patients with consistent management",
            "attributes": {
                "insurance_type": ["commercial", "medicare"],
                "diagnosis_category": ["endocrine", "cardiovascular"],
                "provider_specialty": ["primary_care", "endocrinology", "cardiology"],
                "department": ["primary_care", "outpatient"],
            },
            "baseline": {
                "medication_adherence": "high",
                "clinical_engagement": "high",
                "readmission_risk": "low",
            },
        },
        {
            "name": "pediatric_routine",
            "count": 14,
            "archetype": "flat",
            "label": "Pediatric routine well-child visits",
            "attributes": {
                "insurance_type": ["commercial", "medicaid", "tricare"],
                "diagnosis_category": ["routine", "immunization", "developmental"],
                "provider_specialty": ["pediatrics", "primary_care"],
                "department": ["primary_care", "outpatient"],
            },
            "baseline": {"clinical_engagement": "high", "readmission_risk": "low"},
        },
        {
            "name": "high_risk_post_surgical",
            "count": 12,
            "archetype": "flat > spike_then_crash > flat @ 4 @ 12",
            "label": "Post-surgical patients with peri-operative risk window",
            "attributes": {
                "insurance_type": ["commercial", "medicare"],
                "diagnosis_category": ["surgical", "cardiovascular"],
                "provider_specialty": ["surgery", "cardiology", "primary_care"],
                "department": ["inpatient", "rehab", "outpatient"],
            },
            "baseline": {"readmission_risk": "high", "bp_systolic": "high"},
        },
    ],
    lifecycle={
        "track": "readmission_risk",
        "stages": [{"well": 0.0}, {"watch": 0.25}, {"high_risk": 0.55}, {"critical": 0.8}],
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
            "name": "dim_patient",
            "per": "unit",
            "columns": [
                {"name": "patient_id", "type": "id"},
                {"name": "patient_name", "type": "faker.name"},
                {"name": "patient_email", "type": "faker.email"},
                {"name": "birth_year", "type": "faker.year"},
                {"name": "cohort_size", "type": "segment.count"},
                {"name": "insurance_type", "type": "pool.insurance_type"},
                {"name": "diagnosis_category", "type": "pool.diagnosis_category"},
                {"name": "provider_specialty", "type": "pool.provider_specialty"},
                {"name": "home_country", "type": "geo.country"},
                {"name": "home_country_code", "type": "geo.country_code"},
                {"name": "home_region", "type": "geo.region"},
                {"name": "home_city", "type": "geo.city"},
                {
                    "name": "risk_stratification",
                    "type": "scd",
                    "tracks": "readmission_risk",
                    "tiers": ["low", "moderate", "high", "critical"],
                    "at": [0.25, 0.55, 0.8],
                },
            ],
        },
        {
            "name": "dim_diagnosis",
            "reference": True,
            "columns": [
                {"name": "diagnosis_id", "type": "id"},
                {
                    "name": "diagnosis_name",
                    "type": "static.hypertension,type2_diabetes,asthma,copd,depression,arthritis,coronary_disease,obesity",
                },
                {
                    "name": "chronicity",
                    "type": "static.chronic,chronic,chronic,chronic,chronic,chronic,chronic,chronic",
                },
            ],
        },
        {
            "name": "dim_medication",
            "reference": True,
            "columns": [
                {"name": "medication_id", "type": "id"},
                {
                    "name": "medication_class",
                    "type": "static.analgesic,antibiotic,antihypertensive,statin,ssri,bronchodilator,antidiabetic,anticoagulant,nsaid,beta_blocker",
                },
            ],
        },
    ],
    facts=[
        {
            "name": "fct_clinical_activity",
            "metrics": [
                "encounter_volume",
                "bp_systolic",
                "a1c",
                "medication_adherence",
                "clinical_engagement",
                "readmission_risk",
                "lab_volume",
            ],
            "columns": [
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "patient_id", "type": "ref.dim_patient"},
                {"name": "encounter_volume", "type": "metric.encounter_volume"},
                {"name": "bp_systolic", "type": "metric.bp_systolic"},
                {"name": "a1c", "type": "metric.a1c"},
                {"name": "medication_adherence", "type": "metric.medication_adherence"},
                {"name": "clinical_engagement", "type": "metric.clinical_engagement"},
                {"name": "readmission_risk", "type": "metric.readmission_risk"},
                {"name": "lab_volume", "type": "metric.lab_volume"},
                {
                    "name": "encounter_notes",
                    "type": "narrative",
                    "template": "{stem} {assessment}. {plan}",
                    "lexicons": {
                        "chronic_progressive": _NAR_BLOCK,
                        "recovering": _NAR_BLOCK,
                        "acute_episodic": _NAR_BLOCK,
                        "well_managed": _NAR_BLOCK,
                        "pediatric_routine": _NAR_BLOCK,
                        "high_risk_post_surgical": _NAR_BLOCK,
                    },
                },
            ],
        },
        {
            "name": "fct_encounters",
            "row_count_driver": "encounter_volume",
            "row_count_scale": 1.0,
            "cdc": True,
            "columns": [
                {"name": "encounter_id", "type": "id"},
                {"name": "patient_id", "type": "ref.dim_patient"},
                {"name": "encounter_date", "type": "ref.dim_date"},
                {
                    "name": "admission_type",
                    "type": "static.outpatient,outpatient,outpatient,inpatient,emergency,urgent,scheduled,follow_up",
                },
                {"name": "visit_duration", "type": "range", "range": [10, 240]},
            ],
        },
        {
            "name": "fct_lab_results",
            "parent_table": "fct_encounters",
            "children_per_row": [1, 4],
            "columns": [
                {"name": "lab_id", "type": "id"},
                {"name": "patient_id", "type": "ref.dim_patient"},
                {"name": "encounter_date", "type": "ref.dim_date"},
                {
                    "name": "panel",
                    "type": "static.cbc,cmp,lipid_panel,a1c,tsh,urinalysis,bnp,d_dimer",
                },
                {"name": "result_value", "type": "range", "range": [0.5, 500.0]},
            ],
        },
        {
            "name": "fct_prescriptions",
            "row_count_driver": "encounter_volume",
            "row_count_scale": 0.6,
            "columns": [
                {"name": "prescription_id", "type": "id"},
                {"name": "encounter_id", "type": "ref.fct_encounters"},
                {"name": "patient_id", "type": "ref.dim_patient"},
                {"name": "prescribed_date", "type": "ref.dim_date"},
                {"name": "medication_id", "type": "ref.dim_medication"},
                {"name": "days_supply", "type": "range", "range": [7, 90]},
            ],
        },
    ],
    events=[
        {
            "name": "evt_lab_order",
            "trigger": "proportional",
            "driver": "lab_volume",
            "scale": 3.0,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "patient_id", "type": "ref.dim_patient"},
                {"name": "panel", "type": "static.cbc,cmp,lipid_panel,a1c,tsh,urinalysis"},
                {"name": "event_ts", "type": "timestamp"},
            ],
        },
        {
            "name": "evt_readmission",
            "trigger": "proportional",
            "driver": "readmission_risk",
            "scale": 1.5,
            "columns": [
                {"name": "event_id", "type": "id"},
                {"name": "date_key", "type": "ref.dim_date"},
                {"name": "patient_id", "type": "ref.dim_patient"},
                {"name": "reason", "type": "faker.sentence"},
                {"name": "severity", "type": "static.observation,inpatient,icu"},
                {"name": "event_ts", "type": "timestamp"},
            ],
        },
    ],
    bridges=[
        {
            "name": "bridge_patient_diagnosis",
            "left": "dim_patient",
            "right": "dim_diagnosis",
            "cardinality": [1, 4],
            "driver": "readmission_risk",
        },
    ],
    holdout={"target": "readmission_risk", "periods": 3},
)
