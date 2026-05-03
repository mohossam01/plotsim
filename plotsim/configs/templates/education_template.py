"""University student academic performance — Python builder template.

Mirror of ``education_template.yaml``. Demonstrates:

* ``noise: realistic`` — closer to live LMS data than perfectly clean
* multi-effect ``seasonality`` — fall (Sep-Nov) + spring (Feb-Apr)
  lifts with summer (Jun-Aug) and finals-week (Dec) dips
* SCD2 ``academic_standing`` tracking ``assignment_score``
"""
from plotsim import create

config = create(

    about="University student academic performance and engagement",
    unit="student",

    seed=31337,
    noise="realistic",

    window=("2023-01", "2024-12", "monthly"),

    seasonality=[
        {"months": [9, 10, 11], "strength": 0.20},
        {"months": [2, 3, 4], "strength": 0.15},
        {"months": [6, 7, 8], "strength": -0.30},
        {"months": [12], "strength": -0.10},
    ],

    metrics=[
        {"name": "assignment_score", "label": "Average assignment score",
         "type": "amount", "polarity": "positive", "range": [0, 100]},

        {"name": "attendance_rate", "label": "Class attendance rate",
         "type": "score", "polarity": "positive"},

        {"name": "study_hours", "label": "Weekly study hours",
         "type": "count", "polarity": "positive"},

        {"name": "participation", "label": "Class participation index",
         "type": "score", "polarity": "positive"},

        {"name": "dropout_risk", "label": "Dropout risk score",
         "type": "score", "polarity": "negative"},

        {"name": "stress_level", "label": "Reported stress index",
         "type": "score", "polarity": "negative",
         "follows": "study_hours", "delay": 1},
    ],

    connections=[
        ("attendance_rate", "driven_by", "participation"),
        ("assignment_score", "related", "attendance_rate"),
        ("participation", "opposes", "dropout_risk"),
        ("assignment_score", "resists", "dropout_risk"),
        ("stress_level", "hints_at", "dropout_risk"),
    ],

    segments=[
        {"name": "high_achievers", "count": 25, "archetype": "growth",
         "label": "Steady academic climb across both years",
         "attributes": {"program": ["computer_science", "engineering"],
                        "year": ["sophomore", "junior"]},
         "baseline": {"assignment_score": "high", "attendance_rate": "high",
                      "dropout_risk": "low"}},

        {"name": "late_bloomers", "count": 20,
         "archetype": "flat > growth @ 8",
         "label": "Struggled the first two terms, then found their footing",
         "attributes": {"program": ["biology", "mathematics", "history"],
                        "year": ["freshman", "sophomore"]},
         "baseline": {"assignment_score": "mid", "participation": "mid"}},

        {"name": "early_peakers", "count": 15,
         "archetype": "growth > decline @ 14",
         "label": "Strong start, fade by senior year",
         "attributes": {"program": ["business", "communications"],
                        "year": ["junior", "senior"]},
         "baseline": {"assignment_score": "high", "attendance_rate": "mid"}},

        {"name": "at_risk", "count": 18, "archetype": "decline",
         "label": "Steady decline — at risk of dropout from term one",
         "attributes": {"program": ["undeclared", "business"],
                        "year": ["freshman"]},
         "baseline": {"assignment_score": "low", "attendance_rate": "low",
                      "dropout_risk": "high"}},

        {"name": "exam_burnout", "count": 10,
         "archetype": "growth > spike_then_crash > flat @ 8 @ 16",
         "label": "Pushed hard before finals, crashed, never recovered",
         "attributes": {"program": ["pre_med", "engineering"],
                        "year": ["junior", "senior"]},
         "baseline": {"study_hours": "high", "stress_level": "high",
                      "dropout_risk": "high"}},

        {"name": "seasonal_engagement", "count": 12, "archetype": "seasonal",
         "label": "Cyclical engagement — strong terms, weak summers",
         "attributes": {"program": ["arts", "music", "history"],
                        "year": ["sophomore", "junior"]},
         "baseline": {"participation": "mid", "attendance_rate": "mid"}},
    ],

    lifecycle={
        "track": "dropout_risk",
        "stages": [
            ("thriving", 0.0),
            ("stable", 0.15),
            ("struggling", 0.4),
            ("critical", 0.7),
        ],
    },

    dimensions=[
        {"name": "dim_date", "per": "period",
         "columns": [
             {"name": "date_key", "type": "id"},
             {"name": "date", "type": "date"},
             {"name": "year", "type": "int"},
             {"name": "month", "type": "int"},
             {"name": "quarter", "type": "int"},
         ]},

        {"name": "dim_student", "per": "unit",
         "columns": [
             {"name": "student_id", "type": "id"},
             {"name": "full_name", "type": "faker.name"},
             {"name": "enroll_year", "type": "faker.year"},
             {"name": "cohort_size", "type": "segment.count"},
             {"name": "academic_standing", "type": "scd",
              "tracks": "assignment_score",
              "tiers": ["probation", "good_standing", "deans_list"],
              "at": [0.4, 0.8]},
         ]},

        {"name": "dim_course", "reference": True,
         "columns": [
             {"name": "course_id", "type": "id"},
             {"name": "course_name",
              "type": "static.intro_cs,calculus,history,physics,literature,statistics"},
             {"name": "credits", "type": "static.3,4,3,4,3,4"},
             {"name": "department",
              "type": "static.computing,math,humanities,sciences,humanities,math"},
         ]},

        {"name": "dim_term", "reference": True,
         "columns": [
             {"name": "term_id", "type": "id"},
             {"name": "term_name",
              "type": "static.spring_2023,fall_2023,spring_2024,fall_2024"},
             {"name": "term_type", "type": "static.spring,fall,spring,fall"},
         ]},
    ],

    facts=[
        {"name": "fct_grades",
         "metrics": ["assignment_score", "participation"],
         "columns": [
             {"name": "date_key", "type": "ref.dim_date"},
             {"name": "student_id", "type": "ref.dim_student"},
             {"name": "course_id", "type": "ref.dim_course"},
             {"name": "assignment_score", "type": "metric.assignment_score"},
             {"name": "participation", "type": "metric.participation"},
             {"name": "grade_band", "type": "bucket",
              "labels": ["F", "D", "C", "B", "A"]},
         ]},

        {"name": "fct_engagement",
         "metrics": ["attendance_rate", "study_hours", "stress_level"],
         "columns": [
             {"name": "date_key", "type": "ref.dim_date"},
             {"name": "student_id", "type": "ref.dim_student"},
             {"name": "term_id", "type": "ref.dim_term"},
             {"name": "attendance_rate", "type": "metric.attendance_rate"},
             {"name": "study_hours", "type": "metric.study_hours"},
             {"name": "stress_level", "type": "metric.stress_level"},
         ]},

        {"name": "fct_risk",
         "metrics": ["dropout_risk"],
         "columns": [
             {"name": "date_key", "type": "ref.dim_date"},
             {"name": "student_id", "type": "ref.dim_student"},
             {"name": "dropout_risk", "type": "metric.dropout_risk"},
         ]},
    ],

    events=[
        {"name": "evt_office_hours",
         "trigger": "proportional", "driver": "participation", "scale": 3.0,
         "columns": [
             {"name": "event_id", "type": "id"},
             {"name": "date_key", "type": "ref.dim_date"},
             {"name": "student_id", "type": "ref.dim_student"},
             {"name": "event_ts", "type": "timestamp"},
         ]},

        {"name": "evt_dropout",
         "trigger": "threshold", "metric": "dropout_risk",
         "above": 0.65, "for_periods": 2,
         "columns": [
             {"name": "event_id", "type": "id"},
             {"name": "date_key", "type": "ref.dim_date"},
             {"name": "student_id", "type": "ref.dim_student"},
             {"name": "reason", "type": "faker.sentence"},
             {"name": "dropout_flag", "type": "flag"},
         ]},
    ],
)
