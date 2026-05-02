# M127 verification â€” measurement report

_Seed: 42    Alt seed: 43    Configs: saas, retail, education, marketing, hr, stress, degenerate_

## Statistical profile

### saas

| table               | metric           |         mean |          std |      min |   max |   nulls | dtype   |
|:--------------------|:-----------------|-------------:|-------------:|---------:|------:|--------:|:--------|
| fct_engagement      | feature_adoption |     0.39517  |     0.332888 |    0     |     1 |       0 | float64 |
| fct_revenue         | mrr              | 23632        | 12904.4      |  460.261 | 50000 |       0 | float64 |
| fct_support_tickets | churn_risk       |     0.597251 |     0.329789 |    0     |     1 |       0 | float64 |
| fct_support_tickets | nps              |    -0.530968 |    32.7412   | -100     |   100 |       0 | float64 |

### retail

| table            | metric               |       mean |        std |   min |   max |   nulls | dtype   |
|:-----------------|:---------------------|-----------:|-----------:|------:|------:|--------:|:--------|
| fct_sessions     | conversion_rate      |   0.426008 |   0.27244  |     0 |     1 |       0 | float64 |
| fct_purchases    | cart_value           | 827.257    | 549.117    |    10 |  2000 |       0 | float64 |
| fct_purchases    | return_rate          |   0.535596 |   0.343928 |     0 |     1 |       0 | float64 |
| fct_purchases    | loyalty_score        |   0.474548 |   0.343776 |     0 |     1 |       0 | float64 |
| fct_purchases    | repeat_purchase_rate |   0.406099 |   0.321957 |     0 |     1 |       0 | float64 |
| fct_satisfaction | nps                  |  -0.674912 |  33.5346   |  -100 |   100 |       0 | float64 |

### education

| table          | metric           |      mean |       std |         min |   max |   nulls | dtype   |
|:---------------|:-----------------|----------:|----------:|------------:|------:|--------:|:--------|
| fct_grades     | assignment_score | 45.0004   | 26.7672   | 2.12629e-10 |   100 |       0 | float64 |
| fct_grades     | participation    |  0.462139 |  0.283327 | 0           |     1 |       0 | float64 |
| fct_engagement | attendance_rate  |  0.487332 |  0.3102   | 0           |     1 |       0 | float64 |
| fct_engagement | study_hours      |  2.28625  |  2.23573  | 0           |    13 |       0 | Int64   |
| fct_engagement | stress_level     |  0.582725 |  0.330585 | 0           |     1 |       0 | float64 |
| fct_risk       | dropout_risk     |  0.540661 |  0.346754 | 0           |     1 |       0 | float64 |

### marketing

| table       | metric             |         mean |          std |           min |          max |   nulls | dtype   |
|:------------|:-------------------|-------------:|-------------:|--------------:|-------------:|--------:|:--------|
| fct_spend   | ad_spend           | 23191.9      | 12578.5      | 500           |  50000       |       0 | float64 |
| fct_spend   | impressions        |     2.24462  |     2.2313   |   0           |     11       |       0 | Int64   |
| fct_spend   | click_through_rate |     0.453344 |     0.332855 |   0           |      1       |       0 | float64 |
| fct_funnel  | conversion_rate    |     0.398683 |     0.28697  |   0           |      1       |       0 | float64 |
| fct_funnel  | bounce_rate        |     0.59432  |     0.301519 |   0           |      1       |       0 | float64 |
| fct_funnel  | leads_generated    |     2.20565  |     2.21196  |   0           |     12       |       0 | Int64   |
| fct_revenue | revenue            | 85700        | 70442.6      |   8.07432e-11 | 250000       |       0 | float64 |
| fct_revenue | roi                |     1.14648  |     1.27616  |  -1           |      4.72611 |       0 | float64 |

### hr

| table            | metric            |         mean |         std |   min |   max |   nulls | dtype   |
|:-----------------|:------------------|-------------:|------------:|------:|------:|--------:|:--------|
| fct_performance  | performance_score |     0.585483 |    0.233103 |     0 |     1 |       0 | float64 |
| fct_performance  | engagement        |     0.487617 |    0.248113 |     0 |     1 |       0 | float64 |
| fct_performance  | training_hours    |     1.54912  |    1.86925  |     0 |    10 |       0 | Int64   |
| fct_compensation | compensation      | 11419.4      | 6851.24     |  4000 | 25000 |       0 | float64 |
| fct_compensation | attrition_risk    |     0.555985 |    0.320334 |     0 |     1 |       0 | float64 |
| fct_attendance   | absence_rate      |     0.719976 |    0.279645 |     0 |     1 |       0 | float64 |

### stress

| table       | metric   |     mean |      std |   min |   max |   nulls | dtype   |
|:------------|:---------|---------:|---------:|------:|------:|--------:|:--------|
| fct_account | m00      | 0.465356 | 0.319153 |     0 |     1 |       0 | float64 |
| fct_account | m01      | 0.464093 | 0.319082 |     0 |     1 |       0 | float64 |
| fct_account | m02      | 0.465566 | 0.317844 |     0 |     1 |       0 | float64 |
| fct_account | m03      | 0.464191 | 0.320225 |     0 |     1 |       0 | float64 |
| fct_account | m04      | 0.46547  | 0.318376 |     0 |     1 |       0 | float64 |
| fct_account | m05      | 0.465295 | 0.319073 |     0 |     1 |       0 | float64 |
| fct_account | m06      | 0.46498  | 0.319283 |     0 |     1 |       0 | float64 |
| fct_account | m07      | 0.464294 | 0.318436 |     0 |     1 |       0 | float64 |
| fct_account | m08      | 0.464368 | 0.319388 |     0 |     1 |       0 | float64 |
| fct_account | m09      | 0.463653 | 0.318607 |     0 |     1 |       0 | float64 |
| fct_account | m10      | 0.464676 | 0.317417 |     0 |     1 |       0 | float64 |
| fct_account | m11      | 0.46537  | 0.318813 |     0 |     1 |       0 | float64 |
| fct_account | m12      | 0.465231 | 0.320181 |     0 |     1 |       0 | float64 |
| fct_account | m13      | 0.464824 | 0.319379 |     0 |     1 |       0 | float64 |
| fct_account | m14      | 0.463625 | 0.318771 |     0 |     1 |       0 | float64 |
| fct_account | m15      | 0.465464 | 0.318811 |     0 |     1 |       0 | float64 |
| fct_account | m16      | 0.465647 | 0.31957  |     0 |     1 |       0 | float64 |
| fct_account | m17      | 0.464755 | 0.318103 |     0 |     1 |       0 | float64 |
| fct_account | m18      | 0.466363 | 0.319388 |     0 |     1 |       0 | float64 |
| fct_account | m19      | 0.464214 | 0.319025 |     0 |     1 |       0 | float64 |

### degenerate

| table       | metric     |     mean |      std |   min |   max |   nulls | dtype   |
|:------------|:-----------|---------:|---------:|------:|------:|--------:|:--------|
| fct_account | score_a    | 0.433873 | 0.284832 |     0 |     1 |       0 | float64 |
| fct_account | rare_event | 2.15958  | 1.92547  |     0 |    11 |       0 | Int64   |

## Shape recovery

| template   | archetype            | metric               |   pearson_r |
|:-----------|:---------------------|:---------------------|------------:|
| saas       | promising_client     | feature_adoption     |    0.997392 |
| saas       | steady_enterprise    | feature_adoption     |    0.995949 |
| saas       | slow_churn           | feature_adoption     |    0.988288 |
| saas       | seasonal_accounts    | feature_adoption     |    0.991124 |
| saas       | dormant              | feature_adoption     |  nan        |
| saas       | turnaround           | feature_adoption     |    0.990249 |
| retail     | loyal_climbers       | cart_value           |    0.895967 |
| retail     | loyal_climbers       | return_rate          |   -0.997206 |
| retail     | loyal_climbers       | loyalty_score        |    0.996532 |
| retail     | loyal_climbers       | repeat_purchase_rate |    0.991557 |
| retail     | holiday_shoppers     | cart_value           |    0.933552 |
| retail     | holiday_shoppers     | return_rate          |   -0.993706 |
| retail     | holiday_shoppers     | loyalty_score        |    0.99485  |
| retail     | holiday_shoppers     | repeat_purchase_rate |    0.827028 |
| retail     | cooled_off           | cart_value           |    0.899242 |
| retail     | cooled_off           | return_rate          |   -0.985267 |
| retail     | cooled_off           | loyalty_score        |    0.990779 |
| retail     | cooled_off           | repeat_purchase_rate |    0.699301 |
| retail     | one_and_done         | cart_value           |    0.928841 |
| retail     | one_and_done         | return_rate          |   -0.990347 |
| retail     | one_and_done         | loyalty_score        |    0.994667 |
| retail     | one_and_done         | repeat_purchase_rate |    0.117236 |
| retail     | winback              | cart_value           |    0.940956 |
| retail     | winback              | return_rate          |   -0.987932 |
| retail     | winback              | loyalty_score        |    0.991932 |
| retail     | winback              | repeat_purchase_rate |    0.912222 |
| retail     | escalating_basket    | cart_value           |    0.773502 |
| retail     | escalating_basket    | return_rate          |   -0.984861 |
| retail     | escalating_basket    | loyalty_score        |    0.992186 |
| retail     | escalating_basket    | repeat_purchase_rate |    0.988055 |
| education  | high_achievers       | attendance_rate      |    0.997358 |
| education  | high_achievers       | study_hours          |    0.990901 |
| education  | high_achievers       | stress_level         |   -0.990425 |
| education  | late_bloomers        | attendance_rate      |    0.995628 |
| education  | late_bloomers        | study_hours          |    0.992403 |
| education  | late_bloomers        | stress_level         |   -0.988802 |
| education  | early_peakers        | attendance_rate      |    0.994577 |
| education  | early_peakers        | study_hours          |    0.976792 |
| education  | early_peakers        | stress_level         |   -0.959046 |
| education  | at_risk              | attendance_rate      |    0.989715 |
| education  | at_risk              | study_hours          |    0.96406  |
| education  | at_risk              | stress_level         |   -0.989296 |
| education  | exam_burnout         | attendance_rate      |    0.981789 |
| education  | exam_burnout         | study_hours          |    0.977563 |
| education  | exam_burnout         | stress_level         |   -0.658381 |
| education  | seasonal_engagement  | attendance_rate      |    0.982148 |
| education  | seasonal_engagement  | study_hours          |    0.925699 |
| education  | seasonal_engagement  | stress_level         |   -0.852009 |
| marketing  | awareness_builder    | conversion_rate      |    0.996911 |
| marketing  | awareness_builder    | bounce_rate          |   -0.993831 |
| marketing  | awareness_builder    | leads_generated      |    0.961667 |
| marketing  | paid_burst           | conversion_rate      |    0.995441 |
| marketing  | paid_burst           | bounce_rate          |   -0.993889 |
| marketing  | paid_burst           | leads_generated      |    0.663529 |
| marketing  | seasonal_promo       | conversion_rate      |    0.993864 |
| marketing  | seasonal_promo       | bounce_rate          |   -0.995123 |
| marketing  | seasonal_promo       | leads_generated      |    0.812791 |
| marketing  | delayed_breakthrough | conversion_rate      |    0.992179 |
| marketing  | delayed_breakthrough | bounce_rate          |   -0.989872 |
| marketing  | delayed_breakthrough | leads_generated      |    0.978483 |
| marketing  | viral_compound       | conversion_rate      |    0.993721 |
| marketing  | viral_compound       | bounce_rate          |   -0.98544  |
| marketing  | viral_compound       | leads_generated      |    0.944779 |
| marketing  | end_of_life          | conversion_rate      |    0.97243  |
| marketing  | end_of_life          | bounce_rate          |   -0.970987 |
| marketing  | end_of_life          | leads_generated      |    0.919936 |
| marketing  | retarget_revival     | conversion_rate      |    0.989643 |
| marketing  | retarget_revival     | bounce_rate          |   -0.989634 |
| marketing  | retarget_revival     | leads_generated      |    0.874555 |
| hr         | new_hire_ramp        | absence_rate         |   -0.985526 |
| hr         | core_team            | absence_rate         |  nan        |
| hr         | fast_riser           | absence_rate         |   -0.988196 |
| hr         | quiet_quitter        | absence_rate         |   -0.586398 |
| hr         | burnout_cohort       | absence_rate         |   -0.607708 |
| hr         | comeback             | absence_rate         |   -0.929242 |
| stress     | growth_seg           | m00                  |    0.99902  |
| stress     | growth_seg           | m01                  |    0.998901 |
| stress     | growth_seg           | m02                  |    0.999076 |
| stress     | growth_seg           | m03                  |    0.998785 |
| stress     | growth_seg           | m04                  |    0.999136 |
| stress     | growth_seg           | m05                  |    0.998793 |
| stress     | growth_seg           | m06                  |    0.998995 |
| stress     | growth_seg           | m07                  |    0.999027 |
| stress     | growth_seg           | m08                  |    0.998947 |
| stress     | growth_seg           | m09                  |    0.99907  |
| stress     | growth_seg           | m10                  |    0.999069 |
| stress     | growth_seg           | m11                  |    0.999069 |
| stress     | growth_seg           | m12                  |    0.998936 |
| stress     | growth_seg           | m13                  |    0.998909 |
| stress     | growth_seg           | m14                  |    0.99898  |
| stress     | growth_seg           | m15                  |    0.999066 |
| stress     | growth_seg           | m16                  |    0.998864 |
| stress     | growth_seg           | m17                  |    0.999175 |
| stress     | growth_seg           | m18                  |    0.998852 |
| stress     | growth_seg           | m19                  |    0.99905  |
| stress     | decline_seg          | m00                  |    0.999007 |
| stress     | decline_seg          | m01                  |    0.999096 |
| stress     | decline_seg          | m02                  |    0.998789 |
| stress     | decline_seg          | m03                  |    0.999047 |
| stress     | decline_seg          | m04                  |    0.998979 |
| stress     | decline_seg          | m05                  |    0.998812 |
| stress     | decline_seg          | m06                  |    0.998956 |
| stress     | decline_seg          | m07                  |    0.998853 |
| stress     | decline_seg          | m08                  |    0.998828 |
| stress     | decline_seg          | m09                  |    0.998967 |
| stress     | decline_seg          | m10                  |    0.999167 |
| stress     | decline_seg          | m11                  |    0.998927 |
| stress     | decline_seg          | m12                  |    0.998516 |
| stress     | decline_seg          | m13                  |    0.998888 |
| stress     | decline_seg          | m14                  |    0.999184 |
| stress     | decline_seg          | m15                  |    0.998833 |
| stress     | decline_seg          | m16                  |    0.998931 |
| stress     | decline_seg          | m17                  |    0.999128 |
| stress     | decline_seg          | m18                  |    0.999055 |
| stress     | decline_seg          | m19                  |    0.998797 |
| degenerate | near_zero            | score_a              |    0.997544 |
| degenerate | near_zero            | rare_event           |    0.995191 |

## Correlation signs

| template   | metric_a    | metric_b        |   configured_coef |   realized_r | configured_sign   | realized_sign   | match   |
|:-----------|:------------|:----------------|------------------:|-------------:|:------------------|:----------------|:--------|
| retail     | cart_value  | loyalty_score   |              0.4  |        0.727 | +                 | +               | True    |
| retail     | return_rate | loyalty_score   |             -0.55 |       -0.774 | -                 | -               | True    |
| marketing  | bounce_rate | conversion_rate |             -0.55 |       -0.392 | -                 | -               | True    |
| stress     | m00         | m01             |              0.75 |        0.754 | +                 | +               | True    |
| stress     | m02         | m03             |             -0.55 |        0.617 | -                 | +               | False   |
| stress     | m04         | m05             |              0.55 |        0.717 | +                 | +               | True    |
| stress     | m06         | m07             |             -0.4  |        0.621 | -                 | +               | False   |
| degenerate | score_a     | rare_event      |              0.75 |        0.466 | +                 | +               | True    |

## Degenerate centers

|   rare_event_nan_count |   rare_event_inf_count |   rare_event_mean |   rare_event_std |   rare_event_min |   rare_event_max |   score_x_rare_event_pearson | deterministic_repeat   |
|-----------------------:|-----------------------:|------------------:|-----------------:|-----------------:|-----------------:|-----------------------------:|:-----------------------|
|                      0 |                      0 |           2.15958 |          1.92547 |                0 |               11 |                       0.4658 | True                   |

## Determinism

| template   | same_seed_identical   | alt_seed_differs   | alt_seed_same_shape   |
|:-----------|:----------------------|:-------------------|:----------------------|
| saas       | True                  | True               | True                  |
| retail     | True                  | True               | True                  |
| education  | True                  | True               | True                  |
| marketing  | True                  | True               | True                  |
| hr         | True                  | True               | True                  |
| stress     | True                  | True               | True                  |
| degenerate | True                  | True               | True                  |

## FakerSource parity

| template   | table                | column            | match   | note   |
|:-----------|:---------------------|:------------------|:--------|:-------|
| saas       | dim_plan             | plan_id           | True    |        |
| saas       | dim_plan             | plan_name         | True    |        |
| saas       | dim_company          | company_id        | True    |        |
| saas       | dim_company          | company_name      | True    |        |
| saas       | dim_company          | industry          | True    |        |
| saas       | dim_company          | plan_tier         | True    |        |
| saas       | dim_user             | user_id           | True    |        |
| saas       | dim_user             | company_id        | True    |        |
| saas       | dim_user             | user_name         | True    |        |
| saas       | dim_user             | role              | True    |        |
| retail     | dim_product_category | category_id       | True    |        |
| retail     | dim_product_category | category_name     | True    |        |
| retail     | dim_product_category | margin_tier       | True    |        |
| retail     | dim_channel          | channel_id        | True    |        |
| retail     | dim_channel          | channel_name      | True    |        |
| retail     | dim_channel          | channel_type      | True    |        |
| retail     | dim_promotion        | promotion_id      | True    |        |
| retail     | dim_promotion        | promo_name        | True    |        |
| retail     | dim_promotion        | discount_type     | True    |        |
| retail     | dim_customer         | customer_id       | True    |        |
| retail     | dim_customer         | customer_name     | True    |        |
| retail     | dim_customer         | customer_tier     | True    |        |
| education  | dim_course           | course_id         | True    |        |
| education  | dim_course           | course_name       | True    |        |
| education  | dim_course           | credits           | True    |        |
| education  | dim_course           | department        | True    |        |
| education  | dim_term             | term_id           | True    |        |
| education  | dim_term             | term_name         | True    |        |
| education  | dim_term             | term_type         | True    |        |
| education  | dim_student          | student_id        | True    |        |
| education  | dim_student          | full_name         | True    |        |
| education  | dim_student          | academic_standing | True    |        |
| marketing  | dim_channel          | channel_id        | True    |        |
| marketing  | dim_channel          | channel_name      | True    |        |
| marketing  | dim_channel          | channel_type      | True    |        |
| marketing  | dim_audience         | audience_id       | True    |        |
| marketing  | dim_audience         | audience_name     | True    |        |
| marketing  | dim_audience         | audience_size     | True    |        |
| marketing  | dim_creative         | creative_id       | True    |        |
| marketing  | dim_creative         | format            | True    |        |
| marketing  | dim_creative         | variant           | True    |        |
| marketing  | dim_campaign         | campaign_id       | True    |        |
| marketing  | dim_campaign         | campaign_name     | True    |        |
| marketing  | dim_campaign         | campaign_phase    | True    |        |
| hr         | dim_department       | department_id     | True    |        |
| hr         | dim_department       | department        | True    |        |
| hr         | dim_department       | cost_center       | True    |        |
| hr         | dim_office           | office_id         | True    |        |
| hr         | dim_office           | office            | True    |        |
| hr         | dim_office           | region            | True    |        |
| hr         | dim_employee         | employee_id       | True    |        |
| hr         | dim_employee         | full_name         | True    |        |
| hr         | dim_employee         | job_level         | True    |        |
| stress     | dim_account          | account_id        | True    |        |
| stress     | dim_account          | account_name      | True    |        |
| degenerate | dim_account          | account_id        | True    |        |
| degenerate | dim_account          | account_name      | True    |        |

## Validation parity

| config_name                      | error_type      | error_message                                                                                        |
|:---------------------------------|:----------------|:-----------------------------------------------------------------------------------------------------|
| missing_polarity                 | ValidationError | 1 validation error for UserInput metrics.0.polarity   Field required [type=missing, input_value={'na |
| unknown_archetype_shape          | ValidationError | 1 validation error for UserInput   Value error, segment 's1' archetype 'exploding_donkey': archetype |
| duplicate_metric_name            | ValidationError | 1 validation error for UserInput   Value error, duplicate metric name(s): ['m1'] [type=value_error,  |
| connection_to_nonexistent_metric | ValidationError | 1 validation error for UserInput   Value error, connection 'm1' mirrors 'ghost': endpoint 'ghost' is |
| reserved_plus_in_archetype_dsl   | ValidationError | 1 validation error for UserInput   Value error, segment 's1' archetype 'growth + decline': Layered p |
| count_with_range                 | ValidationError | 1 validation error for UserInput metrics.0   Value error, metric 'm1' of type 'count' must not decla |
| amount_without_range             | ValidationError | 1 validation error for UserInput metrics.0   Value error, metric 'm1' of type 'amount' requires a `r |

## Performance

| config   |   serial_s |   vectorized_s |   speedup |   copula_pct |
|:---------|-----------:|---------------:|----------:|-------------:|
| saas     |      9.196 |          2.41  |      3.82 |        nan   |
| stress   |    661.903 |          3.945 |    167.77 |         99.3 |

## Fixture consistency

| template   | check                                                     | passed   | detail   |
|:-----------|:----------------------------------------------------------|:---------|:---------|
| education  | PK_unique[dim_course.course_id]                           | True     |          |
| education  | PK_unique[dim_student.student_id]                         | False    | 4 dupes  |
| education  | PK_unique[dim_student.dim_row_id]                         | True     |          |
| education  | FK[bridge_enrollment.course_id->dim_course]               | True     |          |
| education  | FK[evt_dropout.student_id->dim_student]                   | True     |          |
| education  | FK[fct_engagement.student_id->dim_student]                | True     |          |
| education  | FK[fct_grades.student_id->dim_student]                    | True     |          |
| education  | FK[fct_grades.course_id->dim_course]                      | True     |          |
| education  | no_nulls[fct_engagement.attendance_rate]                  | True     |          |
| education  | no_nulls[fct_engagement.study_hours]                      | True     |          |
| education  | no_nulls[fct_engagement.dropout_risk]                     | False    | 1 nulls  |
| education  | no_nulls[fct_grades.assignment_score]                     | True     |          |
| education  | no_nulls[fct_grades.participation_index]                  | True     |          |
| hr         | PK_unique[dim_department.dept_id]                         | True     |          |
| hr         | PK_unique[dim_employee.employee_id]                       | True     |          |
| hr         | PK_unique[dim_employee.dept_id]                           | False    | 3 dupes  |
| hr         | FK[evt_attrition.employee_id->dim_employee]               | True     |          |
| hr         | FK[fct_attendance.employee_id->dim_employee]              | True     |          |
| hr         | FK[fct_performance.employee_id->dim_employee]             | True     |          |
| hr         | FK[fct_training.employee_id->dim_employee]                | True     |          |
| hr         | no_nulls[fct_attendance.absence_rate]                     | False    | 2 nulls  |
| hr         | no_nulls[fct_attendance.attrition_risk]                   | False    | 5 nulls  |
| hr         | no_nulls[fct_performance.performance_score]               | False    | 3 nulls  |
| hr         | no_nulls[fct_performance.engagement_index]                | False    | 2 nulls  |
| hr         | no_nulls[fct_training.training_hours]                     | False    | 1 nulls  |
| marketing  | PK_unique[dim_campaign.campaign_id]                       | True     |          |
| marketing  | PK_unique[dim_channel.channel_id]                         | True     |          |
| marketing  | PK_unique[dim_customer.customer_id]                       | False    | 8 dupes  |
| marketing  | PK_unique[dim_customer.dim_row_id]                        | True     |          |
| marketing  | PK_unique[dim_product_category.category_id]               | True     |          |
| marketing  | FK[bridge_customer_campaign.campaign_id->dim_campaign]    | True     |          |
| marketing  | FK[bridge_customer_channel.channel_id->dim_channel]       | True     |          |
| marketing  | FK[evt_churn.customer_id->dim_customer]                   | True     |          |
| marketing  | FK[fct_campaigns.customer_id->dim_customer]               | True     |          |
| marketing  | FK[fct_campaigns.campaign_id->dim_campaign]               | True     |          |
| marketing  | FK[fct_revenue.customer_id->dim_customer]                 | True     |          |
| marketing  | FK[fct_traffic.customer_id->dim_customer]                 | True     |          |
| marketing  | no_nulls[fct_campaigns.ad_spend]                          | True     |          |
| marketing  | no_nulls[fct_campaigns.impressions]                       | False    | 1 nulls  |
| marketing  | no_nulls[fct_campaigns.click_through_rate]                | True     |          |
| marketing  | no_nulls[fct_revenue.revenue]                             | True     |          |
| marketing  | no_nulls[fct_revenue.average_order_value]                 | False    | 2 nulls  |
| marketing  | no_nulls[fct_revenue.customer_lifetime_value]             | False    | 3 nulls  |
| marketing  | no_nulls[fct_traffic.bounce_rate]                         | False    | 1 nulls  |
| marketing  | no_nulls[fct_traffic.conversion_rate]                     | False    | 1 nulls  |
| retail     | PK_unique[dim_customer.segment_id]                        | False    | 5 dupes  |
| retail     | PK_unique[dim_customer.dim_row_id]                        | True     |          |
| retail     | PK_unique[dim_product_category.category_id]               | True     |          |
| retail     | PK_unique[dim_promotion.promotion_id]                     | True     |          |
| retail     | PK_unique[dim_store_type.store_type_id]                   | True     |          |
| retail     | FK[bridge_customer_promotion.promotion_id->dim_promotion] | True     |          |
| retail     | no_nulls[fct_purchases.cart_value]                        | True     |          |
| retail     | no_nulls[fct_purchases.return_rate]                       | True     |          |
| retail     | no_nulls[fct_purchases.loyalty_score]                     | True     |          |
| retail     | no_nulls[fct_purchases.repeat_purchase_rate]              | False    | 1 nulls  |
| retail     | no_nulls[fct_sessions.session_count]                      | True     |          |
| retail     | no_nulls[fct_sessions.conversion_rate]                    | False    | 2 nulls  |
| saas       | PK_unique[dim_company.company_id]                         | False    | 4 dupes  |
| saas       | PK_unique[dim_company.dim_row_id]                         | True     |          |
| saas       | PK_unique[dim_plan.plan_id]                               | True     |          |
| saas       | PK_unique[dim_user.user_id]                               | True     |          |
| saas       | PK_unique[dim_user.company_id]                            | False    | 87 dupes |
| saas       | FK[evt_churn.company_id->dim_company]                     | True     |          |
| saas       | FK[evt_login.user_id->dim_user]                           | True     |          |
| saas       | FK[evt_login.company_id->dim_company]                     | True     |          |
| saas       | FK[fct_engagement.company_id->dim_company]                | True     |          |
| saas       | FK[fct_revenue.company_id->dim_company]                   | True     |          |
| saas       | FK[fct_revenue.plan_id->dim_plan]                         | True     |          |
| saas       | FK[fct_support_tickets.company_id->dim_company]           | True     |          |
| saas       | no_nulls[fct_engagement.feature_adoption]                 | True     |          |
| saas       | no_nulls[fct_revenue.mrr]                                 | True     |          |
| saas       | no_nulls[fct_support_tickets.churn_risk]                  | True     |          |
| saas       | no_nulls[fct_support_tickets.nps]                         | True     |          |
