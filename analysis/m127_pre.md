# M127 verification â€” measurement report

_Seed: 42    Alt seed: 43    Configs: saas, retail, education, marketing, hr, stress, degenerate_

## Statistical profile

### saas

| table               | metric           |         mean |          std |     min |   max |   nulls | dtype   |
|:--------------------|:-----------------|-------------:|-------------:|--------:|------:|--------:|:--------|
| fct_engagement      | feature_adoption |     0.394957 |     0.331    |    0    |     1 |       0 | float64 |
| fct_revenue         | mrr              | 23547.2      | 12831.6      |  249.31 | 50000 |       0 | float64 |
| fct_support_tickets | churn_risk       |     0.599529 |     0.330507 |    0    |     1 |       0 | float64 |
| fct_support_tickets | nps              |    -1.01859  |    33.2768   | -100    |   100 |       0 | float64 |

### retail

| table            | metric               |        mean |        std |   min |   max |   nulls | dtype   |
|:-----------------|:---------------------|------------:|-----------:|------:|------:|--------:|:--------|
| fct_sessions     | conversion_rate      |   0.425683  |   0.270898 |     0 |     1 |       0 | float64 |
| fct_purchases    | cart_value           | 815.498     | 544.624    |    10 |  2000 |       0 | float64 |
| fct_purchases    | return_rate          |   0.534896  |   0.345551 |     0 |     1 |       0 | float64 |
| fct_purchases    | loyalty_score        |   0.477044  |   0.343888 |     0 |     1 |       0 | float64 |
| fct_purchases    | repeat_purchase_rate |   0.404397  |   0.323644 |     0 |     1 |       0 | float64 |
| fct_satisfaction | nps                  |   0.0127546 |  33.775    |  -100 |   100 |       0 | float64 |

### education

| table          | metric           |      mean |       std |         min |   max |   nulls | dtype   |
|:---------------|:-----------------|----------:|----------:|------------:|------:|--------:|:--------|
| fct_grades     | assignment_score | 45.6869   | 26.486    | 2.15938e-10 |   100 |       0 | float64 |
| fct_grades     | participation    |  0.464376 |  0.284688 | 0           |     1 |       0 | float64 |
| fct_engagement | attendance_rate  |  0.485821 |  0.313532 | 0           |     1 |       0 | float64 |
| fct_engagement | study_hours      |  2.38833  |  2.19906  | 0           |    15 |       0 | Int64   |
| fct_engagement | stress_level     |  0.582843 |  0.324995 | 0           |     1 |       0 | float64 |
| fct_risk       | dropout_risk     |  0.542162 |  0.350266 | 0           |     1 |       0 | float64 |

### marketing

| table       | metric             |         mean |          std |           min |    max |   nulls | dtype   |
|:------------|:-------------------|-------------:|-------------:|--------------:|-------:|--------:|:--------|
| fct_spend   | ad_spend           | 23214.2      | 12606.3      | 500           |  50000 |       0 | float64 |
| fct_spend   | impressions        |     2.32661  |     2.23816  |   0           |     13 |       0 | Int64   |
| fct_spend   | click_through_rate |     0.424659 |     0.343288 |   0           |      1 |       0 | float64 |
| fct_funnel  | conversion_rate    |     0.398341 |     0.285392 |   0           |      1 |       0 | float64 |
| fct_funnel  | bounce_rate        |     0.59509  |     0.29821  |   0           |      1 |       0 | float64 |
| fct_funnel  | leads_generated    |     2.31272  |     2.21975  |   0           |     13 |       0 | Int64   |
| fct_revenue | revenue            | 84500.7      | 69093.1      |   1.75057e-10 | 250000 |       0 | float64 |
| fct_revenue | roi                |     1.19108  |     1.29017  |  -1           |      5 |       0 | float64 |

### hr

| table            | metric            |         mean |         std |   min |   max |   nulls | dtype   |
|:-----------------|:------------------|-------------:|------------:|------:|------:|--------:|:--------|
| fct_performance  | performance_score |     0.588641 |    0.233209 |     0 |     1 |       0 | float64 |
| fct_performance  | engagement        |     0.48868  |    0.248739 |     0 |     1 |       0 | float64 |
| fct_performance  | training_hours    |     1.54649  |    1.89078  |     0 |    11 |       0 | Int64   |
| fct_compensation | compensation      | 11465.7      | 6866.42     |  4000 | 25000 |       0 | float64 |
| fct_compensation | attrition_risk    |     0.558661 |    0.324753 |     0 |     1 |       0 | float64 |
| fct_attendance   | absence_rate      |     0.72662  |    0.273937 |     0 |     1 |       0 | float64 |

### stress

| table       | metric   |     mean |      std |   min |   max |   nulls | dtype   |
|:------------|:---------|---------:|---------:|------:|------:|--------:|:--------|
| fct_account | m00      | 0.465319 | 0.319693 |     0 |     1 |       0 | float64 |
| fct_account | m01      | 0.463623 | 0.318444 |     0 |     1 |       0 | float64 |
| fct_account | m02      | 0.46414  | 0.318865 |     0 |     1 |       0 | float64 |
| fct_account | m03      | 0.465406 | 0.319027 |     0 |     1 |       0 | float64 |
| fct_account | m04      | 0.465107 | 0.319243 |     0 |     1 |       0 | float64 |
| fct_account | m05      | 0.465631 | 0.319182 |     0 |     1 |       0 | float64 |
| fct_account | m06      | 0.465299 | 0.319189 |     0 |     1 |       0 | float64 |
| fct_account | m07      | 0.464389 | 0.319339 |     0 |     1 |       0 | float64 |
| fct_account | m08      | 0.464823 | 0.319499 |     0 |     1 |       0 | float64 |
| fct_account | m09      | 0.464418 | 0.318312 |     0 |     1 |       0 | float64 |
| fct_account | m10      | 0.466061 | 0.319422 |     0 |     1 |       0 | float64 |
| fct_account | m11      | 0.464334 | 0.318839 |     0 |     1 |       0 | float64 |
| fct_account | m12      | 0.464789 | 0.318297 |     0 |     1 |       0 | float64 |
| fct_account | m13      | 0.463794 | 0.318095 |     0 |     1 |       0 | float64 |
| fct_account | m14      | 0.464031 | 0.318859 |     0 |     1 |       0 | float64 |
| fct_account | m15      | 0.463982 | 0.319132 |     0 |     1 |       0 | float64 |
| fct_account | m16      | 0.466446 | 0.320187 |     0 |     1 |       0 | float64 |
| fct_account | m17      | 0.463742 | 0.319345 |     0 |     1 |       0 | float64 |
| fct_account | m18      | 0.464097 | 0.318457 |     0 |     1 |       0 | float64 |
| fct_account | m19      | 0.464523 | 0.318624 |     0 |     1 |       0 | float64 |

### degenerate

| table       | metric     |    mean |      std |   min |   max |   nulls | dtype   |
|:------------|:-----------|--------:|---------:|------:|------:|--------:|:--------|
| fct_account | score_a    | 0.42942 | 0.284106 |     0 |     1 |       0 | float64 |
| fct_account | rare_event | 2.67458 | 1.98692  |     0 |    13 |       0 | Int64   |

## Shape recovery

| template   | archetype            | metric               |   pearson_r |
|:-----------|:---------------------|:---------------------|------------:|
| saas       | promising_client     | feature_adoption     |    0.997555 |
| saas       | steady_enterprise    | feature_adoption     |    0.991764 |
| saas       | slow_churn           | feature_adoption     |    0.984156 |
| saas       | seasonal_accounts    | feature_adoption     |    0.993049 |
| saas       | dormant              | feature_adoption     |  nan        |
| saas       | turnaround           | feature_adoption     |    0.989524 |
| retail     | loyal_climbers       | cart_value           |    0.908652 |
| retail     | loyal_climbers       | return_rate          |   -0.996301 |
| retail     | loyal_climbers       | loyalty_score        |    0.996825 |
| retail     | loyal_climbers       | repeat_purchase_rate |    0.994267 |
| retail     | holiday_shoppers     | cart_value           |    0.872876 |
| retail     | holiday_shoppers     | return_rate          |   -0.994139 |
| retail     | holiday_shoppers     | loyalty_score        |    0.994137 |
| retail     | holiday_shoppers     | repeat_purchase_rate |    0.872282 |
| retail     | cooled_off           | cart_value           |    0.967563 |
| retail     | cooled_off           | return_rate          |   -0.981287 |
| retail     | cooled_off           | loyalty_score        |    0.985087 |
| retail     | cooled_off           | repeat_purchase_rate |    0.670516 |
| retail     | one_and_done         | cart_value           |    0.925979 |
| retail     | one_and_done         | return_rate          |   -0.99198  |
| retail     | one_and_done         | loyalty_score        |    0.991243 |
| retail     | one_and_done         | repeat_purchase_rate |    0.156677 |
| retail     | winback              | cart_value           |    0.909812 |
| retail     | winback              | return_rate          |   -0.992376 |
| retail     | winback              | loyalty_score        |    0.991428 |
| retail     | winback              | repeat_purchase_rate |    0.913585 |
| retail     | escalating_basket    | cart_value           |    0.67809  |
| retail     | escalating_basket    | return_rate          |   -0.991922 |
| retail     | escalating_basket    | loyalty_score        |    0.989435 |
| retail     | escalating_basket    | repeat_purchase_rate |    0.982649 |
| education  | high_achievers       | attendance_rate      |    0.997483 |
| education  | high_achievers       | study_hours          |    0.975745 |
| education  | high_achievers       | stress_level         |   -0.992319 |
| education  | late_bloomers        | attendance_rate      |    0.995627 |
| education  | late_bloomers        | study_hours          |    0.97249  |
| education  | late_bloomers        | stress_level         |   -0.984243 |
| education  | early_peakers        | attendance_rate      |    0.989433 |
| education  | early_peakers        | study_hours          |    0.978417 |
| education  | early_peakers        | stress_level         |   -0.961074 |
| education  | at_risk              | attendance_rate      |    0.986459 |
| education  | at_risk              | study_hours          |    0.977479 |
| education  | at_risk              | stress_level         |   -0.978299 |
| education  | exam_burnout         | attendance_rate      |    0.989111 |
| education  | exam_burnout         | study_hours          |    0.971174 |
| education  | exam_burnout         | stress_level         |   -0.630161 |
| education  | seasonal_engagement  | attendance_rate      |    0.988114 |
| education  | seasonal_engagement  | study_hours          |    0.94607  |
| education  | seasonal_engagement  | stress_level         |   -0.835782 |
| marketing  | awareness_builder    | conversion_rate      |    0.996838 |
| marketing  | awareness_builder    | bounce_rate          |   -0.993777 |
| marketing  | awareness_builder    | leads_generated      |    0.974062 |
| marketing  | paid_burst           | conversion_rate      |    0.996305 |
| marketing  | paid_burst           | bounce_rate          |   -0.996371 |
| marketing  | paid_burst           | leads_generated      |    0.623232 |
| marketing  | seasonal_promo       | conversion_rate      |    0.993485 |
| marketing  | seasonal_promo       | bounce_rate          |   -0.992646 |
| marketing  | seasonal_promo       | leads_generated      |    0.879296 |
| marketing  | delayed_breakthrough | conversion_rate      |    0.989715 |
| marketing  | delayed_breakthrough | bounce_rate          |   -0.9953   |
| marketing  | delayed_breakthrough | leads_generated      |    0.960537 |
| marketing  | viral_compound       | conversion_rate      |    0.987669 |
| marketing  | viral_compound       | bounce_rate          |   -0.984821 |
| marketing  | viral_compound       | leads_generated      |    0.964768 |
| marketing  | end_of_life          | conversion_rate      |    0.974297 |
| marketing  | end_of_life          | bounce_rate          |   -0.973805 |
| marketing  | end_of_life          | leads_generated      |    0.955001 |
| marketing  | retarget_revival     | conversion_rate      |    0.986411 |
| marketing  | retarget_revival     | bounce_rate          |   -0.991031 |
| marketing  | retarget_revival     | leads_generated      |    0.935024 |
| hr         | new_hire_ramp        | absence_rate         |   -0.986989 |
| hr         | core_team            | absence_rate         |  nan        |
| hr         | fast_riser           | absence_rate         |   -0.984727 |
| hr         | quiet_quitter        | absence_rate         |   -0.622618 |
| hr         | burnout_cohort       | absence_rate         |   -0.651178 |
| hr         | comeback             | absence_rate         |   -0.929759 |
| stress     | growth_seg           | m00                  |    0.998866 |
| stress     | growth_seg           | m01                  |    0.999098 |
| stress     | growth_seg           | m02                  |    0.998963 |
| stress     | growth_seg           | m03                  |    0.999007 |
| stress     | growth_seg           | m04                  |    0.999031 |
| stress     | growth_seg           | m05                  |    0.998966 |
| stress     | growth_seg           | m06                  |    0.999102 |
| stress     | growth_seg           | m07                  |    0.999024 |
| stress     | growth_seg           | m08                  |    0.999089 |
| stress     | growth_seg           | m09                  |    0.99891  |
| stress     | growth_seg           | m10                  |    0.998758 |
| stress     | growth_seg           | m11                  |    0.998856 |
| stress     | growth_seg           | m12                  |    0.99903  |
| stress     | growth_seg           | m13                  |    0.999222 |
| stress     | growth_seg           | m14                  |    0.99918  |
| stress     | growth_seg           | m15                  |    0.998957 |
| stress     | growth_seg           | m16                  |    0.998724 |
| stress     | growth_seg           | m17                  |    0.998927 |
| stress     | growth_seg           | m18                  |    0.999129 |
| stress     | growth_seg           | m19                  |    0.999022 |
| stress     | decline_seg          | m00                  |    0.998718 |
| stress     | decline_seg          | m01                  |    0.999045 |
| stress     | decline_seg          | m02                  |    0.999047 |
| stress     | decline_seg          | m03                  |    0.999046 |
| stress     | decline_seg          | m04                  |    0.999125 |
| stress     | decline_seg          | m05                  |    0.998926 |
| stress     | decline_seg          | m06                  |    0.999092 |
| stress     | decline_seg          | m07                  |    0.998828 |
| stress     | decline_seg          | m08                  |    0.99904  |
| stress     | decline_seg          | m09                  |    0.999155 |
| stress     | decline_seg          | m10                  |    0.998995 |
| stress     | decline_seg          | m11                  |    0.998948 |
| stress     | decline_seg          | m12                  |    0.999022 |
| stress     | decline_seg          | m13                  |    0.999012 |
| stress     | decline_seg          | m14                  |    0.998663 |
| stress     | decline_seg          | m15                  |    0.998927 |
| stress     | decline_seg          | m16                  |    0.998938 |
| stress     | decline_seg          | m17                  |    0.999032 |
| stress     | decline_seg          | m18                  |    0.998865 |
| stress     | decline_seg          | m19                  |    0.998812 |
| degenerate | near_zero            | score_a              |    0.997776 |
| degenerate | near_zero            | rare_event           |    0.993202 |

## Correlation signs

| template   | metric_a    | metric_b        |   configured_coef |   realized_r | configured_sign   | realized_sign   | match   |
|:-----------|:------------|:----------------|------------------:|-------------:|:------------------|:----------------|:--------|
| retail     | cart_value  | loyalty_score   |              0.4  |        0.713 | +                 | +               | True    |
| retail     | return_rate | loyalty_score   |             -0.55 |       -0.778 | -                 | -               | True    |
| marketing  | bounce_rate | conversion_rate |             -0.55 |       -0.391 | -                 | -               | True    |
| stress     | m00         | m01             |              0.75 |        0.752 | +                 | +               | True    |
| stress     | m02         | m03             |             -0.55 |        0.62  | -                 | +               | False   |
| stress     | m04         | m05             |              0.55 |        0.717 | +                 | +               | True    |
| stress     | m06         | m07             |             -0.4  |        0.617 | -                 | +               | False   |
| degenerate | score_a     | rare_event      |              0.75 |        0.442 | +                 | +               | True    |

## Degenerate centers

|   rare_event_nan_count |   rare_event_inf_count |   rare_event_mean |   rare_event_std |   rare_event_min |   rare_event_max |   score_x_rare_event_pearson | deterministic_repeat   |
|-----------------------:|-----------------------:|------------------:|-----------------:|-----------------:|-----------------:|-----------------------------:|:-----------------------|
|                      0 |                      0 |           2.67458 |          1.98692 |                0 |               13 |                       0.4419 | True                   |

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
| saas     |     16.826 |          4.356 |      3.86 |        nan   |
| stress   |   1014.51  |          5.684 |    178.49 |         99.6 |

## Fixture consistency

_(skipped — RUN_FIXTURE_CHECK = False)_
