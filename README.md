# Healthcare Risk Score Calculation

## Overview

This project develops a machine-learning-based healthcare risk scoring system to identify patients who may be at higher risk of an adverse event within 90 days.

The system combines information from multiple healthcare data sources and generates:

- Adverse-event prediction
- Risk probability
- Risk score from 0 to 100
- Low, Medium, and High risk categories

## Data Sources

The model integrates four datasets:

1. **Electronic Health Records (EHR)**
   - Demographics
   - Laboratory results
   - Vital signs
   - Clinical history

2. **Social Determinants of Health (SDOH)**
   - Income
   - Housing
   - Education
   - Access to care
   - Chronic conditions
   - Healthcare utilization

3. **Behavioral and Lifestyle Data**
   - Step count
   - Sleep duration
   - Heart rate
   - Blood oxygen
   - Screen time
   - Other lifestyle measurements

4. **Healthcare Claims**
   - Procedure codes
   - Diagnosis codes
   - Billed amounts
   - Allowed amounts
   - Paid amounts
   - Insurance information

The datasets contain information for **1,000 patients**.

## Target

The prediction target is:

`adverse_event_90d`

- `0` = No adverse event
- `1` = Adverse event within 90 days

The dataset is imbalanced, with approximately 12.2% positive cases.

## Data Processing

The preprocessing pipeline includes:

- Patient-level dataset integration
- Missing-value handling
- Date processing
- Numerical feature scaling
- Categorical feature encoding
- Train/test stratification
- Class-imbalance handling

After preprocessing and one-hot encoding, the model works with both numerical and categorical patient information.

## Model

The final model is **LightGBM (Light Gradient Boosting Machine)**.

Important model settings include:

- `n_estimators = 300`
- `learning_rate = 0.01`
- `max_depth = 3`
- `num_leaves = 7`
- `min_child_samples = 50`
- Positive class weight = `2`

The class weight was selected using cross-validation on the training data.

## Decision Threshold

The classification threshold was selected using out-of-fold predictions from the training data.

Final threshold:

`0.23`

Patients with a predicted score greater than or equal to this threshold are flagged as predicted adverse-event cases.

## Risk Score

The model probability is converted into a 0-100 model-derived risk score:

`Risk Score = Model Probability × 100`

Risk categories used in this prototype:

| Risk Score | Category |
|---|---|
| < 15 | Low |
| 15 to < 30 | Medium |
| >= 30 | High |

These categories are model-derived prototype risk strata and are not clinically validated thresholds.

## Held-Out Evaluation

The selected model achieved the following results on the held-out evaluation set:

| Metric | Result |
|---|---:|
| Accuracy | 75.00% |
| Precision | 25.93% |
| Recall | 58.33% |
| F1 Score | 35.90% |
| ROC-AUC | 68.28% |
| PR-AUC | 31.96% |

The model correctly detected **14 of 24 adverse-event cases** in the evaluation set.

Because the target is imbalanced, accuracy alone should not be used to evaluate model quality. Recall, F1, ROC-AUC, and PR-AUC are also considered.

## Final Risk Distribution

After training the final deployment model on all 1,000 patients:

| Category | Patients |
|---|---:|
| Low | 477 |
| Medium | 333 |
| High | 190 |

The final model flags 301 patients at or above the adverse-event decision threshold of 0.23.

## Project Structure

```text
Risk-Score-Calculation/
|
|-- data/
|   |-- Behavioral and lifestyle data.csv
|   |-- EHR_Patientlevel_features.csv
|   |-- Social Determinants of Health DataSet.csv
|   `-- healthcare_claims.csv
|
|-- models/
|   |-- final_lightgbm_risk_model_v2.pkl
|   `-- risk_score_config_v2.json
|
|-- outputs/
|   `-- final_patient_risk_scores_v2.csv
|
`-- README.md
