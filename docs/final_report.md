# Ethiopian Fintech Analytics Platform
## Final Project Report

**Author**: Eyob Nebyou
**Institution**: Addis Ababa University, CNCS
**Date**: August 2026
**GitHub**: https://github.com/yeezyyoba/ethiopian-fintech-analytics

---

## 1. Executive Summary

This project builds an end-to-end credit risk prediction system for the Ethiopian
digital finance ecosystem. Using 150,000 real customer records, we developed a
machine learning pipeline that predicts loan default probability with an AUC-ROC
of 0.8842 and a Recall of 0.8284 — meaning the model correctly identifies 82.8%
of all actual defaulters.

---

## 2. Problem Statement

Digital financial services in Ethiopia — mobile banking, microloans, mobile money —
are growing rapidly. Two critical challenges face these platforms:

1. **Credit Risk**: How likely is a customer to default on a loan?
2. **Fraud Detection**: Is a transaction fraudulent?

---

## 3. Dataset

| Property | Detail |
|---|---|
| Name | Give Me Some Credit |
| Source | Kaggle |
| Rows | 150,000 customers |
| Features | 11 raw features |
| Target | SeriousDlqin2yrs (binary default label) |
| Default rate | 6.68% |
| Class imbalance | 14:1 (non-default : default) |

---

## 4. Methodology

### 4.1 Exploratory Data Analysis
- Identified 14:1 class imbalance requiring SMOTE
- Found RevolvingUtilizationOfUnsecuredLines as top correlated feature (0.278)
- Confirmed statistical significance with t-tests (p=0.0000 for age and income)
- Identified under-25s as highest default risk group (11.73% rate)

### 4.2 Feature Engineering

| Feature | Description |
|---|---|
| delinquency_score | Weighted sum of past-due events (90+ days x 3) |
| debt_to_income | Monthly debt payment relative to income |
| is_young_borrower | Flag for age < 30 |
| is_senior_borrower | Flag for age >= 60 |
| high_utilization | Flag for credit utilization > 80% |
| total_past_due | Sum of all delinquency counts |
| has_delinquency | Binary flag for any delinquency |
| has_dependents | Binary flag for dependents > 0 |

### 4.3 Class Imbalance — SMOTE
Applied SMOTE to fix the 14:1 imbalance. Dataset grew from 149,954 to 279,862
samples. Training used SMOTE-balanced data; evaluation used original distribution.

### 4.4 SQL Analysis
- Combined delinquency + high utilization → 36.82% default rate
- Default rate by age group confirmed EDA findings

---

## 5. Model Results

| Model | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.8500 | 0.3160 | 0.1990 | 0.7677 |
| Random Forest | 0.8714 | 0.4356 | 0.3949 | 0.4855 |
| LightGBM | 0.8727 | 0.3692 | 0.5710 | 0.2728 |
| XGBoost | 0.8842 | 0.3326 | 0.2081 | 0.8284 |

**Best model: XGBoost** — AUC-ROC 0.8842, Recall 0.8284

---

## 6. Model Explainability — SHAP

**Global feature importance (SHAP ranking):**
1. has_delinquency — SHAP value 1.4215
2. high_utilization — SHAP value 0.5738
3. delinquency_score
4. RevolvingUtilizationOfUnsecuredLines
5. age

**Business translation — three questions the model asks:**
1. Has this customer missed payments before?
2. Are they overextended on credit?
3. Do they have capacity to repay?

---

## 7. Deployment

### Streamlit Dashboard
Live at: https://ethiopian-credit-risk.streamlit.app

### FastAPI REST Endpoint
POST /predict — accepts customer data, returns default probability and risk tier

---

## 8. Limitations

- Dataset is US-based — Ethiopian-specific data would improve relevance
- Model evaluated on historical data — performance on future data may differ
- Fairness analysis shows higher false positive rates for young borrowers

---

## 9. Conclusion

This project demonstrates a complete, production-ready credit risk pipeline.
The XGBoost model achieves AUC-ROC of 0.8842 with 82.8% recall on defaulters,
and SHAP analysis reveals that past delinquency history is the most powerful signal.

---

## 10. References

- Kaggle: Give Me Some Credit Dataset
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique
