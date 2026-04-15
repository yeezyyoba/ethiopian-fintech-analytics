# Daily GitHub Commit Guide

The goal is one meaningful commit every day for 8 weeks.
These are pre-written commit messages and task ideas — just pick one each day.

---

## Week 1 — Setup & Data Acquisition

| Day | Task | Commit Message |
|-----|------|----------------|
| 1 | Initialize repo, folder structure | `feat: initial project structure` |
| 2 | Add requirements.txt + .gitignore | `chore: add requirements and gitignore` |
| 3 | Write README introduction | `docs: add project overview to README` |
| 4 | Add data download instructions | `docs: add Kaggle data download guide` |
| 5 | Write loader.py | `feat: add data loading utilities` |
| 6 | Download datasets, verify shapes | `data: confirm dataset shapes and structure` |
| 7 | Update README weekly progress | `docs: update Week 1 progress in README` |

---

## Week 2 — EDA

| Day | Task | Commit Message |
|-----|------|----------------|
| 8  | Class distribution analysis | `eda: add class imbalance analysis for credit data` |
| 9  | Missing values heatmap | `eda: add missing value analysis` |
| 10 | Feature distribution plots | `eda: add numeric feature distributions` |
| 11 | Correlation matrix | `eda: add correlation matrix` |
| 12 | Age group default rate analysis | `eda: add default rate by age group` |
| 13 | Fraud EDA notebook started | `eda: start fraud dataset exploration` |
| 14 | EDA findings summary | `docs: write EDA findings and next steps` |

---

## Week 3 — Cleaning & SQL

| Day | Task | Commit Message |
|-----|------|----------------|
| 15 | Write cleaner.py — outlier capping | `feat: add outlier capping to cleaner pipeline` |
| 16 | Write cleaner.py — imputation | `feat: add missing value imputation` |
| 17 | Create SQLite schema | `feat: add SQLite database schema` |
| 18 | SQL query: fraud rate by category | `sql: add fraud rate aggregation query` |
| 19 | SQL query: default rate by age | `sql: add default rate by age group query` |
| 20 | Feature engineering: delinquency score | `feat: add delinquency score feature` |
| 21 | Feature engineering: debt-to-income | `feat: add debt-to-income ratio feature` |

---

## Week 4 — Feature Engineering & SMOTE

| Day | Task | Commit Message |
|-----|------|----------------|
| 22 | Apply SMOTE to credit dataset | `feat: add SMOTE balancing to credit pipeline` |
| 23 | RFE feature selection | `feat: add recursive feature elimination` |
| 24 | Build feature store (parquet) | `feat: create feature store for modeling` |
| 25 | Write data dictionary | `docs: add data dictionary for all features` |
| 26 | Fraud feature engineering | `feat: add time-based fraud features` |
| 27 | Fraud SMOTE / ADASYN | `feat: add ADASYN balancing for fraud data` |
| 28 | Feature pipeline test | `test: add unit tests for feature pipeline` |

---

## Week 5 — Model Training

| Day | Task | Commit Message |
|-----|------|----------------|
| 29 | Train Logistic Regression baseline | `model: train logistic regression baseline` |
| 30 | Train Random Forest | `model: train random forest classifier` |
| 31 | Train XGBoost | `model: train xgboost model` |
| 32 | Train LightGBM | `model: train lightgbm model` |
| 33 | Hyperparameter tuning — XGBoost | `model: xgboost hyperparameter tuning` |
| 34 | Results comparison table | `results: add model comparison table` |
| 35 | Update README with results | `docs: update README with model results` |

---

## Week 6 — Explainability

| Day | Task | Commit Message |
|-----|------|----------------|
| 36 | SHAP summary plot | `explainability: add SHAP summary plot` |
| 37 | SHAP waterfall plot (single prediction) | `explainability: add SHAP waterfall for single prediction` |
| 38 | SHAP force plot | `explainability: add SHAP force plot` |
| 39 | Business translation of SHAP findings | `docs: add business interpretation of model features` |
| 40 | Fairness analysis by age group | `analysis: add model fairness check by age group` |
| 41 | Write model card | `docs: add model card (methodology and limitations)` |
| 42 | Notebook 05 finalized | `notebook: finalize explainability notebook` |

---

## Week 7 — Dashboard

| Day | Task | Commit Message |
|-----|------|----------------|
| 43 | Power BI — Overview page screenshot | `dashboard: add overview page screenshot` |
| 44 | Power BI — Fraud analysis page | `dashboard: add fraud analysis page screenshot` |
| 45 | Power BI — Credit risk page | `dashboard: add credit risk page screenshot` |
| 46 | Export dashboard as PDF | `dashboard: export final dashboard PDF` |
| 47 | Write dashboard commentary | `docs: add dashboard design decisions` |

---

## Week 8 — Deployment & Polish

| Day | Task | Commit Message |
|-----|------|----------------|
| 50 | FastAPI prediction endpoint | `feat: add FastAPI prediction endpoint` |
| 51 | Test API with sample inputs | `test: add API endpoint tests` |
| 52 | Write final report | `docs: add final project report` |
| 53 | Record Loom video, add link to README | `docs: add project walkthrough video` |
| 54 | Final README polish | `docs: final README review and polish` |
| 55 | Tag v1.0 release | `release: tag v1.0 - project complete` |

---

## Good Commit Message Formula

```
<type>: <what you did>

Types:
  feat     — new code or feature
  fix      — bug fix
  docs     — documentation only
  eda      — notebook analysis
  model    — model training or evaluation
  data     — data processing
  sql      — SQL queries
  test     — unit tests
  chore    — setup, dependencies
  results  — results tables or metrics
  dashboard — Power BI work
  explainability — SHAP / interpretability
  release  — version tag
```
