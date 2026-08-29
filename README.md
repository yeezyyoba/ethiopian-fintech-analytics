# Ethiopian Fintech Analytics Platform
### Credit Risk Modeling & Fraud Detection | End-to-End Data Science Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC--ROC%200.8842-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

An end-to-end data science project that builds a **credit risk scoring** system
for the Ethiopian digital finance ecosystem. Covers the full pipeline — raw data
ingestion, feature engineering, machine learning, SHAP explainability, interactive
dashboard, and REST API deployment.

---

## Key Findings

- `has_delinquency` (SHAP=1.4215) is the strongest default predictor
- Customers with any delinquency history default at **22.27%** vs **2.73%** without — 8x difference
- High utilization customers default at **21.08%** vs **3.79%** — 5.6x difference
- Young borrowers (<30) default at **11.73%** — nearly double the overall 6.68% rate
- Combined delinquency + high utilization → **36.82%** default rate (SQL analysis)
- SMOTE applied: 14:1 class imbalance fixed to 1:1 (279,862 total samples)

---

## Model Results

| Model | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.8500 | 0.3160 | 0.1990 | 0.7677 |
| Random Forest | 0.8714 | 0.4356 | 0.3949 | 0.4855 |
| LightGBM | 0.8727 | 0.3692 | 0.5710 | 0.2728 |
| **XGBoost ✓** | **0.8842** | 0.3326 | 0.2081 | **0.8284** |

**Best model: XGBoost** — highest AUC-ROC and Recall (catches 82.8% of all defaulters)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, pandas, NumPy, SQLite |
| Machine Learning | scikit-learn, XGBoost, LightGBM, SMOTE |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn, Streamlit |
| API | FastAPI |

---

## Project Structure

\`\`\`
ethiopian-fintech-analytics/
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Feature store (parquet)
├── notebooks/
│   ├── 01_EDA_credit_risk.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_shap_explainability.ipynb
├── src/
│   ├── data/             # Loaders and cleaners
│   ├── features/         # Feature engineering
│   ├── models/           # Training utilities
│   ├── app.py            # Streamlit dashboard
│   ├── predict.py        # Inference module
│   └── api/              # FastAPI endpoint
├── reports/              # Charts and figures
├── docs/                 # Final report
└── requirements.txt
\`\`\`

---

## Live Demo

🚀 **[Ethiopian Credit Risk Dashboard](https://ethiopian-credit-risk.streamlit.app)**

---

## How to Run

\`\`\`bash
# 1. Clone and install
git clone https://github.com/yeezyyoba/ethiopian-fintech-analytics.git
cd ethiopian-fintech-analytics
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run src/app.py

# 3. Run the API
uvicorn src.api.main:app --reload
\`\`\`

---

## Weekly Progress

- [x] Week 1 — Project setup & repo structure
- [x] Week 2 — Exploratory Data Analysis
- [x] Week 3 & 4 — Feature Engineering & SMOTE
- [x] Week 5 — Model Training (XGBoost AUC-ROC 0.8842)
- [x] Week 6 — Model Explainability (SHAP)
- [x] Week 7 — Streamlit Dashboard
- [x] Week 8 — Final Report & FastAPI Deployment

---

## Author

**Eyob Nebyou**
Computer Science Student, Addis Ababa University
Data Engineering Intern @ Habtech
[LinkedIn](https://linkedin.com/in/eyob-nebyou-2782b8395) | [GitHub](https://github.com/yeezyyoba)

---

## License

MIT License
