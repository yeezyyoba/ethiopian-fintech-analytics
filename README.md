# Ethiopian Fintech Analytics Platform
### Credit Risk Modeling & Fraud Detection | End-to-End Data Science Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

An end-to-end data science project that builds a **credit risk scoring** and **transaction fraud detection** system inspired by the Ethiopian and East African digital finance ecosystem. The project covers the full pipeline — from raw data ingestion to machine learning models, explainability analysis, and a business-facing Power BI dashboard.

This project was built to demonstrate applied data science skills for master's program applications and fintech industry roles.

---

## Problem Statement

Digital financial services in Ethiopia (mobile banking, microloans, mobile money) are growing rapidly. Two critical challenges for these platforms are:

1. **Credit Risk**: How likely is a customer to default on a loan?
2. **Fraud Detection**: Is a transaction fraudulent?

This project builds predictive models for both problems, with a focus on model interpretability and business value.

---

## Project Structure

```
ethiopian-fintech-analytics/
│
├── data/
│   ├── raw/              # Original, unmodified datasets
│   ├── processed/        # Cleaned and feature-engineered data
│   └── external/         # Reference data (e.g., exchange rates, regional data)
│
├── notebooks/
│   ├── 01_EDA_credit_risk.ipynb
│   ├── 02_EDA_fraud_detection.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_explainability.ipynb
│   └── 06_final_report.ipynb
│
├── src/
│   ├── data/             # Data loading and cleaning scripts
│   ├── features/         # Feature engineering pipeline
│   ├── models/           # Model training and evaluation
│   └── visualization/    # Reusable plot functions
│
├── dashboard/            # Power BI files and exported screenshots
├── reports/              # PDF reports and model cards
├── tests/                # Unit tests
├── docs/                 # Data dictionary and documentation
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Datasets

| Dataset | Source | Description |
|---|---|---|
| IEEE-CIS Fraud Detection | Kaggle | 590K+ transactions with fraud labels |
| Give Me Some Credit | Kaggle | 150K customers, credit default prediction |

> Download instructions: see `docs/data_download.md`

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, pandas, NumPy, SQLite |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn, Power BI |
| API | FastAPI |
| Experiment Tracking | MLflow |

---

## Models & Results

| Model | AUC-ROC | F1-Score | Notes |
|---|---|---|---|
| Logistic Regression | TBD | TBD | Baseline |
| Random Forest | TBD | TBD | |
| XGBoost | TBD | TBD | |
| LightGBM | TBD | TBD | Best performer |

*Results will be updated weekly as training progresses.*

---

## Key Findings

> *This section will be updated at the end of Week 6.*

---

## Weekly Progress

- [x] Week 1 — Project setup, repo structure, data acquisition
- [ ] Week 2 — Exploratory Data Analysis
- [ ] Week 3 — Data cleaning & SQL pipeline
- [ ] Week 4 — Feature engineering
- [ ] Week 5 — Model training & comparison
- [ ] Week 6 — Model explainability (SHAP)
- [ ] Week 7 — Power BI dashboard
- [ ] Week 8 — Final report & API deployment

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ethiopian-fintech-analytics.git
cd ethiopian-fintech-analytics

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data (see docs/data_download.md)

# 5. Run notebooks in order
jupyter notebook notebooks/
```

---

## Author

**Eyob Nebyou**  
Computer Science Student, Addis Ababa University  
Software Engineering Intern, IE Networks PLC  
[LinkedIn](https://linkedin.com/in/eyob-nebyou-2782b8395)

---

## License

MIT License — free to use and adapt with attribution.
