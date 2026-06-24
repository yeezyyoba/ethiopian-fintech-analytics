# Ethiopian Fintech Analytics Platform
### Credit Risk Modeling & Fraud Detection | End-to-End Data Science Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview
An end-to-end data science project building a credit risk scoring and 
transaction fraud detection system inspired by the Ethiopian digital 
finance ecosystem. Covers the full pipeline from raw data to ML models, 
explainability, and a business-facing Power BI dashboard.

---

## Key Findings So Far
- `has_delinquency` (correlation 0.3144) is the strongest default predictor
- Customers with delinquency default at **22.27%** vs **2.73%** without — 8x difference
- High utilization customers default at **21.08%** vs **3.79%** — 5.6x difference
- Young borrowers (<30) default at **11.73%** — nearly double the overall 6.68% rate
- SMOTE applied: 14:1 class imbalance fixed to 1:1 (279,862 total samples)

---

## Tech Stack
| Layer | Tools |
|---|---|
| Data Processing | Python, pandas, NumPy, SQLite |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn, Power BI |
| API | FastAPI |

---

## Project Structure

---

## Weekly Progress
- [x] Week 1 — Project setup & repo structure
- [x] Week 2 — Exploratory Data Analysis
- [x] Week 3 & 4 — Feature Engineering & SMOTE
- [ ] Week 5 — Model Training
- [ ] Week 6 — Model Explainability (SHAP)
- [ ] Week 7 — Power BI Dashboard
- [ ] Week 8 — Final Report & API Deployment

---

## Author
**Eyob Nebyou**  
Computer Science Student, Addis Ababa University  
[LinkedIn](https://linkedin.com/in/eyob-nebyou-2782b8395) | [GitHub](https://github.com/yeezyyoba)