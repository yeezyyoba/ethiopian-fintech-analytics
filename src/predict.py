"""
predict.py

Standalone inference module for the Ethiopian Fintech Analytics Platform
credit risk model. Mirrors the exact pipeline built across:
  - Notebook 01 (EDA)            -> sentinel/outlier cleaning rules
  - Notebook 03 (Feature Eng.)   -> engineered features, feature list
  - Notebook 04 (Model Training) -> best_model.joblib, feature_names.joblib
  - Notebook 05 (SHAP)           -> explainer + top-feature extraction

Import this from a Streamlit app (or anything else) instead of
re-deriving feature logic by hand:

    from predict import predict

    result = predict({
        "RevolvingUtilizationOfUnsecuredLines": 0.45,
        "age": 34,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.32,
        "MonthlyIncome": 3200.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
    })
    # result -> {
    #   "probability_of_default": 0.1234,
    #   "shap_top_features": [{"feature": ..., "value": ..., "shap_value": ...}, ...],
    #   "raw_engineered_features": {...},
    # }

This is single-applicant inference (built for a UI form), not the
batch/sample-array flow the notebooks use for evaluation.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

# ── Paths (relative to project root — adjust if predict.py moves) ──────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")

# ── Constants carried over from Notebook 01 / 03 cleaning rules ────────
# These are fit on the training data in the notebooks; hardcoding here
# keeps single-row inference self-contained. If the training data
# changes, regenerate these from Notebook 03 and update below.
RAW_INPUT_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Fitted on training data in Notebook 03 (income_median, debt_ratio_median).
MONTHLY_INCOME_MEDIAN_FOR_IMPUTATION = 5400.0
DEBT_RATIO_MEDIAN_FOR_SENTINEL_FIX = 0.37813977649999997
MONTHLY_INCOME_CLIP_UPPER = 12645.50

_model = None
_feature_names = None
_explainer = None


def _load_artifacts():
    """Lazy-load model/feature list/SHAP explainer once per process."""
    global _model, _feature_names, _explainer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _feature_names = joblib.load(FEATURE_NAMES_PATH)
        _explainer = shap.TreeExplainer(_model)
    return _model, _feature_names, _explainer


def engineer_features(raw: dict[str, Any]) -> pd.DataFrame:
    """Reproduce Notebook 01 cleaning + Notebook 03 feature engineering
    for a single applicant. Input keys must match RAW_INPUT_COLUMNS.
    Returns a one-row DataFrame with the final model feature set.
    """
    missing = [c for c in RAW_INPUT_COLUMNS if c not in raw]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    df = pd.DataFrame([raw])[RAW_INPUT_COLUMNS].copy()

    # ── Notebook 01/03 cleaning ─────────────────────────────────────
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(
        MONTHLY_INCOME_MEDIAN_FOR_IMPUTATION
    )
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    income_sentinel_mask = df["MonthlyIncome"] <= 1.0
    df.loc[income_sentinel_mask, "MonthlyIncome"] = (
        MONTHLY_INCOME_MEDIAN_FOR_IMPUTATION
    )

    debtratio_sentinel_mask = df["DebtRatio"] == 0.0
    df.loc[debtratio_sentinel_mask, "DebtRatio"] = (
        DEBT_RATIO_MEDIAN_FOR_SENTINEL_FIX
    )

    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=1.0)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=1.0)
    df["MonthlyIncome"] = df["MonthlyIncome"].clip(upper=MONTHLY_INCOME_CLIP_UPPER)

    # ── Notebook 03 engineered features ─────────────────────────────
    df["delinquency_score"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] * 1
        + df["NumberOfTime60-89DaysPastDueNotWorse"] * 2
        + df["NumberOfTimes90DaysLate"] * 3
    )

    df["debt_to_income"] = df["DebtRatio"] * df["MonthlyIncome"]
    df["debt_to_income"] = (
        df["debt_to_income"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    df["is_young_borrower"] = (df["age"] < 30).astype(int)
    df["is_senior_borrower"] = (df["age"] >= 60).astype(int)

    df["high_utilization"] = (
        df["RevolvingUtilizationOfUnsecuredLines"] > 0.8
    ).astype(int)

    df["total_past_due"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["has_delinquency"] = (df["total_past_due"] > 0).astype(int)

    df["has_dependents"] = (df["NumberOfDependents"] > 0).astype(int)

    # ── Drop raw delinquency columns + NumberOfDependents, matching
    #    Notebook 03's final feature selection ───────────────────────
    df = df.drop(
        columns=[
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfDependents",
        ]
    )

    return df


def predict(raw: dict[str, Any], top_n_shap: int = 5) -> dict[str, Any]:
    """Run full inference for one applicant: engineer features, predict
    probability of default, and compute the top-N SHAP contributions.
    """
    model, feature_names, explainer = _load_artifacts()

    engineered = engineer_features(raw)
    engineered = engineered[feature_names]  # enforce training column order

    proba = float(model.predict_proba(engineered)[:, 1][0])

    shap_values = explainer.shap_values(engineered)
    # TreeExplainer on binary XGBoost/LightGBM returns a single array;
    # if it returns a list (older SHAP / some model types), take class 1.
    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
    else:
        shap_row = shap_values[0]

    contributions = sorted(
        zip(feature_names, shap_row),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:top_n_shap]

    top_features = [
        {
            "feature": feat,
            "value": engineered.iloc[0][feat].item(),
            "shap_value": round(float(val), 4),
        }
        for feat, val in contributions
    ]

    return {
        "probability_of_default": round(proba, 4),
        "shap_top_features": top_features,
        "raw_engineered_features": engineered.iloc[0].to_dict(),
    }


if __name__ == "__main__":
    example = {
        "RevolvingUtilizationOfUnsecuredLines": 0.98,
        "age": 27,
        "NumberOfTime30-59DaysPastDueNotWorse": 1,
        "DebtRatio": 0.61,
        "MonthlyIncome": 2100.0,
        "NumberOfOpenCreditLinesAndLoans": 4,
        "NumberOfTimes90DaysLate": 1,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 1,
        "NumberOfDependents": 0,
    }
    import json

    print(json.dumps(predict(example), indent=2, default=str))