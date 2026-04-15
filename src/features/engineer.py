"""
src/features/engineer.py
------------------------
Feature engineering for credit risk and fraud detection models.
Each function adds new columns to the DataFrame — no data is removed here.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# CREDIT RISK FEATURES
# ─────────────────────────────────────────────

def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-specific features for credit risk modeling.
    
    New features:
      - debt_to_income_ratio
      - delinquency_score       (weighted sum of past-due events)
      - age_group               (binned age)
      - is_young_borrower       (flag: age < 30)
      - credit_utilization_flag (flag: utilization > 0.8)
      - total_past_due          (sum of all delinquency counts)
    """
    df = df.copy()

    # Debt-to-income ratio (safe division)
    if "DebtRatio" in df.columns and "MonthlyIncome" in df.columns:
        df["debt_to_income"] = df["DebtRatio"] * df["MonthlyIncome"]
        df["debt_to_income"] = df["debt_to_income"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Weighted delinquency score — 90+ days is more serious than 30-59 days
    late_cols = {
        "NumberOfTime30-59DaysPastDueNotWorse": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 2,
        "NumberOfTimes90DaysLate": 3
    }
    available = {col: w for col, w in late_cols.items() if col in df.columns}
    if available:
        df["delinquency_score"] = sum(df[col] * w for col, w in available.items())
        df["total_past_due"] = sum(df[col] for col in available.keys())

    # Age group bins
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"]
        ).astype(str)
        df["is_young_borrower"] = (df["age"] < 30).astype(int)

    # High credit utilization flag
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["high_utilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.8).astype(int)

    # Dependents flag
    if "NumberOfDependents" in df.columns:
        df["has_dependents"] = (df["NumberOfDependents"] > 0).astype(int)

    new_cols = [c for c in df.columns if c not in df.columns]
    print(f"Credit feature engineering complete.")
    print(f"  New features added: debt_to_income, delinquency_score, total_past_due,")
    print(f"                      age_group, is_young_borrower, high_utilization, has_dependents")

    return df


# ─────────────────────────────────────────────
# FRAUD DETECTION FEATURES
# ─────────────────────────────────────────────

def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features for fraud detection modeling.
    
    New features:
      - transaction_hour         (extracted from TransactionDT)
      - transaction_day          (day of week approximation)
      - is_high_value            (flag: TransactionAmt > 95th percentile)
      - amt_log                  (log-transformed transaction amount)
      - card_velocity_proxy      (TransactionAmt / card1 — rough proxy)
    """
    df = df.copy()

    # Time-based features from TransactionDT (seconds from reference point)
    if "TransactionDT" in df.columns:
        df["transaction_hour"] = (df["TransactionDT"] // 3600) % 24
        df["transaction_day"] = (df["TransactionDT"] // (3600 * 24)) % 7
        df["is_weekend"] = (df["transaction_day"] >= 5).astype(int)
        df["is_night"] = ((df["transaction_hour"] >= 22) | (df["transaction_hour"] <= 6)).astype(int)

    # Transaction amount features
    if "TransactionAmt" in df.columns:
        p95 = df["TransactionAmt"].quantile(0.95)
        df["is_high_value"] = (df["TransactionAmt"] > p95).astype(int)
        df["amt_log"] = np.log1p(df["TransactionAmt"])
        df["amt_cents"] = df["TransactionAmt"] % 1  # Fraction part — fraudsters often use round numbers

    print("Fraud feature engineering complete.")
    print("  New features: transaction_hour, transaction_day, is_weekend,")
    print("                is_night, is_high_value, amt_log, amt_cents")

    return df


# ─────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────

def get_feature_names(df: pd.DataFrame, target_col: str) -> list:
    """Return all column names except the target."""
    return [c for c in df.columns if c != target_col]


def feature_summary(df: pd.DataFrame, target_col: str) -> None:
    """Print a quick summary of feature types and target distribution."""
    features = get_feature_names(df, target_col)
    numeric = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical = df[features].select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"\nFeature Summary")
    print(f"  Total features:    {len(features)}")
    print(f"  Numeric features:  {len(numeric)}")
    print(f"  Categorical:       {len(categorical)}")
    print(f"\n  Target: '{target_col}'")
    print(f"  {df[target_col].value_counts().to_string()}")
    print(f"  Class balance: {df[target_col].mean():.2%} positive\n")
