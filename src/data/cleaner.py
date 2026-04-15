"""
src/data/cleaner.py
-------------------
Cleaning and preprocessing pipelines for credit risk and fraud datasets.
Each function is a single, testable transformation step.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
# CREDIT RISK CLEANING
# ─────────────────────────────────────────────

def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the Give Me Some Credit dataset.
    
    Steps:
      1. Drop duplicate rows
      2. Cap extreme outliers using the 99th percentile
      3. Impute missing values (median strategy)
      4. Clip negative values to 0 where they make no sense
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw credit DataFrame (output of load_credit_data)
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()
    original_shape = df.shape

    # Step 1: Drop duplicates
    df.drop_duplicates(inplace=True)
    print(f"Dropped {original_shape[0] - df.shape[0]} duplicate rows")

    # Step 2: Cap extreme outliers at 99th percentile
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "default_target"]
    
    for col in numeric_cols:
        cap = df[col].quantile(0.99)
        outliers = (df[col] > cap).sum()
        if outliers > 0:
            df[col] = df[col].clip(upper=cap)
            print(f"  Capped {outliers} outliers in '{col}' at {cap:.2f}")

    # Step 3: Impute missing values with median
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Imputed {n_missing} missing values in '{col}' with median={median_val:.2f}")

    # Step 4: Clip negative values that should be non-negative
    non_negative_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "NumberOfDependents",
        "age"
    ]
    for col in non_negative_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    print(f"\nCleaning complete. Shape: {original_shape} → {df.shape}")
    return df


# ─────────────────────────────────────────────
# FRAUD DETECTION CLEANING
# ─────────────────────────────────────────────

def clean_fraud_data(df: pd.DataFrame, drop_threshold: float = 0.5) -> pd.DataFrame:
    """
    Cleaning pipeline for the IEEE-CIS fraud dataset.
    
    Steps:
      1. Drop columns with more than `drop_threshold` missing values
      2. Impute remaining numeric columns with median
      3. Encode categorical columns with LabelEncoder
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw merged fraud DataFrame (output of load_fraud_data)
    drop_threshold : float
        Drop columns where missing % exceeds this value. Default 0.5 (50%).
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()
    original_cols = df.shape[1]

    # Step 1: Drop high-missing columns
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > drop_threshold].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped {len(cols_to_drop)} columns with >{drop_threshold*100:.0f}% missing values")
    print(f"  Remaining columns: {original_cols} → {df.shape[1]}")

    # Step 2: Impute numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "isFraud"]
    
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Step 3: Encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    
    if cat_cols:
        print(f"Label-encoded {len(cat_cols)} categorical columns: {cat_cols[:5]}{'...' if len(cat_cols) > 5 else ''}")

    print(f"\nFraud cleaning complete. Final shape: {df.shape}")
    return df


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage by downcasting numeric columns.
    Useful for the large IEEE-CIS dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
        Memory-optimized DataFrame.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1e6
    
    for col in df.select_dtypes(include=[np.integer]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=[np.floating]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    
    end_mem = df.memory_usage(deep=True).sum() / 1e6
    print(f"Memory reduced: {start_mem:.1f} MB → {end_mem:.1f} MB ({(1 - end_mem/start_mem)*100:.1f}% reduction)")
    
    return df
