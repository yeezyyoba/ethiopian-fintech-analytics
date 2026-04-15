"""
src/data/loader.py
------------------
Data loading utilities for the Ethiopian Fintech Analytics project.
Handles loading, basic validation, and initial inspection of raw datasets.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_credit_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the Give Me Some Credit dataset.
    
    Parameters
    ----------
    filepath : str, optional
        Custom path to cs-training.csv. Defaults to data/raw/credit/cs-training.csv.
    
    Returns
    -------
    pd.DataFrame
        Raw credit dataset with basic type fixes applied.
    """
    if filepath is None:
        filepath = DATA_RAW / "credit" / "cs-training.csv"
    
    print(f"Loading credit data from: {filepath}")
    df = pd.read_csv(filepath, index_col=0)
    
    # Rename target column for clarity
    df.rename(columns={"SeriousDlqin2yrs": "default_target"}, inplace=True)
    
    print(f"  Shape: {df.shape}")
    print(f"  Default rate: {df['default_target'].mean():.2%}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")
    
    return df


def load_fraud_data(trans_filepath: str = None, id_filepath: str = None) -> pd.DataFrame:
    """
    Load and merge IEEE-CIS fraud transaction + identity datasets.
    
    Parameters
    ----------
    trans_filepath : str, optional
        Path to train_transaction.csv
    id_filepath : str, optional
        Path to train_identity.csv
    
    Returns
    -------
    pd.DataFrame
        Merged fraud dataset.
    """
    if trans_filepath is None:
        trans_filepath = DATA_RAW / "fraud" / "train_transaction.csv"
    if id_filepath is None:
        id_filepath = DATA_RAW / "fraud" / "train_identity.csv"
    
    print(f"Loading fraud transaction data...")
    trans = pd.read_csv(trans_filepath)
    print(f"  Transactions shape: {trans.shape}")
    
    print(f"Loading fraud identity data...")
    identity = pd.read_csv(id_filepath)
    print(f"  Identity shape: {identity.shape}")
    
    print("Merging on TransactionID (left join)...")
    df = trans.merge(identity, on="TransactionID", how="left")
    print(f"  Merged shape: {df.shape}")
    print(f"  Fraud rate: {df['isFraud'].mean():.2%}\n")
    
    return df


def get_dataset_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print a structured summary of a DataFrame for initial inspection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to summarize.
    name : str
        Display name for the dataset.
    """
    print(f"\n{'='*50}")
    print(f"  {name} Summary")
    print(f"{'='*50}")
    print(f"  Rows:     {df.shape[0]:,}")
    print(f"  Columns:  {df.shape[1]:,}")
    print(f"\n  Data types:")
    print(df.dtypes.value_counts().to_string())
    print(f"\n  Missing values (columns with > 0 missing):")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        print("  None")
    else:
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
        print(missing_df.head(20).to_string())
    print(f"\n  Numeric column stats:")
    print(df.describe().round(2).to_string())
    print(f"{'='*50}\n")


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save a processed DataFrame to data/processed/ as a parquet file."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    filepath = DATA_PROCESSED / filename
    df.to_parquet(filepath, index=False)
    print(f"Saved to {filepath}  ({os.path.getsize(filepath) / 1e6:.1f} MB)")


def load_processed(filename: str) -> pd.DataFrame:
    """Load a processed parquet file from data/processed/."""
    filepath = DATA_PROCESSED / filename
    df = pd.read_parquet(filepath)
    print(f"Loaded {filename}  shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Quick test — run with: python -m src.data.loader
    print("Data loader ready. Use load_credit_data() or load_fraud_data() to get started.")
