"""
src/visualization/plots.py
---------------------------
Reusable plotting functions for EDA and model evaluation.
All functions return matplotlib Figure objects so they can be saved easily.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ── Style defaults ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = {
    "primary": "#534AB7",
    "secondary": "#1D9E75",
    "warning": "#EF9F27",
    "danger": "#E24B4A",
    "neutral": "#888780"
}


def plot_class_distribution(y: pd.Series, title: str = "Class Distribution") -> plt.Figure:
    """Bar chart showing class balance / imbalance."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = y.value_counts()
    axes[0].bar(counts.index.astype(str), counts.values,
                color=[COLORS["secondary"], COLORS["danger"]])
    axes[0].set_title("Count")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Frequency")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + counts.max() * 0.01, f"{v:,}", ha="center", fontsize=10)

    pcts = y.value_counts(normalize=True) * 100
    axes[1].pie(pcts.values, labels=[f"Class {i}\n{p:.1f}%" for i, p in zip(pcts.index, pcts.values)],
                colors=[COLORS["secondary"], COLORS["danger"]], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Proportion")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_missing_heatmap(df: pd.DataFrame, title: str = "Missing Values Heatmap") -> plt.Figure:
    """Heatmap showing missing value patterns across columns."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("No missing values found.")
        return None

    fig, ax = plt.subplots(figsize=(12, max(4, len(missing) * 0.4)))
    missing_pct = (missing / len(df) * 100).values.reshape(-1, 1)
    sns.heatmap(missing_pct, annot=True, fmt=".1f", cmap="Reds",
                yticklabels=missing.index, xticklabels=["Missing %"],
                ax=ax, cbar_kws={"label": "Missing %"})
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, cols: list, target: str = None,
                                ncols: int = 3) -> plt.Figure:
    """
    Grid of histograms for numeric features, optionally colored by target class.
    """
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        if target and target in df.columns:
            for val, color in zip([0, 1], [COLORS["secondary"], COLORS["danger"]]):
                subset = df[df[target] == val][col].dropna()
                ax.hist(subset, bins=40, alpha=0.6, color=color,
                        label=f"Class {val}", density=True)
            ax.legend(fontsize=8)
        else:
            ax.hist(df[col].dropna(), bins=40, color=COLORS["primary"], alpha=0.8)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix") -> plt.Figure:
    """Heatmap of Pearson correlations between numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(10, len(corr) * 0.7), max(8, len(corr) * 0.6)))

    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, annot=len(corr) <= 15,
                fmt=".2f", linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc_curves(results: dict, title: str = "ROC Curves Comparison") -> plt.Figure:
    """
    Plot ROC curves for multiple models on one axis.
    
    Parameters
    ----------
    results : dict
        {model_name: {"y_test": ..., "y_proba": ...}}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["warning"], COLORS["danger"]]

    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})",
                color=palette[i % len(palette)], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list,
                             top_n: int = 20, title: str = "Feature Importance") -> plt.Figure:
    """
    Horizontal bar chart of feature importances for tree-based models.
    Works with RandomForest, XGBoost, and LightGBM.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("Model does not have feature_importances_ attribute.")
        return None

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    bars = ax.barh(fi_df["feature"], fi_df["importance"], color=COLORS["primary"], alpha=0.85)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filename: str, dpi: int = 150) -> None:
    """Save a figure to the reports/ directory."""
    from pathlib import Path
    reports_dir = Path(__file__).resolve().parents[2] / "reports"
    reports_dir.mkdir(exist_ok=True)
    path = reports_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved: {path}")
