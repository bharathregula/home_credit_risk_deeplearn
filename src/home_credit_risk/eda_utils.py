"""
Reusable EDA utility functions for exploratory data analysis.
These functions are designed to be used across multiple EDA notebooks.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import zscore


def display_basic_info(df: pd.DataFrame, df_name: str = "Dataset") -> None:
    """
    Display basic information about the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    df_name : str
        Name of the dataset for display purposes
    """
    print(f"\n{'=' * 60}")
    print(f"Basic Information - {df_name}")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")


def show_missing_patterns(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Display missing data patterns.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    threshold : float
        Only show columns with missing percentage > threshold (default: 0)

    Returns:
    --------
    pd.DataFrame
        Missing data summary sorted by percentage
    """
    print(f"\n{'=' * 60}")
    print("Missing Data Patterns")
    print(f"{'=' * 60}")

    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    missing_summary = pd.DataFrame(
        {"Missing_Count": missing_count, "Missing_Percentage": missing_pct}
    ).sort_values("Missing_Percentage", ascending=False)

    missing_summary = missing_summary[missing_summary["Missing_Percentage"] > threshold]

    if len(missing_summary) == 0:
        print(f"No missing values found (threshold: {threshold}%)")
    else:
        print(missing_summary)

    return missing_summary


def analyze_target_distribution(
    df: pd.DataFrame, target_col: str = "TARGET", figsize: Tuple[int, int] = (8, 5)
) -> None:
    """
    Analyze and visualize target variable distribution.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the target column
    target_col : str
        Name of the target column
    figsize : Tuple[int, int]
        Figure size for the plot
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataframe")
        return

    print(f"\n{'=' * 60}")
    print(f"Target Distribution - {target_col}")
    print(f"{'=' * 60}")

    target_counts = df[target_col].value_counts()
    target_pcts = df[target_col].value_counts(normalize=True) * 100

    print("\nValue Counts:")
    print(target_counts)
    print("\nPercentages:")
    print(target_pcts)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar plot
    target_counts.plot(
        kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"][: len(target_counts)]
    )
    axes[0].set_title(f"Distribution of {target_col}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    # Pie plot
    axes[1].pie(
        target_counts.values,
        labels=target_counts.index,
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"][: len(target_counts)],
    )
    axes[1].set_title(f"{target_col} Distribution (%)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()


def check_duplicates(df: pd.DataFrame, id_col: Optional[str] = None) -> None:
    """
    Check for duplicate records.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to check for duplicates
    id_col : str, optional
        Column name to check for duplicate IDs. If None, checks all rows.
    """
    print(f"\n{'=' * 60}")
    print("Duplicate Records Analysis")
    print(f"{'=' * 60}")

    total_duplicates = df.duplicated().sum()
    print(f"Total Duplicate Rows: {total_duplicates}")

    if id_col and id_col in df.columns:
        id_duplicates = df[id_col].duplicated().sum()
        print(f"Duplicate {id_col}: {id_duplicates}")


def analyze_numeric_features(
    df: pd.DataFrame, figsize: Tuple[int, int] = (18, 6), show_summary: bool = True
) -> pd.DataFrame:
    """
    Analyze numeric features: distribution, skewness, and outliers.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    figsize : Tuple[int, int]
        Figure size for the boxplot
    show_summary : bool
        Whether to print summary statistics and skewness

    Returns:
    --------
    pd.DataFrame
        Numeric columns dataframe
    """
    print(f"\n{'=' * 60}")
    print("Numeric Features Analysis")
    print(f"{'=' * 60}")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if show_summary:
        print("\nDescriptive Statistics:")
        print(numeric_df.describe())

        print("\nSkewness:")
        print(numeric_df.skew().sort_values(ascending=False))

    # Boxplot for outliers
    numeric_df.plot(
        kind="box", figsize=figsize, vert=False, grid=True, patch_artist=True
    )
    plt.title("Box Plot of Numeric Features", fontsize=14, fontweight="bold")
    plt.xlabel("Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return numeric_df


def identify_outliers(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    z_threshold: float = 3.0,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Identify outliers using z-score method.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    numeric_cols : List[str], optional
        Specific numeric columns to analyze. If None, uses all numeric columns.
    z_threshold : float
        Z-score threshold for outlier detection (default: 3.0)
    top_n : int
        Number of top features with outliers to show

    Returns:
    --------
    pd.DataFrame
        Summary of outliers by feature
    """
    print(f"\n{'=' * 60}")
    print(f"Outlier Detection (Z-score > {z_threshold})")
    print(f"{'=' * 60}")

    if numeric_cols is None:
        numeric_df = df.select_dtypes(include=["int64", "float64"])
    else:
        numeric_df = df[numeric_cols].select_dtypes(include=["int64", "float64"])

    # Handle missing values
    numeric_df_clean = numeric_df.fillna(numeric_df.median())

    # Calculate z-scores
    z_scores = numeric_df_clean.apply(zscore)

    # Count outliers
    outlier_summary = (
        (z_scores.abs() > z_threshold).sum(axis=0).sort_values(ascending=False)
    )

    print(f"\nTop {top_n} Features with Most Outliers:")
    print(outlier_summary.head(top_n))

    # Visualization for top features
    top_features = outlier_summary.head(10).index.tolist()
    if top_features:
        df[top_features].plot(kind="box", figsize=(12, 5), vert=False, grid=True)
        plt.title(
            "Box Plot of Top 10 Features with Outliers", fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    return outlier_summary


def analyze_categorical_features(df: pd.DataFrame) -> None:
    """
    Analyze categorical features: unique values and distributions.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    """
    print(f"\n{'=' * 60}")
    print("Categorical Features Analysis")
    print(f"{'=' * 60}")

    categorical_cols = df.select_dtypes(include="object").columns

    if len(categorical_cols) == 0:
        print("No categorical columns found")
        return

    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print("  Value Counts:")
        print(df[col].value_counts())


def analyze_correlations(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
) -> pd.Series:
    """
    Analyze correlation between numeric features and target variable.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    target_col : str
        Name of the target column
    top_n : int
        Number of top correlated features to visualize
    figsize : Tuple[int, int]
        Figure size for heatmap

    Returns:
    --------
    pd.Series
        Correlations sorted by absolute value
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataframe")
        return None

    print(f"\n{'=' * 60}")
    print(f"Correlation Analysis with {target_col}")
    print(f"{'=' * 60}")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_data = df[numeric_cols].fillna(df[numeric_cols].median())

    correlations = numeric_data.corr()[target_col].sort_values(ascending=False)

    print(f"\nTop {top_n} Correlated Features:")
    print(correlations.head(top_n))

    # Heatmap
    top_features = correlations.abs().nlargest(top_n).index.tolist()
    if target_col not in top_features:
        top_features.append(target_col)

    plt.figure(figsize=figsize)
    sns.heatmap(
        df[top_features].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )
    plt.title(
        f"Correlation Matrix - Top {top_n} Features with {target_col}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    return correlations


def check_data_quality(df: pd.DataFrame) -> None:
    """
    Check for common data quality issues.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to check
    """
    print(f"\n{'=' * 60}")
    print("Data Quality Checks")
    print(f"{'=' * 60}")

    # Check for negative values in typically positive columns
    potentially_positive_cols = [
        col
        for col in df.columns
        if any(
            keyword in col.upper() for keyword in ["AMT", "INCOME", "PRICE", "CREDIT"]
        )
    ]

    if potentially_positive_cols:
        print("\nChecking for unexpected negative values:")
        for col in potentially_positive_cols:
            if df[col].dtype in ["int64", "float64"]:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    print(f"  {col}: {neg_count} negative values")

    # Check for logical inconsistencies
    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        credit_gt_price = (df["AMT_CREDIT"] > df["AMT_GOODS_PRICE"]).sum()
        print(f"\nAMT_CREDIT > AMT_GOODS_PRICE: {credit_gt_price} records")

    # Check for impossible values (e.g., age)
    if "DAYS_BIRTH" in df.columns:
        print("\nDAYS_BIRTH Range:")
        print(df["DAYS_BIRTH"].describe())
        print("  Expected: negative values (days before reference date)")
        print(f"  Positive values: {(df['DAYS_BIRTH'] > 0).sum()}")


def identify_feature_engineering_opportunities(
    df: pd.DataFrame, target_col: str = "TARGET"
) -> None:
    """
    Suggest and calculate feature engineering opportunities.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    target_col : str
        Name of the target column for correlation check
    """
    print(f"\n{'=' * 60}")
    print("Feature Engineering Opportunities")
    print(f"{'=' * 60}")

    # Ratio features
    opportunities = []

    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        df["credit_to_goods_ratio"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
        opportunities.append("credit_to_goods_ratio")

    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["annuity_to_income_ratio"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        opportunities.append("annuity_to_income_ratio")

    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["credit_to_income_ratio"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
        opportunities.append("credit_to_income_ratio")

    if "DAYS_BIRTH" in df.columns:
        df["age_years"] = df["DAYS_BIRTH"] / -365
        opportunities.append("age_years")

    # Check correlations with target
    if opportunities and target_col in df.columns:
        print("\nNew Feature Correlations with Target:")
        for feature in opportunities:
            if feature in df.columns:
                corr = df[feature].corr(df[target_col])
                print(f"  {feature}: {corr:.4f}")
