"""
Reusable EDA utility functions for exploratory data analysis.
These functions are designed to be used across multiple EDA notebooks.

All functions operate on eager :class:`polars.DataFrame` objects. When working
with :class:`polars.LazyFrame` pipelines (e.g. ``pl.scan_csv``), call
``.collect()`` before passing the frame to these helpers.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns


def _measurement_numeric_columns(
    df: pl.DataFrame, exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Numeric column names suitable for statistical analysis.

    Excludes ``SK_ID*`` hashed identifiers (SK_ID_CURR, SK_ID_PREV, SK_ID_BUREAU) —
    a describe / skew / correlation / outlier score over an id is meaningless — plus
    any names passed in ``exclude``. Identifiers are intentionally *kept* by the
    null / missing-data helpers, which do not use this function.
    """
    exclude_set = set(exclude or [])
    return [
        c
        for c in df.select(cs.numeric()).columns
        if not c.startswith("SK_ID") and c not in exclude_set
    ]


def display_basic_info(df: pl.DataFrame, df_name: str = "Dataset") -> None:
    """
    Display basic information about the dataset.

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe to analyze
    df_name : str
        Name of the dataset for display purposes
    """
    print(f"\n{'=' * 60}")
    print(f"Basic Information - {df_name}")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    for name, dtype in df.schema.items():
        print(f"  {name}: {dtype}")


def show_missing_patterns(df: pl.DataFrame, threshold: float = 0.0) -> pl.DataFrame:
    """
    Display missing data patterns.

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe to analyze
    threshold : float
        Only show columns with missing percentage > threshold (default: 0)

    Returns:
    --------
    pl.DataFrame
        Missing data summary sorted by percentage, with columns
        ``column``, ``Missing_Count`` and ``Missing_Percentage``.
    """
    print(f"\n{'=' * 60}")
    print("Missing Data Patterns")
    print(f"{'=' * 60}")

    n_rows = df.height
    missing_summary = (
        df.null_count()
        .transpose(
            include_header=True,
            header_name="column",
            column_names=["Missing_Count"],
        )
        .with_columns(
            (pl.col("Missing_Count") / n_rows * 100).alias("Missing_Percentage")
        )
        .filter(pl.col("Missing_Percentage") > threshold)
        .sort("Missing_Percentage", descending=True)
    )

    if missing_summary.height == 0:
        print(f"No missing values found (threshold: {threshold}%)")
    else:
        print(missing_summary)

    return missing_summary


def analyze_target_distribution(
    df: pl.DataFrame, target_col: str = "TARGET", figsize: Tuple[int, int] = (8, 5)
) -> None:
    """
    Analyze and visualize target variable distribution.

    Parameters:
    -----------
    df : pl.DataFrame
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

    counts = df[target_col].value_counts(sort=True)
    labels = [str(v) for v in counts[target_col].to_list()]
    values = counts["count"].to_list()
    total = sum(values)

    print("\nValue Counts:")
    print(counts)
    print("\nPercentages:")
    for label, value in zip(labels, values):
        print(f"  {label}: {value / total * 100:.4f}")

    # Visualization
    colors = ["#2ecc71", "#e74c3c"][: len(values)]
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar plot
    axes[0].bar(labels, values, color=colors)
    axes[0].set_title(f"Distribution of {target_col}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    # Pie plot
    axes[1].pie(values, labels=labels, autopct="%1.1f%%", colors=colors)
    axes[1].set_title(f"{target_col} Distribution (%)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()


def check_duplicates(df: pl.DataFrame, id_col: Optional[str] = None) -> None:
    """
    Check for duplicate records.

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe to check for duplicates
    id_col : str, optional
        Column name to check for duplicate IDs. If None, checks all rows.
    """
    print(f"\n{'=' * 60}")
    print("Duplicate Records Analysis")
    print(f"{'=' * 60}")

    total_duplicates = int(df.is_duplicated().sum())
    print(f"Total Duplicate Rows: {total_duplicates}")

    if id_col and id_col in df.columns:
        id_duplicates = df.height - df[id_col].n_unique()
        print(f"Duplicate {id_col}: {id_duplicates}")


def analyze_numeric_features(
    df: pl.DataFrame,
    figsize: Tuple[int, int] = (18, 6),
    show_summary: bool = True,
    exclude: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Analyze numeric features: distribution, skewness, and outliers.

    ``SK_ID*`` identifier columns are excluded automatically (see
    ``_measurement_numeric_columns``).

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe to analyze
    figsize : Tuple[int, int]
        Figure size for the boxplot
    show_summary : bool
        Whether to print summary statistics and skewness
    exclude : List[str], optional
        Additional numeric column names to exclude from the analysis.

    Returns:
    --------
    pl.DataFrame
        Numeric columns dataframe
    """
    print(f"\n{'=' * 60}")
    print("Numeric Features Analysis")
    print(f"{'=' * 60}")

    numeric_df = df.select(_measurement_numeric_columns(df, exclude))

    if show_summary:
        print("\nDescriptive Statistics:")
        print(numeric_df.describe())

        print("\nSkewness:")
        skewness = numeric_df.select(pl.all().skew()).transpose(
            include_header=True,
            header_name="column",
            column_names=["Skewness"],
        )
        print(skewness.sort("Skewness", descending=True))

    # Boxplot for outliers (drop nulls per column; matplotlib cannot plot NaN)
    box_data = [numeric_df[c].drop_nulls().to_numpy() for c in numeric_df.columns]
    _fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(
        box_data,
        orientation="horizontal",
        tick_labels=numeric_df.columns,
        patch_artist=True,
    )
    ax.grid(True)
    ax.set_title("Box Plot of Numeric Features", fontsize=14, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return numeric_df


def identify_outliers(
    df: pl.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    z_threshold: float = 3.0,
    top_n: int = 20,
    exclude: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Identify outliers using z-score method.

    ``SK_ID*`` identifier columns are excluded automatically (see
    ``_measurement_numeric_columns``).

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe to analyze
    numeric_cols : List[str], optional
        Specific numeric columns to analyze. If None, uses all numeric columns.
    z_threshold : float
        Z-score threshold for outlier detection (default: 3.0)
    top_n : int
        Number of top features with outliers to show
    exclude : List[str], optional
        Additional numeric column names to exclude from the analysis.

    Returns:
    --------
    pl.DataFrame
        Count of outliers per feature (columns ``column`` and
        ``Outlier_Count``), sorted descending.
    """
    print(f"\n{'=' * 60}")
    print(f"Outlier Detection (Z-score > {z_threshold})")
    print(f"{'=' * 60}")

    if numeric_cols is None:
        cols = _measurement_numeric_columns(df, exclude)
    else:
        exclude_set = set(exclude or [])
        cols = [
            c
            for c in df.select(numeric_cols).select(cs.numeric()).columns
            if not c.startswith("SK_ID") and c not in exclude_set
        ]
    numeric_df = df.select(cols)

    # Handle missing values by imputing the per-column median
    clean = numeric_df.with_columns(
        [pl.col(c).fill_null(pl.col(c).median()) for c in cols]
    )

    # Count outliers per column via z-score
    outlier_counts = clean.select(
        [
            (((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).abs() > z_threshold)
            .sum()
            .alias(c)
            for c in cols
        ]
    )

    outlier_summary = outlier_counts.transpose(
        include_header=True,
        header_name="column",
        column_names=["Outlier_Count"],
    ).sort("Outlier_Count", descending=True)

    print(f"\nTop {top_n} Features with Most Outliers:")
    print(outlier_summary.head(top_n))

    # Visualization for top features
    top_features = outlier_summary.head(10)["column"].to_list()
    if top_features:
        # `clean` is already null-free, so the columns plot directly
        box_data = [clean[c].to_numpy() for c in top_features]
        _fig, ax = plt.subplots(figsize=(12, 5))
        ax.boxplot(box_data, orientation="horizontal", tick_labels=top_features)
        ax.grid(True)
        ax.set_title(
            "Box Plot of Top 10 Features with Outliers", fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    return outlier_summary


def analyze_categorical_features(df: pl.DataFrame) -> None:
    """
    Analyze categorical features: unique values and distributions.

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe to analyze
    """
    print(f"\n{'=' * 60}")
    print("Categorical Features Analysis")
    print(f"{'=' * 60}")

    categorical_cols = df.select(cs.string() | cs.categorical()).columns

    if len(categorical_cols) == 0:
        print("No categorical columns found")
        return

    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].n_unique()}")
        print("  Value Counts:")
        print(df[col].value_counts(sort=True))


def analyze_correlations(
    df: pl.DataFrame,
    target_col: str = "TARGET",
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    exclude: Optional[List[str]] = None,
) -> Optional[pl.DataFrame]:
    """
    Analyze correlation between numeric features and target variable.

    ``SK_ID*`` identifier columns are excluded automatically (see
    ``_measurement_numeric_columns``); the ``target_col`` is always retained.

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe to analyze
    target_col : str
        Name of the target column
    top_n : int
        Number of top correlated features to visualize
    figsize : Tuple[int, int]
        Figure size for heatmap
    exclude : List[str], optional
        Additional numeric column names to exclude from the analysis.

    Returns:
    --------
    Optional[pl.DataFrame]
        Correlations (columns ``column`` and ``Correlation``) sorted by value,
        or None if target_col is missing.
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataframe")
        return None

    print(f"\n{'=' * 60}")
    print(f"Correlation Analysis with {target_col}")
    print(f"{'=' * 60}")

    cols = _measurement_numeric_columns(df, exclude)
    if target_col not in cols:
        cols.append(target_col)
    numeric_df = df.select(cols)
    clean = numeric_df.with_columns(
        [pl.col(c).fill_null(pl.col(c).median()) for c in cols]
    )

    correlations = (
        clean.select([pl.corr(c, target_col).alias(c) for c in cols])
        .transpose(
            include_header=True,
            header_name="column",
            column_names=["Correlation"],
        )
        .sort("Correlation", descending=True)
    )

    print(f"\nTop {top_n} Correlated Features:")
    print(correlations.head(top_n))

    # Heatmap of the most correlated features. Build the correlation matrix with
    # numpy so seaborn never needs a pandas/pyarrow round-trip.
    top_features = (
        correlations.with_columns(pl.col("Correlation").abs().alias("abs_corr"))
        .sort("abs_corr", descending=True)
        .head(top_n)["column"]
        .to_list()
    )
    if target_col not in top_features:
        top_features.append(target_col)

    corr_matrix = np.corrcoef(clean.select(top_features).to_numpy(), rowvar=False)
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        xticklabels=top_features,
        yticklabels=top_features,
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


def check_data_quality(df: pl.DataFrame) -> None:
    """
    Run generic, domain-agnostic data-quality checks.

    Reports issues that matter for any table regardless of its columns:
    fully-null columns, constant (single distinct value) columns, and duplicate
    rows. Table-specific business rules (e.g. ``AMT_CREDIT > AMT_GOODS_PRICE``)
    belong in the individual EDA notebooks, not in this shared helper.

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe to check
    """
    print(f"\n{'=' * 60}")
    print("Data Quality Checks")
    print(f"{'=' * 60}")

    # Distinct non-null values per column in a single pass:
    #   0 -> the column is entirely null
    #   1 -> the column is constant (a single value, ignoring nulls)
    distinct = df.select(pl.all().drop_nulls().n_unique())
    fully_null = [c for c in df.columns if distinct[c].item() == 0]
    constant = [c for c in df.columns if distinct[c].item() == 1]

    print(f"\nFully-null columns ({len(fully_null)}):")
    print(f"  {', '.join(fully_null) if fully_null else 'None'}")

    print(f"\nConstant columns ({len(constant)}):")
    print(f"  {', '.join(constant) if constant else 'None'}")

    dup_rows = int(df.is_duplicated().sum())
    print(f"\nDuplicate rows: {dup_rows}")


def identify_feature_engineering_opportunities(
    df: pl.DataFrame, target_col: str = "TARGET"
) -> pl.DataFrame:
    """
    Suggest and calculate feature engineering opportunities.

    Unlike the previous pandas implementation, polars frames are immutable, so
    this returns a new frame with the engineered ratio columns appended rather
    than mutating the input in place.

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe to analyze
    target_col : str
        Name of the target column for correlation check

    Returns:
    --------
    pl.DataFrame
        The input frame with any engineered ratio/age columns appended.
    """
    print(f"\n{'=' * 60}")
    print("Feature Engineering Opportunities")
    print(f"{'=' * 60}")

    opportunities: List[str] = []
    exprs: List[pl.Expr] = []

    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        exprs.append(
            (pl.col("AMT_CREDIT") / pl.col("AMT_GOODS_PRICE")).alias(
                "credit_to_goods_ratio"
            )
        )
        opportunities.append("credit_to_goods_ratio")

    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        exprs.append(
            (pl.col("AMT_ANNUITY") / pl.col("AMT_INCOME_TOTAL")).alias(
                "annuity_to_income_ratio"
            )
        )
        opportunities.append("annuity_to_income_ratio")

    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        exprs.append(
            (pl.col("AMT_CREDIT") / pl.col("AMT_INCOME_TOTAL")).alias(
                "credit_to_income_ratio"
            )
        )
        opportunities.append("credit_to_income_ratio")

    if "DAYS_BIRTH" in df.columns:
        exprs.append((pl.col("DAYS_BIRTH") / -365).alias("age_years"))
        opportunities.append("age_years")

    augmented = df.with_columns(exprs) if exprs else df

    # Check correlations with target
    if opportunities and target_col in augmented.columns:
        print("\nNew Feature Correlations with Target:")
        for feature in opportunities:
            corr = augmented.select(pl.corr(feature, target_col)).item()
            print(f"  {feature}: {corr:.4f}")

    return augmented


def detect_sentinel_values(
    df: pl.DataFrame,
    sentinel: float = 365243,
    columns: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Detect a repeated sentinel/placeholder value in numeric columns.

    Home Credit encodes "not applicable / never" in some numeric columns with a
    single extreme value (notably ``365243`` in the ``DAYS_*`` columns). Such a
    value silently corrupts any mean / describe / correlation, so it must be
    treated as null before numeric analysis. This reports where it occurs.

    Identifier columns (name starting with ``SK_ID``) are skipped automatically
    when scanning all columns: they hold hashed ids where a value equal to the
    sentinel is a coincidence, not missing-value encoding, and would otherwise be
    a false positive.

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe to analyze
    sentinel : float
        The sentinel value to search for (default: 365243)
    columns : List[str], optional
        Specific columns to check. If None, checks all numeric columns except the
        auto-skipped ``SK_ID*`` identifiers.
    exclude : List[str], optional
        Additional column names to skip (applied whether or not ``columns`` is
        given).

    Returns:
    --------
    pl.DataFrame
        Columns ``column``, ``Sentinel_Count`` and ``Sentinel_Percentage`` for
        every column that contains the sentinel, sorted descending.
    """
    print(f"\n{'=' * 60}")
    print(f"Sentinel Value Detection (== {sentinel})")
    print(f"{'=' * 60}")

    if columns is None:
        # Scanning all columns: skip SK_ID* identifiers so a value coinciding with
        # the sentinel is not reported as missing-value encoding.
        cols = _measurement_numeric_columns(df, exclude)
    else:
        exclude_set = set(exclude or [])
        cols = [
            c
            for c in df.select(columns).select(cs.numeric()).columns
            if c not in exclude_set
        ]

    numeric_df = df.select(cols)
    empty = pl.DataFrame(
        schema={
            "column": pl.String,
            "Sentinel_Count": pl.Int64,
            "Sentinel_Percentage": pl.Float64,
        }
    )
    if not cols:
        print("No numeric columns to check")
        return empty

    n_rows = df.height
    summary = (
        numeric_df.select([(pl.col(c) == sentinel).sum().alias(c) for c in cols])
        .transpose(
            include_header=True,
            header_name="column",
            column_names=["Sentinel_Count"],
        )
        .with_columns(
            (pl.col("Sentinel_Count") / n_rows * 100).alias("Sentinel_Percentage")
        )
        .filter(pl.col("Sentinel_Count") > 0)
        .sort("Sentinel_Count", descending=True)
    )

    if summary.height == 0:
        print(f"No occurrences of {sentinel} found")
    else:
        print(summary)
        print(
            f"\nWARNING: replace {sentinel} with null in the columns above before "
            "computing means / describe / correlations."
        )

    return summary


def check_categorical_placeholders(
    df: pl.DataFrame,
    placeholders: Tuple[str, ...] = ("XNA", "XAP"),
) -> pl.DataFrame:
    """
    Count placeholder values in string/categorical columns.

    Home Credit uses tokens such as ``XNA``/``XAP`` for "unknown" or "not
    applicable" instead of null. This is functional missingness that ``.null_count``
    will not catch.

    Parameters:
    -----------
    df : pl.DataFrame
        The dataframe to analyze
    placeholders : Tuple[str, ...]
        Placeholder tokens to count (default: ("XNA", "XAP"))

    Returns:
    --------
    pl.DataFrame
        Long-form summary with columns ``column``, ``placeholder``, ``count`` and
        ``Percentage``, sorted by count descending.
    """
    print(f"\n{'=' * 60}")
    print(f"Categorical Placeholder Detection {placeholders}")
    print(f"{'=' * 60}")

    n_rows = df.height
    string_cols = df.select(cs.string() | cs.categorical()).columns

    col_out: List[str] = []
    plh_out: List[str] = []
    cnt_out: List[int] = []
    for col in string_cols:
        for placeholder in placeholders:
            count = int(df.select((pl.col(col) == placeholder).sum()).item())
            if count > 0:
                col_out.append(col)
                plh_out.append(placeholder)
                cnt_out.append(count)

    summary = pl.DataFrame(
        {"column": col_out, "placeholder": plh_out, "count": cnt_out},
        schema={"column": pl.String, "placeholder": pl.String, "count": pl.Int64},
    )

    if summary.height == 0:
        print("No placeholder values found")
        return summary

    summary = summary.with_columns(
        (pl.col("count") / n_rows * 100).alias("Percentage")
    ).sort("count", descending=True)
    print(summary)

    return summary
