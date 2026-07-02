# EDA Refactoring Summary

## Overview
Your EDA notebooks have been refactored to follow a clean, modular architecture with reusable functions and focused analysis scope.

---

## Key Changes

### 1. **Created Reusable EDA Module** (`src/home_credit_risk/eda_utils.py`)

This module contains 10 reusable functions for exploratory data analysis:

#### Core Analysis Functions:
- **`display_basic_info(df, df_name)`** - Display shape, dtypes, and dataset info
- **`show_missing_patterns(df, threshold)`** - Analyze and visualize missing data
- **`analyze_target_distribution(df, target_col, figsize)`** - Analyze and plot target variable
- **`check_duplicates(df, id_col)`** - Check for duplicate records/IDs

#### Feature Analysis Functions:
- **`analyze_numeric_features(df, figsize, show_summary)`** - Distribution and boxplot analysis
- **`identify_outliers(df, numeric_cols, z_threshold, top_n)`** - Z-score outlier detection with visualization
- **`analyze_categorical_features(df)`** - Unique values and distributions
- **`analyze_correlations(df, target_col, top_n, figsize)`** - Correlation matrix and heatmap

#### Quality & Engineering Functions:
- **`check_data_quality(df)`** - Check for negative values, logical inconsistencies, and data issues
- **`identify_feature_engineering_opportunities(df, target_col)`** - Suggest and calculate engineered features

### 2. **Refactored EDA.ipynb** (Application_train.csv only)

**Scope:** Exclusively analyzes the `application_train.csv` dataset

**Structure:**
```
1. Setup and Data Loading
   └─ Imports, configuration, path setup
2. Basic Dataset Exploration
   └─ Dataset info, dtypes, summary statistics
3. Missing Data Analysis
   └─ Missing patterns, duplicate checks
4. Target Variable Analysis
   └─ Distribution, class imbalance
5. Numeric Features Analysis
   └─ Distributions, skewness, boxplots, outlier detection
6. Categorical Features Analysis
   └─ Unique values, distributions
7. Correlation Analysis
   └─ Feature-target correlations, heatmap
8. Data Quality Checks
   └─ Negative values, logical inconsistencies, impossible values
9. Feature Engineering Opportunities
   └─ Ratio features, derived features
```

**Improvements:**
- ✅ Clean section organization with markdown headers
- ✅ Removed multi-dataset merging code
- ✅ Uses reusable functions from `eda_utils`
- ✅ Focused, readable, maintainable code
- ✅ Better documentation and commented sections

### 3. **Refactored bureau_eda.ipynb** (Bureau datasets)

**Scope:** Analyzes `bureau.csv` and `bureau_balance.csv` with merged analysis

**Structure:**
```
1. Setup and Data Loading
2. Bureau Dataset Exploration
3. Bureau Balance Dataset Exploration
4. Merged Bureau Analysis
5. Missing Data Analysis (for each dataset)
6. Numeric Features Analysis (for each dataset)
7. Categorical Features Analysis (for each dataset)
8. Data Quality Checks (for each dataset)
9. Feature Engineering Opportunities (for each dataset)
```

**Template Pattern:**
- Uses the same functions as `EDA.ipynb`
- Demonstrates how to apply functions to different datasets
- Can be copied for other datasets (previous_application, pos_cash_balance, etc.)

---

## How to Use for Other Datasets

To create EDA notebooks for other datasets (e.g., `previous_application.csv`), follow this template:

```python
# 1. Import the reusable functions
from home_credit_risk.eda_utils import (
    display_basic_info,
    show_missing_patterns,
    analyze_numeric_features,
    identify_outliers,
    analyze_categorical_features,
    check_data_quality,
    identify_feature_engineering_opportunities
)

# 2. Load dataset
df = pd.read_csv(datasets / 'your_dataset.csv')

# 3. Run analyses in order
display_basic_info(df, df_name='YourDataset')
show_missing_patterns(df)
analyze_numeric_features(df)
identify_outliers(df)
analyze_categorical_features(df)
check_data_quality(df)
identify_feature_engineering_opportunities(df, target_col=None)  # None if no target
```

---

## Function Parameters Reference

### `display_basic_info()`
```python
display_basic_info(df: pd.DataFrame, df_name: str = "Dataset")
```

### `show_missing_patterns()`
```python
show_missing_patterns(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame
# threshold: only show columns with missing % > threshold
```

### `analyze_target_distribution()`
```python
analyze_target_distribution(df: pd.DataFrame, target_col: str = 'TARGET', figsize: Tuple = (8, 5))
```

### `analyze_numeric_features()`
```python
analyze_numeric_features(df: pd.DataFrame, figsize: Tuple = (18, 6), show_summary: bool = True) -> pd.DataFrame
```

### `identify_outliers()`
```python
identify_outliers(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,  # If None, uses all numeric columns
    z_threshold: float = 3.0,  # Z-score threshold
    top_n: int = 20  # Top N features to visualize
) -> pd.DataFrame
```

### `analyze_categorical_features()`
```python
analyze_categorical_features(df: pd.DataFrame)
```

### `analyze_correlations()`
```python
analyze_correlations(
    df: pd.DataFrame,
    target_col: str = 'TARGET',  # Set to None if dataset has no target
    top_n: int = 15,  # Number of top features to visualize
    figsize: Tuple = (10, 8)
) -> pd.Series
```

### `check_data_quality()`
```python
check_data_quality(df: pd.DataFrame)
```

### `identify_feature_engineering_opportunities()`
```python
identify_feature_engineering_opportunities(df: pd.DataFrame, target_col: str = 'TARGET')
# Set target_col=None for datasets without a target variable
```

---

## File Structure After Refactoring

```
home_credit_risk_deeplearn/
├── src/
│   └── home_credit_risk/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       ├── main.py
│       ├── eda_utils.py          ← NEW: Reusable EDA functions
│       └── trivial_test.py
├── notebooks/
│   ├── EDA.ipynb                 ← REFACTORED: Application_train only
│   └── bureau_eda.ipynb          ← REFACTORED: Bureau datasets
├── tests/
│   └── test_trivial.py
├── pyproject.toml
└── README.md
```

---

## Benefits of This Refactoring

✅ **DRY Principle**: Common EDA operations are centralized
✅ **Consistency**: All EDA notebooks follow the same structure
✅ **Maintainability**: Bug fixes in one function benefit all notebooks
✅ **Scalability**: Easy to add new datasets with minimal code
✅ **Readability**: Clear, organized, well-documented code
✅ **Reusability**: Functions work across different dataset sizes and structures
✅ **Focused Analysis**: Each notebook has a single, clear scope

---

## Next Steps

1. **Create additional EDA notebooks** for other datasets using the same pattern:
   - `previous_application_eda.ipynb`
   - `pos_cash_balance_eda.ipynb`
   - `installments_payments_eda.ipynb`
   - `credit_card_balance_eda.ipynb`

2. **Consider adding more utility functions** for:
   - Feature selection analysis
   - Distribution comparison across datasets
   - Correlation to target variable analysis

3. **Document findings** in each notebook's summary section for reference during modeling

---

## Running the Notebooks

The notebooks use absolute paths, so ensure the dataset path is correct:
```python
datasets = Path('/Users/bharathregula/code/datasets/home-credit-default-risk')
```

If your dataset location changes, update this path in all notebooks.
