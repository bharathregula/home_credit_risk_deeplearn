# EDA Utilities Guide (`eda_utils`)

The EDA notebooks share a small **polars** helper module,
[`src/home_credit_risk/eda_utils.py`](src/home_credit_risk/eda_utils.py). Notebooks
call these helpers instead of duplicating analysis code, so every dataset notebook
follows the same structure.

For the dataset-by-dataset **findings**, see
[`docs/INDIVIDUAL_DATASET_EDA_SUMMARY.md`](docs/INDIVIDUAL_DATASET_EDA_SUMMARY.md).

> All helpers operate on eager `pl.DataFrame` objects. When working with a
> `pl.LazyFrame` (e.g. `pl.scan_csv`), call `.collect()` before passing it in.

---

## Functions

### Overview / structure
- **`display_basic_info(df, df_name="Dataset") -> None`** — shape and dtypes.
- **`show_missing_patterns(df, threshold=0.0) -> pl.DataFrame`** — per-column
  missing count / percentage (identifier columns are **kept** here).
- **`analyze_target_distribution(df, target_col="TARGET", figsize=(8, 5)) -> None`**
  — value counts + bar/pie plot of the target.
- **`check_duplicates(df, id_col=None) -> None`** — full-row duplicates, plus
  duplicate `id_col` values if given.

### Feature analysis
- **`analyze_numeric_features(df, figsize=(18, 6), show_summary=True, exclude=None) -> pl.DataFrame`**
  — describe, skewness, boxplots.
- **`identify_outliers(df, numeric_cols=None, z_threshold=3.0, top_n=20, exclude=None) -> pl.DataFrame`**
  — z-score outlier counts per feature.
- **`analyze_categorical_features(df) -> None`** — unique values and value counts
  for string/categorical columns.
- **`analyze_correlations(df, target_col="TARGET", top_n=15, figsize=(10, 8), exclude=None) -> Optional[pl.DataFrame]`**
  — correlation of numeric features with the target (+ heatmap); returns `None` if
  `target_col` is absent.

> The four feature-analysis helpers above **auto-exclude `SK_ID*` identifier
> columns** — describe / correlation / outlier scores over a hashed id are
> meaningless. Pass `exclude=[...]` to drop additional columns. The `target_col`
> is always retained by `analyze_correlations`.

### Quality & engineering
- **`check_data_quality(df) -> None`** — **generic**, domain-agnostic checks:
  fully-null columns, constant (single-value) columns, and duplicate rows. It does
  **not** hard-code any dataset's column names; table-specific business rules live in
  the individual notebooks.
- **`identify_feature_engineering_opportunities(df, target_col="TARGET") -> pl.DataFrame`**
  — builds candidate ratio/age features and **returns a new frame** with them
  appended (polars frames are immutable — the input is not mutated).
- **`detect_sentinel_values(df, sentinel=365243, columns=None, exclude=None) -> pl.DataFrame`**
  — finds a repeated numeric sentinel (notably Home Credit's `365243` "not
  applicable / never" placeholder in `DAYS_*` columns). When scanning all columns it
  **auto-skips `SK_ID*`** identifiers so an id coinciding with the sentinel is not a
  false positive.
- **`check_categorical_placeholders(df, placeholders=("XNA", "XAP")) -> pl.DataFrame`**
  — counts placeholder tokens in string/categorical columns (functional missingness
  not stored as null).

---

## Notebook structure

Each notebook in [`notebooks/individual_dataset_eda/`](notebooks/individual_dataset_eda)
follows the same skeleton:

```
1. Setup and Data Loading        (polars read_csv / scan_csv)
2. Basic Dataset Exploration     (display_basic_info, describe)
3. Missing Data Analysis         (show_missing_patterns, check_duplicates)
4. Numeric Features Analysis     (analyze_numeric_features, identify_outliers)
5. Categorical Features Analysis (analyze_categorical_features)
6. Standard Data Quality Checks  (check_data_quality — generic)
7. Functional / Business-Logic Quality Checks   (table-specific, inline)
```

`application_eda.ipynb` additionally has target-distribution, correlation-with-TARGET,
and feature-engineering sections (it is the only table with a `TARGET`), and it holds
the application-specific business rules that used to live in `check_data_quality`.

---

## Template for a new dataset notebook

```python
import polars as pl
from home_credit_risk.eda_utils import (
    display_basic_info,
    show_missing_patterns,
    check_duplicates,
    analyze_numeric_features,
    identify_outliers,
    analyze_categorical_features,
    check_data_quality,
    detect_sentinel_values,
    check_categorical_placeholders,
)

# The whole frame is reused by every check, so collect it once.
df = pl.read_csv(datasets / "your_dataset.csv", infer_schema_length=100000)

display_basic_info(df, df_name="YourDataset")
show_missing_patterns(df)
check_duplicates(df)                 # or check_duplicates(df, id_col="SK_ID_PREV")
analyze_numeric_features(df)         # SK_ID* auto-excluded
identify_outliers(df)
analyze_categorical_features(df)
check_data_quality(df)               # generic
detect_sentinel_values(df)           # SK_ID* auto-excluded
check_categorical_placeholders(df)

# Then a "Functional / Business-Logic Quality Checks" section with inline polars
# expressions encoding this table's own consistency rules.
```

For very large tables (e.g. `bureau_balance`), prefer lazy `pl.scan_csv(...)` and
`.collect()` at display points; see `bureau_eda.ipynb`.

---

## Current layout

```
home_credit_risk_deeplearn/
├── src/home_credit_risk/
│   ├── config.py
│   ├── logger.py
│   ├── eda_utils.py                     # shared polars EDA helpers
│   ├── main.py
│   └── trivial_test.py
├── notebooks/
│   └── individual_dataset_eda/
│       ├── application_eda.ipynb
│       ├── bureau_eda.ipynb             # bureau.csv + bureau_balance.csv
│       ├── previous_application_eda.ipynb
│       ├── credit_card_balance_eda.ipynb
│       ├── installments_payments_eda.ipynb
│       └── pos_cash_balance_eda.ipynb
├── docs/
│   └── INDIVIDUAL_DATASET_EDA_SUMMARY.md
├── tests/
│   ├── test_trivial.py
│   └── test_eda_utils.py
├── pyproject.toml
└── README.md
```

---

## Conventions

- **DRY** — common operations live in `eda_utils`; a fix there benefits every notebook.
- **Report, don't mutate** — helpers print a summary and return a `pl.DataFrame` (or
  `None`); cleaning/remediation is a separate, explicit step in the notebook.
- **Generic vs table-specific** — `eda_utils` stays table-agnostic; column-specific
  business rules live inline in each notebook's functional-checks section.
- **Datasets are not in the repo** — notebooks load them from a local path
  (`~/code/datasets/home-credit-default-risk`); update it if your location differs.
