# Feature Dictionary — every feature, its definition, and whether the baseline GBT kept it

Complete reference for the **969 candidate features** produced by [`features.build_feature_matrix`](../src/home_credit_risk/features.py), each annotated with whether it made the **top-200 gain shortlist** from the LightGBM baseline in [`baseline_trees.ipynb`](../notebooks/modeling/baseline_trees.ipynb).

| | |
|---|---|
| Rows analysed | `application_train`, 307,511 customers (one row per `SK_ID_CURR`) |
| Matrix shape | 307,511 × 971 columns |
| Candidate features | 969 (= 971 − `TARGET` − `SK_ID_CURR`) |
| Dropped as zero-variance or duplicate | 24 → **945 fed to the model** |
| Selected (top-200 by gain) | **200** — 82.7% of total model gain |
| Target base rate | 8.07% positive |
| Machine-readable version | [`feature_dictionary.csv`](feature_dictionary.csv) — same content, one row per feature, with the full definition sentence and every percentile |

**Companion docs:** [FEATURE_ENGINEERING_SUMMARY.md](FEATURE_ENGINEERING_SUMMARY.md) (how the pipeline is built) · [MODELING_SUMMARY.md](MODELING_SUMMARY.md) (how the baseline was fit) · [INDIVIDUAL_DATASET_EDA_SUMMARY.md](INDIVIDUAL_DATASET_EDA_SUMMARY.md) (the raw tables).

---

## 1. Inventory

| Block | Grain / source | Features | Selected | Hit rate | Share of model gain |
|---|---|---:|---:|---:|---:|
| `DRIVER` | `application_train`, as-is | 120 | 29 | 24% | 46.5% |
| `DRIVER_DERIVED` | `application_train`, derived | 7 | 6 | 86% | 5.5% |
| `PREV` | `previous_application` → customer | 162 | 37 | 23% | 11.8% |
| `CC` | `credit_card_balance` → loan → customer | 332 | 9 | 3% | 4.0% |
| `INST` | `installments_payments` → loan → customer | 116 | 42 | 36% | 11.4% |
| `POS` | `POS_CASH_balance` → loan → customer | 88 | 14 | 16% | 4.5% |
| `BUREAU` | `bureau` (+`bureau_balance`) → customer | 79 | 31 | 39% | 8.6% |
| `CC_L12M` | `credit_card_balance`, last 12 months | 16 | 4 | 25% | 0.8% |
| `POS_L12M` | `POS_CASH_balance`, last 12 months | 10 | 7 | 70% | 1.4% |
| `INST_L365D` | `installments_payments`, last 365 days | 9 | 8 | 89% | 1.9% |
| `PREV_L365D` | `previous_application`, last 365 days | 14 | 3 | 21% | 0.8% |
| `BUREAU_L730D` | `bureau`, last 730 days | 14 | 10 | 71% | 2.7% |
| `FLAG` | coverage flags | 2 | 0 | 0% | 0.0% |
| **Total** | | **969** | **200** | **21%** | **100%** |

The shape of this table is the main finding: the **application form and the six ratios derived from it are half the model** (52% of gain from 127 columns), the two-level ledger blocks are large and dilute (`CC` is 34% of the features for 4% of the gain), and the small hand-curated blocks — installment behaviour and the recency windows — are where the aggregation work paid off. The five recency windows are 63 features for 7.6% of gain with **32 of 63 selected**, and the seven driver-derived features are 5.5% of gain with **6 of 7 selected** — by far the best return per feature in the matrix.

## 2. How to read a feature name

Every engineered name is compositional:

```
[BLOCK]_[SOURCE COLUMN]_[INNER AGG]_[OUTER AGG]
```

| Piece | Values | Meaning |
|---|---|---|
| Block prefix | `PREV_`, `PREV_CC_`, `PREV_INST_`, `PREV_POS_`, `BUREAU_`, `BUREAU_BB_`, `CC_L12M_`, `POS_L12M_`, `INST_L365D_`, `PREV_L365D_`, `BUREAU_L730D_`, `HAS_` | which source table and which lineage |
| Source column | any raw column of that table, or a derived metric (`INST_LATE_RATE`, `CC_UTILIZATION_MEAN`, `BB_DPD_MONTHS`, …) | what is being measured |
| Agg suffix | `_MEAN`, `_MIN`, `_MAX`, `_SUM`, `_COUNT`, `_NUNIQUE`, `_RATE` | how it was collapsed |

Two aggregation suffixes means a **two-level** roll-up — inner first (per prior loan, `SK_ID_PREV`), outer second (per customer, `SK_ID_CURR`):

```
PREV_CC_AMT_BALANCE_MEAN_MAX
 |    |  |          |    └── outer: MAX across the customer's prior loans
 |    |  |          └─────── inner: MEAN over that loan's monthly card records
 |    |  └────────────────── source column AMT_BALANCE
 |    └───────────────────── credit_card_balance ledger
 └─────────────────────────── reached through the previous_application hub
```

So `PREV_CC_AMT_BALANCE_MEAN_MAX` = *"of all this customer's past Home Credit loans, the highest average monthly card balance any one of them carried."*

## 3. Null semantics — and why two null columns are reported

Almost every null here means **"no such history"**, not "measurement missing". That splits the columns in two, and the dictionary reports both states:

| Column in the tables below | Meaning |
|---|---|
| **Null %** | missingness in the raw matrix, straight out of `build_feature_matrix` — i.e. genuine coverage |
| **Null after enc.** | missingness after `apply_structural_encoding`, which is what the model actually saw |

`apply_structural_encoding` fills `*_COUNT` / `*_SUM` / `*_NUNIQUE` / `*_RATE` / `HAS_*` nulls with `0`, because zero records really is zero — that is feature **construction**, not statistical imputation. Behavioural `*_MEAN` / `*_MIN` / `*_MAX` are left null on purpose: the average balance of a card you never had is genuinely undefined, and LightGBM learns a default split direction for it. **All distribution statistics below are computed post-encoding** (the values the model saw), so a structural column's mean includes its zero-fills; its raw coverage is still visible in the Null % column.

Coverage at the top of each lineage:

| Flag | Meaning | Rate |
|---|---|---|
| `HAS_PREV` | has ≥1 previous Home Credit application | **94.6%** |
| `HAS_BUREAU` | has ≥1 credit-bureau record | **85.7%** |
| `PREV_CC_COUNT_SUM` > 0 | has any credit-card history | **25.3%** |

One diagnostic worth reading in the CSV: `default_rate_when_null` — the observed default rate among customers for whom the feature is null. Where it deviates sharply from the 8.1% base rate, *the missingness itself is the signal*, which is exactly what a null-native tree can exploit and a median-imputed linear model destroys.

## 4. Encoding logic

Three distinct mechanisms turn categories into numbers; which one applies depends on where the category lived.

### 4.1 Driver categoricals → integer codes + LightGBM native handling

The 16 string columns of `application_train` are **not** one-hot encoded. [`to_tree_matrix`](../src/home_credit_risk/modeling.py) casts each to `pl.Categorical`, takes `.to_physical()` to get integer codes, and returns their column indices, which are passed to LightGBM as `categorical_feature=cat_idx`. The trees then partition the *set* of categories at each split rather than treating the codes as ordered — so code 3 < code 7 never implies an ordering. Nulls stay null instead of becoming a spurious extra category.

| Column | Distinct | Selected | Rank | Modal value (share, default rate) |
|---|---:|:---:|---:|---|
| `ORGANIZATION_TYPE` | 58 | **✔** | 3 | Business Entity Type 3 — 22.1%, 9.3% |
| `OCCUPATION_TYPE` | 19 | **✔** | 8 | *(null)* — 31.3%, 6.5% |
| `CODE_GENDER` | 3 | **✔** | 14 | F — 65.8%, 7.0% |
| `NAME_EDUCATION_TYPE` | 5 | **✔** | 15 | Secondary / secondary special — 71.0%, 8.9% |
| `NAME_FAMILY_STATUS` | 6 | **✔** | 48 | Married — 63.9%, 7.6% |
| `WEEKDAY_APPR_PROCESS_START` | 7 | **✔** | 178 | TUESDAY — 17.5%, 8.3% |
| `NAME_INCOME_TYPE` | 8 | **✔** | 195 | Working — 51.6%, 9.6% |
| `NAME_CONTRACT_TYPE` | 2 | · | 205 | Cash loans — 90.5%, 8.3% |
| `FLAG_OWN_CAR` | 2 | · | 315 | N — 66.0%, 8.5% |
| `WALLSMATERIAL_MODE` | 8 | · | 316 | *(null)* — 50.8%, 9.1% |
| `NAME_HOUSING_TYPE` | 6 | · | 413 | House / apartment — 88.7%, 7.8% |
| `NAME_TYPE_SUITE` | 8 | · | 532 | Unaccompanied — 80.8%, 8.2% |
| `FLAG_OWN_REALTY` | 2 | · | 627 | Y — 69.4%, 8.0% |
| `FONDKAPREMONT_MODE` | 5 | · | 667 | *(null)* — 68.4%, 8.6% |
| `HOUSETYPE_MODE` | 4 | · | 751 | *(null)* — 50.2%, 9.2% |
| `EMERGENCYSTATE_MODE` | 3 | · | 784 | No — 51.8%, 7.0% |

> **Caveat for scoring `application_test`.** Polars assigns categorical codes by order of appearance *within the frame being encoded*. `to_tree_matrix` is called on one frame at a time, so encoding train and test separately can map the same string to different integers. Cross-validation in the baseline notebook is unaffected (single frame), but a future inference path must encode train and test from a shared mapping — e.g. `pl.concat` before coding, or a persisted category→code dictionary.

### 4.2 Aggregated categoricals → one-hot **rates** and distinct-counts

A category in a *child* table has many values per customer, so there is no single value to encode. `aggregate_categorical` instead emits, per customer:

- **`<BLOCK>_<COL>_<VALUE>_RATE`** (70 features) — the share of that customer's records in that category, `mean(col == value)`. A continuous 0–1 profile, not a 0/1 indicator: `PREV_NAME_CONTRACT_STATUS_Refused_RATE = 0.4` means 40% of this customer's past applications were refused. Emitted **only for columns with ≤ 10 distinct values**, which bounds the block.
- **`<BLOCK>_<COL>_NUNIQUE`** (19 features) — the count of distinct values the customer shows. This is what high-cardinality columns (`NAME_GOODS_CATEGORY`, `CREDIT_TYPE`, …) get *instead of* rates: a breadth signal rather than a membership signal.

Both are numeric by construction, so no further encoding happens at model time, and both are structural (null → 0).

### 4.3 Hand-written domain flags → counts and rates

A few status categories were considered important enough to encode explicitly, in both count and rate form (`PREV_APPROVED_COUNT` / `PREV_APPROVED_RATE`, `PREV_REFUSED_*`, `BUREAU_ACTIVE_*`, `BUREAU_CLOSED_COUNT`), plus the per-loan coverage rates `PREV_HAS_CC_RATE` / `PREV_HAS_INST_RATE` / `PREV_HAS_POS_RATE`. These duplicate some of §4.2 by design — the count and the rate carry different information (volume vs. propensity) and the trees pick whichever splits better.

## 5. What "selected" means

The **Selected** column throughout is membership in `gbt_shortlist.txt`: the **top 200 features by summed LightGBM gain importance**, accumulated across all 5 CV folds (`booster_.feature_importance(importance_type='gain')`). Gain is the total loss reduction a feature's splits delivered — not how often it was split on, which over-rewards high-cardinality continuous columns.

| | |
|---|---|
| Baseline OOF ROC-AUC | ~0.79 (5-fold stratified) |
| Features with gain > 0 | 873 of 945 |
| Cumulative gain kept by the top 200 | **82.7%** |
| Cutoff | rank 200 = `PREV_L365D_AMT_ANNUITY_MAX` |

**Why selection was done here rather than on correlations or missingness.** The trees consumed the nulls directly, so gain is measured on the data as it really is. A missingness filter would have dropped `PREV_CC_UTILIZATION_MEAN_MAX` (74.9% null, rank 85) and a median-imputed correlation would have understated it. The 200-feature cut is a pragmatic top-K, not a significance test — every kept feature has gain > 0, and the shortlist exists to give the deferred neural-net round a tractable input space where per-fold imputation and scaling are affordable.

**Selected ≠ individually important.** Gain is split among correlated features, so a feature can rank low because a near-duplicate absorbed its splits (the `CC` block's four-agg cross product is full of these). Read a low rank as "redundant or weak here", not as "no relationship with default".

---

## 6. The 200 selected features

Ranked by gain. **Null %** is pre-encoding (true coverage); statistics are post-encoding. **Corr** is the Pearson correlation with `TARGET` over non-null rows — shown for context only, it played no part in selection.

| # | Feature | Gain % | Definition | Null % | Median | Mean | Corr |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | `EXT_SOURCE_3` | 11.85 | Normalized score from external data source | 19.8 | 0.54 | 0.51 | -0.179 |
| 2 | `EXT_SOURCE_2` | 11.41 | Normalized score from external data source | 0.2 | 0.57 | 0.51 | -0.160 |
| 3 | `ORGANIZATION_TYPE` | 5.72 | Type of organization where client works | 0.0 | – | – | – |
| 4 | `EXT_SOURCE_1` | 3.79 | Normalized score from external data source | 56.4 | 0.51 | 0.50 | -0.155 |
| 5 | `RATIO_PAYMENT_RATE` | 2.12 | Affordability ratio `AMT_ANNUITY` / `AMT_CREDIT` — effective repayment speed: the fraction of principal repaid per annuity period (its reciprocal is the term in periods). A tree splits on one column at a time and cannot express a quotient, so this is built explicitly on the driver frame. | 0.0 | 0.05 | 0.05 | 0.013 |
| 6 | `RATIO_CREDIT_TO_GOODS` | 1.87 | Affordability ratio `AMT_CREDIT` / `AMT_GOODS_PRICE` — how much was borrowed above the financed item's price (>1 means cash on top). A tree splits on one column at a time and cannot express a quotient, so this is built explicitly on the driver frame. | 0.1 | 1.12 | 1.12 | 0.069 |
| 7 | `DAYS_EMPLOYED` | 1.54 | How many days before the application the person started current employment | 18.0 | -1648 | -2384.17 | 0.075 |
| 8 | `OCCUPATION_TYPE` | 1.17 | What kind of occupation does the client have | 31.3 | – | – | – |
| 9 | `DAYS_BIRTH` | 1.07 | Client's age in days at the time of application | 0.0 | -15,750 | -16,037 | 0.078 |
| 10 | `PREV_REFUSED_RATE` | 1.05 | Share of the customer's previous Home Credit applications with `NAME_CONTRACT_STATUS` = "Refused". | 5.4 | 0 | 0.10 | 0.078 |
| 11 | `AMT_ANNUITY` | 1.03 | Loan annuity | 0.0 | 24,903 | 27,109 | -0.013 |
| 12 | `PREV_INST_AMT_PAYMENT_MIN_MEAN` | 0.97 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the minimum of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 8272.27 | 13,438 | -0.024 |
| 13 | `INST_L365D_LATE_RATE` | 0.85 | Share of installments due in the last 365 days that were paid after their due date. | 29.3 | 0 | 0.04 | 0.083 |
| 14 | `CODE_GENDER` | 0.80 | Gender of the client | 0.0 | – | – | – |
| 15 | `NAME_EDUCATION_TYPE` | 0.79 | Level of highest education the client achieved | 0.0 | – | – | – |
| 16 | `OWN_CAR_AGE` | 0.66 | Age of client's car | 66.0 | 9 | 12.06 | 0.038 |
| 17 | `PREV_DAYS_LAST_DUE_1ST_VERSION_MAX` | 0.65 | The maximum of `DAYS_LAST_DUE_1ST_VERSION` across the customer's previous Home Credit applications. Source column: relative to application date of current application when was the first due of the previous application | 6.7 | 12 | -77.75 | 0.011 |
| 18 | `PREV_INST_LATE_RATE_MEAN` | 0.61 | The average, across the customer's previous Home Credit applications, of a per-loan derived metric: the share of installments paid after their due date (DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT), computed over that loan's installment ledger. | 5.9 | 0.01 | 0.07 | 0.075 |
| 19 | `RATIO_ANNUITY_TO_INCOME` | 0.60 | Affordability ratio `AMT_ANNUITY` / `AMT_INCOME_TOTAL` — debt-service burden: the share of annual income consumed by this loan's annuity. A tree splits on one column at a time and cannot express a quotient, so this is built explicitly on the driver frame. | 0.0 | 0.16 | 0.18 | 0.014 |
| 20 | `BUREAU_DAYS_CREDIT_MEAN` | 0.58 | The average of `DAYS_CREDIT` across the customer's credit-bureau records (loans at other institutions). Source column: how many days before current application did client apply for Credit Bureau credit | 14.3 | -1050.57 | -1083.05 | 0.090 |
| 21 | `BUREAU_DAYS_CREDIT_MAX` | 0.58 | The maximum of `DAYS_CREDIT` across the customer's credit-bureau records (loans at other institutions). Source column: how many days before current application did client apply for Credit Bureau credit | 14.3 | -300 | -489.30 | 0.050 |
| 22 | `PREV_POS_CNT_INSTALMENT_FUTURE_MEAN_MEAN` | 0.54 | Two-level roll-up of `CNT_INSTALMENT_FUTURE` from the POS_CASH_balance ledger — installments left to pay on the previous credit. Step 1: per prior loan, the average of `CNT_INSTALMENT_FUTURE` over that loan's monthly POS/cash records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 6.7 | 7 | 9.07 | 0.033 |
| 23 | `AMT_GOODS_PRICE` | 0.52 | For consumer loans it is the price of the goods for which the loan is given | 0.1 | 450,000 | 538,396 | -0.040 |
| 24 | `PREV_INST_LATE_RATE_MIN` | 0.49 | The minimum, across the customer's previous Home Credit applications, of a per-loan derived metric: the share of installments paid after their due date (DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT), computed over that loan's installment ledger. | 5.9 | 0 | 0.03 | 0.072 |
| 25 | `DAYS_ID_PUBLISH` | 0.48 | How many days before the application did client change the identity document with which he applied for the loan | 0.0 | -3254 | -2994.20 | 0.051 |
| 26 | `RATIO_EMPLOYED_TO_AGE` | 0.48 | Affordability ratio `DAYS_EMPLOYED` / `DAYS_BIRTH` — fraction of life spent in the current job (both inputs are negative day offsets, so the quotient is positive). A tree splits on one column at a time and cannot express a quotient, so this is built explicitly on the driver frame. | 18.0 | 0.12 | 0.16 | -0.068 |
| 27 | `PREV_INST_AMT_PAYMENT_MIN_MAX` | 0.46 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the minimum of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 12,608 | 27,500 | -0.014 |
| 28 | `BUREAU_L730D_DAYS_CREDIT_MAX` | 0.45 | The maximum of `DAYS_CREDIT` over bureau records within the last 730 days (`DAYS_CREDIT` >= -730). Source column: how many days before current application did client apply for Credit Bureau credit | 31.8 | -228 | -266.67 | 0.066 |
| 29 | `PREV_INST_AMT_PAYMENT_MIN_SUM` | 0.43 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the minimum of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the total of those per-loan values across all of the customer's previous Home Credit applications. | 5.4 | 17,062 | 38,783 | -0.024 |
| 30 | `AMT_CREDIT` | 0.42 | Credit amount of the loan | 0.0 | 513,531 | 599,026 | -0.030 |
| 31 | `BUREAU_L730D_AMT_CREDIT_SUM_DEBT_MEAN` | 0.40 | The average of `AMT_CREDIT_SUM_DEBT` over bureau records within the last 730 days (`DAYS_CREDIT` >= -730). Source column: current debt on Credit Bureau credit | 35.5 | 83,243 | 292,463 | -0.009 |
| 32 | `BUREAU_DAYS_CREDIT_ENDDATE_MAX` | 0.40 | The maximum of `DAYS_CREDIT_ENDDATE` across the customer's credit-bureau records (loans at other institutions). Source column: remaining duration of CB credit (in days) at the time of application in Home Credit | 15.0 | 909 | 4433.04 | 0.037 |
| 33 | `BUREAU_AMT_CREDIT_SUM_DEBT_MEAN` | 0.39 | The average of `AMT_CREDIT_SUM_DEBT` across the customer's credit-bureau records (loans at other institutions). Source column: current debt on Credit Bureau credit | 16.7 | 44,208 | 160,390 | -6.40e-04 |
| 34 | `POS_L12M_CNT_INSTALMENT_FUTURE_MEAN` | 0.38 | The average of `CNT_INSTALMENT_FUTURE` over POS_CASH_balance records within the last 12 months (`MONTHS_BALANCE` >= -12). Source column: installments left to pay on the previous credit | 32.8 | 7.50 | 11.24 | 0.025 |
| 35 | `BUREAU_AMT_CREDIT_SUM_MAX` | 0.36 | The maximum of `AMT_CREDIT_SUM` across the customer's credit-bureau records (loans at other institutions). Source column: current credit amount for the Credit Bureau credit | 14.3 | 450,000 | 976,004 | -0.020 |
| 36 | `PREV_CNT_PAYMENT_MEAN` | 0.34 | The average of `CNT_PAYMENT` across the customer's previous Home Credit applications. Source column: term of previous credit at application of the previous application | 5.5 | 12 | 14.53 | 0.028 |
| 37 | `BUREAU_AMT_CREDIT_SUM_MEAN` | 0.34 | The average of `AMT_CREDIT_SUM` across the customer's credit-bureau records (loans at other institutions). Source column: current credit amount for the Credit Bureau credit | 14.3 | 195,507 | 378,080 | -0.020 |
| 38 | `BUREAU_DAYS_ENDDATE_FACT_MAX` | 0.33 | The maximum of `DAYS_ENDDATE_FACT` across the customer's credit-bureau records (loans at other institutions). Source column: days since CB credit ended at the time of application in Home Credit (only for closed credit) | 25.1 | -345 | -526.99 | 0.020 |
| 39 | `PREV_NAME_YIELD_GROUP_high_RATE` | 0.33 | Share of the customer's previous Home Credit applications whose `NAME_YIELD_GROUP` equals "high" (0 = never, 1 = always). Source column: grouped interest rate into small medium and high of the previous application | 5.4 | 0.11 | 0.23 | 0.038 |
| 40 | `BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN` | 0.33 | The average of `AMT_CREDIT_MAX_OVERDUE` across the customer's credit-bureau records (loans at other institutions). Source column: maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample) | 40.2 | 0 | 5242.43 | 0.002 |
| 41 | `PREV_AMT_DOWN_PAYMENT_MAX` | 0.32 | The maximum of `AMT_DOWN_PAYMENT` across the customer's previous Home Credit applications. Source column: down payment on the previous application | 11.0 | 5850 | 11,681 | -0.025 |
| 42 | `PREV_DAYS_LAST_DUE_MAX` | 0.32 | The maximum of `DAYS_LAST_DUE` across the customer's previous Home Credit applications. Source column: relative to application date of current application when was the last due date of the previous application | 12.5 | -346 | -559.99 | 0.001 |
| 43 | `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MEAN_MEAN` | 0.31 | Two-level roll-up of `CNT_DRAWINGS_ATM_CURRENT` from the credit_card_balance ledger — number of drawings at ATM during this month on the previous credit. Step 1: per prior loan, the average of `CNT_DRAWINGS_ATM_CURRENT` over that loan's monthly credit-card records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 82.8 | 0.33 | 0.63 | 0.103 |
| 44 | `BUREAU_CREDIT_ACTIVE_Closed_RATE` | 0.31 | Share of the customer's credit-bureau records (loans at other institutions) whose `CREDIT_ACTIVE` equals "Closed" (0 = never, 1 = always). Source column: status of the Credit Bureau (CB) reported credits | 14.3 | 0.52 | 0.50 | -0.079 |
| 45 | `PREV_AMT_DOWN_PAYMENT_SUM` | 0.31 | The total of `AMT_DOWN_PAYMENT` across the customer's previous Home Credit applications. Source column: down payment on the previous application | 5.4 | 5152.50 | 14,374 | -0.027 |
| 46 | `DAYS_REGISTRATION` | 0.31 | How many days before the application did client change his registration | 0.0 | -4504 | -4986.12 | 0.042 |
| 47 | `BUREAU_L730D_AMT_CREDIT_SUM_MAX` | 0.30 | The maximum of `AMT_CREDIT_SUM` over bureau records within the last 730 days (`DAYS_CREDIT` >= -730). Source column: current credit amount for the Credit Bureau credit | 31.8 | 337,500 | 797,227 | -0.010 |
| 48 | `NAME_FAMILY_STATUS` | 0.30 | Family status of the client | 0.0 | – | – | – |
| 49 | `BUREAU_ACTIVE_COUNT` | 0.29 | Count of the customer's credit-bureau records (loans at other institutions) with `CREDIT_ACTIVE` = "Active". | 14.3 | 1 | 1.76 | 0.067 |
| 50 | `PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN` | 0.28 | The average of `DAYS_LAST_DUE_1ST_VERSION` across the customer's previous Home Credit applications. Source column: relative to application date of current application when was the first due of the previous application | 6.7 | -515 | -608.41 | 0.041 |
| 51 | `BUREAU_L730D_AMT_CREDIT_SUM_DEBT_MAX` | 0.28 | The maximum of `AMT_CREDIT_SUM_DEBT` over bureau records within the last 730 days (`DAYS_CREDIT` >= -730). Source column: current debt on Credit Bureau credit | 35.5 | 151,088 | 526,421 | 2.70e-04 |
| 52 | `RATIO_CREDIT_TO_INCOME` | 0.27 | Affordability ratio `AMT_CREDIT` / `AMT_INCOME_TOTAL` — leverage: how many years of gross income the loan principal represents. A tree splits on one column at a time and cannot express a quotient, so this is built explicitly on the driver frame. | 0.0 | 3.27 | 3.96 | -0.008 |
| 53 | `DAYS_LAST_PHONE_CHANGE` | 0.27 | How many days before application did client change phone | 0.0 | -757 | -962.86 | 0.055 |
| 54 | `REGION_POPULATION_RELATIVE` | 0.26 | Normalized population of region where client lives (higher number means the client lives in more populated region) | 0.0 | 0.02 | 0.02 | -0.037 |
| 55 | `BUREAU_L730D_AMT_CREDIT_SUM_DEBT_SUM` | 0.26 | The total of `AMT_CREDIT_SUM_DEBT` over bureau records within the last 730 days (`DAYS_CREDIT` >= -730). Source column: current debt on Credit Bureau credit | 31.8 | 8383.50 | 399,054 | 0.011 |
| 56 | `PREV_INST_AMT_PAYMENT_SUM_SUM` | 0.25 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the total of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the total of those per-loan values across all of the customer's previous Home Credit applications. | 5.4 | 279,962 | 627,702 | -0.023 |
| 57 | `BUREAU_L730D_AMT_CREDIT_SUM_SUM` | 0.25 | The total of `AMT_CREDIT_SUM` over bureau records within the last 730 days (`DAYS_CREDIT` >= -730). Source column: current credit amount for the Credit Bureau credit | 31.8 | 205,754 | 780,526 | 0.004 |
| 58 | `BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX` | 0.25 | The maximum of `AMT_CREDIT_MAX_OVERDUE` across the customer's credit-bureau records (loans at other institutions). Source column: maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample) | 40.2 | 0 | 9396.23 | 0.003 |
| 59 | `REGION_RATING_CLIENT_W_CITY` | 0.25 | Our rating of the region where client lives with taking city into account (1,2,3) | 0.0 | 2 | 2.03 | 0.061 |
| 60 | `FLAG_DOCUMENT_3` | 0.24 | Did client provide document 3 | 0.0 | 1 | 0.71 | 0.044 |
| 61 | `PREV_HOUR_APPR_PROCESS_START_MEAN` | 0.24 | The average of `HOUR_APPR_PROCESS_START` across the customer's previous Home Credit applications. Source column: approximately at what day hour did the client apply for the previous application | 5.4 | 12.75 | 12.66 | -0.036 |
| 62 | `BUREAU_L730D_DAYS_CREDIT_MEAN` | 0.24 | The average of `DAYS_CREDIT` over bureau records within the last 730 days (`DAYS_CREDIT` >= -730). Source column: how many days before current application did client apply for Credit Bureau credit | 31.8 | -372.33 | -370.57 | 0.054 |
| 63 | `BUREAU_AMT_CREDIT_SUM_MIN` | 0.23 | The minimum of `AMT_CREDIT_SUM` across the customer's credit-bureau records (loans at other institutions). Source column: current credit amount for the Credit Bureau credit | 14.3 | 39,535 | 119,631 | -0.011 |
| 64 | `PREV_DAYS_LAST_DUE_1ST_VERSION_SUM` | 0.23 | The total of `DAYS_LAST_DUE_1ST_VERSION` across the customer's previous Home Credit applications. Source column: relative to application date of current application when was the first due of the previous application | 5.4 | -983 | -1692.30 | 0.052 |
| 65 | `PREV_DAYS_DECISION_MEAN` | 0.23 | The average of `DAYS_DECISION` across the customer's previous Home Credit applications. Source column: relative to current application when was the decision about previous application made | 5.4 | -789 | -919.57 | 0.047 |
| 66 | `BUREAU_DAYS_CREDIT_ENDDATE_MEAN` | 0.23 | The average of `DAYS_CREDIT_ENDDATE` across the customer's credit-bureau records (loans at other institutions). Source column: remaining duration of CB credit (in days) at the time of application in Home Credit | 15.0 | -135.56 | 651.81 | 0.047 |
| 67 | `BUREAU_AMT_CREDIT_SUM_SUM` | 0.23 | The total of `AMT_CREDIT_SUM` across the customer's credit-bureau records (loans at other institutions). Source column: current credit amount for the Credit Bureau credit | 14.3 | 711,000 | 1.68M | -0.014 |
| 68 | `PREV_INST_DAYS_LATE_MEAN_MIN` | 0.22 | The minimum, across the customer's previous Home Credit applications, of a per-loan derived metric: the average days late per installment (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT; negative = paid early), computed over that loan's installment ledger. | 5.9 | -15.25 | -18.39 | 0.031 |
| 69 | `BUREAU_L730D_COUNT` | 0.22 | Number of bureau records within the last 730 days (`DAYS_CREDIT` >= -730) — a recent-activity volume signal. | 31.8 | 1 | 1.83 | 0.072 |
| 70 | `POS_L12M_CNT_INSTALMENT_FUTURE_MAX` | 0.22 | The maximum of `CNT_INSTALMENT_FUTURE` over POS_CASH_balance records within the last 12 months (`MONTHS_BALANCE` >= -12). Source column: installments left to pay on the previous credit | 32.8 | 12 | 17.17 | 0.023 |
| 71 | `PREV_INST_AMT_PAYMENT_SUM_MAX` | 0.22 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the total of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 191,284 | 396,692 | -0.025 |
| 72 | `BUREAU_DAYS_CREDIT_UPDATE_MEAN` | 0.22 | The average of `DAYS_CREDIT_UPDATE` across the customer's credit-bureau records (loans at other institutions). Source column: how many days before loan application did last information about the Credit Bureau credit come | 14.3 | -481.83 | -546.63 | 0.069 |
| 73 | `PREV_INST_AMT_PAYMENT_SUM_MEAN` | 0.21 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the total of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 133,511 | 214,213 | -0.023 |
| 74 | `PREV_RATE_DOWN_PAYMENT_MAX` | 0.21 | The maximum of `RATE_DOWN_PAYMENT` across the customer's previous Home Credit applications. Source column: down payment rate normalized on previous credit | 11.0 | 0.11 | 0.13 | -0.040 |
| 75 | `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MEAN_SUM` | 0.20 | Two-level roll-up of `CNT_DRAWINGS_ATM_CURRENT` from the credit_card_balance ledger — number of drawings at ATM during this month on the previous credit. Step 1: per prior loan, the average of `CNT_DRAWINGS_ATM_CURRENT` over that loan's monthly credit-card records. Step 2: the total of those per-loan values across all of the customer's previous Home Credit applications. | 5.4 | 0 | 0.11 | 0.065 |
| 76 | `PREV_POS_CNT_INSTALMENT_FUTURE_MEAN_MAX` | 0.20 | Two-level roll-up of `CNT_INSTALMENT_FUTURE` from the POS_CASH_balance ledger — installments left to pay on the previous credit. Step 1: per prior loan, the average of `CNT_INSTALMENT_FUTURE` over that loan's monthly POS/cash records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 6.7 | 9 | 14.11 | 0.018 |
| 77 | `BUREAU_DAYS_CREDIT_UPDATE_MAX` | 0.20 | The maximum of `DAYS_CREDIT_UPDATE` across the customer's credit-bureau records (loans at other institutions). Source column: how many days before loan application did last information about the Credit Bureau credit come | 14.3 | -19 | -140.99 | 0.028 |
| 78 | `DEF_30_CNT_SOCIAL_CIRCLE` | 0.20 | How many observation of client's social surroundings defaulted on 30 DPD (days past due) | 0.3 | 0 | 0.14 | 0.032 |
| 79 | `PREV_POS_CNT_INSTALMENT_FUTURE_MIN_MEAN` | 0.20 | Two-level roll-up of `CNT_INSTALMENT_FUTURE` from the POS_CASH_balance ledger — installments left to pay on the previous credit. Step 1: per prior loan, the minimum of `CNT_INSTALMENT_FUTURE` over that loan's monthly POS/cash records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 6.7 | 0 | 2.29 | 0.019 |
| 80 | `PREV_INST_DAYS_ENTRY_PAYMENT_MAX_MAX` | 0.19 | Two-level roll-up of `DAYS_ENTRY_PAYMENT` from the installments_payments ledger — when was the installments of previous credit paid actually (relative to application date of current loan). Step 1: per prior loan, the maximum of `DAYS_ENTRY_PAYMENT` over that loan's monthly installment records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | -70 | -323.66 | -1.80e-04 |
| 81 | `PREV_INST_AMT_INSTALMENT_MIN_MIN` | 0.19 | Two-level roll-up of `AMT_INSTALMENT` from the installments_payments ledger — what was the prescribed installment amount of previous credit on this installment. Step 1: per prior loan, the minimum of `AMT_INSTALMENT` over that loan's monthly installment records. Step 2: the minimum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 5171.76 | 7417.86 | -0.021 |
| 82 | `PREV_INST_AMT_PAYMENT_MIN_MIN` | 0.19 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the minimum of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the minimum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 2743.34 | 5542.47 | -0.026 |
| 83 | `PREV_DAYS_TERMINATION_MAX` | 0.19 | The maximum of `DAYS_TERMINATION` across the customer's previous Home Credit applications. Source column: relative to application date of current application when was the expected termination of the previous application | 13.2 | -337 | -554.63 | 0.002 |
| 84 | `PREV_INST_DAYS_LATE_MEAN_MAX` | 0.19 | The maximum, across the customer's previous Home Credit applications, of a per-loan derived metric: the average days late per installment (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT; negative = paid early), computed over that loan's installment ledger. | 5.9 | -5.80 | -7.05 | 0.013 |
| 85 | `PREV_CC_UTILIZATION_MEAN_MAX` | 0.19 | The maximum, across the customer's previous Home Credit applications, of a per-loan derived metric: the average credit-card utilisation (AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL, months with a zero/negative limit excluded as null), computed over that loan's credit-card ledger. | 74.9 | 0.24 | 0.33 | 0.144 |
| 86 | `PREV_INST_NUM_INSTALMENT_VERSION_SUM_MEAN` | 0.19 | Two-level roll-up of `NUM_INSTALMENT_VERSION` from the installments_payments ledger — version of installment calendar (0 is for credit card) of previous credit. Change of installment version from month to month signifies that some parameter of payment calendar has changed. Step 1: per prior loan, the total of `NUM_INSTALMENT_VERSION` over that loan's monthly installment records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 10 | 11.82 | -0.013 |
| 87 | `INST_L365D_UNDERPAY_RATE` | 0.19 | Share of installments due in the last 365 days that were paid short of the scheduled amount. | 29.3 | 0 | 0.04 | 0.072 |
| 88 | `PREV_NAME_YIELD_GROUP_low_action_RATE` | 0.19 | Share of the customer's previous Home Credit applications whose `NAME_YIELD_GROUP` equals "low_action" (0 = never, 1 = always). Source column: grouped interest rate into small medium and high of the previous application | 5.4 | 0 | 0.06 | -0.029 |
| 89 | `PREV_INST_AMT_INSTALMENT_MIN_SUM` | 0.18 | Two-level roll-up of `AMT_INSTALMENT` from the installments_payments ledger — what was the prescribed installment amount of previous credit on this installment. Step 1: per prior loan, the minimum of `AMT_INSTALMENT` over that loan's monthly installment records. Step 2: the total of those per-loan values across all of the customer's previous Home Credit applications. | 5.4 | 22,082 | 44,783 | -0.021 |
| 90 | `PREV_AMT_DOWN_PAYMENT_MEAN` | 0.18 | The average of `AMT_DOWN_PAYMENT` across the customer's previous Home Credit applications. Source column: down payment on the previous application | 11.0 | 3375 | 7080.50 | -0.025 |
| 91 | `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MEAN_MAX` | 0.18 | Two-level roll-up of `CNT_DRAWINGS_ATM_CURRENT` from the credit_card_balance ledger — number of drawings at ATM during this month on the previous credit. Step 1: per prior loan, the average of `CNT_DRAWINGS_ATM_CURRENT` over that loan's monthly credit-card records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 82.8 | 0.33 | 0.63 | 0.103 |
| 92 | `BUREAU_DAYS_CREDIT_MIN` | 0.17 | The minimum of `DAYS_CREDIT` across the customer's credit-bureau records (loans at other institutions). Source column: how many days before current application did client apply for Credit Bureau credit | 14.3 | -1827 | -1762.37 | 0.075 |
| 93 | `PREV_INST_SHORTFALL_MEAN_MEAN` | 0.17 | The average, across the customer's previous Home Credit applications, of a per-loan derived metric: the average payment shortfall per installment (AMT_INSTALMENT - AMT_PAYMENT; negative = overpaid), computed over that loan's installment ledger. | 5.9 | 0 | -537.93 | 0.025 |
| 94 | `PREV_INST_DAYS_LATE_MEAN_MEAN` | 0.16 | The average, across the customer's previous Home Credit applications, of a per-loan derived metric: the average days late per installment (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT; negative = paid early), computed over that loan's installment ledger. | 5.9 | -10.73 | -12.29 | 0.028 |
| 95 | `PREV_INST_AMT_INSTALMENT_MIN_MEAN` | 0.16 | Two-level roll-up of `AMT_INSTALMENT` from the installments_payments ledger — what was the prescribed installment amount of previous credit on this installment. Step 1: per prior loan, the minimum of `AMT_INSTALMENT` over that loan's monthly installment records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 10,285 | 15,507 | -0.018 |
| 96 | `BUREAU_L730D_AMT_CREDIT_SUM_MEAN` | 0.16 | The average of `AMT_CREDIT_SUM` over bureau records within the last 730 days (`DAYS_CREDIT` >= -730). Source column: current credit amount for the Credit Bureau credit | 31.8 | 218,095 | 476,459 | -0.022 |
| 97 | `PREV_AMT_ANNUITY_MEAN` | 0.16 | The average of `AMT_ANNUITY` across the customer's previous Home Credit applications. Source column: annuity of previous application | 5.5 | 11,984 | 14,532 | -0.035 |
| 98 | `INST_L365D_AMT_INSTALMENT_MAX` | 0.16 | The maximum of `AMT_INSTALMENT` over installments_payments records within the last 365 days (`DAYS_INSTALMENT` >= -365). Source column: what was the prescribed installment amount of previous credit on this installment | 29.3 | 25,996 | 108,045 | 0.006 |
| 99 | `BUREAU_DAYS_ENDDATE_FACT_MIN` | 0.16 | The minimum of `DAYS_ENDDATE_FACT` across the customer's credit-bureau records (loans at other institutions). Source column: days since CB credit ended at the time of application in Home Credit (only for closed credit) | 25.1 | -1452 | -1467.34 | 0.056 |
| 100 | `INST_L365D_AMT_PAYMENT_MAX` | 0.16 | The maximum of `AMT_PAYMENT` over installments_payments records within the last 365 days (`DAYS_INSTALMENT` >= -365). Source column: what the client actually paid on previous credit on this installment | 29.3 | 26,342 | 109,618 | 0.005 |
| 101 | `PREV_CNT_PAYMENT_MAX` | 0.16 | The maximum of `CNT_PAYMENT` across the customer's previous Home Credit applications. Source column: term of previous credit at application of the previous application | 5.5 | 18 | 24.09 | 0.029 |
| 102 | `PREV_SELLERPLACE_AREA_MEAN` | 0.15 | The average of `SELLERPLACE_AREA` across the customer's previous Home Credit applications. Source column: selling area of seller place of the previous application | 5.4 | 79 | 413.11 | -0.002 |
| 103 | `POS_L12M_SK_DPD_DEF_MEAN` | 0.15 | The average of `SK_DPD_DEF` over POS_CASH_balance records within the last 12 months (`MONTHS_BALANCE` >= -12). Source column: dPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit | 32.7 | 0 | 0.31 | 0.007 |
| 104 | `PREV_INST_AMT_PAYMENT_MEAN_MIN` | 0.15 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the average of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the minimum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 6743.32 | 10,789 | -0.015 |
| 105 | `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MEAN_MIN` | 0.15 | Two-level roll-up of `CNT_DRAWINGS_ATM_CURRENT` from the credit_card_balance ledger — number of drawings at ATM during this month on the previous credit. Step 1: per prior loan, the average of `CNT_DRAWINGS_ATM_CURRENT` over that loan's monthly credit-card records. Step 2: the minimum of those per-loan values across all of the customer's previous Home Credit applications. | 82.8 | 0.33 | 0.63 | 0.103 |
| 106 | `PREV_DAYS_DECISION_MAX` | 0.15 | The maximum of `DAYS_DECISION` across the customer's previous Home Credit applications. Source column: relative to current application when was the decision about previous application made | 5.4 | -299 | -480.82 | 0.016 |
| 107 | `RATIO_INCOME_PER_PERSON` | 0.15 | Affordability ratio `AMT_INCOME_TOTAL` / `CNT_FAM_MEMBERS` — household income per family member. A tree splits on one column at a time and cannot express a quotient, so this is built explicitly on the driver frame. | 0.0 | 75,000 | 93,106 | -0.007 |
| 108 | `TOTALAREA_MODE` | 0.15 | Building profile of the client's home — total area (building mode, normalized 0–1) | 48.3 | 0.07 | 0.10 | -0.033 |
| 109 | `BUREAU_AMT_CREDIT_MAX_OVERDUE_SUM` | 0.15 | The total of `AMT_CREDIT_MAX_OVERDUE` across the customer's credit-bureau records (loans at other institutions). Source column: maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample) | 14.3 | 0 | 6626.06 | 0.002 |
| 110 | `BUREAU_AMT_CREDIT_SUM_DEBT_MAX` | 0.15 | The maximum of `AMT_CREDIT_SUM_DEBT` across the customer's credit-bureau records (loans at other institutions). Source column: current debt on Credit Bureau credit | 16.7 | 149,458 | 521,627 | -0.002 |
| 111 | `PREV_AMT_ANNUITY_MIN` | 0.14 | The minimum of `AMT_ANNUITY` across the customer's previous Home Credit applications. Source column: annuity of previous application | 5.5 | 5721.98 | 7864.30 | -0.032 |
| 112 | `PREV_RATE_DOWN_PAYMENT_MEAN` | 0.14 | The average of `RATE_DOWN_PAYMENT` across the customer's previous Home Credit applications. Source column: down payment rate normalized on previous credit | 11.0 | 0.07 | 0.08 | -0.034 |
| 113 | `BUREAU_DAYS_CREDIT_ENDDATE_SUM` | 0.14 | The total of `DAYS_CREDIT_ENDDATE` across the customer's credit-bureau records (loans at other institutions). Source column: remaining duration of CB credit (in days) at the time of application in Home Credit | 14.3 | 0 | 2206.95 | 0.054 |
| 114 | `PREV_CC_UTILIZATION_MEAN_SUM` | 0.14 | The total, across the customer's previous Home Credit applications, of a per-loan derived metric: the average credit-card utilisation (AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL, months with a zero/negative limit excluded as null), computed over that loan's credit-card ledger. | 5.4 | 0 | 0.08 | 0.071 |
| 115 | `PREV_CC_UTILIZATION_MEAN_MEAN` | 0.14 | The average, across the customer's previous Home Credit applications, of a per-loan derived metric: the average credit-card utilisation (AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL, months with a zero/negative limit excluded as null), computed over that loan's credit-card ledger. | 74.9 | 0.24 | 0.33 | 0.144 |
| 116 | `PREV_POS_CNT_INSTALMENT_FUTURE_MIN_SUM` | 0.14 | Two-level roll-up of `CNT_INSTALMENT_FUTURE` from the POS_CASH_balance ledger — installments left to pay on the previous credit. Step 1: per prior loan, the minimum of `CNT_INSTALMENT_FUTURE` over that loan's monthly POS/cash records. Step 2: the total of those per-loan values across all of the customer's previous Home Credit applications. | 5.4 | 0 | 5.84 | 0.009 |
| 117 | `BUREAU_DAYS_ENDDATE_FACT_MEAN` | 0.14 | The average of `DAYS_ENDDATE_FACT` across the customer's credit-bureau records (loans at other institutions). Source column: days since CB credit ended at the time of application in Home Credit (only for closed credit) | 25.1 | -928.40 | -970.30 | 0.053 |
| 118 | `PREV_CODE_REJECT_REASON_HC_RATE` | 0.14 | Share of the customer's previous Home Credit applications whose `CODE_REJECT_REASON` equals "HC" (0 = never, 1 = always). Source column: why was the previous application rejected | 5.4 | 0 | 0.06 | 0.055 |
| 119 | `POS_L12M_COUNT` | 0.14 | Number of POS_CASH_balance records within the last 12 months (`MONTHS_BALANCE` >= -12) — a recent-activity volume signal. | 32.7 | 6 | 6.56 | -0.019 |
| 120 | `INST_L365D_AMT_PAYMENT_SUM` | 0.14 | The total of `AMT_PAYMENT` over installments_payments records within the last 365 days (`DAYS_INSTALMENT` >= -365). Source column: what the client actually paid on previous credit on this installment | 29.3 | 83,957 | 231,074 | -0.015 |
| 121 | `PREV_INST_NUM_INSTALMENT_VERSION_MEAN_MAX` | 0.13 | Two-level roll-up of `NUM_INSTALMENT_VERSION` from the installments_payments ledger — version of installment calendar (0 is for credit card) of previous credit. Change of installment version from month to month signifies that some parameter of payment calendar has changed. Step 1: per prior loan, the average of `NUM_INSTALMENT_VERSION` over that loan's monthly installment records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 1.12 | 1.40 | -0.025 |
| 122 | `PREV_SELLERPLACE_AREA_MAX` | 0.13 | The maximum of `SELLERPLACE_AREA` across the customer's previous Home Credit applications. Source column: selling area of seller place of the previous application | 5.4 | 158 | 967.98 | -0.003 |
| 123 | `FLAG_WORK_PHONE` | 0.13 | Did client provide home phone (1=YES, 0=NO) | 0.0 | 0 | 0.20 | 0.029 |
| 124 | `PREV_NAME_PRODUCT_TYPE_walk_in_RATE` | 0.13 | Share of the customer's previous Home Credit applications whose `NAME_PRODUCT_TYPE` equals "walk-in" (0 = never, 1 = always). Source column: was the previous application x-sell o walk-in | 5.4 | 0 | 0.08 | 0.057 |
| 125 | `BUREAU_DAYS_CREDIT_UPDATE_MIN` | 0.13 | The minimum of `DAYS_CREDIT_UPDATE` across the customer's credit-bureau records (loans at other institutions). Source column: how many days before loan application did last information about the Credit Bureau credit come | 14.3 | -1055 | -1181.36 | 0.043 |
| 126 | `INST_L365D_AMT_PAYMENT_MEAN` | 0.13 | The average of `AMT_PAYMENT` over installments_payments records within the last 365 days (`DAYS_INSTALMENT` >= -365). Source column: what the client actually paid on previous credit on this installment | 29.3 | 14,825 | 31,340 | -0.013 |
| 127 | `CC_L12M_AMT_DRAWINGS_CURRENT_MEAN` | 0.13 | The average of `AMT_DRAWINGS_CURRENT` over credit_card_balance records within the last 12 months (`MONTHS_BALANCE` >= -12). Source column: amount drawing during the month of the previous credit | 71.7 | 0 | 11,895 | 0.054 |
| 128 | `BUREAU_DAYS_CREDIT_ENDDATE_MIN` | 0.13 | The minimum of `DAYS_CREDIT_ENDDATE` across the customer's credit-bureau records (loans at other institutions). Source column: remaining duration of CB credit (in days) at the time of application in Home Credit | 15.0 | -1268 | -1029.58 | 0.034 |
| 129 | `PREV_INST_AMT_INSTALMENT_MEAN_MIN` | 0.13 | Two-level roll-up of `AMT_INSTALMENT` from the installments_payments ledger — what was the prescribed installment amount of previous credit on this installment. Step 1: per prior loan, the average of `AMT_INSTALMENT` over that loan's monthly installment records. Step 2: the minimum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 7015.02 | 10,903 | -0.012 |
| 130 | `PREV_AMT_GOODS_PRICE_MIN` | 0.13 | The minimum of `AMT_GOODS_PRICE` across the customer's previous Home Credit applications. Source column: goods price of good that client asked for (if applicable) on the previous application | 5.7 | 45,630 | 79,415 | -0.021 |
| 131 | `PREV_POS_COUNT_SUM` | 0.13 | The total, across the customer's previous Home Credit applications, of the number of monthly POS/cash ledger records a prior loan has. A null means the customer has no prior loan with any POS/cash history. | 5.4 | 20 | 26.83 | -0.036 |
| 132 | `PREV_RATE_DOWN_PAYMENT_SUM` | 0.13 | The total of `RATE_DOWN_PAYMENT` across the customer's previous Home Credit applications. Source column: down payment rate normalized on previous credit | 5.4 | 0.10 | 0.17 | -0.042 |
| 133 | `PREV_INST_DAYS_LATE_MEAN_SUM` | 0.13 | The total, across the customer's previous Home Credit applications, of a per-loan derived metric: the average days late per installment (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT; negative = paid early), computed over that loan's installment ledger. | 5.4 | -23.24 | -31.73 | 0.040 |
| 134 | `CC_L12M_AMT_BALANCE_MEAN` | 0.12 | The average of `AMT_BALANCE` over credit_card_balance records within the last 12 months (`MONTHS_BALANCE` >= -12). Source column: balance during the month of previous credit | 71.7 | 0 | 66,949 | 0.085 |
| 135 | `PREV_INST_LATE_RATE_MAX` | 0.12 | The maximum, across the customer's previous Home Credit applications, of a per-loan derived metric: the share of installments paid after their due date (DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT), computed over that loan's installment ledger. | 5.9 | 0.04 | 0.13 | 0.051 |
| 136 | `DEF_60_CNT_SOCIAL_CIRCLE` | 0.12 | How many observation of client's social surroundings defaulted on 60 (days past due) DPD | 0.3 | 0 | 0.10 | 0.031 |
| 137 | `PREV_NAME_YIELD_GROUP_low_normal_RATE` | 0.12 | Share of the customer's previous Home Credit applications whose `NAME_YIELD_GROUP` equals "low_normal" (0 = never, 1 = always). Source column: grouped interest rate into small medium and high of the previous application | 5.4 | 0.07 | 0.20 | -0.041 |
| 138 | `POS_L12M_SK_DPD_MEAN` | 0.12 | The average of `SK_DPD` over POS_CASH_balance records within the last 12 months (`MONTHS_BALANCE` >= -12). Source column: dPD (days past due) during the month of previous credit | 32.7 | 0 | 6.41 | 0.007 |
| 139 | `PREV_INST_AMT_INSTALMENT_SUM_MEAN` | 0.12 | Two-level roll-up of `AMT_INSTALMENT` from the installments_payments ledger — what was the prescribed installment amount of previous credit on this installment. Step 1: per prior loan, the total of `AMT_INSTALMENT` over that loan's monthly installment records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 137,117 | 212,416 | -0.016 |
| 140 | `PREV_CC_OVERLIMIT_MONTHS_MEAN` | 0.12 | The average, across the customer's previous Home Credit applications, of a per-loan derived metric: the number of months the card balance exceeded the credit limit, computed over that loan's credit-card ledger. | 74.7 | 0 | 3.06 | 0.045 |
| 141 | `PREV_INST_AMT_INSTALMENT_SUM_MIN` | 0.12 | Two-level roll-up of `AMT_INSTALMENT` from the installments_payments ledger — what was the prescribed installment amount of previous credit on this installment. Step 1: per prior loan, the total of `AMT_INSTALMENT` over that loan's monthly installment records. Step 2: the minimum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 58,570 | 101,878 | -0.006 |
| 142 | `PREV_INST_NUM_INSTALMENT_VERSION_SUM_SUM` | 0.12 | Two-level roll-up of `NUM_INSTALMENT_VERSION` from the installments_payments ledger — version of installment calendar (0 is for credit card) of previous credit. Change of installment version from month to month signifies that some parameter of payment calendar has changed. Step 1: per prior loan, the total of `NUM_INSTALMENT_VERSION` over that loan's monthly installment records. Step 2: the total of those per-loan values across all of the customer's previous Home Credit applications. | 5.4 | 21 | 31.61 | -0.029 |
| 143 | `PREV_APPROVED_RATE` | 0.12 | Share of the customer's previous Home Credit applications with `NAME_CONTRACT_STATUS` = "Approved". | 5.4 | 0.75 | 0.71 | -0.064 |
| 144 | `PREV_CNT_PAYMENT_SUM` | 0.12 | The total of `CNT_PAYMENT` across the customer's previous Home Credit applications. Source column: term of previous credit at application of the previous application | 5.4 | 34 | 57.76 | 0.028 |
| 145 | `PREV_INST_AMT_INSTALMENT_MAX_MIN` | 0.12 | Two-level roll-up of `AMT_INSTALMENT` from the installments_payments ledger — what was the prescribed installment amount of previous credit on this installment. Step 1: per prior loan, the maximum of `AMT_INSTALMENT` over that loan's monthly installment records. Step 2: the minimum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 8978.62 | 21,738 | 0.002 |
| 146 | `AMT_INCOME_TOTAL` | 0.12 | Income of the client | 0.0 | 147,150 | 168,798 | -0.004 |
| 147 | `PREV_L365D_CNT_PAYMENT_MEAN` | 0.12 | The average of `CNT_PAYMENT` over previous_application records within the last 365 days (`DAYS_DECISION` >= -365). Source column: term of previous credit at application of the previous application | 54.1 | 12 | 18.70 | 0.022 |
| 148 | `BUREAU_L730D_DAYS_CREDIT_SUM` | 0.12 | The total of `DAYS_CREDIT` over bureau records within the last 730 days (`DAYS_CREDIT` >= -730). Source column: how many days before current application did client apply for Credit Bureau credit | 31.8 | -447 | -682.61 | -0.036 |
| 149 | `CC_L12M_AMT_BALANCE_MAX` | 0.11 | The maximum of `AMT_BALANCE` over credit_card_balance records within the last 12 months (`MONTHS_BALANCE` >= -12). Source column: balance during the month of previous credit | 71.7 | 0 | 101,402 | 0.084 |
| 150 | `BUREAU_DAYS_CREDIT_SUM` | 0.11 | The total of `DAYS_CREDIT` across the customer's credit-bureau records (loans at other institutions). Source column: how many days before current application did client apply for Credit Bureau credit | 14.3 | -3541 | -5449.39 | 0.042 |
| 151 | `INST_L365D_AMT_INSTALMENT_MEAN` | 0.11 | The average of `AMT_INSTALMENT` over installments_payments records within the last 365 days (`DAYS_INSTALMENT` >= -365). Source column: what was the prescribed installment amount of previous credit on this installment | 29.3 | 15,063 | 30,165 | -0.010 |
| 152 | `BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN` | 0.11 | The average of `AMT_CREDIT_SUM_LIMIT` across the customer's credit-bureau records (loans at other institutions). Source column: current credit limit of credit card reported in Credit Bureau | 21.2 | 0 | 5901.48 | -0.011 |
| 153 | `PREV_POS_SK_DPD_DEF_MAX_MEAN` | 0.11 | Two-level roll-up of `SK_DPD_DEF` from the POS_CASH_balance ledger — dPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit. Step 1: per prior loan, the maximum of `SK_DPD_DEF` over that loan's monthly POS/cash records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 6.7 | 0 | 0.44 | 0.049 |
| 154 | `PREV_INST_AMT_PAYMENT_MAX_MIN` | 0.11 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the maximum of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the minimum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 8996.08 | 22,002 | 0.001 |
| 155 | `PREV_POS_COUNT_MEAN` | 0.11 | The average, across the customer's previous Home Credit applications, of the number of monthly POS/cash ledger records a prior loan has. A null means the customer has no prior loan with any POS/cash history. | 6.7 | 10 | 10.77 | -0.005 |
| 156 | `PREV_POS_MONTHS_BALANCE_SUM_MIN` | 0.11 | Two-level roll-up of `MONTHS_BALANCE` from the POS_CASH_balance ledger — month of balance relative to application date (-1 means the information to the freshest monthly snapshot, 0 means the information at application - often it will be the same as -1 as many banks are not updating the information to Credit Bureau regularly ). Step 1: per prior loan, the total of `MONTHS_BALANCE` over that loan's monthly POS/cash records. Step 2: the minimum of those per-loan values across all of the customer's previous Home Credit applications. | 6.7 | -440 | -518.70 | 0.037 |
| 157 | `POS_L12M_SK_DPD_DEF_MAX` | 0.11 | The maximum of `SK_DPD_DEF` over POS_CASH_balance records within the last 12 months (`MONTHS_BALANCE` >= -12). Source column: dPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit | 32.7 | 0 | 0.62 | 0.009 |
| 158 | `PREV_INST_AMT_PAYMENT_SUM_MIN` | 0.11 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the total of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the minimum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 56,842 | 100,287 | -0.010 |
| 159 | `PREV_INST_AMT_INSTALMENT_MIN_MAX` | 0.11 | Two-level roll-up of `AMT_INSTALMENT` from the installments_payments ledger — what was the prescribed installment amount of previous credit on this installment. Step 1: per prior loan, the minimum of `AMT_INSTALMENT` over that loan's monthly installment records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 14,584 | 29,380 | -0.011 |
| 160 | `PREV_DAYS_FIRST_DRAWING_MEAN` | 0.11 | The average of `DAYS_FIRST_DRAWING` across the customer's previous Home Credit applications. Source column: relative to application date of current application when was the first disbursement of the previous application | 82.7 | -632 | -1052.85 | 0.096 |
| 161 | `PREV_INST_SHORTFALL_MEAN_SUM` | 0.11 | The total, across the customer's previous Home Credit applications, of a per-loan derived metric: the average payment shortfall per installment (AMT_INSTALMENT - AMT_PAYMENT; negative = overpaid), computed over that loan's installment ledger. | 5.4 | 0 | -1731.22 | 0.021 |
| 162 | `POS_L12M_CNT_INSTALMENT_FUTURE_SUM` | 0.11 | The total of `CNT_INSTALMENT_FUTURE` over POS_CASH_balance records within the last 12 months (`MONTHS_BALANCE` >= -12). Source column: installments left to pay on the previous credit | 32.7 | 27 | 83.64 | 0.014 |
| 163 | `PREV_HOUR_APPR_PROCESS_START_MAX` | 0.11 | The maximum of `HOUR_APPR_PROCESS_START` across the customer's previous Home Credit applications. Source column: approximately at what day hour did the client apply for the previous application | 5.4 | 15 | 14.77 | -0.031 |
| 164 | `PREV_POS_MONTHS_BALANCE_MAX_MAX` | 0.11 | Two-level roll-up of `MONTHS_BALANCE` from the POS_CASH_balance ledger — month of balance relative to application date (-1 means the information to the freshest monthly snapshot, 0 means the information at application - often it will be the same as -1 as many banks are not updating the information to Credit Bureau regularly ). Step 1: per prior loan, the maximum of `MONTHS_BALANCE` over that loan's monthly POS/cash records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 6.7 | -4 | -11.99 | -0.004 |
| 165 | `PREV_HAS_POS_RATE` | 0.11 | Share of the customer's previous Home Credit applications that have any POS/cash ledger history attached (a mix of product type and data coverage). | 5.4 | 0.67 | 0.63 | -0.063 |
| 166 | `PREV_POS_CNT_INSTALMENT_FUTURE_MIN_MAX` | 0.11 | Two-level roll-up of `CNT_INSTALMENT_FUTURE` from the POS_CASH_balance ledger — installments left to pay on the previous credit. Step 1: per prior loan, the minimum of `CNT_INSTALMENT_FUTURE` over that loan's monthly POS/cash records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 6.7 | 0 | 5.78 | 0.009 |
| 167 | `PREV_SELLERPLACE_AREA_SUM` | 0.10 | The total of `SELLERPLACE_AREA` across the customer's previous Home Credit applications. Source column: selling area of seller place of the previous application | 5.4 | 227 | 1448.07 | -0.004 |
| 168 | `PREV_POS_SK_DPD_DEF_MEAN_MEAN` | 0.10 | Two-level roll-up of `SK_DPD_DEF` from the POS_CASH_balance ledger — dPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit. Step 1: per prior loan, the average of `SK_DPD_DEF` over that loan's monthly POS/cash records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 6.7 | 0 | 0.06 | 0.039 |
| 169 | `PREV_INST_DAYS_INSTALMENT_MAX_MAX` | 0.10 | Two-level roll-up of `DAYS_INSTALMENT` from the installments_payments ledger — when the installment of previous credit was supposed to be paid (relative to application date of current loan). Step 1: per prior loan, the maximum of `DAYS_INSTALMENT` over that loan's monthly installment records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | -54 | -315.12 | -0.001 |
| 170 | `PREV_INST_AMT_INSTALMENT_SUM_MAX` | 0.10 | Two-level roll-up of `AMT_INSTALMENT` from the installments_payments ledger — what was the prescribed installment amount of previous credit on this installment. Step 1: per prior loan, the total of `AMT_INSTALMENT` over that loan's monthly installment records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 197,528 | 385,039 | -0.018 |
| 171 | `PREV_INST_NUM_INSTALMENT_NUMBER_MAX_MAX` | 0.10 | Two-level roll-up of `NUM_INSTALMENT_NUMBER` from the installments_payments ledger — on which installment we observe payment. Step 1: per prior loan, the maximum of `NUM_INSTALMENT_NUMBER` over that loan's monthly installment records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 12 | 21.69 | 0.019 |
| 172 | `PREV_INST_AMT_PAYMENT_MEAN_MAX` | 0.10 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the average of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the maximum of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 18,644 | 47,629 | -0.010 |
| 173 | `BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN` | 0.10 | The average of `AMT_CREDIT_SUM_OVERDUE` across the customer's credit-bureau records (loans at other institutions). Source column: current amount overdue on Credit Bureau credit | 14.3 | 0 | 49.55 | 0.007 |
| 174 | `BUREAU_AMT_CREDIT_SUM_OVERDUE_MAX` | 0.10 | The maximum of `AMT_CREDIT_SUM_OVERDUE` across the customer's credit-bureau records (loans at other institutions). Source column: current amount overdue on Credit Bureau credit | 14.3 | 0 | 184.97 | 0.011 |
| 175 | `PREV_INST_AMT_INSTALMENT_MAX_MEAN` | 0.10 | Two-level roll-up of `AMT_INSTALMENT` from the installments_payments ledger — what was the prescribed installment amount of previous credit on this installment. Step 1: per prior loan, the maximum of `AMT_INSTALMENT` over that loan's monthly installment records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 23,334 | 62,583 | 0.004 |
| 176 | `PREV_POS_CNT_INSTALMENT_FUTURE_MEAN_SUM` | 0.10 | Two-level roll-up of `CNT_INSTALMENT_FUTURE` from the POS_CASH_balance ledger — installments left to pay on the previous credit. Step 1: per prior loan, the average of `CNT_INSTALMENT_FUTURE` over that loan's monthly POS/cash records. Step 2: the total of those per-loan values across all of the customer's previous Home Credit applications. | 5.4 | 14.79 | 23.61 | 0.001 |
| 177 | `BUREAU_AMT_CREDIT_SUM_DEBT_MIN` | 0.10 | The minimum of `AMT_CREDIT_SUM_DEBT` across the customer's credit-bureau records (loans at other institutions). Source column: current debt on Credit Bureau credit | 16.7 | 0 | 39,806 | 2.40e-04 |
| 178 | `WEEKDAY_APPR_PROCESS_START` | 0.10 | On which day of the week did the client apply for the loan | 0.0 | – | – | – |
| 179 | `CC_L12M_AMT_DRAWINGS_CURRENT_MAX` | 0.10 | The maximum of `AMT_DRAWINGS_CURRENT` over credit_card_balance records within the last 12 months (`MONTHS_BALANCE` >= -12). Source column: amount drawing during the month of the previous credit | 71.7 | 0 | 49,521 | 0.069 |
| 180 | `PREV_INST_AMT_PAYMENT_MEAN_MEAN` | 0.10 | Two-level roll-up of `AMT_PAYMENT` from the installments_payments ledger — what the client actually paid on previous credit on this installment. Step 1: per prior loan, the average of `AMT_PAYMENT` over that loan's monthly installment records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 13,430 | 23,922 | -0.015 |
| 181 | `PREV_NAME_CLIENT_TYPE_New_RATE` | 0.10 | Share of the customer's previous Home Credit applications whose `NAME_CLIENT_TYPE` equals "New" (0 = never, 1 = always). Source column: was the client old or new client when applying for the previous application | 5.4 | 0.20 | 0.33 | 0.021 |
| 182 | `PREV_INST_UNDERPAY_RATE_MEAN` | 0.10 | The average, across the customer's previous Home Credit applications, of a per-loan derived metric: the share of installments paid short (AMT_PAYMENT < AMT_INSTALMENT), computed over that loan's installment ledger. | 5.9 | 0 | 0.07 | 0.068 |
| 183 | `REG_CITY_NOT_LIVE_CITY` | 0.10 | Flag if client's permanent address does not match contact address (1=different, 0=same, at city level) | 0.0 | 0 | 0.08 | 0.044 |
| 184 | `BUREAU_AMT_CREDIT_SUM_DEBT_SUM` | 0.10 | The total of `AMT_CREDIT_SUM_DEBT` across the customer's credit-bureau records (loans at other institutions). Source column: current debt on Credit Bureau credit | 14.3 | 87,584 | 548,942 | 0.007 |
| 185 | `BUREAU_AMT_CREDIT_SUM_LIMIT_MAX` | 0.10 | The maximum of `AMT_CREDIT_SUM_LIMIT` across the customer's credit-bureau records (loans at other institutions). Source column: current credit limit of credit card reported in Credit Bureau | 21.2 | 0 | 22,281 | -0.011 |
| 186 | `PREV_CODE_REJECT_REASON_XAP_RATE` | 0.09 | Share of the customer's previous Home Credit applications whose `CODE_REJECT_REASON` equals "XAP" (0 = never, 1 = always). Source column: why was the previous application rejected | 5.4 | 1 | 0.83 | -0.074 |
| 187 | `PREV_INST_NUM_INSTALMENT_VERSION_MEAN_MEAN` | 0.09 | Two-level roll-up of `NUM_INSTALMENT_VERSION` from the installments_payments ledger — version of installment calendar (0 is for credit card) of previous credit. Change of installment version from month to month signifies that some parameter of payment calendar has changed. Step 1: per prior loan, the average of `NUM_INSTALMENT_VERSION` over that loan's monthly installment records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 1.04 | 1.13 | -0.032 |
| 188 | `INST_L365D_COUNT` | 0.09 | Number of installments_payments records within the last 365 days (`DAYS_INSTALMENT` >= -365) — a recent-activity volume signal. | 29.3 | 7 | 9.60 | 0.032 |
| 189 | `PREV_POS_MONTHS_BALANCE_SUM_MEAN` | 0.09 | Two-level roll-up of `MONTHS_BALANCE` from the POS_CASH_balance ledger — month of balance relative to application date (-1 means the information to the freshest monthly snapshot, 0 means the information at application - often it will be the same as -1 as many banks are not updating the information to Credit Bureau regularly ). Step 1: per prior loan, the total of `MONTHS_BALANCE` over that loan's monthly POS/cash records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 6.7 | -281 | -332.27 | 0.025 |
| 190 | `PREV_INST_LATE_RATE_SUM` | 0.09 | The total, across the customer's previous Home Credit applications, of a per-loan derived metric: the share of installments paid after their due date (DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT), computed over that loan's installment ledger. | 5.4 | 0 | 0.18 | 0.040 |
| 191 | `PREV_POS_SK_DPD_DEF_SUM_MEAN` | 0.09 | Two-level roll-up of `SK_DPD_DEF` from the POS_CASH_balance ledger — dPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit. Step 1: per prior loan, the total of `SK_DPD_DEF` over that loan's monthly POS/cash records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 6.7 | 0 | 0.80 | 0.010 |
| 192 | `LIVINGAREA_AVG` | 0.09 | Building profile of the client's home — living area (building average, normalized 0–1) | 50.2 | 0.07 | 0.11 | -0.033 |
| 193 | `PREV_L365D_AMT_ANNUITY_MEAN` | 0.09 | The average of `AMT_ANNUITY` over previous_application records within the last 365 days (`DAYS_DECISION` >= -365). Source column: annuity of previous application | 54.1 | 13,500 | 17,579 | -0.041 |
| 194 | `PREV_INST_AMT_INSTALMENT_SUM_SUM` | 0.09 | Two-level roll-up of `AMT_INSTALMENT` from the installments_payments ledger — what was the prescribed installment amount of previous credit on this installment. Step 1: per prior loan, the total of `AMT_INSTALMENT` over that loan's monthly installment records. Step 2: the total of those per-loan values across all of the customer's previous Home Credit applications. | 5.4 | 288,986 | 621,375 | -0.019 |
| 195 | `NAME_INCOME_TYPE` | 0.09 | Clients income type (businessman, working, maternity leave, ) | 0.0 | – | – | – |
| 196 | `PREV_AMT_ANNUITY_MAX` | 0.09 | The maximum of `AMT_ANNUITY` across the customer's previous Home Credit applications. Source column: annuity of previous application | 5.5 | 17,908 | 23,723 | -0.029 |
| 197 | `PREV_CC_UTILIZATION_MEAN_MIN` | 0.09 | The minimum, across the customer's previous Home Credit applications, of a per-loan derived metric: the average credit-card utilisation (AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL, months with a zero/negative limit excluded as null), computed over that loan's credit-card ledger. | 74.9 | 0.24 | 0.33 | 0.143 |
| 198 | `PREV_INST_NUM_INSTALMENT_NUMBER_MAX_MEAN` | 0.09 | Two-level roll-up of `NUM_INSTALMENT_NUMBER` from the installments_payments ledger — on which installment we observe payment. Step 1: per prior loan, the maximum of `NUM_INSTALMENT_NUMBER` over that loan's monthly installment records. Step 2: the average of those per-loan values across all of the customer's previous Home Credit applications. | 5.9 | 10 | 13.07 | 0.034 |
| 199 | `PREV_AMT_CREDIT_MEAN` | 0.09 | The average of `AMT_CREDIT` across the customer's previous Home Credit applications. Source column: final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client initially applied for, but during our approval process he could have received different amount - AMT_CREDIT | 5.4 | 117,450 | 170,070 | -0.016 |
| 200 | `PREV_L365D_AMT_ANNUITY_MAX` | 0.09 | The maximum of `AMT_ANNUITY` over previous_application records within the last 365 days (`DAYS_DECISION` >= -365). Source column: annuity of previous application | 54.1 | 15,885 | 20,845 | -0.034 |

---

## 7. Complete catalog by block

Every candidate feature, selected or not. Within a block the definition follows the block's template, so each table lists the source column and its official meaning once in a glossary, then every feature derived from it. The fully composed English sentence for each individual feature is in [`feature_dictionary.csv`](feature_dictionary.csv).

### 7.1 Application driver columns (`application_train`)

*120 features · 29 selected · 46.5% of model gain*

The 120 usable columns of `application_train` itself (all 122 minus `TARGET` and `SK_ID_CURR`). No aggregation — one row per customer already. These are **46.5% of total model gain**; the three `EXT_SOURCE_*` external credit scores alone are 27.1%.

| Feature | Sel | Rank | Definition | Type | Null % | Median | Mean | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|
| `EXT_SOURCE_3` | **✔** | 1 | Normalized score from external data source | numeric | 19.8 | 0.54 | 0.51 | -0.179 |
| `EXT_SOURCE_2` | **✔** | 2 | Normalized score from external data source | numeric | 0.2 | 0.57 | 0.51 | -0.160 |
| `ORGANIZATION_TYPE` | **✔** | 3 | Type of organization where client works | categorical | 0.0 | – | – | – |
| `EXT_SOURCE_1` | **✔** | 4 | Normalized score from external data source | numeric | 56.4 | 0.51 | 0.50 | -0.155 |
| `DAYS_EMPLOYED` | **✔** | 7 | How many days before the application the person started current employment | numeric | 18.0 | -1648 | -2384.17 | 0.075 |
| `OCCUPATION_TYPE` | **✔** | 8 | What kind of occupation does the client have | categorical | 31.3 | – | – | – |
| `DAYS_BIRTH` | **✔** | 9 | Client's age in days at the time of application | numeric | 0.0 | -15,750 | -16,037 | 0.078 |
| `AMT_ANNUITY` | **✔** | 11 | Loan annuity | numeric | 0.0 | 24,903 | 27,109 | -0.013 |
| `CODE_GENDER` | **✔** | 14 | Gender of the client | categorical | 0.0 | – | – | – |
| `NAME_EDUCATION_TYPE` | **✔** | 15 | Level of highest education the client achieved | categorical | 0.0 | – | – | – |
| `OWN_CAR_AGE` | **✔** | 16 | Age of client's car | numeric | 66.0 | 9 | 12.06 | 0.038 |
| `AMT_GOODS_PRICE` | **✔** | 23 | For consumer loans it is the price of the goods for which the loan is given | numeric | 0.1 | 450,000 | 538,396 | -0.040 |
| `DAYS_ID_PUBLISH` | **✔** | 25 | How many days before the application did client change the identity document with which he applied for the loan | numeric | 0.0 | -3254 | -2994.20 | 0.051 |
| `AMT_CREDIT` | **✔** | 30 | Credit amount of the loan | numeric | 0.0 | 513,531 | 599,026 | -0.030 |
| `DAYS_REGISTRATION` | **✔** | 46 | How many days before the application did client change his registration | numeric | 0.0 | -4504 | -4986.12 | 0.042 |
| `NAME_FAMILY_STATUS` | **✔** | 48 | Family status of the client | categorical | 0.0 | – | – | – |
| `DAYS_LAST_PHONE_CHANGE` | **✔** | 53 | How many days before application did client change phone | numeric | 0.0 | -757 | -962.86 | 0.055 |
| `REGION_POPULATION_RELATIVE` | **✔** | 54 | Normalized population of region where client lives (higher number means the client lives in more populated region) | numeric | 0.0 | 0.02 | 0.02 | -0.037 |
| `REGION_RATING_CLIENT_W_CITY` | **✔** | 59 | Our rating of the region where client lives with taking city into account (1,2,3) | numeric | 0.0 | 2 | 2.03 | 0.061 |
| `FLAG_DOCUMENT_3` | **✔** | 60 | Did client provide document 3 | numeric | 0.0 | 1 | 0.71 | 0.044 |
| `DEF_30_CNT_SOCIAL_CIRCLE` | **✔** | 78 | How many observation of client's social surroundings defaulted on 30 DPD (days past due) | numeric | 0.3 | 0 | 0.14 | 0.032 |
| `TOTALAREA_MODE` | **✔** | 108 | Building profile of the client's home — total area (building mode, normalized 0–1) | numeric | 48.3 | 0.07 | 0.10 | -0.033 |
| `FLAG_WORK_PHONE` | **✔** | 123 | Did client provide home phone (1=YES, 0=NO) | numeric | 0.0 | 0 | 0.20 | 0.029 |
| `DEF_60_CNT_SOCIAL_CIRCLE` | **✔** | 136 | How many observation of client's social surroundings defaulted on 60 (days past due) DPD | numeric | 0.3 | 0 | 0.10 | 0.031 |
| `AMT_INCOME_TOTAL` | **✔** | 146 | Income of the client | numeric | 0.0 | 147,150 | 168,798 | -0.004 |
| `WEEKDAY_APPR_PROCESS_START` | **✔** | 178 | On which day of the week did the client apply for the loan | categorical | 0.0 | – | – | – |
| `REG_CITY_NOT_LIVE_CITY` | **✔** | 183 | Flag if client's permanent address does not match contact address (1=different, 0=same, at city level) | numeric | 0.0 | 0 | 0.08 | 0.044 |
| `LIVINGAREA_AVG` | **✔** | 192 | Building profile of the client's home — living area (building average, normalized 0–1) | numeric | 50.2 | 0.07 | 0.11 | -0.033 |
| `NAME_INCOME_TYPE` | **✔** | 195 | Clients income type (businessman, working, maternity leave, ) | categorical | 0.0 | – | – | – |
| `NAME_CONTRACT_TYPE` | · | 205 | Identification if loan is cash or revolving | categorical | 0.0 | – | – | – |
| `YEARS_BEGINEXPLUATATION_MEDI` | · | 206 | Building profile of the client's home — years since the building entered service (building median, normalized 0–1) | numeric | 48.8 | 0.98 | 0.98 | -0.010 |
| `YEARS_BEGINEXPLUATATION_MODE` | · | 209 | Building profile of the client's home — years since the building entered service (building mode, normalized 0–1) | numeric | 48.8 | 0.98 | 0.98 | -0.009 |
| `LIVINGAREA_MEDI` | · | 214 | Building profile of the client's home — living area (building median, normalized 0–1) | numeric | 50.2 | 0.07 | 0.11 | -0.033 |
| `YEARS_BEGINEXPLUATATION_AVG` | · | 233 | Building profile of the client's home — years since the building entered service (building average, normalized 0–1) | numeric | 48.8 | 0.98 | 0.98 | -0.010 |
| `LANDAREA_AVG` | · | 241 | Building profile of the client's home — land area (building average, normalized 0–1) | numeric | 59.4 | 0.05 | 0.07 | -0.011 |
| `APARTMENTS_MODE` | · | 243 | Building profile of the client's home — apartment size (building mode, normalized 0–1) | numeric | 50.7 | 0.08 | 0.11 | -0.027 |
| `LIVINGAREA_MODE` | · | 245 | Building profile of the client's home — living area (building mode, normalized 0–1) | numeric | 50.2 | 0.07 | 0.11 | -0.031 |
| `HOUR_APPR_PROCESS_START` | · | 249 | Approximately at what hour did the client apply for the loan | numeric | 0.0 | 12 | 12.06 | -0.024 |
| `APARTMENTS_AVG` | · | 253 | Building profile of the client's home — apartment size (building average, normalized 0–1) | numeric | 50.7 | 0.09 | 0.12 | -0.029 |
| `LANDAREA_MODE` | · | 259 | Building profile of the client's home — land area (building mode, normalized 0–1) | numeric | 59.4 | 0.05 | 0.06 | -0.010 |
| `FLOORSMAX_AVG` | · | 263 | Building profile of the client's home — highest floor (building average, normalized 0–1) | numeric | 49.8 | 0.17 | 0.23 | -0.044 |
| `APARTMENTS_MEDI` | · | 279 | Building profile of the client's home — apartment size (building median, normalized 0–1) | numeric | 50.7 | 0.09 | 0.12 | -0.029 |
| `AMT_REQ_CREDIT_BUREAU_QRT` | · | 289 | Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application) | numeric | 13.5 | 0 | 0.27 | -0.002 |
| `BASEMENTAREA_MODE` | · | 291 | Building profile of the client's home — basement area (building mode, normalized 0–1) | numeric | 58.5 | 0.07 | 0.09 | -0.020 |
| `REGION_RATING_CLIENT` | · | 294 | Our rating of the region where client lives (1,2,3) | numeric | 0.0 | 2 | 2.05 | 0.059 |
| `LANDAREA_MEDI` | · | 306 | Building profile of the client's home — land area (building median, normalized 0–1) | numeric | 59.4 | 0.05 | 0.07 | -0.011 |
| `ENTRANCES_AVG` | · | 311 | Building profile of the client's home — number of entrances (building average, normalized 0–1) | numeric | 50.3 | 0.14 | 0.15 | -0.019 |
| `FLAG_OWN_CAR` | · | 315 | Flag if the client owns a car | categorical | 0.0 | – | – | – |
| `WALLSMATERIAL_MODE` | · | 316 | Building profile of the client's home — walls material (building mode, normalized 0–1) | categorical | 50.8 | – | – | – |
| `BASEMENTAREA_AVG` | · | 332 | Building profile of the client's home — basement area (building average, normalized 0–1) | numeric | 58.5 | 0.08 | 0.09 | -0.023 |
| `NONLIVINGAREA_AVG` | · | 334 | Building profile of the client's home — non-living area (building average, normalized 0–1) | numeric | 55.2 | 0.00 | 0.03 | -0.014 |
| `OBS_30_CNT_SOCIAL_CIRCLE` | · | 340 | How many observation of client's social surroundings with observable 30 DPD (days past due) default | numeric | 0.3 | 0 | 1.42 | 0.009 |
| `OBS_60_CNT_SOCIAL_CIRCLE` | · | 349 | How many observation of client's social surroundings with observable 60 DPD (days past due) default | numeric | 0.3 | 0 | 1.41 | 0.009 |
| `BASEMENTAREA_MEDI` | · | 351 | Building profile of the client's home — basement area (building median, normalized 0–1) | numeric | 58.5 | 0.08 | 0.09 | -0.022 |
| `LIVINGAPARTMENTS_AVG` | · | 356 | Building profile of the client's home — number of living apartments (building average, normalized 0–1) | numeric | 68.4 | 0.08 | 0.10 | -0.025 |
| `COMMONAREA_MODE` | · | 358 | Building profile of the client's home — common area (building mode, normalized 0–1) | numeric | 69.9 | 0.02 | 0.04 | -0.016 |
| `NONLIVINGAREA_MODE` | · | 363 | Building profile of the client's home — non-living area (building mode, normalized 0–1) | numeric | 55.2 | 0.00 | 0.03 | -0.013 |
| `LIVINGAPARTMENTS_MODE` | · | 388 | Building profile of the client's home — number of living apartments (building mode, normalized 0–1) | numeric | 68.4 | 0.08 | 0.11 | -0.023 |
| `COMMONAREA_AVG` | · | 395 | Building profile of the client's home — common area (building average, normalized 0–1) | numeric | 69.9 | 0.02 | 0.04 | -0.019 |
| `YEARS_BUILD_MODE` | · | 398 | Building profile of the client's home — building age (building mode, normalized 0–1) | numeric | 66.5 | 0.76 | 0.76 | -0.022 |
| `COMMONAREA_MEDI` | · | 406 | Building profile of the client's home — common area (building median, normalized 0–1) | numeric | 69.9 | 0.02 | 0.04 | -0.019 |
| `YEARS_BUILD_AVG` | · | 408 | Building profile of the client's home — building age (building average, normalized 0–1) | numeric | 66.5 | 0.76 | 0.75 | -0.022 |
| `NAME_HOUSING_TYPE` | · | 413 | What is the housing situation of the client (renting, living with parents, ...) | categorical | 0.0 | – | – | – |
| `LIVINGAPARTMENTS_MEDI` | · | 419 | Building profile of the client's home — number of living apartments (building median, normalized 0–1) | numeric | 68.4 | 0.08 | 0.10 | -0.025 |
| `NONLIVINGAREA_MEDI` | · | 435 | Building profile of the client's home — non-living area (building median, normalized 0–1) | numeric | 55.2 | 0.00 | 0.03 | -0.013 |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | · | 442 | Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application) | numeric | 13.5 | 1 | 1.90 | 0.020 |
| `FLAG_DOCUMENT_18` | · | 445 | Did client provide document 18 | numeric | 0.0 | 0 | 0.01 | -0.008 |
| `CNT_FAM_MEMBERS` | · | 448 | How many family members does client have | numeric | 0.0 | 2 | 2.15 | 0.009 |
| `ENTRANCES_MEDI` | · | 451 | Building profile of the client's home — number of entrances (building median, normalized 0–1) | numeric | 50.3 | 0.14 | 0.15 | -0.019 |
| `NONLIVINGAPARTMENTS_AVG` | · | 455 | Building profile of the client's home — number of non-living apartments (building average, normalized 0–1) | numeric | 69.4 | 0 | 0.01 | -0.003 |
| `ENTRANCES_MODE` | · | 461 | Building profile of the client's home — number of entrances (building mode, normalized 0–1) | numeric | 50.3 | 0.14 | 0.15 | -0.017 |
| `FLOORSMAX_MEDI` | · | 465 | Building profile of the client's home — highest floor (building median, normalized 0–1) | numeric | 49.8 | 0.17 | 0.23 | -0.044 |
| `ELEVATORS_AVG` | · | 472 | Building profile of the client's home — number of elevators (building average, normalized 0–1) | numeric | 53.3 | 0 | 0.08 | -0.034 |
| `YEARS_BUILD_MEDI` | · | 474 | Building profile of the client's home — building age (building median, normalized 0–1) | numeric | 66.5 | 0.76 | 0.76 | -0.022 |
| `FLOORSMIN_AVG` | · | 509 | Building profile of the client's home — lowest floor (building average, normalized 0–1) | numeric | 67.8 | 0.21 | 0.23 | -0.034 |
| `FLOORSMAX_MODE` | · | 511 | Building profile of the client's home — highest floor (building mode, normalized 0–1) | numeric | 49.8 | 0.17 | 0.22 | -0.043 |
| `FLOORSMIN_MODE` | · | 524 | Building profile of the client's home — lowest floor (building mode, normalized 0–1) | numeric | 67.8 | 0.21 | 0.23 | -0.033 |
| `FLAG_DOCUMENT_16` | · | 527 | Did client provide document 16 | numeric | 0.0 | 0 | 0.01 | -0.012 |
| `NONLIVINGAPARTMENTS_MODE` | · | 528 | Building profile of the client's home — number of non-living apartments (building mode, normalized 0–1) | numeric | 69.4 | 0 | 0.01 | -0.002 |
| `NAME_TYPE_SUITE` | · | 532 | Who was accompanying client when he was applying for the loan | categorical | 0.4 | – | – | – |
| `NONLIVINGAPARTMENTS_MEDI` | · | 548 | Building profile of the client's home — number of non-living apartments (building median, normalized 0–1) | numeric | 69.4 | 0 | 0.01 | -0.003 |
| `CNT_CHILDREN` | · | 555 | Number of children the client has | numeric | 0.0 | 0 | 0.42 | 0.019 |
| `AMT_REQ_CREDIT_BUREAU_MON` | · | 572 | Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application) | numeric | 13.5 | 0 | 0.27 | -0.012 |
| `ELEVATORS_MEDI` | · | 577 | Building profile of the client's home — number of elevators (building median, normalized 0–1) | numeric | 53.3 | 0 | 0.08 | -0.034 |
| `FLOORSMIN_MEDI` | · | 584 | Building profile of the client's home — lowest floor (building median, normalized 0–1) | numeric | 67.8 | 0.21 | 0.23 | -0.033 |
| `ELEVATORS_MODE` | · | 598 | Building profile of the client's home — number of elevators (building mode, normalized 0–1) | numeric | 53.3 | 0 | 0.07 | -0.032 |
| `FLAG_PHONE` | · | 602 | Did client provide home phone (1=YES, 0=NO) | numeric | 0.0 | 0 | 0.28 | -0.024 |
| `FLAG_OWN_REALTY` | · | 627 | Flag if client owns a house or flat | categorical | 0.0 | – | – | – |
| `FLAG_DOCUMENT_13` | · | 646 | Did client provide document 13 | numeric | 0.0 | 0 | 0.00 | -0.012 |
| `FONDKAPREMONT_MODE` | · | 667 | Building profile of the client's home — maintenance-fund type (building mode, normalized 0–1) | categorical | 68.4 | – | – | – |
| `REG_CITY_NOT_WORK_CITY` | · | 680 | Flag if client's permanent address does not match work address (1=different, 0=same, at city level) | numeric | 0.0 | 0 | 0.23 | 0.051 |
| `FLAG_DOCUMENT_8` | · | 701 | Did client provide document 8 | numeric | 0.0 | 0 | 0.08 | -0.008 |
| `LIVE_CITY_NOT_WORK_CITY` | · | 718 | Flag if client's contact address does not match work address (1=different, 0=same, at city level) | numeric | 0.0 | 0 | 0.18 | 0.033 |
| `HOUSETYPE_MODE` | · | 751 | Building profile of the client's home — house type (building mode, normalized 0–1) | categorical | 50.2 | – | – | – |
| `AMT_REQ_CREDIT_BUREAU_DAY` | · | 759 | Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application) | numeric | 13.5 | 0 | 0.01 | 0.003 |
| `AMT_REQ_CREDIT_BUREAU_WEEK` | · | 767 | Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application) | numeric | 13.5 | 0 | 0.03 | 7.90e-04 |
| `FLAG_EMAIL` | · | 780 | Did client provide email (1=YES, 0=NO) | numeric | 0.0 | 0 | 0.06 | -0.002 |
| `EMERGENCYSTATE_MODE` | · | 784 | Building profile of the client's home — emergency-state flag (building mode, normalized 0–1) | categorical | 47.4 | – | – | – |
| `FLAG_DOCUMENT_6` | · | 807 | Did client provide document 6 | numeric | 0.0 | 0 | 0.09 | -0.029 |
| `FLAG_EMP_PHONE` | · | 820 | Did client provide work phone (1=YES, 0=NO) | numeric | 0.0 | 1 | 0.82 | 0.046 |
| `LIVE_REGION_NOT_WORK_REGION` | · | 821 | Flag if client's contact address does not match work address (1=different, 0=same, at region level) | numeric | 0.0 | 0 | 0.04 | 0.003 |
| `FLAG_DOCUMENT_11` | · | 831 | Did client provide document 11 | numeric | 0.0 | 0 | 0.00 | -0.004 |
| `REG_REGION_NOT_LIVE_REGION` | · | 832 | Flag if client's permanent address does not match contact address (1=different, 0=same, at region level) | numeric | 0.0 | 0 | 0.02 | 0.006 |
| `REG_REGION_NOT_WORK_REGION` | · | 833 | Flag if client's permanent address does not match work address (1=different, 0=same, at region level) | numeric | 0.0 | 0 | 0.05 | 0.007 |
| `AMT_REQ_CREDIT_BUREAU_HOUR` | · | 840 | Number of enquiries to Credit Bureau about the client one hour before application | numeric | 13.5 | 0 | 0.01 | 9.30e-04 |
| `FLAG_DOCUMENT_14` | · | 861 | Did client provide document 14 | numeric | 0.0 | 0 | 0.00 | -0.009 |
| `FLAG_DOCUMENT_5` | · | 866 | Did client provide document 5 | numeric | 0.0 | 0 | 0.02 | -3.20e-04 |
| `FLAG_MOBIL` | · | 874 | Did client provide mobile phone (1=YES, 0=NO) | numeric | 0.0 | 1 | 1.00 | 5.30e-04 |
| `FLAG_CONT_MOBILE` | · | 875 | Was mobile phone reachable (1=YES, 0=NO) | numeric | 0.0 | 1 | 1.00 | 3.70e-04 |
| `FLAG_DOCUMENT_2` | · | 876 | Did client provide document 2 | numeric | 0.0 | 0 | 4.23e-05 | 0.005 |
| `FLAG_DOCUMENT_4` | · | 877 | Did client provide document 4 | numeric | 0.0 | 0 | 8.13e-05 | -0.003 |
| `FLAG_DOCUMENT_7` | · | 878 | Did client provide document 7 | numeric | 0.0 | 0 | 1.92e-04 | -0.002 |
| `FLAG_DOCUMENT_9` | · | 879 | Did client provide document 9 | numeric | 0.0 | 0 | 0.00 | -0.004 |
| `FLAG_DOCUMENT_10` | · | 880 | Did client provide document 10 | numeric | 0.0 | 0 | 2.28e-05 | -0.001 |
| `FLAG_DOCUMENT_12` | · | 881 | Did client provide document 12 | numeric | 0.0 | 0 | 6.50e-06 | -7.60e-04 |
| `FLAG_DOCUMENT_15` | · | 882 | Did client provide document 15 | numeric | 0.0 | 0 | 0.00 | -0.007 |
| `FLAG_DOCUMENT_17` | · | 883 | Did client provide document 17 | numeric | 0.0 | 0 | 2.67e-04 | -0.003 |
| `FLAG_DOCUMENT_19` | · | 884 | Did client provide document 19 | numeric | 0.0 | 0 | 5.95e-04 | -0.001 |
| `FLAG_DOCUMENT_20` | · | 885 | Did client provide document 20 | numeric | 0.0 | 0 | 5.07e-04 | 2.20e-04 |
| `FLAG_DOCUMENT_21` | · | 886 | Did client provide document 21 | numeric | 0.0 | 0 | 3.35e-04 | 0.004 |

### 7.2 Driver-derived features (`RATIO_*`, `DAYS_EMPLOYED_ANOMALY`)

*7 features · 6 selected · 5.5% of model gain*

Built by `features.py` on the driver frame before any join — the two things the aggregation pipeline cannot supply. **Ratios**: a tree splits on one column at a time, so a quotient of two of its own inputs is inexpressible no matter how deep it grows; every input here is already a high-gain driver column. **Sentinel flag**: `DAYS_EMPLOYED` uses `365243` for "never employed" in 18% of rows, which is nulled out (it would otherwise wreck any imputation or scaling) with its membership preserved as a flag.

<details><summary><b>Source-column glossary</b> (4 columns)</summary>

| Source column | What it measures |
|---|---|
| `AMT_ANNUITY` | Loan annuity |
| `AMT_CREDIT` | Credit amount of the loan |
| `AMT_INCOME_TOTAL` | Income of the client |
| `DAYS_EMPLOYED` | How many days before the application the person started current employment |

</details>

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `RATIO_PAYMENT_RATE` | **✔** | 5 | `AMT_ANNUITY` | ratio | 0.0 | 0.0 | 0.05 | 0.05 | 0.11 | 0.013 |
| `RATIO_CREDIT_TO_GOODS` | **✔** | 6 | `AMT_CREDIT` | ratio | 0.1 | 0.1 | 1.12 | 1.12 | 1.48 | 0.069 |
| `RATIO_ANNUITY_TO_INCOME` | **✔** | 19 | `AMT_ANNUITY` | ratio | 0.0 | 0.0 | 0.16 | 0.18 | 0.48 | 0.014 |
| `RATIO_EMPLOYED_TO_AGE` | **✔** | 26 | `DAYS_EMPLOYED` | ratio | 18.0 | 18.0 | 0.12 | 0.16 | 0.59 | -0.068 |
| `RATIO_CREDIT_TO_INCOME` | **✔** | 52 | `AMT_CREDIT` | ratio | 0.0 | 0.0 | 3.27 | 3.96 | 13.03 | -0.008 |
| `RATIO_INCOME_PER_PERSON` | **✔** | 107 | `AMT_INCOME_TOTAL` | ratio | 0.0 | 0.0 | 75,000 | 93,106 | 337,500 | -0.007 |
| `DAYS_EMPLOYED_ANOMALY` | · | 856 | `DAYS_EMPLOYED` | sentinel-flag | 0.0 | 0.0 | 0 | 0.18 | 1 | -0.046 |

### 7.3 Previous-application roll-up (`PREV_*`)

*162 features · 37 selected · 11.8% of model gain*

One row per customer from `previous_application`, which also acts as the hub carrying rolled-up ledger behaviour (the `PREV_CC_*` / `PREV_INST_*` / `PREV_POS_*` blocks below). This block is `previous_application`'s **own** columns.

**Definition template** — `PREV_<COL>_<AGG>` = the *<agg>* of `<COL>` across all of the customer's previous Home Credit applications.

**Calculation** — `previous_application.group_by('SK_ID_CURR').<agg>(COL)`.

The `365243` "never / not applicable" sentinel is nulled out in every `DAYS_*` column *before* aggregating (it is 56% of `DAYS_FIRST_DRAWING`), so these means are not poisoned by a fake 1,000-year offset.

<details><summary><b>Source-column glossary</b> (35 columns)</summary>

| Source column | What it measures |
|---|---|
| `AMT_ANNUITY` | Annuity of previous application |
| `AMT_APPLICATION` | For how much credit did client ask on the previous application |
| `AMT_CREDIT` | Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client initially applied for, but during our approval p… |
| `AMT_DOWN_PAYMENT` | Down payment on the previous application |
| `AMT_GOODS_PRICE` | Goods price of good that client asked for (if applicable) on the previous application |
| `CHANNEL_TYPE` | Through which channel we acquired the client on the previous application |
| `CNT_PAYMENT` | Term of previous credit at application of the previous application |
| `CODE_REJECT_REASON` | Why was the previous application rejected |
| `DAYS_DECISION` | Relative to current application when was the decision about previous application made |
| `DAYS_FIRST_DRAWING` | Relative to application date of current application when was the first disbursement of the previous application |
| `DAYS_FIRST_DUE` | Relative to application date of current application when was the first due supposed to be of the previous application |
| `DAYS_LAST_DUE` | Relative to application date of current application when was the last due date of the previous application |
| `DAYS_LAST_DUE_1ST_VERSION` | Relative to application date of current application when was the first due of the previous application |
| `DAYS_TERMINATION` | Relative to application date of current application when was the expected termination of the previous application |
| `FLAG_LAST_APPL_PER_CONTRACT` | Flag if it was last application for the previous contract. Sometimes by mistake of client or our clerk there could be more applications for one single contract |
| `HOUR_APPR_PROCESS_START` | Approximately at what day hour did the client apply for the previous application |
| `NAME_CASH_LOAN_PURPOSE` | Purpose of the cash loan |
| `NAME_CLIENT_TYPE` | Was the client old or new client when applying for the previous application |
| `NAME_CONTRACT_STATUS` | Contract status (approved, cancelled, ...) of previous application |
| `NAME_CONTRACT_TYPE` | Contract product type (Cash loan, consumer loan [POS] ,...) of the previous application |
| `NAME_GOODS_CATEGORY` | What kind of goods did the client apply for in the previous application |
| `NAME_PAYMENT_TYPE` | Payment method that client chose to pay for the previous application |
| `NAME_PORTFOLIO` | Was the previous application for CASH, POS, CAR, |
| `NAME_PRODUCT_TYPE` | Was the previous application x-sell o walk-in |
| `NAME_SELLER_INDUSTRY` | The industry of the seller |
| `NAME_TYPE_SUITE` | Who accompanied client when applying for the previous application |
| `NAME_YIELD_GROUP` | Grouped interest rate into small medium and high of the previous application |
| `NFLAG_INSURED_ON_APPROVAL` | Did the client requested insurance during the previous application |
| `NFLAG_LAST_APPL_IN_DAY` | Flag if the application was the last application per day of the client. Sometimes clients apply for more applications a day. Rarely it could also be error in our system that one application is in the … |
| `PRODUCT_COMBINATION` | Detailed product combination of the previous application |
| `RATE_DOWN_PAYMENT` | Down payment rate normalized on previous credit |
| `RATE_INTEREST_PRIMARY` | Interest rate normalized on previous credit |
| `RATE_INTEREST_PRIVILEGED` | Interest rate normalized on previous credit |
| `SELLERPLACE_AREA` | Selling area of seller place of the previous application |
| `WEEKDAY_APPR_PROCESS_START` | On which day of the week did the client apply for previous application |

</details>

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `PREV_REFUSED_RATE` | **✔** | 10 | `NAME_CONTRACT_STATUS` | DOMAIN | 5.4 | 0.0 | 0 | 0.10 | 0.71 | 0.078 |
| `PREV_DAYS_LAST_DUE_1ST_VERSION_MAX` | **✔** | 17 | `DAYS_LAST_DUE_1ST_VERSION` | MAX | 6.7 | 6.7 | 12 | -77.75 | 1560 | 0.011 |
| `PREV_CNT_PAYMENT_MEAN` | **✔** | 36 | `CNT_PAYMENT` | MEAN | 5.5 | 5.5 | 12 | 14.53 | 42 | 0.028 |
| `PREV_NAME_YIELD_GROUP_high_RATE` | **✔** | 39 | `NAME_YIELD_GROUP` | RATE of "high" | 5.4 | 0.0 | 0.11 | 0.23 | 1 | 0.038 |
| `PREV_AMT_DOWN_PAYMENT_MAX` | **✔** | 41 | `AMT_DOWN_PAYMENT` | MAX | 11.0 | 11.0 | 5850 | 11,681 | 90,000 | -0.025 |
| `PREV_DAYS_LAST_DUE_MAX` | **✔** | 42 | `DAYS_LAST_DUE` | MAX | 12.5 | 12.5 | -346 | -559.99 | -9 | 0.001 |
| `PREV_AMT_DOWN_PAYMENT_SUM` | **✔** | 45 | `AMT_DOWN_PAYMENT` | SUM | 5.4 | 0.0 | 5152.50 | 14,374 | 121,504 | -0.027 |
| `PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN` | **✔** | 50 | `DAYS_LAST_DUE_1ST_VERSION` | MEAN | 6.7 | 6.7 | -515 | -608.41 | 655 | 0.041 |
| `PREV_HOUR_APPR_PROCESS_START_MEAN` | **✔** | 61 | `HOUR_APPR_PROCESS_START` | MEAN | 5.4 | 5.4 | 12.75 | 12.66 | 19 | -0.036 |
| `PREV_DAYS_LAST_DUE_1ST_VERSION_SUM` | **✔** | 64 | `DAYS_LAST_DUE_1ST_VERSION` | SUM | 5.4 | 0.0 | -983 | -1692.30 | 1232 | 0.052 |
| `PREV_DAYS_DECISION_MEAN` | **✔** | 65 | `DAYS_DECISION` | MEAN | 5.4 | 5.4 | -789 | -919.57 | -139 | 0.047 |
| `PREV_RATE_DOWN_PAYMENT_MAX` | **✔** | 74 | `RATE_DOWN_PAYMENT` | MAX | 11.0 | 11.0 | 0.11 | 0.13 | 0.56 | -0.040 |
| `PREV_DAYS_TERMINATION_MAX` | **✔** | 83 | `DAYS_TERMINATION` | MAX | 13.2 | 13.2 | -337 | -554.63 | -8 | 0.002 |
| `PREV_NAME_YIELD_GROUP_low_action_RATE` | **✔** | 88 | `NAME_YIELD_GROUP` | RATE of "low_action" | 5.4 | 0.0 | 0 | 0.06 | 1 | -0.029 |
| `PREV_AMT_DOWN_PAYMENT_MEAN` | **✔** | 90 | `AMT_DOWN_PAYMENT` | MEAN | 11.0 | 11.0 | 3375 | 7080.50 | 57,375 | -0.025 |
| `PREV_AMT_ANNUITY_MEAN` | **✔** | 97 | `AMT_ANNUITY` | MEAN | 5.5 | 5.5 | 11,984 | 14,532 | 50,078 | -0.035 |
| `PREV_CNT_PAYMENT_MAX` | **✔** | 101 | `CNT_PAYMENT` | MAX | 5.5 | 5.5 | 18 | 24.09 | 60 | 0.029 |
| `PREV_SELLERPLACE_AREA_MEAN` | **✔** | 102 | `SELLERPLACE_AREA` | MEAN | 5.4 | 5.4 | 79 | 413.11 | 3333 | -0.002 |
| `PREV_DAYS_DECISION_MAX` | **✔** | 106 | `DAYS_DECISION` | MAX | 5.4 | 5.4 | -299 | -480.82 | -5 | 0.016 |
| `PREV_AMT_ANNUITY_MIN` | **✔** | 111 | `AMT_ANNUITY` | MIN | 5.5 | 5.5 | 5721.98 | 7864.30 | 38,250 | -0.032 |
| `PREV_RATE_DOWN_PAYMENT_MEAN` | **✔** | 112 | `RATE_DOWN_PAYMENT` | MEAN | 11.0 | 11.0 | 0.07 | 0.08 | 0.42 | -0.034 |
| `PREV_CODE_REJECT_REASON_HC_RATE` | **✔** | 118 | `CODE_REJECT_REASON` | RATE of "HC" | 5.4 | 0.0 | 0 | 0.06 | 0.59 | 0.055 |
| `PREV_SELLERPLACE_AREA_MAX` | **✔** | 122 | `SELLERPLACE_AREA` | MAX | 5.4 | 5.4 | 158 | 967.98 | 6000 | -0.003 |
| `PREV_NAME_PRODUCT_TYPE_walk_in_RATE` | **✔** | 124 | `NAME_PRODUCT_TYPE` | RATE of "walk-in" | 5.4 | 0.0 | 0 | 0.08 | 1 | 0.057 |
| `PREV_AMT_GOODS_PRICE_MIN` | **✔** | 130 | `AMT_GOODS_PRICE` | MIN | 5.7 | 5.7 | 45,630 | 79,415 | 512,550 | -0.021 |
| `PREV_RATE_DOWN_PAYMENT_SUM` | **✔** | 132 | `RATE_DOWN_PAYMENT` | SUM | 5.4 | 0.0 | 0.10 | 0.17 | 1.05 | -0.042 |
| `PREV_NAME_YIELD_GROUP_low_normal_RATE` | **✔** | 137 | `NAME_YIELD_GROUP` | RATE of "low_normal" | 5.4 | 0.0 | 0.07 | 0.20 | 1 | -0.041 |
| `PREV_APPROVED_RATE` | **✔** | 143 | `NAME_CONTRACT_STATUS` | DOMAIN | 5.4 | 0.0 | 0.75 | 0.71 | 1 | -0.064 |
| `PREV_CNT_PAYMENT_SUM` | **✔** | 144 | `CNT_PAYMENT` | SUM | 5.4 | 0.0 | 34 | 57.76 | 352 | 0.028 |
| `PREV_DAYS_FIRST_DRAWING_MEAN` | **✔** | 160 | `DAYS_FIRST_DRAWING` | MEAN | 82.7 | 82.7 | -632 | -1052.85 | -33 | 0.096 |
| `PREV_HOUR_APPR_PROCESS_START_MAX` | **✔** | 163 | `HOUR_APPR_PROCESS_START` | MAX | 5.4 | 5.4 | 15 | 14.77 | 21 | -0.031 |
| `PREV_HAS_POS_RATE` | **✔** | 165 | `–` | DOMAIN | 5.4 | 0.0 | 0.67 | 0.63 | 1 | -0.063 |
| `PREV_SELLERPLACE_AREA_SUM` | **✔** | 167 | `SELLERPLACE_AREA` | SUM | 5.4 | 0.0 | 227 | 1448.07 | 12,840 | -0.004 |
| `PREV_NAME_CLIENT_TYPE_New_RATE` | **✔** | 181 | `NAME_CLIENT_TYPE` | RATE of "New" | 5.4 | 0.0 | 0.20 | 0.33 | 1 | 0.021 |
| `PREV_CODE_REJECT_REASON_XAP_RATE` | **✔** | 186 | `CODE_REJECT_REASON` | RATE of "XAP" | 5.4 | 0.0 | 1 | 0.83 | 1 | -0.074 |
| `PREV_AMT_ANNUITY_MAX` | **✔** | 196 | `AMT_ANNUITY` | MAX | 5.5 | 5.5 | 17,908 | 23,723 | 90,224 | -0.029 |
| `PREV_AMT_CREDIT_MEAN` | **✔** | 199 | `AMT_CREDIT` | MEAN | 5.4 | 5.4 | 117,450 | 170,070 | 787,617 | -0.016 |
| `PREV_AMT_GOODS_PRICE_MEAN` | · | 204 | `AMT_GOODS_PRICE` | MEAN | 5.7 | 5.7 | 128,595 | 193,896 | 914,600 | -0.016 |
| `PREV_DAYS_LAST_DUE_MEAN` | · | 207 | `DAYS_LAST_DUE` | MEAN | 12.5 | 12.5 | -840.43 | -919.60 | -26 | 0.034 |
| `PREV_DAYS_LAST_DUE_1ST_VERSION_MIN` | · | 210 | `DAYS_LAST_DUE_1ST_VERSION` | MIN | 6.7 | 6.7 | -1107 | -1153.71 | 524 | 0.052 |
| `PREV_DAYS_TERMINATION_MEAN` | · | 212 | `DAYS_TERMINATION` | MEAN | 13.2 | 13.2 | -823 | -907.01 | -25 | 0.033 |
| `PREV_DAYS_DECISION_MIN` | · | 215 | `DAYS_DECISION` | MIN | 5.4 | 5.4 | -1503 | -1539.82 | -172 | 0.053 |
| `PREV_REFUSED_COUNT` | · | 218 | `NAME_CONTRACT_STATUS` | DOMAIN | 5.4 | 0.0 | 0 | 0.80 | 8 | 0.064 |
| `PREV_AMT_APPLICATION_MEAN` | · | 228 | `AMT_APPLICATION` | MEAN | 5.4 | 5.4 | 106,002 | 154,005 | 732,375 | -0.022 |
| `PREV_NAME_TYPE_SUITE_Unaccompanied_RATE` | · | 229 | `NAME_TYPE_SUITE` | RATE of "Unaccompanied" | 20.1 | 0.0 | 0.50 | 0.45 | 1 | 0.016 |
| `PREV_RATE_DOWN_PAYMENT_MIN` | · | 231 | `RATE_DOWN_PAYMENT` | MIN | 11.0 | 11.0 | 0 | 0.05 | 0.42 | -0.015 |
| `PREV_NAME_YIELD_GROUP_middle_RATE` | · | 232 | `NAME_YIELD_GROUP` | RATE of "middle" | 5.4 | 0.0 | 0.17 | 0.24 | 1 | -0.016 |
| `PREV_DAYS_FIRST_DRAWING_MAX` | · | 234 | `DAYS_FIRST_DRAWING` | MAX | 82.7 | 82.7 | -628 | -1050.80 | -32 | 0.096 |
| `PREV_DAYS_DECISION_SUM` | · | 235 | `DAYS_DECISION` | SUM | 5.4 | 0.0 | -2779 | -4047.26 | 0 | 0.020 |
| `PREV_AMT_CREDIT_MAX` | · | 247 | `AMT_CREDIT` | MAX | 5.4 | 5.4 | 225,000 | 413,866 | 2.07M | -0.008 |
| `PREV_HOUR_APPR_PROCESS_START_SUM` | · | 252 | `HOUR_APPR_PROCESS_START` | SUM | 5.4 | 0.0 | 43 | 57.37 | 248 | 0.011 |
| `PREV_HAS_INST_RATE` | · | 258 | `–` | DOMAIN | 5.4 | 0.0 | 0.67 | 0.67 | 1 | -0.055 |
| `PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_RATE` | · | 264 | `WEEKDAY_APPR_PROCESS_START` | RATE of "FRIDAY" | 5.4 | 0.0 | 0 | 0.14 | 1 | 0.005 |
| `PREV_CHANNEL_TYPE_AP_Cash_loan_RATE` | · | 268 | `CHANNEL_TYPE` | RATE of "AP+ (Cash loan)" | 5.4 | 0.0 | 0 | 0.03 | 0.67 | 0.035 |
| `PREV_HOUR_APPR_PROCESS_START_MIN` | · | 269 | `HOUR_APPR_PROCESS_START` | MIN | 5.4 | 5.4 | 10 | 10.73 | 19 | -0.031 |
| `PREV_CHANNEL_TYPE_Credit_and_cash_offices_RATE` | · | 271 | `CHANNEL_TYPE` | RATE of "Credit and cash offices" | 5.4 | 0.0 | 0 | 0.28 | 1 | 0.010 |
| `PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_RATE` | · | 273 | `WEEKDAY_APPR_PROCESS_START` | RATE of "THURSDAY" | 5.4 | 0.0 | 0 | 0.14 | 1 | 0.003 |
| `PREV_AMT_ANNUITY_SUM` | · | 274 | `AMT_ANNUITY` | SUM | 5.4 | 0.0 | 33,300 | 56,985 | 343,125 | -0.007 |
| `PREV_AMT_APPLICATION_MAX` | · | 278 | `AMT_APPLICATION` | MAX | 5.4 | 5.4 | 202,500 | 374,590 | 1.94M | -0.013 |
| `PREV_AMT_CREDIT_MIN` | · | 280 | `AMT_CREDIT` | MIN | 5.4 | 5.4 | 23,805 | 48,691 | 429,196 | -0.019 |
| `PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_RATE` | · | 281 | `WEEKDAY_APPR_PROCESS_START` | RATE of "MONDAY" | 5.4 | 0.0 | 0 | 0.14 | 1 | 0.006 |
| `PREV_SELLERPLACE_AREA_MIN` | · | 282 | `SELLERPLACE_AREA` | MIN | 5.4 | 5.4 | -1 | 169.78 | 2586 | -0.001 |
| `PREV_DAYS_TERMINATION_MIN` | · | 283 | `DAYS_TERMINATION` | MIN | 13.2 | 13.2 | -1253 | -1293.49 | -25 | 0.048 |
| `PREV_AMT_DOWN_PAYMENT_MIN` | · | 286 | `AMT_DOWN_PAYMENT` | MIN | 11.0 | 11.0 | 0 | 3961.75 | 45,765 | -0.017 |
| `PREV_CHANNEL_TYPE_Channel_of_corporate_sales_RATE` | · | 292 | `CHANNEL_TYPE` | RATE of "Channel of corporate sales" | 5.4 | 0.0 | 0 | 0.00 | 0 | -0.009 |
| `PREV_WEEKDAY_APPR_PROCESS_START_TUESDAY_RATE` | · | 298 | `WEEKDAY_APPR_PROCESS_START` | RATE of "TUESDAY" | 5.4 | 0.0 | 0 | 0.14 | 1 | 3.30e-04 |
| `PREV_CHANNEL_TYPE_Country_wide_RATE` | · | 299 | `CHANNEL_TYPE` | RATE of "Country-wide" | 5.4 | 0.0 | 0.25 | 0.35 | 1 | -0.001 |
| `PREV_CODE_REJECT_REASON_SCOFR_RATE` | · | 300 | `CODE_REJECT_REASON` | RATE of "SCOFR" | 5.4 | 0.0 | 0 | 0.01 | 0.21 | 0.056 |
| `PREV_DAYS_LAST_DUE_MIN` | · | 302 | `DAYS_LAST_DUE` | MIN | 12.5 | 12.5 | -1278 | -1313.71 | -26 | 0.048 |
| `PREV_NAME_PAYMENT_TYPE_XNA_RATE` | · | 304 | `NAME_PAYMENT_TYPE` | RATE of "XNA" | 5.4 | 0.0 | 0.25 | 0.30 | 1 | 0.035 |
| `PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_RATE` | · | 308 | `WEEKDAY_APPR_PROCESS_START` | RATE of "WEDNESDAY" | 5.4 | 0.0 | 0 | 0.14 | 1 | 0.004 |
| `PREV_AMT_CREDIT_SUM` | · | 317 | `AMT_CREDIT` | SUM | 5.4 | 0.0 | 383,778 | 902,686 | 6.81M | 0.008 |
| `PREV_DAYS_FIRST_DRAWING_MIN` | · | 318 | `DAYS_FIRST_DRAWING` | MIN | 82.7 | 82.7 | -632 | -1054.92 | -33 | 0.096 |
| `PREV_NAME_CONTRACT_TYPE_Cash_loans_RATE` | · | 333 | `NAME_CONTRACT_TYPE` | RATE of "Cash loans" | 5.4 | 0.0 | 0.25 | 0.30 | 1 | 0.015 |
| `PREV_NAME_CLIENT_TYPE_Refreshed_RATE` | · | 335 | `NAME_CLIENT_TYPE` | RATE of "Refreshed" | 5.4 | 0.0 | 0 | 0.09 | 1 | -0.027 |
| `PREV_DAYS_FIRST_DUE_MAX` | · | 337 | `DAYS_FIRST_DUE` | MAX | 5.9 | 5.9 | -427 | -602.50 | -20 | 0.004 |
| `PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_RATE` | · | 345 | `WEEKDAY_APPR_PROCESS_START` | RATE of "SATURDAY" | 5.4 | 0.0 | 0 | 0.15 | 1 | -0.008 |
| `PREV_AMT_APPLICATION_SUM` | · | 346 | `AMT_APPLICATION` | SUM | 5.4 | 0.0 | 343,098 | 805,636 | 6.19M | 0.005 |
| `PREV_CHANNEL_TYPE_Stone_RATE` | · | 350 | `CHANNEL_TYPE` | RATE of "Stone" | 5.4 | 0.0 | 0 | 0.17 | 1 | -0.021 |
| `PREV_NAME_PRODUCT_TYPE_XNA_RATE` | · | 360 | `NAME_PRODUCT_TYPE` | RATE of "XNA" | 5.4 | 0.0 | 0.71 | 0.68 | 1 | -0.021 |
| `PREV_AMT_APPLICATION_MIN` | · | 364 | `AMT_APPLICATION` | MIN | 5.4 | 5.4 | 18,368 | 44,361 | 416,250 | -0.021 |
| `PREV_NFLAG_INSURED_ON_APPROVAL_MEAN` | · | 366 | `NFLAG_INSURED_ON_APPROVAL` | MEAN | 5.8 | 5.8 | 0.20 | 0.28 | 1 | -5.30e-04 |
| `PREV_NAME_PAYMENT_TYPE_Cash_through_the_bank_RATE` | · | 367 | `NAME_PAYMENT_TYPE` | RATE of "Cash through the bank" | 5.4 | 0.0 | 0.67 | 0.65 | 1 | -0.035 |
| `PREV_CODE_REJECT_REASON_LIMIT_RATE` | · | 368 | `CODE_REJECT_REASON` | RATE of "LIMIT" | 5.4 | 0.0 | 0 | 0.02 | 0.50 | 0.040 |
| `PREV_AMT_GOODS_PRICE_MAX` | · | 370 | `AMT_GOODS_PRICE` | MAX | 5.7 | 5.7 | 202,622 | 375,880 | 1.94M | -0.013 |
| `PREV_DAYS_FIRST_DUE_MEAN` | · | 371 | `DAYS_FIRST_DUE` | MEAN | 5.9 | 5.9 | -938.50 | -1025.02 | -139 | 0.040 |
| `PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_RATE` | · | 373 | `WEEKDAY_APPR_PROCESS_START` | RATE of "SUNDAY" | 5.4 | 0.0 | 0 | 0.11 | 1 | -0.010 |
| `PREV_DAYS_FIRST_DUE_MIN` | · | 377 | `DAYS_FIRST_DUE` | MIN | 5.9 | 5.9 | -1454 | -1494.82 | -143 | 0.054 |
| `PREV_NAME_PORTFOLIO_Cash_RATE` | · | 378 | `NAME_PORTFOLIO` | RATE of "Cash" | 5.4 | 0.0 | 0 | 0.20 | 1 | 0.006 |
| `PREV_NAME_YIELD_GROUP_XNA_RATE` | · | 379 | `NAME_YIELD_GROUP` | RATE of "XNA" | 5.4 | 0.0 | 0.17 | 0.22 | 0.84 | 0.043 |
| `PREV_NAME_CONTRACT_STATUS_Canceled_RATE` | · | 381 | `NAME_CONTRACT_STATUS` | RATE of "Canceled" | 5.4 | 0.0 | 0 | 0.12 | 0.67 | 0.012 |
| `PREV_NAME_CONTRACT_TYPE_Consumer_loans_RATE` | · | 384 | `NAME_CONTRACT_TYPE` | RATE of "Consumer loans" | 5.4 | 0.0 | 0.50 | 0.55 | 1 | -0.033 |
| `PREV_CODE_REJECT_REASON_SCO_RATE` | · | 385 | `CODE_REJECT_REASON` | RATE of "SCO" | 5.4 | 0.0 | 0 | 0.02 | 0.33 | 0.012 |
| `PREV_NAME_PRODUCT_TYPE_x_sell_RATE` | · | 402 | `NAME_PRODUCT_TYPE` | RATE of "x-sell" | 5.4 | 0.0 | 0 | 0.19 | 1 | -0.018 |
| `PREV_NAME_PORTFOLIO_POS_RATE` | · | 424 | `NAME_PORTFOLIO` | RATE of "POS" | 5.4 | 0.0 | 0.50 | 0.53 | 1 | -0.034 |
| `PREV_NAME_TYPE_SUITE_Family_RATE` | · | 426 | `NAME_TYPE_SUITE` | RATE of "Family" | 20.1 | 0.0 | 0 | 0.22 | 1 | -0.016 |
| `PREV_CHANNEL_TYPE_Contact_center_RATE` | · | 428 | `CHANNEL_TYPE` | RATE of "Contact center" | 5.4 | 0.0 | 0 | 0.03 | 0.50 | 0.012 |
| `PREV_NAME_CLIENT_TYPE_Repeater_RATE` | · | 431 | `NAME_CLIENT_TYPE` | RATE of "Repeater" | 5.4 | 0.0 | 0.67 | 0.52 | 1 | -0.005 |
| `PREV_DAYS_FIRST_DUE_SUM` | · | 440 | `DAYS_FIRST_DUE` | SUM | 5.4 | 0.0 | -2106 | -2939.57 | 0 | 0.043 |
| `PREV_NAME_PORTFOLIO_XNA_RATE` | · | 447 | `NAME_PORTFOLIO` | RATE of "XNA" | 5.4 | 0.0 | 0 | 0.14 | 0.67 | 0.029 |
| `PREV_NAME_CASH_LOAN_PURPOSE_NUNIQUE` | · | 449 | `NAME_CASH_LOAN_PURPOSE` | NUNIQUE | 5.4 | 0.0 | 2 | 1.59 | 4 | 0.036 |
| `PREV_NAME_PORTFOLIO_Cards_RATE` | · | 463 | `NAME_PORTFOLIO` | RATE of "Cards" | 5.4 | 0.0 | 0 | 0.07 | 0.50 | 0.032 |
| `PREV_NAME_CONTRACT_TYPE_Revolving_loans_RATE` | · | 478 | `NAME_CONTRACT_TYPE` | RATE of "Revolving loans" | 5.4 | 0.0 | 0 | 0.09 | 0.67 | 0.039 |
| `PREV_CNT_PAYMENT_MIN` | · | 487 | `CNT_PAYMENT` | MIN | 5.5 | 5.5 | 6 | 6.78 | 36 | -0.011 |
| `PREV_NAME_TYPE_SUITE_Spouse_partner_RATE` | · | 488 | `NAME_TYPE_SUITE` | RATE of "Spouse, partner" | 20.1 | 0.0 | 0 | 0.07 | 1 | 9.70e-04 |
| `PREV_AMT_GOODS_PRICE_SUM` | · | 500 | `AMT_GOODS_PRICE` | SUM | 5.4 | 0.0 | 343,102 | 805,752 | 6.19M | 0.005 |
| `PREV_DAYS_LAST_DUE_SUM` | · | 507 | `DAYS_LAST_DUE` | SUM | 5.4 | 0.0 | -1445 | -2173.18 | 0 | 0.044 |
| `PREV_CHANNEL_TYPE_Regional_Local_RATE` | · | 513 | `CHANNEL_TYPE` | RATE of "Regional / Local" | 5.4 | 0.0 | 0 | 0.09 | 1 | -0.009 |
| `PREV_CODE_REJECT_REASON_NUNIQUE` | · | 518 | `CODE_REJECT_REASON` | NUNIQUE | 5.4 | 0.0 | 1 | 1.43 | 4 | 0.064 |
| `PREV_DAYS_TERMINATION_SUM` | · | 520 | `DAYS_TERMINATION` | SUM | 5.4 | 0.0 | -1379 | -2091.76 | 0 | 0.043 |
| `PREV_NAME_TYPE_SUITE_Other_B_RATE` | · | 539 | `NAME_TYPE_SUITE` | RATE of "Other_B" | 20.1 | 0.0 | 0 | 0.02 | 0.50 | 0.007 |
| `PREV_COUNT` | · | 540 | `–` | COUNT | 5.4 | 0.0 | 3 | 4.60 | 20 | 0.020 |
| `PREV_NAME_CONTRACT_STATUS_Unused_offer_RATE` | · | 554 | `NAME_CONTRACT_STATUS` | RATE of "Unused offer" | 5.4 | 0.0 | 0 | 0.01 | 0.33 | 9.20e-04 |
| `PREV_NAME_TYPE_SUITE_NUNIQUE` | · | 574 | `NAME_TYPE_SUITE` | NUNIQUE | 5.4 | 0.0 | 2 | 2.01 | 4 | -0.014 |
| `PREV_NFLAG_INSURED_ON_APPROVAL_SUM` | · | 579 | `NFLAG_INSURED_ON_APPROVAL` | SUM | 5.4 | 0.0 | 0 | 0.92 | 6 | -0.016 |
| `PREV_NAME_SELLER_INDUSTRY_NUNIQUE` | · | 582 | `NAME_SELLER_INDUSTRY` | NUNIQUE | 5.4 | 0.0 | 2 | 1.92 | 4 | -0.015 |
| `PREV_NFLAG_LAST_APPL_IN_DAY_SUM` | · | 583 | `NFLAG_LAST_APPL_IN_DAY` | SUM | 5.4 | 0.0 | 3 | 4.58 | 19 | 0.020 |
| `PREV_APPROVED_COUNT` | · | 594 | `NAME_CONTRACT_STATUS` | DOMAIN | 5.4 | 0.0 | 2 | 2.88 | 10 | -0.032 |
| `PREV_NAME_PORTFOLIO_NUNIQUE` | · | 595 | `NAME_PORTFOLIO` | NUNIQUE | 5.4 | 0.0 | 2 | 2.09 | 4 | 0.027 |
| `PREV_PRODUCT_COMBINATION_NUNIQUE` | · | 600 | `PRODUCT_COMBINATION` | NUNIQUE | 5.4 | 0.0 | 3 | 2.92 | 8 | 0.009 |
| `PREV_CHANNEL_TYPE_NUNIQUE` | · | 605 | `CHANNEL_TYPE` | NUNIQUE | 5.4 | 0.0 | 2 | 1.96 | 5 | 0.002 |
| `PREV_WEEKDAY_APPR_PROCESS_START_NUNIQUE` | · | 607 | `WEEKDAY_APPR_PROCESS_START` | NUNIQUE | 5.4 | 0.0 | 2 | 2.62 | 7 | 0.001 |
| `PREV_NAME_GOODS_CATEGORY_NUNIQUE` | · | 614 | `NAME_GOODS_CATEGORY` | NUNIQUE | 5.4 | 0.0 | 2 | 2.15 | 6 | -0.022 |
| `PREV_NAME_TYPE_SUITE_Children_RATE` | · | 618 | `NAME_TYPE_SUITE` | RATE of "Children" | 20.1 | 0.0 | 0 | 0.03 | 1 | -0.017 |
| `PREV_NAME_CONTRACT_TYPE_NUNIQUE` | · | 619 | `NAME_CONTRACT_TYPE` | NUNIQUE | 5.4 | 0.0 | 2 | 1.77 | 3 | 0.025 |
| `PREV_HAS_CC_RATE` | · | 631 | `–` | DOMAIN | 5.4 | 0.0 | 0 | 0.05 | 0.50 | 0.013 |
| `PREV_NAME_TYPE_SUITE_Other_A_RATE` | · | 634 | `NAME_TYPE_SUITE` | RATE of "Other_A" | 20.1 | 0.0 | 0 | 0.01 | 0.33 | 0.008 |
| `PREV_NAME_PRODUCT_TYPE_NUNIQUE` | · | 635 | `NAME_PRODUCT_TYPE` | NUNIQUE | 5.4 | 0.0 | 2 | 1.64 | 3 | 0.028 |
| `PREV_CODE_REJECT_REASON_VERIF_RATE` | · | 642 | `CODE_REJECT_REASON` | RATE of "VERIF" | 5.4 | 0.0 | 0 | 0.00 | 0 | 6.80e-04 |
| `PREV_NAME_YIELD_GROUP_NUNIQUE` | · | 660 | `NAME_YIELD_GROUP` | NUNIQUE | 5.4 | 0.0 | 2 | 2.34 | 5 | -0.008 |
| `PREV_NAME_CONTRACT_STATUS_NUNIQUE` | · | 664 | `NAME_CONTRACT_STATUS` | NUNIQUE | 5.4 | 0.0 | 2 | 1.69 | 4 | 0.041 |
| `PREV_NAME_CLIENT_TYPE_NUNIQUE` | · | 677 | `NAME_CLIENT_TYPE` | NUNIQUE | 5.4 | 0.0 | 2 | 1.74 | 3 | 0.001 |
| `PREV_NAME_PAYMENT_TYPE_Non_cash_from_your_account_RATE` | · | 695 | `NAME_PAYMENT_TYPE` | RATE of "Non-cash from your account" | 5.4 | 0.0 | 0 | 0.01 | 0.20 | 2.50e-04 |
| `PREV_RATE_INTEREST_PRIMARY_MAX` | · | 735 | `RATE_INTEREST_PRIMARY` | MAX | 98.5 | 98.5 | 0.19 | 0.19 | 0.70 | -5.10e-04 |
| `PREV_NFLAG_INSURED_ON_APPROVAL_MAX` | · | 776 | `NFLAG_INSURED_ON_APPROVAL` | MAX | 5.8 | 5.8 | 1 | 0.53 | 1 | -0.011 |
| `PREV_DAYS_FIRST_DRAWING_SUM` | · | 777 | `DAYS_FIRST_DRAWING` | SUM | 5.4 | 0.0 | 0 | -183.16 | 0 | 5.50e-04 |
| `PREV_FLAG_LAST_APPL_PER_CONTRACT_N_RATE` | · | 786 | `FLAG_LAST_APPL_PER_CONTRACT` | RATE of "N" | 5.4 | 0.0 | 0 | 0.00 | 0.12 | 0.009 |
| `PREV_NFLAG_LAST_APPL_IN_DAY_MEAN` | · | 793 | `NFLAG_LAST_APPL_IN_DAY` | MEAN | 5.4 | 5.4 | 1 | 1.00 | 1 | -0.004 |
| `PREV_CODE_REJECT_REASON_XNA_RATE` | · | 796 | `CODE_REJECT_REASON` | RATE of "XNA" | 5.4 | 0.0 | 0 | 0.00 | 0.08 | 0.004 |
| `PREV_RATE_INTEREST_PRIMARY_SUM` | · | 798 | `RATE_INTEREST_PRIMARY` | SUM | 5.4 | 0.0 | 0 | 0.00 | 0.16 | -2.90e-04 |
| `PREV_RATE_INTEREST_PRIMARY_MIN` | · | 817 | `RATE_INTEREST_PRIMARY` | MIN | 98.5 | 98.5 | 0.19 | 0.19 | 0.70 | 2.00e-05 |
| `PREV_RATE_INTEREST_PRIVILEGED_MAX` | · | 829 | `RATE_INTEREST_PRIVILEGED` | MAX | 98.5 | 98.5 | 0.84 | 0.77 | 0.87 | 0.031 |
| `PREV_FLAG_LAST_APPL_PER_CONTRACT_Y_RATE` | · | 834 | `FLAG_LAST_APPL_PER_CONTRACT` | RATE of "Y" | 5.4 | 0.0 | 1 | 0.94 | 1 | -0.009 |
| `PREV_RATE_INTEREST_PRIMARY_MEAN` | · | 845 | `RATE_INTEREST_PRIMARY` | MEAN | 98.5 | 98.5 | 0.19 | 0.19 | 0.70 | -2.70e-04 |
| `PREV_RATE_INTEREST_PRIVILEGED_MEAN` | · | 846 | `RATE_INTEREST_PRIVILEGED` | MEAN | 98.5 | 98.5 | 0.84 | 0.77 | 0.87 | 0.031 |
| `PREV_NAME_CLIENT_TYPE_XNA_RATE` | · | 847 | `NAME_CLIENT_TYPE` | RATE of "XNA" | 5.4 | 0.0 | 0 | 8.92e-04 | 0 | 0.002 |
| `PREV_NFLAG_INSURED_ON_APPROVAL_MIN` | · | 855 | `NFLAG_INSURED_ON_APPROVAL` | MIN | 5.8 | 5.8 | 0 | 0.07 | 1 | 0.012 |
| `PREV_NAME_PAYMENT_TYPE_NUNIQUE` | · | 863 | `NAME_PAYMENT_TYPE` | NUNIQUE | 5.4 | 0.0 | 2 | 1.53 | 3 | 0.015 |
| `PREV_RATE_INTEREST_PRIVILEGED_MIN` | · | 868 | `RATE_INTEREST_PRIVILEGED` | MIN | 98.5 | 98.5 | 0.84 | 0.77 | 0.87 | 0.031 |
| `PREV_NFLAG_LAST_APPL_IN_DAY_MIN` | · | 887 | `NFLAG_LAST_APPL_IN_DAY` | MIN | 5.4 | 5.4 | 1 | 0.99 | 1 | -0.005 |
| `PREV_NFLAG_LAST_APPL_IN_DAY_MAX` | · | 888 | `NFLAG_LAST_APPL_IN_DAY` | MAX | 5.4 | 5.4 | 1 | 1.00 | 1 | -5.80e-04 |
| `PREV_RATE_INTEREST_PRIVILEGED_SUM` | · | 889 | `RATE_INTEREST_PRIVILEGED` | SUM | 5.4 | 0.0 | 0 | 0.01 | 0.72 | 2.20e-04 |
| `PREV_NAME_CONTRACT_TYPE_XNA_RATE` | · | 928 | `NAME_CONTRACT_TYPE` | RATE of "XNA" | 5.4 | 0.0 | 0 | 1.69e-04 | 0 | 0.009 |
| `PREV_FLAG_LAST_APPL_PER_CONTRACT_NUNIQUE` | · | 929 | `FLAG_LAST_APPL_PER_CONTRACT` | NUNIQUE | 5.4 | 0.0 | 1 | 0.96 | 2 | 0.007 |
| `PREV_NAME_PAYMENT_TYPE_Cashless_from_the_account_of_the_employer_RATE` | · | 930 | `NAME_PAYMENT_TYPE` | RATE of "Cashless from the account of the employer" | 5.4 | 0.0 | 0 | 6.66e-04 | 0 | 6.00e-05 |
| `PREV_CODE_REJECT_REASON_SYSTEM_RATE` | · | 931 | `CODE_REJECT_REASON` | RATE of "SYSTEM" | 5.4 | 0.0 | 0 | 3.18e-04 | 0 | -6.70e-04 |
| `PREV_NAME_TYPE_SUITE_Group_of_people_RATE` | · | 932 | `NAME_TYPE_SUITE` | RATE of "Group of people" | 20.1 | 0.0 | 0 | 0.00 | 0 | 0.002 |
| `PREV_NAME_PORTFOLIO_Cars_RATE` | · | 933 | `NAME_PORTFOLIO` | RATE of "Cars" | 5.4 | 0.0 | 0 | 3.02e-04 | 0 | -0.003 |
| `PREV_CHANNEL_TYPE_Car_dealer_RATE` | · | 934 | `CHANNEL_TYPE` | RATE of "Car dealer" | 5.4 | 0.0 | 0 | 3.20e-04 | 0 | -0.003 |
| `PREV_NAME_CONTRACT_STATUS_Approved_RATE` | · | — | `NAME_CONTRACT_STATUS` | RATE of "Approved" | 5.4 | 0.0 | 0.75 | 0.71 | 1 | -0.064 |
| `PREV_NAME_CONTRACT_STATUS_Refused_RATE` | · | — | `NAME_CONTRACT_STATUS` | RATE of "Refused" | 5.4 | 0.0 | 0 | 0.10 | 0.71 | 0.078 |
| `PREV_CODE_REJECT_REASON_CLIENT_RATE` | · | — | `CODE_REJECT_REASON` | RATE of "CLIENT" | 5.4 | 0.0 | 0 | 0.01 | 0.33 | 9.20e-04 |

### 7.4 Credit-card ledger via the previous-application hub (`PREV_CC_*`)

*332 features · 9 selected · 4.0% of model gain*

**Two-level roll-up.** `credit_card_balance` is monthly, keyed by `SK_ID_PREV`. It is first consolidated **per prior loan** (inner agg), joined onto that loan's row in `previous_application`, then rolled up **per customer** (outer agg). This equal-weights prior loans; aggregating the ledger straight to the customer would instead weight customers by how many monthly rows they happen to have.

**Definition template** — `PREV_CC_<COL>_<AGG1>_<AGG2>` = the *<agg2>*, across the customer's prior loans, of (the *<agg1>* of `<COL>` over that loan's monthly card records).

**Calculation** — `credit_card_balance.group_by('SK_ID_PREV').<agg1>(COL)` → join → `.group_by('SK_ID_CURR').<agg2>()`.

The biggest block (332 features) and the weakest per feature — only 9 selected, 4.0% of gain. Card history covers just ~25% of customers, and the four-agg cross product is highly redundant.

<details><summary><b>Source-column glossary</b> (22 columns)</summary>

| Source column | What it measures |
|---|---|
| `AMT_BALANCE` | Balance during the month of previous credit |
| `AMT_CREDIT_LIMIT_ACTUAL` | Credit card limit during the month of the previous credit |
| `AMT_DRAWINGS_ATM_CURRENT` | Amount drawing at ATM during the month of the previous credit |
| `AMT_DRAWINGS_CURRENT` | Amount drawing during the month of the previous credit |
| `AMT_DRAWINGS_OTHER_CURRENT` | Amount of other drawings during the month of the previous credit |
| `AMT_DRAWINGS_POS_CURRENT` | Amount drawing or buying goods during the month of the previous credit |
| `AMT_INST_MIN_REGULARITY` | Minimal installment for this month of the previous credit |
| `AMT_PAYMENT_CURRENT` | How much did the client pay during the month on the previous credit |
| `AMT_PAYMENT_TOTAL_CURRENT` | How much did the client pay during the month in total on the previous credit |
| `AMT_RECEIVABLE_PRINCIPAL` | Amount receivable for principal on the previous credit |
| `AMT_RECIVABLE` | Amount receivable on the previous credit |
| `AMT_TOTAL_RECEIVABLE` | Total amount receivable on the previous credit |
| `CC_OVERLIMIT_MONTHS` | the number of months the card balance exceeded the credit limit, computed over that loan's credit-card ledger |
| `CC_UTILIZATION_MEAN` | the average credit-card utilisation (AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL, months with a zero/negative limit excluded as null), computed over that loan's credit-card ledger |
| `CNT_DRAWINGS_ATM_CURRENT` | Number of drawings at ATM during this month on the previous credit |
| `CNT_DRAWINGS_CURRENT` | Number of drawings during this month on the previous credit |
| `CNT_DRAWINGS_OTHER_CURRENT` | Number of other drawings during this month on the previous credit |
| `CNT_DRAWINGS_POS_CURRENT` | Number of drawings for goods during this month on the previous credit |
| `CNT_INSTALMENT_MATURE_CUM` | Number of paid installments on the previous credit |
| `MONTHS_BALANCE` | Month of balance relative to application date (-1 means the freshest balance date) |
| `SK_DPD` | DPD (Days past due) during the month on the previous credit |
| `SK_DPD_DEF` | DPD (Days past due) during the month with tolerance (debts with low loan amounts are ignored) of the previous credit |

</details>

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MEAN_MEAN` | **✔** | 43 | `CNT_DRAWINGS_ATM_CURRENT` | MEAN → MEAN | 82.8 | 82.8 | 0.33 | 0.63 | 4 | 0.103 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MEAN_SUM` | **✔** | 75 | `CNT_DRAWINGS_ATM_CURRENT` | MEAN → SUM | 5.4 | 0.0 | 0 | 0.11 | 2.11 | 0.065 |
| `PREV_CC_UTILIZATION_MEAN_MAX` | **✔** | 85 | `CC_UTILIZATION_MEAN` | DOMAIN → MAX | 74.9 | 74.9 | 0.24 | 0.33 | 1.01 | 0.144 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MEAN_MAX` | **✔** | 91 | `CNT_DRAWINGS_ATM_CURRENT` | MEAN → MAX | 82.8 | 82.8 | 0.33 | 0.63 | 4 | 0.103 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MEAN_MIN` | **✔** | 105 | `CNT_DRAWINGS_ATM_CURRENT` | MEAN → MIN | 82.8 | 82.8 | 0.33 | 0.63 | 4 | 0.103 |
| `PREV_CC_UTILIZATION_MEAN_SUM` | **✔** | 114 | `CC_UTILIZATION_MEAN` | DOMAIN → SUM | 5.4 | 0.0 | 0 | 0.08 | 0.95 | 0.071 |
| `PREV_CC_UTILIZATION_MEAN_MEAN` | **✔** | 115 | `CC_UTILIZATION_MEAN` | DOMAIN → MEAN | 74.9 | 74.9 | 0.24 | 0.33 | 1.01 | 0.144 |
| `PREV_CC_OVERLIMIT_MONTHS_MEAN` | **✔** | 140 | `CC_OVERLIMIT_MONTHS` | DOMAIN → MEAN | 74.7 | 74.7 | 0 | 3.06 | 34 | 0.045 |
| `PREV_CC_UTILIZATION_MEAN_MIN` | **✔** | 197 | `CC_UTILIZATION_MEAN` | DOMAIN → MIN | 74.9 | 74.9 | 0.24 | 0.33 | 1.01 | 0.143 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MEAN_MEAN` | · | 222 | `CNT_DRAWINGS_CURRENT` | MEAN → MEAN | 74.7 | 74.7 | 0.27 | 1.70 | 18.64 | 0.081 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MEAN_MIN` | · | 236 | `CNT_DRAWINGS_CURRENT` | MEAN → MIN | 74.7 | 74.7 | 0.27 | 1.70 | 18.62 | 0.081 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MEAN_MAX` | · | 251 | `CNT_DRAWINGS_CURRENT` | MEAN → MAX | 74.7 | 74.7 | 0.27 | 1.71 | 18.67 | 0.081 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MAX_MEAN` | · | 276 | `CNT_DRAWINGS_CURRENT` | MAX → MEAN | 74.7 | 74.7 | 4 | 7.19 | 52 | 0.101 |
| `PREV_CC_OVERLIMIT_MONTHS_MAX` | · | 313 | `CC_OVERLIMIT_MONTHS` | DOMAIN → MAX | 74.7 | 74.7 | 0 | 3.07 | 34 | 0.046 |
| `PREV_CC_OVERLIMIT_MONTHS_MIN` | · | 328 | `CC_OVERLIMIT_MONTHS` | DOMAIN → MIN | 74.7 | 74.7 | 0 | 3.05 | 34 | 0.045 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MEAN_MEAN` | · | 374 | `AMT_DRAWINGS_ATM_CURRENT` | MEAN → MEAN | 82.8 | 82.8 | 6107.14 | 13,883 | 97,500 | 0.052 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MEAN_MEAN` | · | 386 | `AMT_PAYMENT_CURRENT` | MEAN → MEAN | 82.8 | 82.8 | 11,548 | 20,074 | 129,904 | -0.006 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MEAN_MIN` | · | 390 | `AMT_DRAWINGS_ATM_CURRENT` | MEAN → MIN | 82.8 | 82.8 | 6066.46 | 13,865 | 97,500 | 0.052 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN_MEAN` | · | 405 | `AMT_CREDIT_LIMIT_ACTUAL` | MEAN → MEAN | 74.7 | 74.7 | 159,808 | 222,543 | 900,000 | -0.019 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MAX_MIN` | · | 407 | `CNT_DRAWINGS_CURRENT` | MAX → MIN | 74.7 | 74.7 | 4 | 7.17 | 52 | 0.101 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MAX_MAX` | · | 411 | `CNT_DRAWINGS_CURRENT` | MAX → MAX | 74.7 | 74.7 | 4 | 7.20 | 53 | 0.101 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MAX_MEAN` | · | 417 | `AMT_PAYMENT_CURRENT` | MAX → MEAN | 82.8 | 82.8 | 67,500 | 125,164 | 778,275 | -0.008 |
| `PREV_CC_OVERLIMIT_MONTHS_SUM` | · | 418 | `CC_OVERLIMIT_MONTHS` | DOMAIN → SUM | 5.4 | 0.0 | 0 | 0.78 | 18 | 0.030 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MEAN_MIN` | · | 450 | `AMT_PAYMENT_CURRENT` | MEAN → MIN | 82.8 | 82.8 | 11,519 | 20,046 | 129,904 | -0.006 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MEAN_SUM` | · | 456 | `CNT_DRAWINGS_CURRENT` | MEAN → SUM | 5.4 | 0.0 | 0 | 0.43 | 9.67 | 0.048 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MAX_SUM` | · | 457 | `CNT_DRAWINGS_CURRENT` | MAX → SUM | 5.4 | 0.0 | 0 | 1.83 | 33 | 0.057 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MEAN_MEAN` | · | 468 | `CNT_DRAWINGS_POS_CURRENT` | MEAN → MEAN | 82.8 | 82.8 | 0.14 | 2.02 | 21.33 | 0.045 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_SUM_MEAN` | · | 469 | `CNT_DRAWINGS_CURRENT` | SUM → MEAN | 74.7 | 74.7 | 9 | 26.89 | 247 | 0.050 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_SUM_MEAN` | · | 475 | `AMT_CREDIT_LIMIT_ACTUAL` | SUM → MEAN | 74.7 | 74.7 | 3.60M | 5.35M | 19.12M | -0.045 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN_MIN` | · | 476 | `AMT_CREDIT_LIMIT_ACTUAL` | MEAN → MIN | 74.7 | 74.7 | 159,545 | 222,217 | 900,000 | -0.019 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN_MEAN` | · | 481 | `AMT_PAYMENT_TOTAL_CURRENT` | MEAN → MEAN | 74.7 | 74.7 | 4429.74 | 11,165 | 93,425 | 0.019 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MAX_MEAN` | · | 486 | `AMT_PAYMENT_TOTAL_CURRENT` | MAX → MEAN | 74.7 | 74.7 | 22,500 | 82,995 | 663,007 | 0.027 |
| `PREV_CC_SK_DPD_DEF_MEAN_MEAN` | · | 490 | `SK_DPD_DEF` | MEAN → MEAN | 74.7 | 74.7 | 0 | 0.03 | 0.58 | 0.022 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MAX_MEAN` | · | 491 | `AMT_DRAWINGS_ATM_CURRENT` | MAX → MEAN | 82.8 | 82.8 | 90,000 | 112,488 | 495,000 | 0.015 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_SUM_MEAN` | · | 493 | `AMT_DRAWINGS_ATM_CURRENT` | SUM → MEAN | 74.7 | 74.7 | 63,000 | 179,623 | 1.29M | 0.040 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MEAN_MAX` | · | 496 | `AMT_PAYMENT_CURRENT` | MEAN → MAX | 82.8 | 82.8 | 11,559 | 20,103 | 130,606 | -0.006 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN_MAX` | · | 497 | `AMT_PAYMENT_TOTAL_CURRENT` | MEAN → MAX | 74.7 | 74.7 | 4441.19 | 11,202 | 93,837 | 0.019 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN_MIN` | · | 499 | `AMT_PAYMENT_TOTAL_CURRENT` | MEAN → MIN | 74.7 | 74.7 | 4396.88 | 11,128 | 93,206 | 0.019 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_SUM_MEAN` | · | 501 | `CNT_DRAWINGS_ATM_CURRENT` | SUM → MEAN | 74.7 | 74.7 | 3 | 8.95 | 66 | 0.057 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MEAN_MAX` | · | 502 | `AMT_DRAWINGS_ATM_CURRENT` | MEAN → MAX | 82.8 | 82.8 | 6114.13 | 13,901 | 97,532 | 0.052 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MEAN_MIN` | · | 504 | `AMT_DRAWINGS_CURRENT` | MEAN → MIN | 74.7 | 74.7 | 4505.44 | 15,246 | 118,282 | 0.055 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MAX_MEAN` | · | 506 | `CNT_DRAWINGS_ATM_CURRENT` | MAX → MEAN | 82.8 | 82.8 | 3 | 4.13 | 17 | 0.062 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_SUM_MIN` | · | 508 | `AMT_CREDIT_LIMIT_ACTUAL` | SUM → MIN | 74.7 | 74.7 | 3.51M | 5.33M | 19.12M | -0.045 |
| `PREV_CC_AMT_PAYMENT_CURRENT_SUM_MEAN` | · | 514 | `AMT_PAYMENT_CURRENT` | SUM → MEAN | 74.7 | 74.7 | 119,525 | 289,376 | 1.94M | -0.003 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_SUM_MEAN` | · | 516 | `AMT_PAYMENT_TOTAL_CURRENT` | SUM → MEAN | 74.7 | 74.7 | 102,797 | 268,149 | 1.87M | -0.003 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MAX_MAX` | · | 517 | `AMT_PAYMENT_CURRENT` | MAX → MAX | 82.8 | 82.8 | 67,500 | 125,323 | 778,398 | -0.008 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MAX_MEAN` | · | 523 | `CNT_DRAWINGS_POS_CURRENT` | MAX → MEAN | 82.8 | 82.8 | 2 | 7.25 | 57 | 0.061 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MAX_MEAN` | · | 525 | `AMT_RECEIVABLE_PRINCIPAL` | MAX → MEAN | 74.7 | 74.7 | 90,000 | 141,107 | 764,852 | 0.067 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN_MAX` | · | 533 | `AMT_CREDIT_LIMIT_ACTUAL` | MEAN → MAX | 74.7 | 74.7 | 160,281 | 222,872 | 900,000 | -0.019 |
| `PREV_CC_AMT_BALANCE_MAX_MEAN` | · | 534 | `AMT_BALANCE` | MAX → MEAN | 74.7 | 74.7 | 96,468 | 147,696 | 802,031 | 0.069 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MAX_MIN` | · | 535 | `AMT_PAYMENT_CURRENT` | MAX → MIN | 82.8 | 82.8 | 67,455 | 125,006 | 778,275 | -0.008 |
| `PREV_CC_AMT_BALANCE_MIN_MEAN` | · | 536 | `AMT_BALANCE` | MIN → MEAN | 74.7 | 74.7 | 0 | 11,956 | 233,347 | 0.064 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MEAN_SUM` | · | 538 | `AMT_DRAWINGS_ATM_CURRENT` | MEAN → SUM | 5.4 | 0.0 | 0 | 2393.93 | 46,575 | 0.043 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MAX_MIN` | · | 542 | `CNT_DRAWINGS_ATM_CURRENT` | MAX → MIN | 82.8 | 82.8 | 3 | 4.13 | 17 | 0.062 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MAX_MEAN` | · | 544 | `AMT_DRAWINGS_POS_CURRENT` | MAX → MEAN | 82.8 | 82.8 | 14,216 | 48,597 | 402,913 | -0.020 |
| `PREV_CC_AMT_PAYMENT_CURRENT_SUM_MIN` | · | 545 | `AMT_PAYMENT_CURRENT` | SUM → MIN | 74.7 | 74.7 | 117,909 | 288,770 | 1.94M | -0.003 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MAX_MIN` | · | 546 | `CNT_DRAWINGS_POS_CURRENT` | MAX → MIN | 82.8 | 82.8 | 2 | 7.24 | 57 | 0.061 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MIN_MEAN` | · | 547 | `AMT_PAYMENT_CURRENT` | MIN → MEAN | 82.8 | 82.8 | 0 | 2454.79 | 28,350 | 0.009 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MEAN_MAX` | · | 549 | `AMT_DRAWINGS_CURRENT` | MEAN → MAX | 74.7 | 74.7 | 4609.43 | 15,354 | 119,061 | 0.055 |
| `PREV_CC_MONTHS_BALANCE_MEAN_MAX` | · | 550 | `MONTHS_BALANCE` | MEAN → MAX | 74.7 | 74.7 | -10 | -16.01 | -2 | 0.051 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_SUM_MIN` | · | 551 | `CNT_DRAWINGS_CURRENT` | SUM → MIN | 74.7 | 74.7 | 9 | 26.84 | 247 | 0.049 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MEAN_MEAN` | · | 556 | `AMT_DRAWINGS_CURRENT` | MEAN → MEAN | 74.7 | 74.7 | 4597.06 | 15,300 | 118,421 | 0.055 |
| `PREV_CC_AMT_RECIVABLE_MAX_MEAN` | · | 558 | `AMT_RECIVABLE` | MAX → MEAN | 74.7 | 74.7 | 95,749 | 146,538 | 796,590 | 0.069 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MAX_MEAN` | · | 560 | `AMT_INST_MIN_REGULARITY` | MAX → MEAN | 74.7 | 74.7 | 4633.74 | 7318.85 | 39,769 | 0.065 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_SUM_MIN` | · | 562 | `AMT_PAYMENT_TOTAL_CURRENT` | SUM → MIN | 74.7 | 74.7 | 101,484 | 267,581 | 1.87M | -0.003 |
| `PREV_CC_AMT_BALANCE_MAX_MAX` | · | 563 | `AMT_BALANCE` | MAX → MAX | 74.7 | 74.7 | 96,666 | 147,944 | 802,081 | 0.069 |
| `PREV_CC_MONTHS_BALANCE_SUM_MAX` | · | 565 | `MONTHS_BALANCE` | SUM → MAX | 74.7 | 74.7 | -171 | -885.24 | -3 | 0.047 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MAX_MAX` | · | 567 | `CNT_DRAWINGS_POS_CURRENT` | MAX → MAX | 82.8 | 82.8 | 2 | 7.26 | 57 | 0.061 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MEAN_MIN` | · | 568 | `CNT_DRAWINGS_POS_CURRENT` | MEAN → MIN | 82.8 | 82.8 | 0.13 | 2.02 | 21.33 | 0.045 |
| `PREV_CC_MONTHS_BALANCE_SUM_MEAN` | · | 569 | `MONTHS_BALANCE` | SUM → MEAN | 74.7 | 74.7 | -171 | -888.91 | -3 | 0.047 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_SUM_MAX` | · | 570 | `AMT_CREDIT_LIMIT_ACTUAL` | SUM → MAX | 74.7 | 74.7 | 3.60M | 5.36M | 19.12M | -0.045 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MEAN_MEAN` | · | 571 | `CNT_INSTALMENT_MATURE_CUM` | MEAN → MEAN | 74.7 | 74.7 | 3.11 | 8.00 | 43.15 | -0.017 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_SUM_MEAN` | · | 573 | `AMT_DRAWINGS_POS_CURRENT` | SUM → MEAN | 74.7 | 74.7 | 0 | 93,972 | 1.17M | -0.007 |
| `PREV_CC_AMT_BALANCE_MAX_SUM` | · | 575 | `AMT_BALANCE` | MAX → SUM | 5.4 | 0.0 | 0 | 37,538 | 515,053 | 0.041 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MEAN_MEAN` | · | 576 | `AMT_INST_MIN_REGULARITY` | MEAN → MEAN | 74.7 | 74.7 | 1714.58 | 3738.61 | 24,285 | 0.074 |
| `PREV_CC_AMT_BALANCE_MIN_MIN` | · | 578 | `AMT_BALANCE` | MIN → MIN | 74.7 | 74.7 | 0 | 11,921 | 233,347 | 0.064 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MAX_MEAN` | · | 580 | `AMT_DRAWINGS_CURRENT` | MAX → MEAN | 74.7 | 74.7 | 75,326 | 103,798 | 569,983 | 0.051 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MIN_MIN` | · | 581 | `AMT_PAYMENT_CURRENT` | MIN → MIN | 82.8 | 82.8 | 0 | 2448.93 | 28,190 | 0.009 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MEAN_MEAN` | · | 585 | `AMT_DRAWINGS_POS_CURRENT` | MEAN → MEAN | 82.8 | 82.8 | 1018.03 | 9589.33 | 93,902 | -0.013 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MEAN_MAX` | · | 586 | `CNT_DRAWINGS_POS_CURRENT` | MEAN → MAX | 82.8 | 82.8 | 0.14 | 2.03 | 21.33 | 0.045 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MAX_MAX` | · | 587 | `AMT_INST_MIN_REGULARITY` | MAX → MAX | 74.7 | 74.7 | 4651.20 | 7331.30 | 39,771 | 0.065 |
| `PREV_CC_AMT_BALANCE_MEAN_SUM` | · | 588 | `AMT_BALANCE` | MEAN → SUM | 5.4 | 0.0 | 0 | 19,214 | 345,143 | 0.050 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_SUM_MAX` | · | 589 | `CNT_DRAWINGS_ATM_CURRENT` | SUM → MAX | 74.7 | 74.7 | 3 | 8.97 | 66 | 0.057 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_SUM_MEAN` | · | 590 | `AMT_INST_MIN_REGULARITY` | SUM → MEAN | 74.7 | 74.7 | 30,654 | 109,123 | 718,102 | 0.011 |
| `PREV_CC_MONTHS_BALANCE_MEAN_MEAN` | · | 591 | `MONTHS_BALANCE` | MEAN → MEAN | 74.7 | 74.7 | -10 | -16.05 | -2 | 0.051 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MIN_MEAN` | · | 592 | `AMT_CREDIT_LIMIT_ACTUAL` | MIN → MEAN | 74.7 | 74.7 | 67,500 | 163,864 | 900,000 | -0.009 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MAX_MIN` | · | 596 | `AMT_DRAWINGS_ATM_CURRENT` | MAX → MIN | 82.8 | 82.8 | 90,000 | 112,399 | 495,000 | 0.015 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_SUM_MEAN` | · | 601 | `CNT_DRAWINGS_POS_CURRENT` | SUM → MEAN | 74.7 | 74.7 | 0 | 17.82 | 232 | 0.036 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_SUM_MAX` | · | 603 | `AMT_DRAWINGS_ATM_CURRENT` | SUM → MAX | 74.7 | 74.7 | 63,000 | 179,955 | 1.29M | 0.040 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_SUM_MEAN` | · | 604 | `AMT_DRAWINGS_CURRENT` | SUM → MEAN | 74.7 | 74.7 | 151,650 | 283,419 | 1.92M | 0.023 |
| `PREV_CC_AMT_RECIVABLE_MIN_MAX` | · | 606 | `AMT_RECIVABLE` | MIN → MAX | 74.7 | 74.7 | 0 | 11,606 | 228,227 | 0.063 |
| `PREV_CC_AMT_PAYMENT_CURRENT_SUM_MAX` | · | 608 | `AMT_PAYMENT_CURRENT` | SUM → MAX | 74.7 | 74.7 | 119,896 | 289,984 | 1.95M | -0.003 |
| `PREV_CC_AMT_RECIVABLE_MAX_SUM` | · | 609 | `AMT_RECIVABLE` | MAX → SUM | 5.4 | 0.0 | 0 | 37,243 | 511,812 | 0.041 |
| `PREV_CC_AMT_RECIVABLE_MAX_MAX` | · | 610 | `AMT_RECIVABLE` | MAX → MAX | 74.7 | 74.7 | 95,987 | 146,783 | 796,628 | 0.069 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_SUM_MAX` | · | 611 | `CNT_DRAWINGS_CURRENT` | SUM → MAX | 74.7 | 74.7 | 9 | 26.94 | 248 | 0.050 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MAX_MIN` | · | 612 | `AMT_DRAWINGS_CURRENT` | MAX → MIN | 74.7 | 74.7 | 74,727 | 103,584 | 569,876 | 0.051 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MEAN_MAX` | · | 613 | `AMT_RECEIVABLE_PRINCIPAL` | MEAN → MAX | 74.7 | 74.7 | 27,721 | 72,914 | 516,387 | 0.086 |
| `PREV_CC_AMT_RECIVABLE_MIN_MEAN` | · | 615 | `AMT_RECIVABLE` | MIN → MEAN | 74.7 | 74.7 | 0 | 11,571 | 228,136 | 0.063 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_SUM_MIN` | · | 616 | `CNT_DRAWINGS_ATM_CURRENT` | SUM → MIN | 74.7 | 74.7 | 3 | 8.93 | 66 | 0.057 |
| `PREV_CC_AMT_BALANCE_MEAN_MEAN` | · | 617 | `AMT_BALANCE` | MEAN → MEAN | 74.7 | 74.7 | 28,876 | 75,635 | 534,055 | 0.087 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_SUM_MEAN` | · | 620 | `CNT_INSTALMENT_MATURE_CUM` | SUM → MEAN | 74.7 | 74.7 | 36 | 479.20 | 3843 | -0.033 |
| `PREV_CC_AMT_BALANCE_MEAN_MAX` | · | 622 | `AMT_BALANCE` | MEAN → MAX | 74.7 | 74.7 | 28,986 | 75,762 | 534,055 | 0.087 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_SUM_MIN` | · | 623 | `AMT_DRAWINGS_ATM_CURRENT` | SUM → MIN | 74.7 | 74.7 | 60,750 | 179,292 | 1.29M | 0.040 |
| `PREV_CC_AMT_BALANCE_MEAN_MIN` | · | 624 | `AMT_BALANCE` | MEAN → MIN | 74.7 | 74.7 | 28,582 | 75,507 | 534,055 | 0.086 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MEAN_MIN` | · | 625 | `CNT_INSTALMENT_MATURE_CUM` | MEAN → MIN | 74.7 | 74.7 | 3.05 | 7.97 | 43.15 | -0.017 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MAX_MAX` | · | 626 | `AMT_DRAWINGS_POS_CURRENT` | MAX → MAX | 82.8 | 82.8 | 14,252 | 48,677 | 403,582 | -0.019 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MAX_MIN` | · | 628 | `AMT_DRAWINGS_POS_CURRENT` | MAX → MIN | 82.8 | 82.8 | 14,075 | 48,517 | 402,913 | -0.020 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_SUM_MAX` | · | 629 | `AMT_DRAWINGS_POS_CURRENT` | SUM → MAX | 74.7 | 74.7 | 0 | 94,194 | 1.17M | -0.007 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MEAN_MIN` | · | 630 | `AMT_DRAWINGS_POS_CURRENT` | MEAN → MIN | 82.8 | 82.8 | 1000 | 9564.22 | 93,850 | -0.013 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MIN_MIN` | · | 632 | `AMT_CREDIT_LIMIT_ACTUAL` | MIN → MIN | 74.7 | 74.7 | 67,500 | 163,509 | 900,000 | -0.009 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MAX_MIN` | · | 633 | `AMT_RECEIVABLE_PRINCIPAL` | MAX → MIN | 74.7 | 74.7 | 90,000 | 140,870 | 764,852 | 0.067 |
| `PREV_CC_SK_DPD_MEAN_MEAN` | · | 636 | `SK_DPD` | MEAN → MEAN | 74.7 | 74.7 | 0 | 3.14 | 24.29 | -0.005 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MAX_MAX` | · | 637 | `AMT_RECEIVABLE_PRINCIPAL` | MAX → MAX | 74.7 | 74.7 | 90,000 | 141,345 | 764,853 | 0.067 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MAX_MIN` | · | 638 | `AMT_INST_MIN_REGULARITY` | MAX → MIN | 74.7 | 74.7 | 4611.51 | 7306.42 | 39,769 | 0.065 |
| `PREV_CC_COUNT_MIN` | · | 639 | `–` | COUNT → MIN | 74.7 | 74.7 | 18 | 30.01 | 95 | -0.049 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_SUM_MAX` | · | 640 | `AMT_PAYMENT_TOTAL_CURRENT` | SUM → MAX | 74.7 | 74.7 | 103,196 | 268,718 | 1.87M | -0.003 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MAX_MAX` | · | 641 | `CNT_DRAWINGS_ATM_CURRENT` | MAX → MAX | 82.8 | 82.8 | 3 | 4.14 | 17 | 0.062 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MEAN_MIN` | · | 643 | `AMT_INST_MIN_REGULARITY` | MEAN → MIN | 74.7 | 74.7 | 1703.98 | 3732.80 | 24,285 | 0.074 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_SUM_MIN` | · | 644 | `AMT_DRAWINGS_CURRENT` | SUM → MIN | 74.7 | 74.7 | 150,750 | 282,925 | 1.92M | 0.023 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MAX_MAX` | · | 647 | `AMT_DRAWINGS_CURRENT` | MAX → MAX | 74.7 | 74.7 | 76,115 | 104,013 | 571,157 | 0.051 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN_SUM` | · | 648 | `AMT_CREDIT_LIMIT_ACTUAL` | MEAN → SUM | 5.4 | 0.0 | 0 | 56,597 | 765,000 | 0.006 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MAX_SUM` | · | 649 | `AMT_INST_MIN_REGULARITY` | MAX → SUM | 5.4 | 0.0 | 0 | 1860.14 | 25,545 | 0.040 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MAX_MEAN` | · | 650 | `AMT_CREDIT_LIMIT_ACTUAL` | MAX → MEAN | 74.7 | 74.7 | 180,000 | 262,791 | 900,000 | -0.021 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MAX_MIN` | · | 651 | `AMT_PAYMENT_TOTAL_CURRENT` | MAX → MIN | 74.7 | 74.7 | 22,500 | 82,764 | 663,007 | 0.027 |
| `PREV_CC_AMT_BALANCE_MAX_MIN` | · | 652 | `AMT_BALANCE` | MAX → MIN | 74.7 | 74.7 | 96,294 | 147,448 | 802,031 | 0.069 |
| `PREV_CC_MONTHS_BALANCE_MIN_MEAN` | · | 653 | `MONTHS_BALANCE` | MIN → MEAN | 74.7 | 74.7 | -18 | -30.60 | -2 | 0.050 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MAX_MEAN` | · | 654 | `CNT_INSTALMENT_MATURE_CUM` | MAX → MEAN | 74.7 | 74.7 | 6 | 13.40 | 81 | -0.007 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MIN_MAX` | · | 655 | `CNT_DRAWINGS_POS_CURRENT` | MIN → MAX | 82.8 | 82.8 | 0 | 0.19 | 5 | 0.015 |
| `PREV_CC_SK_DPD_DEF_MEAN_MIN` | · | 656 | `SK_DPD_DEF` | MEAN → MIN | 74.7 | 74.7 | 0 | 0.03 | 0.58 | 0.022 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MAX_MEAN` | · | 657 | `AMT_DRAWINGS_OTHER_CURRENT` | MAX → MEAN | 82.8 | 82.8 | 0 | 10,683 | 262,080 | 0.002 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MIN_MAX` | · | 659 | `AMT_CREDIT_LIMIT_ACTUAL` | MIN → MAX | 74.7 | 74.7 | 67,500 | 164,222 | 900,000 | -0.009 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_SUM_MIN` | · | 661 | `CNT_DRAWINGS_POS_CURRENT` | SUM → MIN | 74.7 | 74.7 | 0 | 17.78 | 232 | 0.036 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MIN_MIN` | · | 663 | `CNT_DRAWINGS_POS_CURRENT` | MIN → MIN | 82.8 | 82.8 | 0 | 0.19 | 5 | 0.015 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MEAN_SUM` | · | 666 | `AMT_PAYMENT_CURRENT` | MEAN → SUM | 5.4 | 0.0 | 0 | 3457.63 | 58,500 | 0.020 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MEAN_SUM` | · | 668 | `AMT_RECEIVABLE_PRINCIPAL` | MEAN → SUM | 5.4 | 0.0 | 0 | 18,492 | 333,067 | 0.050 |
| `PREV_CC_AMT_RECIVABLE_MAX_MIN` | · | 669 | `AMT_RECIVABLE` | MAX → MIN | 74.7 | 74.7 | 95,543 | 146,293 | 796,590 | 0.068 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_SUM_MAX` | · | 670 | `AMT_DRAWINGS_CURRENT` | SUM → MAX | 74.7 | 74.7 | 152,378 | 283,914 | 1.92M | 0.023 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MAX_SUM` | · | 672 | `AMT_RECEIVABLE_PRINCIPAL` | MAX → SUM | 5.4 | 0.0 | 0 | 35,863 | 494,835 | 0.041 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MIN_MAX` | · | 673 | `AMT_PAYMENT_CURRENT` | MIN → MAX | 82.8 | 82.8 | 0 | 2460.64 | 28,531 | 0.009 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MEAN_MIN` | · | 674 | `AMT_RECEIVABLE_PRINCIPAL` | MEAN → MIN | 74.7 | 74.7 | 27,291 | 72,668 | 516,387 | 0.085 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN_MEAN` | · | 675 | `CNT_DRAWINGS_OTHER_CURRENT` | MEAN → MEAN | 82.8 | 82.8 | 0 | 0.01 | 0.18 | 0.015 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_SUM_MIN` | · | 678 | `CNT_INSTALMENT_MATURE_CUM` | SUM → MIN | 74.7 | 74.7 | 35 | 477.27 | 3843 | -0.033 |
| `PREV_CC_AMT_PAYMENT_CURRENT_SUM_SUM` | · | 679 | `AMT_PAYMENT_CURRENT` | SUM → SUM | 5.4 | 0.0 | 0 | 73,557 | 1.20M | 0.007 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_SUM_MIN` | · | 681 | `AMT_DRAWINGS_POS_CURRENT` | SUM → MIN | 74.7 | 74.7 | 0 | 93,750 | 1.17M | -0.007 |
| `PREV_CC_COUNT_MEAN` | · | 682 | `–` | COUNT → MEAN | 74.7 | 74.7 | 18 | 30.09 | 95 | -0.049 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MAX_MIN` | · | 683 | `AMT_CREDIT_LIMIT_ACTUAL` | MAX → MIN | 74.7 | 74.7 | 180,000 | 262,405 | 900,000 | -0.021 |
| `PREV_CC_AMT_RECIVABLE_MEAN_MEAN` | · | 685 | `AMT_RECIVABLE` | MEAN → MEAN | 74.7 | 74.7 | 28,553 | 75,101 | 530,886 | 0.086 |
| `PREV_CC_MONTHS_BALANCE_MEAN_MIN` | · | 686 | `MONTHS_BALANCE` | MEAN → MIN | 74.7 | 74.7 | -10 | -16.10 | -2 | 0.051 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MAX_MAX` | · | 687 | `AMT_PAYMENT_TOTAL_CURRENT` | MAX → MAX | 74.7 | 74.7 | 22,500 | 83,225 | 664,551 | 0.027 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MEAN_MAX` | · | 688 | `AMT_INST_MIN_REGULARITY` | MEAN → MAX | 74.7 | 74.7 | 1719.29 | 3744.42 | 24,285 | 0.074 |
| `PREV_CC_MONTHS_BALANCE_MEAN_SUM` | · | 689 | `MONTHS_BALANCE` | MEAN → SUM | 5.4 | 0.0 | 0 | -4.08 | 0 | 0.007 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN_SUM` | · | 691 | `AMT_PAYMENT_TOTAL_CURRENT` | MEAN → SUM | 5.4 | 0.0 | 0 | 2841.54 | 49,349 | 0.017 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MIN_MIN` | · | 692 | `AMT_DRAWINGS_POS_CURRENT` | MIN → MIN | 82.8 | 82.8 | 0 | 705.19 | 15,968 | -0.007 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MAX_MAX` | · | 693 | `AMT_CREDIT_LIMIT_ACTUAL` | MAX → MAX | 74.7 | 74.7 | 180,000 | 263,178 | 900,000 | -0.021 |
| `PREV_CC_AMT_BALANCE_MIN_MAX` | · | 694 | `AMT_BALANCE` | MIN → MAX | 74.7 | 74.7 | 0 | 11,991 | 233,866 | 0.064 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_SUM_SUM` | · | 696 | `AMT_DRAWINGS_POS_CURRENT` | SUM → SUM | 5.4 | 0.0 | 0 | 23,888 | 537,258 | 0.001 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MEAN_MAX` | · | 697 | `CNT_INSTALMENT_MATURE_CUM` | MEAN → MAX | 74.7 | 74.7 | 3.11 | 8.02 | 43.17 | -0.016 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_SUM_MIN` | · | 698 | `AMT_INST_MIN_REGULARITY` | SUM → MIN | 74.7 | 74.7 | 30,345 | 108,919 | 718,102 | 0.011 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MAX_MAX` | · | 699 | `CNT_INSTALMENT_MATURE_CUM` | MAX → MAX | 74.7 | 74.7 | 6 | 13.43 | 81 | -0.007 |
| `PREV_CC_AMT_RECIVABLE_MEAN_MAX` | · | 702 | `AMT_RECIVABLE` | MEAN → MAX | 74.7 | 74.7 | 28,659 | 75,228 | 530,886 | 0.086 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MEAN_MEAN` | · | 703 | `AMT_RECEIVABLE_PRINCIPAL` | MEAN → MEAN | 74.7 | 74.7 | 27,590 | 72,791 | 516,387 | 0.085 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MAX_MAX` | · | 704 | `AMT_DRAWINGS_ATM_CURRENT` | MAX → MAX | 82.8 | 82.8 | 90,000 | 112,578 | 495,000 | 0.015 |
| `PREV_CC_AMT_BALANCE_SUM_MAX` | · | 705 | `AMT_BALANCE` | SUM → MAX | 74.7 | 74.7 | 622,693 | 2.04M | 13.69M | 0.024 |
| `PREV_CC_AMT_RECIVABLE_MEAN_MIN` | · | 706 | `AMT_RECIVABLE` | MEAN → MIN | 74.7 | 74.7 | 28,286 | 74,975 | 530,886 | 0.086 |
| `PREV_CC_AMT_BALANCE_SUM_MIN` | · | 708 | `AMT_BALANCE` | SUM → MIN | 74.7 | 74.7 | 614,027 | 2.04M | 13.69M | 0.024 |
| `PREV_CC_MONTHS_BALANCE_MAX_MEAN` | · | 709 | `MONTHS_BALANCE` | MAX → MEAN | 74.7 | 74.7 | -1 | -1.50 | -1 | 0.029 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MEAN_SUM` | · | 711 | `AMT_DRAWINGS_CURRENT` | MEAN → SUM | 5.4 | 0.0 | 0 | 3894.04 | 67,500 | 0.035 |
| `PREV_CC_AMT_RECIVABLE_MIN_MIN` | · | 712 | `AMT_RECIVABLE` | MIN → MIN | 74.7 | 74.7 | 0 | 11,536 | 228,136 | 0.063 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_SUM_MEAN` | · | 714 | `AMT_RECEIVABLE_PRINCIPAL` | SUM → MEAN | 74.7 | 74.7 | 594,048 | 1.96M | 13.16M | 0.024 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_SUM_MEAN` | · | 715 | `AMT_DRAWINGS_OTHER_CURRENT` | SUM → MEAN | 74.7 | 74.7 | 0 | 9292.41 | 257,400 | 0.008 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MAX_SUM` | · | 717 | `CNT_DRAWINGS_ATM_CURRENT` | MAX → SUM | 5.4 | 0.0 | 0 | 0.71 | 10 | 0.052 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MAX_SUM` | · | 720 | `CNT_DRAWINGS_POS_CURRENT` | MAX → SUM | 5.4 | 0.0 | 0 | 1.25 | 30 | 0.046 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_SUM_MAX` | · | 721 | `AMT_INST_MIN_REGULARITY` | SUM → MAX | 74.7 | 74.7 | 30,744 | 109,328 | 718,102 | 0.011 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_SUM_MIN` | · | 722 | `AMT_RECEIVABLE_PRINCIPAL` | SUM → MIN | 74.7 | 74.7 | 587,484 | 1.96M | 13.16M | 0.024 |
| `PREV_CC_MONTHS_BALANCE_SUM_MIN` | · | 723 | `MONTHS_BALANCE` | SUM → MIN | 74.7 | 74.7 | -171 | -892.60 | -3 | 0.047 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MIN_MEAN` | · | 724 | `CNT_DRAWINGS_POS_CURRENT` | MIN → MEAN | 82.8 | 82.8 | 0 | 0.19 | 5 | 0.015 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MIN_SUM` | · | 725 | `AMT_CREDIT_LIMIT_ACTUAL` | MIN → SUM | 5.4 | 0.0 | 0 | 41,673 | 765,000 | 0.006 |
| `PREV_CC_AMT_BALANCE_SUM_MEAN` | · | 726 | `AMT_BALANCE` | SUM → MEAN | 74.7 | 74.7 | 620,479 | 2.04M | 13.69M | 0.024 |
| `PREV_CC_SK_DPD_DEF_MEAN_MAX` | · | 727 | `SK_DPD_DEF` | MEAN → MAX | 74.7 | 74.7 | 0 | 0.03 | 0.58 | 0.022 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MAX_MIN` | · | 728 | `AMT_DRAWINGS_OTHER_CURRENT` | MAX → MIN | 82.8 | 82.8 | 0 | 10,660 | 262,080 | 0.002 |
| `PREV_CC_COUNT_SUM` | · | 729 | `–` | COUNT → SUM | 5.4 | 0.0 | 0 | 7.66 | 91 | -0.007 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_SUM_SUM` | · | 730 | `CNT_DRAWINGS_ATM_CURRENT` | SUM → SUM | 5.4 | 0.0 | 0 | 2.28 | 41 | 0.036 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_MAX_SUM` | · | 731 | `AMT_CREDIT_LIMIT_ACTUAL` | MAX → SUM | 5.4 | 0.0 | 0 | 66,851 | 765,000 | 0.006 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_SUM_SUM` | · | 732 | `AMT_DRAWINGS_CURRENT` | SUM → SUM | 5.4 | 0.0 | 0 | 72,028 | 1.15M | 0.020 |
| `PREV_CC_MONTHS_BALANCE_MIN_MIN` | · | 734 | `MONTHS_BALANCE` | MIN → MIN | 74.7 | 74.7 | -18 | -30.69 | -2 | 0.050 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MAX_MIN` | · | 736 | `AMT_TOTAL_RECEIVABLE` | MAX → MIN | 74.7 | 74.7 | 95,573 | 146,300 | 796,590 | 0.068 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MEAN_SUM` | · | 737 | `CNT_DRAWINGS_POS_CURRENT` | MEAN → SUM | 5.4 | 0.0 | 0 | 0.35 | 9.36 | 0.037 |
| `PREV_CC_MONTHS_BALANCE_MAX_SUM` | · | 738 | `MONTHS_BALANCE` | MAX → SUM | 5.4 | 0.0 | 0 | -0.38 | 0 | -0.009 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MEAN_SUM` | · | 739 | `CNT_INSTALMENT_MATURE_CUM` | MEAN → SUM | 5.4 | 0.0 | 0 | 2.03 | 37.30 | 0.002 |
| `PREV_CC_AMT_CREDIT_LIMIT_ACTUAL_SUM_SUM` | · | 740 | `AMT_CREDIT_LIMIT_ACTUAL` | SUM → SUM | 5.4 | 0.0 | 0 | 1.36M | 15.75M | -0.005 |
| `PREV_CC_MONTHS_BALANCE_SUM_SUM` | · | 741 | `MONTHS_BALANCE` | SUM → SUM | 5.4 | 0.0 | 0 | -226.27 | 0 | 0.013 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_SUM_SUM` | · | 742 | `AMT_PAYMENT_TOTAL_CURRENT` | SUM → SUM | 5.4 | 0.0 | 0 | 68,164 | 1.14M | 0.007 |
| `PREV_CC_COUNT_MAX` | · | 743 | `–` | COUNT → MAX | 74.7 | 74.7 | 18 | 30.18 | 95 | -0.049 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_SUM_MAX` | · | 744 | `CNT_DRAWINGS_POS_CURRENT` | SUM → MAX | 74.7 | 74.7 | 0 | 17.86 | 232 | 0.036 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MEAN_MAX` | · | 745 | `AMT_DRAWINGS_POS_CURRENT` | MEAN → MAX | 82.8 | 82.8 | 1019.29 | 9614.45 | 94,269 | -0.013 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MAX_SUM` | · | 746 | `AMT_DRAWINGS_POS_CURRENT` | MAX → SUM | 5.4 | 0.0 | 0 | 8386.52 | 180,477 | 0.011 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MEAN_SUM` | · | 747 | `AMT_TOTAL_RECEIVABLE` | MEAN → SUM | 5.4 | 0.0 | 0 | 19,079 | 343,386 | 0.050 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MIN_MIN` | · | 748 | `AMT_RECEIVABLE_PRINCIPAL` | MIN → MIN | 74.7 | 74.7 | 0 | 11,436 | 225,000 | 0.063 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MAX_SUM` | · | 749 | `AMT_TOTAL_RECEIVABLE` | MAX → SUM | 5.4 | 0.0 | 0 | 37,245 | 511,812 | 0.041 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MAX_SUM` | · | 752 | `AMT_DRAWINGS_ATM_CURRENT` | MAX → SUM | 5.4 | 0.0 | 0 | 19,397 | 297,000 | 0.035 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MAX_SUM` | · | 753 | `AMT_PAYMENT_TOTAL_CURRENT` | MAX → SUM | 5.4 | 0.0 | 0 | 21,117 | 443,718 | 0.022 |
| `PREV_CC_AMT_RECIVABLE_SUM_MAX` | · | 754 | `AMT_RECIVABLE` | SUM → MAX | 74.7 | 74.7 | 613,086 | 2.03M | 13.67M | 0.024 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MAX_MIN` | · | 755 | `CNT_INSTALMENT_MATURE_CUM` | MAX → MIN | 74.7 | 74.7 | 6 | 13.37 | 81 | -0.007 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MEAN_SUM` | · | 757 | `AMT_DRAWINGS_POS_CURRENT` | MEAN → SUM | 5.4 | 0.0 | 0 | 1655.84 | 38,312 | 0.009 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MIN_MEAN` | · | 760 | `AMT_RECEIVABLE_PRINCIPAL` | MIN → MEAN | 74.7 | 74.7 | 0 | 11,469 | 225,000 | 0.063 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_SUM_MAX` | · | 761 | `CNT_INSTALMENT_MATURE_CUM` | SUM → MAX | 74.7 | 74.7 | 36 | 481.13 | 3844 | -0.033 |
| `PREV_CC_SK_DPD_SUM_MIN` | · | 762 | `SK_DPD` | SUM → MIN | 74.7 | 74.7 | 0 | 246.26 | 1108 | -0.005 |
| `PREV_CC_AMT_RECIVABLE_SUM_MEAN` | · | 763 | `AMT_RECIVABLE` | SUM → MEAN | 74.7 | 74.7 | 611,274 | 2.03M | 13.67M | 0.024 |
| `PREV_CC_AMT_RECIVABLE_MIN_SUM` | · | 764 | `AMT_RECIVABLE` | MIN → SUM | 5.4 | 0.0 | 0 | 2940.88 | 105,750 | 0.037 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MAX_SUM` | · | 765 | `AMT_DRAWINGS_CURRENT` | MAX → SUM | 5.4 | 0.0 | 0 | 26,397 | 389,619 | 0.033 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MAX_SUM` | · | 768 | `AMT_PAYMENT_CURRENT` | MAX → SUM | 5.4 | 0.0 | 0 | 21,560 | 450,000 | 0.022 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_MIN_SUM` | · | 769 | `CNT_DRAWINGS_POS_CURRENT` | MIN → SUM | 5.4 | 0.0 | 0 | 0.03 | 0 | 0.012 |
| `PREV_CC_SK_DPD_SUM_MEAN` | · | 770 | `SK_DPD` | SUM → MEAN | 74.7 | 74.7 | 0 | 246.41 | 1113 | -0.005 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MEAN_SUM` | · | 771 | `AMT_INST_MIN_REGULARITY` | MEAN → SUM | 5.4 | 0.0 | 0 | 949.67 | 16,689 | 0.044 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_SUM_MAX` | · | 772 | `AMT_RECEIVABLE_PRINCIPAL` | SUM → MAX | 74.7 | 74.7 | 595,984 | 1.96M | 13.16M | 0.024 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_SUM_SUM` | · | 773 | `AMT_DRAWINGS_ATM_CURRENT` | SUM → SUM | 5.4 | 0.0 | 0 | 45,641 | 819,000 | 0.028 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MIN_MEAN` | · | 774 | `AMT_DRAWINGS_POS_CURRENT` | MIN → MEAN | 82.8 | 82.8 | 0 | 709.68 | 16,034 | -0.007 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN_MEAN` | · | 775 | `AMT_DRAWINGS_OTHER_CURRENT` | MEAN → MEAN | 82.8 | 82.8 | 0 | 695.94 | 18,562 | 0.007 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_SUM_SUM` | · | 778 | `AMT_INST_MIN_REGULARITY` | SUM → SUM | 5.4 | 0.0 | 0 | 27,722 | 518,201 | 0.014 |
| `PREV_CC_AMT_RECIVABLE_MEAN_SUM` | · | 779 | `AMT_RECIVABLE` | MEAN → SUM | 5.4 | 0.0 | 0 | 19,078 | 343,386 | 0.050 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MAX_MAX` | · | 781 | `AMT_TOTAL_RECEIVABLE` | MAX → MAX | 74.7 | 74.7 | 96,021 | 146,790 | 796,628 | 0.069 |
| `PREV_CC_SK_DPD_MEAN_MIN` | · | 782 | `SK_DPD` | MEAN → MIN | 74.7 | 74.7 | 0 | 3.14 | 24.27 | -0.005 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_SUM_MIN` | · | 783 | `AMT_DRAWINGS_OTHER_CURRENT` | SUM → MIN | 74.7 | 74.7 | 0 | 9268.03 | 257,400 | 0.008 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MEAN_MIN` | · | 785 | `AMT_TOTAL_RECEIVABLE` | MEAN → MIN | 74.7 | 74.7 | 28,286 | 74,975 | 530,886 | 0.086 |
| `PREV_CC_MONTHS_BALANCE_MIN_MAX` | · | 787 | `MONTHS_BALANCE` | MIN → MAX | 74.7 | 74.7 | -18 | -30.51 | -2 | 0.050 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MAX_SUM` | · | 788 | `CNT_INSTALMENT_MATURE_CUM` | MAX → SUM | 5.4 | 0.0 | 0 | 3.41 | 62 | 0.007 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MAX_MAX` | · | 789 | `AMT_DRAWINGS_OTHER_CURRENT` | MAX → MAX | 82.8 | 82.8 | 0 | 10,707 | 266,760 | 0.002 |
| `PREV_CC_CNT_DRAWINGS_POS_CURRENT_SUM_SUM` | · | 790 | `CNT_DRAWINGS_POS_CURRENT` | SUM → SUM | 5.4 | 0.0 | 0 | 4.53 | 113 | 0.024 |
| `PREV_CC_AMT_RECIVABLE_SUM_MIN` | · | 791 | `AMT_RECIVABLE` | SUM → MIN | 74.7 | 74.7 | 603,913 | 2.03M | 13.67M | 0.023 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MIN_MEAN` | · | 792 | `AMT_INST_MIN_REGULARITY` | MIN → MEAN | 74.7 | 74.7 | 0 | 258.16 | 7746.48 | 0.002 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_SUM_SUM` | · | 794 | `AMT_RECEIVABLE_PRINCIPAL` | SUM → SUM | 5.4 | 0.0 | 0 | 497,679 | 9.42M | 0.020 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MIN_MAX` | · | 795 | `AMT_RECEIVABLE_PRINCIPAL` | MIN → MAX | 74.7 | 74.7 | 0 | 11,503 | 225,000 | 0.063 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MAX_MEAN` | · | 797 | `CNT_DRAWINGS_OTHER_CURRENT` | MAX → MEAN | 82.8 | 82.8 | 0 | 0.11 | 2 | 0.003 |
| `PREV_CC_MONTHS_BALANCE_MIN_SUM` | · | 799 | `MONTHS_BALANCE` | MIN → SUM | 5.4 | 0.0 | 0 | -7.79 | 0 | 0.007 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_SUM_SUM` | · | 800 | `CNT_INSTALMENT_MATURE_CUM` | SUM → SUM | 5.4 | 0.0 | 0 | 121.95 | 3081 | -0.009 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MIN_MAX` | · | 801 | `AMT_PAYMENT_TOTAL_CURRENT` | MIN → MAX | 74.7 | 74.7 | 0 | 253.77 | 4500 | 0.002 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_SUM_MAX` | · | 803 | `AMT_DRAWINGS_OTHER_CURRENT` | SUM → MAX | 74.7 | 74.7 | 0 | 9316.80 | 257,400 | 0.008 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN_MIN` | · | 804 | `AMT_DRAWINGS_OTHER_CURRENT` | MEAN → MIN | 82.8 | 82.8 | 0 | 694.33 | 18,554 | 0.007 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MIN_MEAN` | · | 805 | `CNT_DRAWINGS_CURRENT` | MIN → MEAN | 74.7 | 74.7 | 0 | 0.11 | 3 | 0.024 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MIN_MAX` | · | 806 | `AMT_DRAWINGS_CURRENT` | MIN → MAX | 74.7 | 74.7 | 0 | 683.46 | 13,077 | 0.009 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN_SUM` | · | 808 | `CNT_DRAWINGS_OTHER_CURRENT` | MEAN → SUM | 5.4 | 0.0 | 0 | 0.00 | 0.02 | 0.013 |
| `PREV_CC_AMT_RECEIVABLE_PRINCIPAL_MIN_SUM` | · | 809 | `AMT_RECEIVABLE_PRINCIPAL` | MIN → SUM | 5.4 | 0.0 | 0 | 2915.26 | 103,500 | 0.037 |
| `PREV_CC_SK_DPD_MAX_MEAN` | · | 810 | `SK_DPD` | MAX → MEAN | 74.7 | 74.7 | 0 | 12.44 | 244 | -0.007 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MIN_MAX` | · | 811 | `AMT_DRAWINGS_POS_CURRENT` | MIN → MAX | 82.8 | 82.8 | 0 | 714.18 | 16,116 | -0.008 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN_MIN` | · | 812 | `CNT_DRAWINGS_OTHER_CURRENT` | MEAN → MIN | 82.8 | 82.8 | 0 | 0.01 | 0.18 | 0.015 |
| `PREV_CC_MONTHS_BALANCE_MAX_MIN` | · | 813 | `MONTHS_BALANCE` | MAX → MIN | 74.7 | 74.7 | -1 | -1.50 | -1 | 0.029 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_SUM_MIN` | · | 814 | `CNT_DRAWINGS_OTHER_CURRENT` | SUM → MIN | 74.7 | 74.7 | 0 | 0.12 | 3 | 0.007 |
| `PREV_CC_SK_DPD_MEAN_MAX` | · | 815 | `SK_DPD` | MEAN → MAX | 74.7 | 74.7 | 0 | 3.14 | 24.83 | -0.005 |
| `PREV_CC_SK_DPD_SUM_MAX` | · | 816 | `SK_DPD` | SUM → MAX | 74.7 | 74.7 | 0 | 246.55 | 1113 | -0.005 |
| `PREV_CC_SK_DPD_DEF_MAX_MEAN` | · | 818 | `SK_DPD_DEF` | MAX → MEAN | 74.7 | 74.7 | 0 | 0.45 | 8 | 0.016 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_SUM_SUM` | · | 819 | `CNT_DRAWINGS_CURRENT` | SUM → SUM | 5.4 | 0.0 | 0 | 6.83 | 129 | 0.032 |
| `PREV_CC_AMT_RECIVABLE_SUM_SUM` | · | 822 | `AMT_RECIVABLE` | SUM → SUM | 5.4 | 0.0 | 0 | 515,948 | 9.77M | 0.020 |
| `PREV_CC_AMT_BALANCE_SUM_SUM` | · | 823 | `AMT_BALANCE` | SUM → SUM | 5.4 | 0.0 | 0 | 518,117 | 9.79M | 0.020 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MIN_MIN` | · | 824 | `CNT_DRAWINGS_CURRENT` | MIN → MIN | 74.7 | 74.7 | 0 | 0.11 | 3 | 0.024 |
| `PREV_CC_SK_DPD_MAX_MIN` | · | 825 | `SK_DPD` | MAX → MIN | 74.7 | 74.7 | 0 | 12.42 | 244 | -0.007 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_SUM_MIN` | · | 826 | `AMT_TOTAL_RECEIVABLE` | SUM → MIN | 74.7 | 74.7 | 603,913 | 2.03M | 13.67M | 0.023 |
| `PREV_CC_SK_DPD_MAX_MAX` | · | 827 | `SK_DPD` | MAX → MAX | 74.7 | 74.7 | 0 | 12.46 | 245 | -0.007 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MIN_MIN` | · | 828 | `AMT_INST_MIN_REGULARITY` | MIN → MIN | 74.7 | 74.7 | 0 | 258.12 | 7746.48 | 0.002 |
| `PREV_CC_SK_DPD_DEF_SUM_MEAN` | · | 830 | `SK_DPD_DEF` | SUM → MEAN | 74.7 | 74.7 | 0 | 1.31 | 23 | 0.009 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MAX_SUM` | · | 835 | `AMT_DRAWINGS_OTHER_CURRENT` | MAX → SUM | 5.4 | 0.0 | 0 | 1843.03 | 46,350 | 0.009 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_SUM_MEAN` | · | 836 | `CNT_DRAWINGS_OTHER_CURRENT` | SUM → MEAN | 74.7 | 74.7 | 0 | 0.12 | 3 | 0.007 |
| `PREV_CC_SK_DPD_MEAN_SUM` | · | 837 | `SK_DPD` | MEAN → SUM | 5.4 | 0.0 | 0 | 0.80 | 0.64 | -0.001 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MIN_MEAN` | · | 838 | `AMT_DRAWINGS_ATM_CURRENT` | MIN → MEAN | 82.8 | 82.8 | 0 | 748.84 | 9000 | 0.012 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MIN_MEAN` | · | 839 | `AMT_DRAWINGS_CURRENT` | MIN → MEAN | 74.7 | 74.7 | 0 | 671.72 | 12,946 | 0.010 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MIN_SUM` | · | 841 | `AMT_INST_MIN_REGULARITY` | MIN → SUM | 5.4 | 0.0 | 0 | 65.44 | 2250 | 0.004 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN_MAX` | · | 842 | `AMT_DRAWINGS_OTHER_CURRENT` | MEAN → MAX | 82.8 | 82.8 | 0 | 697.54 | 18,562 | 0.007 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_SUM_MAX` | · | 843 | `AMT_TOTAL_RECEIVABLE` | SUM → MAX | 74.7 | 74.7 | 613,086 | 2.03M | 13.67M | 0.024 |
| `PREV_CC_SK_DPD_SUM_SUM` | · | 844 | `SK_DPD` | SUM → SUM | 5.4 | 0.0 | 0 | 62.49 | 21 | -0.002 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MEAN_MEAN` | · | 848 | `AMT_TOTAL_RECEIVABLE` | MEAN → MEAN | 74.7 | 74.7 | 28,553 | 75,102 | 530,886 | 0.086 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_SUM_SUM` | · | 849 | `AMT_DRAWINGS_OTHER_CURRENT` | SUM → SUM | 5.4 | 0.0 | 0 | 2361.46 | 52,613 | 0.007 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MAX_MEAN` | · | 850 | `AMT_TOTAL_RECEIVABLE` | MAX → MEAN | 74.7 | 74.7 | 95,779 | 146,545 | 796,590 | 0.069 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MIN_MEAN` | · | 851 | `AMT_PAYMENT_TOTAL_CURRENT` | MIN → MEAN | 74.7 | 74.7 | 0 | 250.54 | 4500 | 0.002 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MIN_SUM` | · | 852 | `AMT_DRAWINGS_CURRENT` | MIN → SUM | 5.4 | 0.0 | 0 | 173.21 | 0 | 0.006 |
| `PREV_CC_AMT_BALANCE_MIN_SUM` | · | 853 | `AMT_BALANCE` | MIN → SUM | 5.4 | 0.0 | 0 | 3038.91 | 108,458 | 0.038 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MIN_MAX` | · | 854 | `CNT_DRAWINGS_CURRENT` | MIN → MAX | 74.7 | 74.7 | 0 | 0.12 | 3 | 0.024 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MEAN_MAX` | · | 857 | `AMT_TOTAL_RECEIVABLE` | MEAN → MAX | 74.7 | 74.7 | 28,659 | 75,229 | 530,886 | 0.086 |
| `PREV_CC_AMT_PAYMENT_CURRENT_MIN_SUM` | · | 858 | `AMT_PAYMENT_CURRENT` | MIN → SUM | 5.4 | 0.0 | 0 | 422.83 | 10,800 | 0.012 |
| `PREV_CC_AMT_DRAWINGS_POS_CURRENT_MIN_SUM` | · | 859 | `AMT_DRAWINGS_POS_CURRENT` | MIN → SUM | 5.4 | 0.0 | 0 | 122.93 | 0 | -2.20e-04 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MIN_MIN` | · | 860 | `AMT_PAYMENT_TOTAL_CURRENT` | MIN → MIN | 74.7 | 74.7 | 0 | 247.31 | 4500 | 0.002 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN_SUM` | · | 862 | `AMT_DRAWINGS_OTHER_CURRENT` | MEAN → SUM | 5.4 | 0.0 | 0 | 120.07 | 1063.64 | 0.009 |
| `PREV_CC_CNT_DRAWINGS_CURRENT_MIN_SUM` | · | 864 | `CNT_DRAWINGS_CURRENT` | MIN → SUM | 5.4 | 0.0 | 0 | 0.03 | 0 | 0.014 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MAX_MAX` | · | 865 | `CNT_DRAWINGS_OTHER_CURRENT` | MAX → MAX | 82.8 | 82.8 | 0 | 0.11 | 2 | 0.003 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN_MAX` | · | 867 | `CNT_DRAWINGS_OTHER_CURRENT` | MEAN → MAX | 82.8 | 82.8 | 0 | 0.01 | 0.18 | 0.015 |
| `PREV_CC_AMT_INST_MIN_REGULARITY_MIN_MAX` | · | 869 | `AMT_INST_MIN_REGULARITY` | MIN → MAX | 74.7 | 74.7 | 0 | 258.20 | 7746.48 | 0.001 |
| `PREV_CC_AMT_PAYMENT_TOTAL_CURRENT_MIN_SUM` | · | 870 | `AMT_PAYMENT_TOTAL_CURRENT` | MIN → SUM | 5.4 | 0.0 | 0 | 64.31 | 0 | 0.002 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_SUM_MAX` | · | 871 | `CNT_DRAWINGS_OTHER_CURRENT` | SUM → MAX | 74.7 | 74.7 | 0 | 0.12 | 3 | 0.007 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MIN_MEAN` | · | 872 | `CNT_DRAWINGS_ATM_CURRENT` | MIN → MEAN | 82.8 | 82.8 | 0 | 0.03 | 1 | 0.022 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MIN_MIN` | · | 890 | `AMT_DRAWINGS_ATM_CURRENT` | MIN → MIN | 82.8 | 82.8 | 0 | 747.03 | 9000 | 0.012 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MIN_MAX` | · | 891 | `AMT_DRAWINGS_ATM_CURRENT` | MIN → MAX | 82.8 | 82.8 | 0 | 750.64 | 9000 | 0.012 |
| `PREV_CC_AMT_DRAWINGS_ATM_CURRENT_MIN_SUM` | · | 892 | `AMT_DRAWINGS_ATM_CURRENT` | MIN → SUM | 5.4 | 0.0 | 0 | 129.20 | 0 | 0.008 |
| `PREV_CC_AMT_DRAWINGS_CURRENT_MIN_MIN` | · | 893 | `AMT_DRAWINGS_CURRENT` | MIN → MIN | 74.7 | 74.7 | 0 | 659.98 | 12,822 | 0.010 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MIN_MEAN` | · | 894 | `AMT_DRAWINGS_OTHER_CURRENT` | MIN → MEAN | 82.8 | 82.8 | 0 | 6.47 | 0 | -0.003 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MIN_SUM` | · | 895 | `AMT_DRAWINGS_OTHER_CURRENT` | MIN → SUM | 5.4 | 0.0 | 0 | 1.11 | 0 | -0.001 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_SUM_MEAN` | · | 896 | `AMT_TOTAL_RECEIVABLE` | SUM → MEAN | 74.7 | 74.7 | 611,274 | 2.03M | 13.67M | 0.024 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_SUM_SUM` | · | 897 | `AMT_TOTAL_RECEIVABLE` | SUM → SUM | 5.4 | 0.0 | 0 | 515,964 | 9.77M | 0.020 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MIN_MIN` | · | 898 | `CNT_DRAWINGS_ATM_CURRENT` | MIN → MIN | 82.8 | 82.8 | 0 | 0.03 | 1 | 0.022 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MIN_MAX` | · | 899 | `CNT_DRAWINGS_ATM_CURRENT` | MIN → MAX | 82.8 | 82.8 | 0 | 0.03 | 1 | 0.022 |
| `PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MIN_SUM` | · | 900 | `CNT_DRAWINGS_ATM_CURRENT` | MIN → SUM | 5.4 | 0.0 | 0 | 0.01 | 0 | 0.014 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MIN_MEAN` | · | 901 | `CNT_DRAWINGS_OTHER_CURRENT` | MIN → MEAN | 82.8 | 82.8 | 0 | 1.13e-04 | 0 | -0.003 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MIN_SUM` | · | 902 | `CNT_DRAWINGS_OTHER_CURRENT` | MIN → SUM | 5.4 | 0.0 | 0 | 1.95e-05 | 0 | -0.001 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MAX_MIN` | · | 903 | `CNT_DRAWINGS_OTHER_CURRENT` | MAX → MIN | 82.8 | 82.8 | 0 | 0.11 | 2 | 0.002 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MAX_SUM` | · | 904 | `CNT_DRAWINGS_OTHER_CURRENT` | MAX → SUM | 5.4 | 0.0 | 0 | 0.02 | 1 | 0.011 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_SUM_SUM` | · | 905 | `CNT_DRAWINGS_OTHER_CURRENT` | SUM → SUM | 5.4 | 0.0 | 0 | 0.03 | 1 | 0.006 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MIN_MEAN` | · | 906 | `CNT_INSTALMENT_MATURE_CUM` | MIN → MEAN | 74.7 | 74.7 | 0 | 0.30 | 1 | -0.032 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MIN_MIN` | · | 907 | `CNT_INSTALMENT_MATURE_CUM` | MIN → MIN | 74.7 | 74.7 | 0 | 0.30 | 1 | -0.032 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MIN_MAX` | · | 908 | `CNT_INSTALMENT_MATURE_CUM` | MIN → MAX | 74.7 | 74.7 | 0 | 0.31 | 1 | -0.032 |
| `PREV_CC_CNT_INSTALMENT_MATURE_CUM_MIN_SUM` | · | 909 | `CNT_INSTALMENT_MATURE_CUM` | MIN → SUM | 5.4 | 0.0 | 0 | 0.08 | 1 | -0.006 |
| `PREV_CC_SK_DPD_MAX_SUM` | · | 910 | `SK_DPD` | MAX → SUM | 5.4 | 0.0 | 0 | 3.16 | 11 | -0.002 |
| `PREV_CC_SK_DPD_DEF_MEAN_SUM` | · | 911 | `SK_DPD_DEF` | MEAN → SUM | 5.4 | 0.0 | 0 | 0.01 | 0.14 | 0.013 |
| `PREV_CC_SK_DPD_DEF_MAX_MIN` | · | 912 | `SK_DPD_DEF` | MAX → MIN | 74.7 | 74.7 | 0 | 0.45 | 8 | 0.016 |
| `PREV_CC_SK_DPD_DEF_MAX_MAX` | · | 913 | `SK_DPD_DEF` | MAX → MAX | 74.7 | 74.7 | 0 | 0.45 | 8 | 0.017 |
| `PREV_CC_SK_DPD_DEF_MAX_SUM` | · | 914 | `SK_DPD_DEF` | MAX → SUM | 5.4 | 0.0 | 0 | 0.11 | 1 | 0.011 |
| `PREV_CC_SK_DPD_DEF_SUM_MIN` | · | 915 | `SK_DPD_DEF` | SUM → MIN | 74.7 | 74.7 | 0 | 1.31 | 23 | 0.009 |
| `PREV_CC_SK_DPD_DEF_SUM_MAX` | · | 916 | `SK_DPD_DEF` | SUM → MAX | 74.7 | 74.7 | 0 | 1.31 | 23 | 0.009 |
| `PREV_CC_SK_DPD_DEF_SUM_SUM` | · | 917 | `SK_DPD_DEF` | SUM → SUM | 5.4 | 0.0 | 0 | 0.33 | 8 | 0.007 |
| `PREV_CC_MONTHS_BALANCE_MAX_MAX` | · | — | `MONTHS_BALANCE` | MAX → MAX | 74.7 | 74.7 | -1 | -1.50 | -1 | 0.029 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MIN_MIN` | · | — | `AMT_DRAWINGS_OTHER_CURRENT` | MIN → MIN | 82.8 | 82.8 | 0 | 6.47 | 0 | -0.003 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MIN_MAX` | · | — | `AMT_DRAWINGS_OTHER_CURRENT` | MIN → MAX | 82.8 | 82.8 | 0 | 6.47 | 0 | -0.003 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MIN_MEAN` | · | — | `AMT_TOTAL_RECEIVABLE` | MIN → MEAN | 74.7 | 74.7 | 0 | 11,571 | 228,136 | 0.063 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MIN_MIN` | · | — | `AMT_TOTAL_RECEIVABLE` | MIN → MIN | 74.7 | 74.7 | 0 | 11,536 | 228,136 | 0.063 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MIN_MAX` | · | — | `AMT_TOTAL_RECEIVABLE` | MIN → MAX | 74.7 | 74.7 | 0 | 11,606 | 228,227 | 0.063 |
| `PREV_CC_AMT_TOTAL_RECEIVABLE_MIN_SUM` | · | — | `AMT_TOTAL_RECEIVABLE` | MIN → SUM | 5.4 | 0.0 | 0 | 2940.88 | 105,750 | 0.037 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MIN_MIN` | · | — | `CNT_DRAWINGS_OTHER_CURRENT` | MIN → MIN | 82.8 | 82.8 | 0 | 1.13e-04 | 0 | -0.003 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MIN_MAX` | · | — | `CNT_DRAWINGS_OTHER_CURRENT` | MIN → MAX | 82.8 | 82.8 | 0 | 1.13e-04 | 0 | -0.003 |
| `PREV_CC_SK_DPD_MIN_MEAN` | ✗ const | — | `SK_DPD` | MIN → MEAN | 74.7 | 74.7 | 0 | 0 | 0 | – |
| `PREV_CC_SK_DPD_MIN_MIN` | ✗ const | — | `SK_DPD` | MIN → MIN | 74.7 | 74.7 | 0 | 0 | 0 | – |
| `PREV_CC_SK_DPD_MIN_MAX` | ✗ const | — | `SK_DPD` | MIN → MAX | 74.7 | 74.7 | 0 | 0 | 0 | – |
| `PREV_CC_SK_DPD_MIN_SUM` | ✗ const | — | `SK_DPD` | MIN → SUM | 5.4 | 0.0 | 0 | 0 | 0 | – |
| `PREV_CC_SK_DPD_DEF_MIN_MEAN` | ✗ const | — | `SK_DPD_DEF` | MIN → MEAN | 74.7 | 74.7 | 0 | 0 | 0 | – |
| `PREV_CC_SK_DPD_DEF_MIN_MIN` | ✗ const | — | `SK_DPD_DEF` | MIN → MIN | 74.7 | 74.7 | 0 | 0 | 0 | – |
| `PREV_CC_SK_DPD_DEF_MIN_MAX` | ✗ const | — | `SK_DPD_DEF` | MIN → MAX | 74.7 | 74.7 | 0 | 0 | 0 | – |
| `PREV_CC_SK_DPD_DEF_MIN_SUM` | ✗ const | — | `SK_DPD_DEF` | MIN → SUM | 5.4 | 0.0 | 0 | 0 | 0 | – |

### 7.5 Installments ledger via the previous-application hub (`PREV_INST_*`)

*116 features · 42 selected · 11.4% of model gain*

Same two-level shape as the credit-card block, over `installments_payments` (every scheduled vs. actual payment).

**Definition template** — `PREV_INST_<COL>_<AGG1>_<AGG2>` = the *<agg2>* across prior loans of (the *<agg1>* of `<COL>` over that loan's installments).

The four curated per-loan derived metrics (`INST_LATE_RATE`, `INST_DAYS_LATE_MEAN`, `INST_UNDERPAY_RATE`, `INST_SHORTFALL_MEAN`) roll up as `PREV_INST_<METRIC>_<AGG2>` and are the strongest engineered features in the whole matrix. **42 of 116 selected — more than any other engineered block.**

<details><summary><b>Source-column glossary</b> (10 columns)</summary>

| Source column | What it measures |
|---|---|
| `AMT_INSTALMENT` | What was the prescribed installment amount of previous credit on this installment |
| `AMT_PAYMENT` | What the client actually paid on previous credit on this installment |
| `DAYS_ENTRY_PAYMENT` | When was the installments of previous credit paid actually (relative to application date of current loan) |
| `DAYS_INSTALMENT` | When the installment of previous credit was supposed to be paid (relative to application date of current loan) |
| `INST_DAYS_LATE_MEAN` | the average days late per installment (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT; negative = paid early), computed over that loan's installment ledger |
| `INST_LATE_RATE` | the share of installments paid after their due date (DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT), computed over that loan's installment ledger |
| `INST_SHORTFALL_MEAN` | the average payment shortfall per installment (AMT_INSTALMENT - AMT_PAYMENT; negative = overpaid), computed over that loan's installment ledger |
| `INST_UNDERPAY_RATE` | the share of installments paid short (AMT_PAYMENT < AMT_INSTALMENT), computed over that loan's installment ledger |
| `NUM_INSTALMENT_NUMBER` | On which installment we observe payment |
| `NUM_INSTALMENT_VERSION` | Version of installment calendar (0 is for credit card) of previous credit. Change of installment version from month to month signifies that some parameter of payment calendar has changed |

</details>

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `PREV_INST_AMT_PAYMENT_MIN_MEAN` | **✔** | 12 | `AMT_PAYMENT` | MIN → MEAN | 5.9 | 5.9 | 8272.27 | 13,438 | 113,999 | -0.024 |
| `PREV_INST_LATE_RATE_MEAN` | **✔** | 18 | `INST_LATE_RATE` | DOMAIN → MEAN | 5.9 | 5.9 | 0.01 | 0.07 | 0.47 | 0.075 |
| `PREV_INST_LATE_RATE_MIN` | **✔** | 24 | `INST_LATE_RATE` | DOMAIN → MIN | 5.9 | 5.9 | 0 | 0.03 | 0.44 | 0.072 |
| `PREV_INST_AMT_PAYMENT_MIN_MAX` | **✔** | 27 | `AMT_PAYMENT` | MIN → MAX | 5.9 | 5.9 | 12,608 | 27,500 | 360,005 | -0.014 |
| `PREV_INST_AMT_PAYMENT_MIN_SUM` | **✔** | 29 | `AMT_PAYMENT` | MIN → SUM | 5.4 | 0.0 | 17,062 | 38,783 | 410,764 | -0.024 |
| `PREV_INST_AMT_PAYMENT_SUM_SUM` | **✔** | 56 | `AMT_PAYMENT` | SUM → SUM | 5.4 | 0.0 | 279,962 | 627,702 | 4.20M | -0.023 |
| `PREV_INST_DAYS_LATE_MEAN_MIN` | **✔** | 68 | `INST_DAYS_LATE_MEAN` | DOMAIN → MIN | 5.9 | 5.9 | -15.25 | -18.39 | 0 | 0.031 |
| `PREV_INST_AMT_PAYMENT_SUM_MAX` | **✔** | 71 | `AMT_PAYMENT` | SUM → MAX | 5.9 | 5.9 | 191,284 | 396,692 | 2.36M | -0.025 |
| `PREV_INST_AMT_PAYMENT_SUM_MEAN` | **✔** | 73 | `AMT_PAYMENT` | SUM → MEAN | 5.9 | 5.9 | 133,511 | 214,213 | 1.13M | -0.023 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MAX_MAX` | **✔** | 80 | `DAYS_ENTRY_PAYMENT` | MAX → MAX | 5.9 | 5.9 | -70 | -323.66 | -4 | -1.80e-04 |
| `PREV_INST_AMT_INSTALMENT_MIN_MIN` | **✔** | 81 | `AMT_INSTALMENT` | MIN → MIN | 5.9 | 5.9 | 5171.76 | 7417.86 | 41,643 | -0.021 |
| `PREV_INST_AMT_PAYMENT_MIN_MIN` | **✔** | 82 | `AMT_PAYMENT` | MIN → MIN | 5.9 | 5.9 | 2743.34 | 5542.47 | 38,786 | -0.026 |
| `PREV_INST_DAYS_LATE_MEAN_MAX` | **✔** | 84 | `INST_DAYS_LATE_MEAN` | DOMAIN → MAX | 5.9 | 5.9 | -5.80 | -7.05 | 19.38 | 0.013 |
| `PREV_INST_NUM_INSTALMENT_VERSION_SUM_MEAN` | **✔** | 86 | `NUM_INSTALMENT_VERSION` | SUM → MEAN | 5.9 | 5.9 | 10 | 11.82 | 47 | -0.013 |
| `PREV_INST_AMT_INSTALMENT_MIN_SUM` | **✔** | 89 | `AMT_INSTALMENT` | MIN → SUM | 5.4 | 0.0 | 22,082 | 44,783 | 410,183 | -0.021 |
| `PREV_INST_SHORTFALL_MEAN_MEAN` | **✔** | 93 | `INST_SHORTFALL_MEAN` | DOMAIN → MEAN | 5.9 | 5.9 | 0 | -537.93 | 5571.19 | 0.025 |
| `PREV_INST_DAYS_LATE_MEAN_MEAN` | **✔** | 94 | `INST_DAYS_LATE_MEAN` | DOMAIN → MEAN | 5.9 | 5.9 | -10.73 | -12.29 | 4.75 | 0.028 |
| `PREV_INST_AMT_INSTALMENT_MIN_MEAN` | **✔** | 95 | `AMT_INSTALMENT` | MIN → MEAN | 5.9 | 5.9 | 10,285 | 15,507 | 111,514 | -0.018 |
| `PREV_INST_AMT_PAYMENT_MEAN_MIN` | **✔** | 104 | `AMT_PAYMENT` | MEAN → MIN | 5.9 | 5.9 | 6743.32 | 10,789 | 69,364 | -0.015 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MEAN_MAX` | **✔** | 121 | `NUM_INSTALMENT_VERSION` | MEAN → MAX | 5.9 | 5.9 | 1.12 | 1.40 | 5.24 | -0.025 |
| `PREV_INST_AMT_INSTALMENT_MEAN_MIN` | **✔** | 129 | `AMT_INSTALMENT` | MEAN → MIN | 5.9 | 5.9 | 7015.02 | 10,903 | 65,781 | -0.012 |
| `PREV_INST_DAYS_LATE_MEAN_SUM` | **✔** | 133 | `INST_DAYS_LATE_MEAN` | DOMAIN → SUM | 5.4 | 0.0 | -23.24 | -31.73 | 6.93 | 0.040 |
| `PREV_INST_LATE_RATE_MAX` | **✔** | 135 | `INST_LATE_RATE` | DOMAIN → MAX | 5.9 | 5.9 | 0.04 | 0.13 | 0.70 | 0.051 |
| `PREV_INST_AMT_INSTALMENT_SUM_MEAN` | **✔** | 139 | `AMT_INSTALMENT` | SUM → MEAN | 5.9 | 5.9 | 137,117 | 212,416 | 1.05M | -0.016 |
| `PREV_INST_AMT_INSTALMENT_SUM_MIN` | **✔** | 141 | `AMT_INSTALMENT` | SUM → MIN | 5.9 | 5.9 | 58,570 | 101,878 | 750,280 | -0.006 |
| `PREV_INST_NUM_INSTALMENT_VERSION_SUM_SUM` | **✔** | 142 | `NUM_INSTALMENT_VERSION` | SUM → SUM | 5.4 | 0.0 | 21 | 31.61 | 158 | -0.029 |
| `PREV_INST_AMT_INSTALMENT_MAX_MIN` | **✔** | 145 | `AMT_INSTALMENT` | MAX → MIN | 5.9 | 5.9 | 8978.62 | 21,738 | 246,082 | 0.002 |
| `PREV_INST_AMT_PAYMENT_MAX_MIN` | **✔** | 154 | `AMT_PAYMENT` | MAX → MIN | 5.9 | 5.9 | 8996.08 | 22,002 | 250,193 | 0.001 |
| `PREV_INST_AMT_PAYMENT_SUM_MIN` | **✔** | 158 | `AMT_PAYMENT` | SUM → MIN | 5.9 | 5.9 | 56,842 | 100,287 | 754,875 | -0.010 |
| `PREV_INST_AMT_INSTALMENT_MIN_MAX` | **✔** | 159 | `AMT_INSTALMENT` | MIN → MAX | 5.9 | 5.9 | 14,584 | 29,380 | 346,804 | -0.011 |
| `PREV_INST_SHORTFALL_MEAN_SUM` | **✔** | 161 | `INST_SHORTFALL_MEAN` | DOMAIN → SUM | 5.4 | 0.0 | 0 | -1731.22 | 16,671 | 0.021 |
| `PREV_INST_DAYS_INSTALMENT_MAX_MAX` | **✔** | 169 | `DAYS_INSTALMENT` | MAX → MAX | 5.9 | 5.9 | -54 | -315.12 | -3 | -0.001 |
| `PREV_INST_AMT_INSTALMENT_SUM_MAX` | **✔** | 170 | `AMT_INSTALMENT` | SUM → MAX | 5.9 | 5.9 | 197,528 | 385,039 | 2.03M | -0.018 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MAX_MAX` | **✔** | 171 | `NUM_INSTALMENT_NUMBER` | MAX → MAX | 5.9 | 5.9 | 12 | 21.69 | 112 | 0.019 |
| `PREV_INST_AMT_PAYMENT_MEAN_MAX` | **✔** | 172 | `AMT_PAYMENT` | MEAN → MAX | 5.9 | 5.9 | 18,644 | 47,629 | 492,497 | -0.010 |
| `PREV_INST_AMT_INSTALMENT_MAX_MEAN` | **✔** | 175 | `AMT_INSTALMENT` | MAX → MEAN | 5.9 | 5.9 | 23,334 | 62,583 | 500,886 | 0.004 |
| `PREV_INST_AMT_PAYMENT_MEAN_MEAN` | **✔** | 180 | `AMT_PAYMENT` | MEAN → MEAN | 5.9 | 5.9 | 13,430 | 23,922 | 179,794 | -0.015 |
| `PREV_INST_UNDERPAY_RATE_MEAN` | **✔** | 182 | `INST_UNDERPAY_RATE` | DOMAIN → MEAN | 5.9 | 5.9 | 0 | 0.07 | 0.64 | 0.068 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MEAN_MEAN` | **✔** | 187 | `NUM_INSTALMENT_VERSION` | MEAN → MEAN | 5.9 | 5.9 | 1.04 | 1.13 | 2.88 | -0.032 |
| `PREV_INST_LATE_RATE_SUM` | **✔** | 190 | `INST_LATE_RATE` | DOMAIN → SUM | 5.4 | 0.0 | 0 | 0.18 | 1.43 | 0.040 |
| `PREV_INST_AMT_INSTALMENT_SUM_SUM` | **✔** | 194 | `AMT_INSTALMENT` | SUM → SUM | 5.4 | 0.0 | 288,986 | 621,375 | 4.03M | -0.019 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MAX_MEAN` | **✔** | 198 | `NUM_INSTALMENT_NUMBER` | MAX → MEAN | 5.9 | 5.9 | 10 | 13.07 | 70.50 | 0.034 |
| `PREV_INST_AMT_PAYMENT_MAX_MEAN` | · | 201 | `AMT_PAYMENT` | MAX → MEAN | 5.9 | 5.9 | 23,660 | 63,443 | 504,738 | 0.004 |
| `PREV_INST_SHORTFALL_MEAN_MAX` | · | 203 | `INST_SHORTFALL_MEAN` | DOMAIN → MAX | 5.9 | 5.9 | 0 | 975.54 | 12,224 | 0.022 |
| `PREV_INST_COUNT_MEAN` | · | 208 | `–` | COUNT → MEAN | 5.9 | 5.9 | 10.14 | 12.60 | 51 | 0.004 |
| `PREV_INST_DAYS_INSTALMENT_SUM_MIN` | · | 211 | `DAYS_INSTALMENT` | SUM → MIN | 5.9 | 5.9 | -13,230 | -21,006 | -408 | 0.025 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_SUM_MAX` | · | 213 | `DAYS_ENTRY_PAYMENT` | SUM → MAX | 5.9 | 5.9 | -2499 | -5416.03 | -32 | 1.40e-04 |
| `PREV_INST_DAYS_INSTALMENT_SUM_MEAN` | · | 217 | `DAYS_INSTALMENT` | SUM → MEAN | 5.9 | 5.9 | -8139 | -11,492 | -395 | 0.021 |
| `PREV_INST_AMT_INSTALMENT_MAX_MAX` | · | 220 | `AMT_INSTALMENT` | MAX → MAX | 5.9 | 5.9 | 34,959 | 136,645 | 1.19M | 0.003 |
| `PREV_INST_UNDERPAY_RATE_MIN` | · | 224 | `INST_UNDERPAY_RATE` | DOMAIN → MIN | 5.9 | 5.9 | 0 | 0.03 | 0.60 | 0.063 |
| `PREV_INST_AMT_INSTALMENT_MEAN_MEAN` | · | 237 | `AMT_INSTALMENT` | MEAN → MEAN | 5.9 | 5.9 | 13,692 | 23,388 | 169,910 | -0.012 |
| `PREV_INST_NUM_INSTALMENT_VERSION_SUM_MAX` | · | 238 | `NUM_INSTALMENT_VERSION` | SUM → MAX | 5.9 | 5.9 | 12 | 18.54 | 100 | -0.017 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MEAN_MAX` | · | 246 | `DAYS_ENTRY_PAYMENT` | MEAN → MAX | 5.9 | 5.9 | -289.44 | -478.47 | -29 | 0.003 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_SUM_MEAN` | · | 250 | `NUM_INSTALMENT_NUMBER` | SUM → MEAN | 5.9 | 5.9 | 62.50 | 180.62 | 2310.33 | -0.005 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_SUM_MIN` | · | 256 | `DAYS_ENTRY_PAYMENT` | SUM → MIN | 5.9 | 5.9 | -13,375 | -21,146 | -470 | 0.025 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_SUM_MEAN` | · | 260 | `DAYS_ENTRY_PAYMENT` | SUM → MEAN | 5.9 | 5.9 | -8262 | -11,613 | -455 | 0.021 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MEAN_SUM` | · | 266 | `NUM_INSTALMENT_NUMBER` | MEAN → SUM | 5.4 | 0.0 | 12 | 17.90 | 83.30 | -0.013 |
| `PREV_INST_DAYS_INSTALMENT_SUM_MAX` | · | 272 | `DAYS_INSTALMENT` | SUM → MAX | 5.9 | 5.9 | -2387 | -5314.63 | -21 | -1.60e-04 |
| `PREV_INST_AMT_PAYMENT_MAX_MAX` | · | 277 | `AMT_PAYMENT` | MAX → MAX | 5.9 | 5.9 | 35,627 | 138,506 | 1.19M | 0.002 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MEAN_MAX` | · | 284 | `NUM_INSTALMENT_NUMBER` | MEAN → MAX | 5.9 | 5.9 | 6.50 | 10.42 | 56.50 | 2.00e-04 |
| `PREV_INST_SHORTFALL_MEAN_MIN` | · | 285 | `INST_SHORTFALL_MEAN` | DOMAIN → MIN | 5.9 | 5.9 | 0 | -2940.33 | 3782.55 | 0.018 |
| `PREV_INST_AMT_PAYMENT_MEAN_SUM` | · | 293 | `AMT_PAYMENT` | MEAN → SUM | 5.4 | 0.0 | 28,912 | 71,493 | 671,788 | -0.015 |
| `PREV_INST_AMT_INSTALMENT_MEAN_MAX` | · | 295 | `AMT_INSTALMENT` | MEAN → MAX | 5.9 | 5.9 | 18,943 | 45,724 | 474,006 | -0.008 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MEAN_MEAN` | · | 296 | `NUM_INSTALMENT_NUMBER` | MEAN → MEAN | 5.9 | 5.9 | 5.50 | 6.61 | 26 | 0.012 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MEAN_SUM` | · | 297 | `NUM_INSTALMENT_VERSION` | MEAN → SUM | 5.4 | 0.0 | 2.20 | 3.01 | 12.12 | -0.040 |
| `PREV_INST_UNDERPAY_RATE_MAX` | · | 301 | `INST_UNDERPAY_RATE` | DOMAIN → MAX | 5.9 | 5.9 | 0 | 0.15 | 0.91 | 0.046 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MAX_MEAN` | · | 303 | `NUM_INSTALMENT_VERSION` | MAX → MEAN | 5.9 | 5.9 | 1.33 | 1.54 | 5 | -0.016 |
| `PREV_INST_UNDERPAY_RATE_SUM` | · | 310 | `INST_UNDERPAY_RATE` | DOMAIN → SUM | 5.4 | 0.0 | 0 | 0.20 | 1.74 | 0.037 |
| `PREV_INST_DAYS_INSTALMENT_MEAN_MAX` | · | 314 | `DAYS_INSTALMENT` | MEAN → MAX | 5.9 | 5.9 | -277 | -466.28 | -19 | 0.002 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MAX_SUM` | · | 319 | `NUM_INSTALMENT_NUMBER` | MAX → SUM | 5.4 | 0.0 | 21 | 35.09 | 175 | 0.003 |
| `PREV_INST_AMT_PAYMENT_MAX_SUM` | · | 321 | `AMT_PAYMENT` | MAX → SUM | 5.4 | 0.0 | 49,736 | 197,653 | 1.88M | 0.002 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_SUM_SUM` | · | 322 | `NUM_INSTALMENT_NUMBER` | SUM → SUM | 5.4 | 0.0 | 132 | 511.44 | 6625 | -0.009 |
| `PREV_INST_DAYS_INSTALMENT_SUM_SUM` | · | 324 | `DAYS_INSTALMENT` | SUM → SUM | 5.4 | 0.0 | -17,472 | -33,326 | 0 | 0.031 |
| `PREV_INST_COUNT_SUM` | · | 338 | `–` | COUNT → SUM | 5.4 | 0.0 | 22 | 34.38 | 166 | -0.015 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MIN_MAX` | · | 342 | `DAYS_ENTRY_PAYMENT` | MIN → MAX | 5.9 | 5.9 | -439 | -614.67 | -31 | 0.004 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_SUM_MAX` | · | 347 | `NUM_INSTALMENT_NUMBER` | SUM → MAX | 5.9 | 5.9 | 84 | 441.86 | 6441 | -0.009 |
| `PREV_INST_AMT_INSTALMENT_MAX_SUM` | · | 348 | `AMT_INSTALMENT` | MAX → SUM | 5.4 | 0.0 | 49,187 | 195,147 | 1.86M | 0.002 |
| `PREV_INST_AMT_INSTALMENT_MEAN_SUM` | · | 357 | `AMT_INSTALMENT` | MEAN → SUM | 5.4 | 0.0 | 29,534 | 69,776 | 640,454 | -0.013 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_SUM_SUM` | · | 361 | `DAYS_ENTRY_PAYMENT` | SUM → SUM | 5.4 | 0.0 | -17,737 | -33,643 | 0 | 0.031 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MAX_MEAN` | · | 362 | `DAYS_ENTRY_PAYMENT` | MAX → MEAN | 5.9 | 5.9 | -645 | -733.88 | -14 | 0.038 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MEAN_MIN` | · | 369 | `NUM_INSTALMENT_NUMBER` | MEAN → MIN | 5.9 | 5.9 | 3.50 | 4.22 | 15.73 | 0.021 |
| `PREV_INST_DAYS_INSTALMENT_MEAN_MEAN` | · | 376 | `DAYS_INSTALMENT` | MEAN → MEAN | 5.9 | 5.9 | -790 | -876.19 | -82 | 0.040 |
| `PREV_INST_COUNT_MAX` | · | 383 | `–` | COUNT → MAX | 5.9 | 5.9 | 13 | 20.45 | 114 | -0.005 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MIN_MEAN` | · | 393 | `DAYS_ENTRY_PAYMENT` | MIN → MEAN | 5.9 | 5.9 | -950.12 | -1036.98 | -152 | 0.040 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MIN_MIN` | · | 400 | `DAYS_ENTRY_PAYMENT` | MIN → MIN | 5.9 | 5.9 | -1467 | -1507.15 | -156 | 0.054 |
| `PREV_INST_DAYS_INSTALMENT_MIN_MAX` | · | 401 | `DAYS_INSTALMENT` | MIN → MAX | 5.9 | 5.9 | -427 | -603.02 | -20 | 0.004 |
| `PREV_INST_DAYS_INSTALMENT_MAX_MEAN` | · | 409 | `DAYS_INSTALMENT` | MAX → MEAN | 5.9 | 5.9 | -638 | -726.99 | -8 | 0.037 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_SUM_MIN` | · | 412 | `NUM_INSTALMENT_NUMBER` | SUM → MIN | 5.9 | 5.9 | 21 | 60.92 | 435 | 0.004 |
| `PREV_INST_DAYS_INSTALMENT_MIN_MEAN` | · | 421 | `DAYS_INSTALMENT` | MIN → MEAN | 5.9 | 5.9 | -938.25 | -1025.10 | -139 | 0.040 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MAX_MIN` | · | 423 | `DAYS_ENTRY_PAYMENT` | MAX → MIN | 5.9 | 5.9 | -1138 | -1217.35 | -14 | 0.052 |
| `PREV_INST_DAYS_INSTALMENT_MEAN_MIN` | · | 427 | `DAYS_INSTALMENT` | MEAN → MIN | 5.9 | 5.9 | -1292.31 | -1349.33 | -84 | 0.053 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MEAN_MEAN` | · | 439 | `DAYS_ENTRY_PAYMENT` | MEAN → MEAN | 5.9 | 5.9 | -802.34 | -888.57 | -94.14 | 0.040 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MAX_MIN` | · | 446 | `NUM_INSTALMENT_NUMBER` | MAX → MIN | 5.9 | 5.9 | 6 | 7.79 | 43 | 0.036 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MIN_MEAN` | · | 453 | `NUM_INSTALMENT_VERSION` | MIN → MEAN | 5.9 | 5.9 | 1 | 0.96 | 1.50 | -0.034 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MEAN_MIN` | · | 458 | `DAYS_ENTRY_PAYMENT` | MEAN → MIN | 5.9 | 5.9 | -1306.50 | -1361.65 | -97 | 0.053 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MEAN_MIN` | · | 462 | `NUM_INSTALMENT_VERSION` | MEAN → MIN | 5.9 | 5.9 | 1 | 0.88 | 2.15 | -0.034 |
| `PREV_INST_NUM_INSTALMENT_VERSION_SUM_MIN` | · | 466 | `NUM_INSTALMENT_VERSION` | SUM → MIN | 5.9 | 5.9 | 6 | 6.97 | 32 | -0.004 |
| `PREV_INST_DAYS_INSTALMENT_MIN_MIN` | · | 473 | `DAYS_INSTALMENT` | MIN → MIN | 5.9 | 5.9 | -1454 | -1495.15 | -144 | 0.054 |
| `PREV_INST_DAYS_INSTALMENT_MEAN_SUM` | · | 477 | `DAYS_INSTALMENT` | MEAN → SUM | 5.4 | 0.0 | -1775.03 | -2541.72 | 0 | 0.043 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MIN_SUM` | · | 480 | `DAYS_ENTRY_PAYMENT` | MIN → SUM | 5.4 | 0.0 | -2138 | -2976.18 | 0 | 0.043 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MAX_SUM` | · | 484 | `NUM_INSTALMENT_VERSION` | MAX → SUM | 5.4 | 0.0 | 3 | 4.13 | 19 | -0.032 |
| `PREV_INST_DAYS_INSTALMENT_MAX_SUM` | · | 485 | `DAYS_INSTALMENT` | MAX → SUM | 5.4 | 0.0 | -1422 | -2136.03 | 0 | 0.043 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MAX_SUM` | · | 498 | `DAYS_ENTRY_PAYMENT` | MAX → SUM | 5.4 | 0.0 | -1435 | -2155.41 | 0 | 0.043 |
| `PREV_INST_DAYS_ENTRY_PAYMENT_MEAN_SUM` | · | 503 | `DAYS_ENTRY_PAYMENT` | MEAN → SUM | 5.4 | 0.0 | -1802.57 | -2573.49 | 0 | 0.043 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MAX_MAX` | · | 505 | `NUM_INSTALMENT_VERSION` | MAX → MAX | 5.9 | 5.9 | 2 | 2.11 | 10 | -0.019 |
| `PREV_INST_COUNT_MIN` | · | 512 | `–` | COUNT → MIN | 5.9 | 5.9 | 6 | 7.65 | 30 | 0.015 |
| `PREV_INST_DAYS_INSTALMENT_MIN_SUM` | · | 515 | `DAYS_INSTALMENT` | MIN → SUM | 5.4 | 0.0 | -2111 | -2945.42 | 0 | 0.042 |
| `PREV_INST_DAYS_INSTALMENT_MAX_MIN` | · | 522 | `DAYS_INSTALMENT` | MAX → MIN | 5.9 | 5.9 | -1134 | -1213.66 | -8 | 0.052 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MIN_SUM` | · | 671 | `NUM_INSTALMENT_NUMBER` | MIN → SUM | 5.4 | 0.0 | 2 | 2.67 | 9 | -0.029 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MAX_MIN` | · | 684 | `NUM_INSTALMENT_VERSION` | MAX → MIN | 5.9 | 5.9 | 1 | 1.07 | 4 | -0.015 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MIN_SUM` | · | 719 | `NUM_INSTALMENT_VERSION` | MIN → SUM | 5.4 | 0.0 | 2 | 2.56 | 9 | -0.037 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MIN_MAX` | · | 756 | `NUM_INSTALMENT_VERSION` | MIN → MAX | 5.9 | 5.9 | 1 | 1.05 | 2 | -0.003 |
| `PREV_INST_NUM_INSTALMENT_VERSION_MIN_MIN` | · | 918 | `NUM_INSTALMENT_VERSION` | MIN → MIN | 5.9 | 5.9 | 1 | 0.82 | 1 | -0.041 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MIN_MEAN` | · | 919 | `NUM_INSTALMENT_NUMBER` | MIN → MEAN | 5.9 | 5.9 | 1 | 1.00 | 1 | 0.004 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MIN_MAX` | · | 920 | `NUM_INSTALMENT_NUMBER` | MIN → MAX | 5.9 | 5.9 | 1 | 1.00 | 1 | 0.003 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MIN_MIN` | ✗ const | — | `NUM_INSTALMENT_NUMBER` | MIN → MIN | 5.9 | 5.9 | 1 | 1 | 1 | – |

### 7.6 POS/cash ledger via the previous-application hub (`PREV_POS_*`)

*88 features · 14 selected · 4.5% of model gain*

Same two-level shape, over `POS_CASH_balance` (monthly point-of-sale / cash loan status).

**Definition template** — `PREV_POS_<COL>_<AGG1>_<AGG2>`, plus the derived `POS_COMPLETED_MONTHS` (count of months with status 'Completed').

<details><summary><b>Source-column glossary</b> (6 columns)</summary>

| Source column | What it measures |
|---|---|
| `CNT_INSTALMENT` | Term of previous credit (can change over time) |
| `CNT_INSTALMENT_FUTURE` | Installments left to pay on the previous credit |
| `MONTHS_BALANCE` | Month of balance relative to application date (-1 means the information to the freshest monthly snapshot, 0 means the information at application - often it will be the same as -1 as many banks are not… |
| `POS_COMPLETED_MONTHS` | the number of monthly records with contract status 'Completed', computed over that loan's POS/cash ledger |
| `SK_DPD` | DPD (days past due) during the month of previous credit |
| `SK_DPD_DEF` | DPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit |

</details>

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `PREV_POS_CNT_INSTALMENT_FUTURE_MEAN_MEAN` | **✔** | 22 | `CNT_INSTALMENT_FUTURE` | MEAN → MEAN | 6.7 | 6.7 | 7 | 9.07 | 31.81 | 0.033 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_MEAN_MAX` | **✔** | 76 | `CNT_INSTALMENT_FUTURE` | MEAN → MAX | 6.7 | 6.7 | 9 | 14.11 | 56.50 | 0.018 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_MIN_MEAN` | **✔** | 79 | `CNT_INSTALMENT_FUTURE` | MIN → MEAN | 6.7 | 6.7 | 0 | 2.29 | 21 | 0.019 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_MIN_SUM` | **✔** | 116 | `CNT_INSTALMENT_FUTURE` | MIN → SUM | 5.4 | 0.0 | 0 | 5.84 | 55 | 0.009 |
| `PREV_POS_COUNT_SUM` | **✔** | 131 | `–` | COUNT → SUM | 5.4 | 0.0 | 20 | 26.83 | 111 | -0.036 |
| `PREV_POS_SK_DPD_DEF_MAX_MEAN` | **✔** | 153 | `SK_DPD_DEF` | MAX → MEAN | 6.7 | 6.7 | 0 | 0.44 | 9 | 0.049 |
| `PREV_POS_COUNT_MEAN` | **✔** | 155 | `–` | COUNT → MEAN | 6.7 | 6.7 | 10 | 10.77 | 26 | -0.005 |
| `PREV_POS_MONTHS_BALANCE_SUM_MIN` | **✔** | 156 | `MONTHS_BALANCE` | SUM → MIN | 6.7 | 6.7 | -440 | -518.70 | -20 | 0.037 |
| `PREV_POS_MONTHS_BALANCE_MAX_MAX` | **✔** | 164 | `MONTHS_BALANCE` | MAX → MAX | 6.7 | 6.7 | -4 | -11.99 | -1 | -0.004 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_MIN_MAX` | **✔** | 166 | `CNT_INSTALMENT_FUTURE` | MIN → MAX | 6.7 | 6.7 | 0 | 5.78 | 53 | 0.009 |
| `PREV_POS_SK_DPD_DEF_MEAN_MEAN` | **✔** | 168 | `SK_DPD_DEF` | MEAN → MEAN | 6.7 | 6.7 | 0 | 0.06 | 1.28 | 0.039 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_MEAN_SUM` | **✔** | 176 | `CNT_INSTALMENT_FUTURE` | MEAN → SUM | 5.4 | 0.0 | 14.79 | 23.61 | 119.58 | 0.001 |
| `PREV_POS_MONTHS_BALANCE_SUM_MEAN` | **✔** | 189 | `MONTHS_BALANCE` | SUM → MEAN | 6.7 | 6.7 | -281 | -332.27 | -18 | 0.025 |
| `PREV_POS_SK_DPD_DEF_SUM_MEAN` | **✔** | 191 | `SK_DPD_DEF` | SUM → MEAN | 6.7 | 6.7 | 0 | 0.80 | 16.67 | 0.010 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_MAX_MEAN` | · | 202 | `CNT_INSTALMENT_FUTURE` | MAX → MEAN | 6.7 | 6.7 | 12 | 14.43 | 41.50 | 0.029 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_MEAN_MIN` | · | 216 | `CNT_INSTALMENT_FUTURE` | MEAN → MIN | 6.7 | 6.7 | 4.50 | 5.58 | 28 | 0.034 |
| `PREV_POS_CNT_INSTALMENT_MIN_SUM` | · | 219 | `CNT_INSTALMENT` | MIN → SUM | 5.4 | 0.0 | 21 | 29.43 | 126 | -0.024 |
| `PREV_POS_CNT_INSTALMENT_SUM_MIN` | · | 226 | `CNT_INSTALMENT` | SUM → MIN | 6.7 | 6.7 | 50 | 99.22 | 612 | 0.016 |
| `PREV_POS_CNT_INSTALMENT_SUM_MEAN` | · | 227 | `CNT_INSTALMENT` | SUM → MEAN | 6.7 | 6.7 | 125.50 | 179.68 | 831 | 0.003 |
| `PREV_POS_CNT_INSTALMENT_MIN_MEAN` | · | 240 | `CNT_INSTALMENT` | MIN → MEAN | 6.7 | 6.7 | 10 | 11.69 | 36 | 0.007 |
| `PREV_POS_MONTHS_BALANCE_SUM_MAX` | · | 242 | `MONTHS_BALANCE` | SUM → MAX | 6.7 | 6.7 | -95 | -179.63 | -1 | -0.004 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_SUM_MEAN` | · | 255 | `CNT_INSTALMENT_FUTURE` | SUM → MEAN | 6.7 | 6.7 | 72 | 110.47 | 555.67 | 0.011 |
| `PREV_POS_COUNT_MAX` | · | 257 | `–` | COUNT → MAX | 6.7 | 6.7 | 13 | 14.18 | 39 | -0.019 |
| `PREV_POS_CNT_INSTALMENT_MAX_MEAN` | · | 265 | `CNT_INSTALMENT` | MAX → MEAN | 6.7 | 6.7 | 12 | 14.51 | 42 | 0.029 |
| `PREV_POS_CNT_INSTALMENT_SUM_MAX` | · | 267 | `CNT_INSTALMENT` | SUM → MAX | 6.7 | 6.7 | 156 | 301.47 | 1536 | -0.008 |
| `PREV_POS_CNT_INSTALMENT_MEAN_MEAN` | · | 270 | `CNT_INSTALMENT` | MEAN → MEAN | 6.7 | 6.7 | 12 | 13.89 | 37.64 | 0.026 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_SUM_MIN` | · | 309 | `CNT_INSTALMENT_FUTURE` | SUM → MIN | 6.7 | 6.7 | 33 | 58.33 | 435 | 0.019 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_SUM_MAX` | · | 312 | `CNT_INSTALMENT_FUTURE` | SUM → MAX | 6.7 | 6.7 | 78 | 191.85 | 1089 | 1.00e-05 |
| `PREV_POS_SK_DPD_DEF_MEAN_MAX` | · | 326 | `SK_DPD_DEF` | MEAN → MAX | 6.7 | 6.7 | 0 | 0.13 | 2.70 | 0.031 |
| `PREV_POS_MONTHS_BALANCE_SUM_SUM` | · | 336 | `MONTHS_BALANCE` | SUM → SUM | 5.4 | 0.0 | -570 | -895.99 | 0 | 0.040 |
| `PREV_POS_CNT_INSTALMENT_MIN_MIN` | · | 339 | `CNT_INSTALMENT` | MIN → MIN | 6.7 | 6.7 | 6 | 7.91 | 30 | 0.017 |
| `PREV_POS_MONTHS_BALANCE_MIN_MEAN` | · | 343 | `MONTHS_BALANCE` | MIN → MEAN | 6.7 | 6.7 | -32 | -34.77 | -5.50 | 0.034 |
| `PREV_POS_MONTHS_BALANCE_MEAN_MAX` | · | 344 | `MONTHS_BALANCE` | MEAN → MAX | 6.7 | 6.7 | -10.50 | -16.71 | -1 | -0.004 |
| `PREV_POS_CNT_INSTALMENT_SUM_SUM` | · | 352 | `CNT_INSTALMENT` | SUM → SUM | 5.4 | 0.0 | 252 | 463.01 | 2718 | -0.015 |
| `PREV_POS_MONTHS_BALANCE_MEAN_MEAN` | · | 359 | `MONTHS_BALANCE` | MEAN → MEAN | 6.7 | 6.7 | -27 | -29.89 | -3.50 | 0.034 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_SUM_SUM` | · | 382 | `CNT_INSTALMENT_FUTURE` | SUM → SUM | 5.4 | 0.0 | 145 | 288.10 | 1777 | -0.006 |
| `PREV_POS_SK_DPD_DEF_MEAN_SUM` | · | 389 | `SK_DPD_DEF` | MEAN → SUM | 5.4 | 0.0 | 0 | 0.14 | 3 | 0.032 |
| `PREV_POS_MONTHS_BALANCE_MAX_MEAN` | · | 391 | `MONTHS_BALANCE` | MAX → MEAN | 6.7 | 6.7 | -22 | -25.00 | -1 | 0.034 |
| `PREV_POS_CNT_INSTALMENT_MEAN_MIN` | · | 396 | `CNT_INSTALMENT` | MEAN → MIN | 6.7 | 6.7 | 8.11 | 9.61 | 36 | 0.032 |
| `PREV_POS_CNT_INSTALMENT_MEAN_MAX` | · | 397 | `CNT_INSTALMENT` | MEAN → MAX | 6.7 | 6.7 | 12 | 19.67 | 60 | 0.011 |
| `PREV_POS_CNT_INSTALMENT_MEAN_SUM` | · | 399 | `CNT_INSTALMENT` | MEAN → SUM | 5.4 | 0.0 | 24 | 35.72 | 160.91 | -0.010 |
| `PREV_POS_COMPLETED_MONTHS_MEAN` | · | 404 | `POS_COMPLETED_MONTHS` | DOMAIN → MEAN | 6.7 | 6.7 | 0.80 | 0.76 | 1 | -2.00e-05 |
| `PREV_POS_SK_DPD_DEF_MEAN_MIN` | · | 410 | `SK_DPD_DEF` | MEAN → MIN | 6.7 | 6.7 | 0 | 0.02 | 0.62 | 0.039 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_MAX_SUM` | · | 416 | `CNT_INSTALMENT_FUTURE` | MAX → SUM | 5.4 | 0.0 | 24 | 37.29 | 172 | -0.008 |
| `PREV_POS_MONTHS_BALANCE_MEAN_MIN` | · | 422 | `MONTHS_BALANCE` | MEAN → MIN | 6.7 | 6.7 | -42 | -44.66 | -3.50 | 0.051 |
| `PREV_POS_MONTHS_BALANCE_MIN_MAX` | · | 425 | `MONTHS_BALANCE` | MIN → MAX | 6.7 | 6.7 | -16 | -21.32 | -1 | -0.003 |
| `PREV_POS_SK_DPD_DEF_MAX_MAX` | · | 429 | `SK_DPD_DEF` | MAX → MAX | 6.7 | 6.7 | 0 | 1.01 | 19 | 0.039 |
| `PREV_POS_COUNT_MIN` | · | 430 | `–` | COUNT → MIN | 6.7 | 6.7 | 7 | 7.92 | 25 | 0.013 |
| `PREV_POS_CNT_INSTALMENT_MAX_SUM` | · | 432 | `CNT_INSTALMENT` | MAX → SUM | 5.4 | 0.0 | 24 | 37.51 | 172 | -0.008 |
| `PREV_POS_MONTHS_BALANCE_MIN_SUM` | · | 436 | `MONTHS_BALANCE` | MIN → SUM | 5.4 | 0.0 | -67 | -93.60 | 0 | 0.044 |
| `PREV_POS_CNT_INSTALMENT_MIN_MAX` | · | 437 | `CNT_INSTALMENT` | MIN → MAX | 6.7 | 6.7 | 12 | 17.00 | 60 | -0.002 |
| `PREV_POS_SK_DPD_MAX_MEAN` | · | 454 | `SK_DPD` | MAX → MEAN | 6.7 | 6.7 | 0 | 5.11 | 49.80 | 0.006 |
| `PREV_POS_MONTHS_BALANCE_MEAN_SUM` | · | 464 | `MONTHS_BALANCE` | MEAN → SUM | 5.4 | 0.0 | -56.50 | -81.43 | 0 | 0.044 |
| `PREV_POS_SK_DPD_MEAN_MEAN` | · | 471 | `SK_DPD` | MEAN → MEAN | 6.7 | 6.7 | 0 | 1.47 | 10.59 | 0.006 |
| `PREV_POS_SK_DPD_MEAN_MAX` | · | 479 | `SK_DPD` | MEAN → MAX | 6.7 | 6.7 | 0 | 2.69 | 28.83 | 0.005 |
| `PREV_POS_CNT_INSTALMENT_MAX_MAX` | · | 482 | `CNT_INSTALMENT` | MAX → MAX | 6.7 | 6.7 | 12 | 20.49 | 60 | 0.014 |
| `PREV_POS_MONTHS_BALANCE_MAX_SUM` | · | 483 | `MONTHS_BALANCE` | MAX → SUM | 5.4 | 0.0 | -46 | -69.25 | 0 | 0.043 |
| `PREV_POS_SK_DPD_DEF_SUM_MAX` | · | 489 | `SK_DPD_DEF` | SUM → MAX | 6.7 | 6.7 | 0 | 1.84 | 35 | 0.008 |
| `PREV_POS_SK_DPD_DEF_MAX_SUM` | · | 492 | `SK_DPD_DEF` | MAX → SUM | 5.4 | 0.0 | 0 | 1.07 | 21 | 0.039 |
| `PREV_POS_MONTHS_BALANCE_MIN_MIN` | · | 494 | `MONTHS_BALANCE` | MIN → MIN | 6.7 | 6.7 | -48 | -49.64 | -6 | 0.051 |
| `PREV_POS_SK_DPD_DEF_MAX_MIN` | · | 510 | `SK_DPD_DEF` | MAX → MIN | 6.7 | 6.7 | 0 | 0.16 | 6 | 0.044 |
| `PREV_POS_MONTHS_BALANCE_MAX_MIN` | · | 526 | `MONTHS_BALANCE` | MAX → MIN | 6.7 | 6.7 | -37 | -39.75 | -1 | 0.051 |
| `PREV_POS_SK_DPD_MEAN_SUM` | · | 530 | `SK_DPD` | MEAN → SUM | 5.4 | 0.0 | 0 | 2.56 | 23.92 | 0.005 |
| `PREV_POS_SK_DPD_SUM_MEAN` | · | 531 | `SK_DPD` | SUM → MEAN | 6.7 | 6.7 | 0 | 80.93 | 162.33 | 0.003 |
| `PREV_POS_CNT_INSTALMENT_MAX_MIN` | · | 537 | `CNT_INSTALMENT` | MAX → MIN | 6.7 | 6.7 | 9 | 10.02 | 36 | 0.033 |
| `PREV_POS_SK_DPD_DEF_SUM_SUM` | · | 541 | `SK_DPD_DEF` | SUM → SUM | 5.4 | 0.0 | 0 | 1.94 | 38 | 0.010 |
| `PREV_POS_SK_DPD_DEF_SUM_MIN` | · | 543 | `SK_DPD_DEF` | SUM → MIN | 6.7 | 6.7 | 0 | 0.28 | 8 | 0.036 |
| `PREV_POS_SK_DPD_SUM_SUM` | · | 552 | `SK_DPD` | SUM → SUM | 5.4 | 0.0 | 0 | 124.32 | 353 | 0.003 |
| `PREV_POS_SK_DPD_SUM_MAX` | · | 553 | `SK_DPD` | SUM → MAX | 6.7 | 6.7 | 0 | 132.58 | 447 | 0.003 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_MAX_MAX` | · | 557 | `CNT_INSTALMENT_FUTURE` | MAX → MAX | 6.7 | 6.7 | 12 | 20.44 | 60 | 0.014 |
| `PREV_POS_SK_DPD_MAX_MAX` | · | 564 | `SK_DPD` | MAX → MAX | 6.7 | 6.7 | 0 | 9.27 | 149 | 0.005 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_MAX_MIN` | · | 566 | `CNT_INSTALMENT_FUTURE` | MAX → MIN | 6.7 | 6.7 | 9 | 9.93 | 36 | 0.033 |
| `PREV_POS_SK_DPD_MAX_SUM` | · | 593 | `SK_DPD` | MAX → SUM | 5.4 | 0.0 | 0 | 8.92 | 130 | 0.005 |
| `PREV_POS_SK_DPD_MEAN_MIN` | · | 597 | `SK_DPD` | MEAN → MIN | 6.7 | 6.7 | 0 | 0.80 | 1.82 | 0.005 |
| `PREV_POS_COMPLETED_MONTHS_SUM` | · | 658 | `POS_COMPLETED_MONTHS` | DOMAIN → SUM | 5.4 | 0.0 | 1 | 1.92 | 8 | -0.022 |
| `PREV_POS_SK_DPD_MAX_MIN` | · | 662 | `SK_DPD` | MAX → MIN | 6.7 | 6.7 | 0 | 2.84 | 14 | 0.005 |
| `PREV_POS_CNT_INSTALMENT_FUTURE_MIN_MIN` | · | 690 | `CNT_INSTALMENT_FUTURE` | MIN → MIN | 6.7 | 6.7 | 0 | 0.62 | 15 | 0.018 |
| `PREV_POS_COMPLETED_MONTHS_MIN` | · | 750 | `POS_COMPLETED_MONTHS` | DOMAIN → MIN | 6.7 | 6.7 | 0 | 0.48 | 1 | 0.012 |
| `PREV_POS_SK_DPD_SUM_MIN` | · | 758 | `SK_DPD` | SUM → MIN | 6.7 | 6.7 | 0 | 50.29 | 22 | 0.003 |
| `PREV_POS_SK_DPD_MIN_MEAN` | · | 921 | `SK_DPD` | MIN → MEAN | 6.7 | 6.7 | 0 | 2.90e-06 | 0 | -7.70e-04 |
| `PREV_POS_SK_DPD_MIN_MAX` | · | 922 | `SK_DPD` | MIN → MAX | 6.7 | 6.7 | 0 | 6.97e-06 | 0 | -7.90e-04 |
| `PREV_POS_SK_DPD_MIN_SUM` | · | 923 | `SK_DPD` | MIN → SUM | 5.4 | 0.0 | 0 | 6.50e-06 | 0 | -7.80e-04 |
| `PREV_POS_SK_DPD_DEF_MIN_MEAN` | · | 924 | `SK_DPD_DEF` | MIN → MEAN | 6.7 | 6.7 | 0 | 1.74e-06 | 0 | -5.60e-04 |
| `PREV_POS_SK_DPD_DEF_MIN_MAX` | · | 925 | `SK_DPD_DEF` | MIN → MAX | 6.7 | 6.7 | 0 | 3.48e-06 | 0 | -5.60e-04 |
| `PREV_POS_SK_DPD_DEF_MIN_SUM` | · | 926 | `SK_DPD_DEF` | MIN → SUM | 5.4 | 0.0 | 0 | 3.25e-06 | 0 | -5.50e-04 |
| `PREV_POS_COMPLETED_MONTHS_MAX` | · | 927 | `POS_COMPLETED_MONTHS` | DOMAIN → MAX | 6.7 | 6.7 | 1 | 0.96 | 1 | -0.004 |
| `PREV_POS_SK_DPD_MIN_MIN` | ✗ const | — | `SK_DPD` | MIN → MIN | 6.7 | 6.7 | 0 | 0 | 0 | – |
| `PREV_POS_SK_DPD_DEF_MIN_MIN` | ✗ const | — | `SK_DPD_DEF` | MIN → MIN | 6.7 | 6.7 | 0 | 0 | 0 | – |

### 7.7 Credit-bureau lineage (`BUREAU_*`)

*79 features · 31 selected · 8.6% of model gain*

Credit at *other* institutions, reported to the bureau. `bureau_balance` (monthly status per bureau credit) is aggregated per `SK_ID_BUREAU` into four `BB_*` metrics, bridged onto `bureau` via `SK_ID_BUREAU`, then rolled up to the customer.

**Definition template** — `BUREAU_<COL>_<AGG>` = the *<agg>* of `<COL>` across the customer's bureau credits. `BUREAU_BB_<METRIC>_<AGG>` = the *<agg>* across bureau credits of a per-credit monthly-history metric.

**Calculation** — `bureau_balance.group_by('SK_ID_BUREAU')` → join on `SK_ID_BUREAU` → `bureau.group_by('SK_ID_CURR').<agg>()`. `STATUS` is cast to string before comparison ('0'–'5', 'C', 'X') because a digit-only CSV chunk can be schema-inferred as an integer.

<details><summary><b>Source-column glossary</b> (19 columns)</summary>

| Source column | What it measures |
|---|---|
| `AMT_ANNUITY` | Annuity of the Credit Bureau credit |
| `AMT_CREDIT_MAX_OVERDUE` | Maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample) |
| `AMT_CREDIT_SUM` | Current credit amount for the Credit Bureau credit |
| `AMT_CREDIT_SUM_DEBT` | Current debt on Credit Bureau credit |
| `AMT_CREDIT_SUM_LIMIT` | Current credit limit of credit card reported in Credit Bureau |
| `AMT_CREDIT_SUM_OVERDUE` | Current amount overdue on Credit Bureau credit |
| `BB_CLOSED_MONTHS` | the number of months in 'C' (closed) status |
| `BB_DPD_MONTHS` | the number of months in days-past-due status (STATUS in 1..5) |
| `BB_MONTHS_BALANCE_MIN` | the oldest observed month offset (min MONTHS_BALANCE, i.e. how far back the history goes) |
| `BB_MONTHS_COUNT` | the number of monthly bureau-balance records observed |
| `CNT_CREDIT_PROLONG` | How many times was the Credit Bureau credit prolonged |
| `CREDIT_ACTIVE` | Status of the Credit Bureau (CB) reported credits |
| `CREDIT_CURRENCY` | Recoded currency of the Credit Bureau credit |
| `CREDIT_DAY_OVERDUE` | Number of days past due on CB credit at the time of application for related loan in our sample |
| `CREDIT_TYPE` | Type of Credit Bureau credit (Car, cash,...) |
| `DAYS_CREDIT` | How many days before current application did client apply for Credit Bureau credit |
| `DAYS_CREDIT_ENDDATE` | Remaining duration of CB credit (in days) at the time of application in Home Credit |
| `DAYS_CREDIT_UPDATE` | How many days before loan application did last information about the Credit Bureau credit come |
| `DAYS_ENDDATE_FACT` | Days since CB credit ended at the time of application in Home Credit (only for closed credit) |

</details>

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `BUREAU_DAYS_CREDIT_MEAN` | **✔** | 20 | `DAYS_CREDIT` | MEAN | 14.3 | 14.3 | -1050.57 | -1083.05 | -83 | 0.090 |
| `BUREAU_DAYS_CREDIT_MAX` | **✔** | 21 | `DAYS_CREDIT` | MAX | 14.3 | 14.3 | -300 | -489.30 | -14 | 0.050 |
| `BUREAU_DAYS_CREDIT_ENDDATE_MAX` | **✔** | 32 | `DAYS_CREDIT_ENDDATE` | MAX | 15.0 | 15.0 | 909 | 4433.04 | 31,171 | 0.037 |
| `BUREAU_AMT_CREDIT_SUM_DEBT_MEAN` | **✔** | 33 | `AMT_CREDIT_SUM_DEBT` | MEAN | 16.7 | 16.7 | 44,208 | 160,390 | 1.94M | -6.40e-04 |
| `BUREAU_AMT_CREDIT_SUM_MAX` | **✔** | 35 | `AMT_CREDIT_SUM` | MAX | 14.3 | 14.3 | 450,000 | 976,004 | 8.55M | -0.020 |
| `BUREAU_AMT_CREDIT_SUM_MEAN` | **✔** | 37 | `AMT_CREDIT_SUM` | MEAN | 14.3 | 14.3 | 195,507 | 378,080 | 2.97M | -0.020 |
| `BUREAU_DAYS_ENDDATE_FACT_MAX` | **✔** | 38 | `DAYS_ENDDATE_FACT` | MAX | 25.1 | 25.1 | -345 | -526.99 | -11 | 0.020 |
| `BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN` | **✔** | 40 | `AMT_CREDIT_MAX_OVERDUE` | MEAN | 40.2 | 40.2 | 0 | 5242.43 | 41,270 | 0.002 |
| `BUREAU_CREDIT_ACTIVE_Closed_RATE` | **✔** | 44 | `CREDIT_ACTIVE` | RATE of "Closed" | 14.3 | 0.0 | 0.52 | 0.50 | 1 | -0.079 |
| `BUREAU_ACTIVE_COUNT` | **✔** | 49 | `CREDIT_ACTIVE` | DOMAIN | 14.3 | 0.0 | 1 | 1.76 | 8 | 0.067 |
| `BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX` | **✔** | 58 | `AMT_CREDIT_MAX_OVERDUE` | MAX | 40.2 | 40.2 | 0 | 9396.23 | 75,627 | 0.003 |
| `BUREAU_AMT_CREDIT_SUM_MIN` | **✔** | 63 | `AMT_CREDIT_SUM` | MIN | 14.3 | 14.3 | 39,535 | 119,631 | 1.45M | -0.011 |
| `BUREAU_DAYS_CREDIT_ENDDATE_MEAN` | **✔** | 66 | `DAYS_CREDIT_ENDDATE` | MEAN | 15.0 | 15.0 | -135.56 | 651.81 | 15,266 | 0.047 |
| `BUREAU_AMT_CREDIT_SUM_SUM` | **✔** | 67 | `AMT_CREDIT_SUM` | SUM | 14.3 | 0.0 | 711,000 | 1.68M | 13.28M | -0.014 |
| `BUREAU_DAYS_CREDIT_UPDATE_MEAN` | **✔** | 72 | `DAYS_CREDIT_UPDATE` | MEAN | 14.3 | 14.3 | -481.83 | -546.63 | -7 | 0.069 |
| `BUREAU_DAYS_CREDIT_UPDATE_MAX` | **✔** | 77 | `DAYS_CREDIT_UPDATE` | MAX | 14.3 | 14.3 | -19 | -140.99 | -1 | 0.028 |
| `BUREAU_DAYS_CREDIT_MIN` | **✔** | 92 | `DAYS_CREDIT` | MIN | 14.3 | 14.3 | -1827 | -1762.37 | -88 | 0.075 |
| `BUREAU_DAYS_ENDDATE_FACT_MIN` | **✔** | 99 | `DAYS_ENDDATE_FACT` | MIN | 25.1 | 25.1 | -1452 | -1467.34 | -51 | 0.056 |
| `BUREAU_AMT_CREDIT_MAX_OVERDUE_SUM` | **✔** | 109 | `AMT_CREDIT_MAX_OVERDUE` | SUM | 14.3 | 0.0 | 0 | 6626.06 | 72,464 | 0.002 |
| `BUREAU_AMT_CREDIT_SUM_DEBT_MAX` | **✔** | 110 | `AMT_CREDIT_SUM_DEBT` | MAX | 16.7 | 16.7 | 149,458 | 521,627 | 6.28M | -0.002 |
| `BUREAU_DAYS_CREDIT_ENDDATE_SUM` | **✔** | 113 | `DAYS_CREDIT_ENDDATE` | SUM | 14.3 | 0.0 | 0 | 2206.95 | 54,598 | 0.054 |
| `BUREAU_DAYS_ENDDATE_FACT_MEAN` | **✔** | 117 | `DAYS_ENDDATE_FACT` | MEAN | 25.1 | 25.1 | -928.40 | -970.30 | -49 | 0.053 |
| `BUREAU_DAYS_CREDIT_UPDATE_MIN` | **✔** | 125 | `DAYS_CREDIT_UPDATE` | MIN | 14.3 | 14.3 | -1055 | -1181.36 | -7 | 0.043 |
| `BUREAU_DAYS_CREDIT_ENDDATE_MIN` | **✔** | 128 | `DAYS_CREDIT_ENDDATE` | MIN | 15.0 | 15.0 | -1268 | -1029.58 | 3710 | 0.034 |
| `BUREAU_DAYS_CREDIT_SUM` | **✔** | 150 | `DAYS_CREDIT` | SUM | 14.3 | 0.0 | -3541 | -5449.39 | 0 | 0.042 |
| `BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN` | **✔** | 152 | `AMT_CREDIT_SUM_LIMIT` | MEAN | 21.2 | 21.2 | 0 | 5901.48 | 112,818 | -0.011 |
| `BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN` | **✔** | 173 | `AMT_CREDIT_SUM_OVERDUE` | MEAN | 14.3 | 14.3 | 0 | 49.55 | 4.50 | 0.007 |
| `BUREAU_AMT_CREDIT_SUM_OVERDUE_MAX` | **✔** | 174 | `AMT_CREDIT_SUM_OVERDUE` | MAX | 14.3 | 14.3 | 0 | 184.97 | 31.50 | 0.011 |
| `BUREAU_AMT_CREDIT_SUM_DEBT_MIN` | **✔** | 177 | `AMT_CREDIT_SUM_DEBT` | MIN | 16.7 | 16.7 | 0 | 39,806 | 744,700 | 2.40e-04 |
| `BUREAU_AMT_CREDIT_SUM_DEBT_SUM` | **✔** | 184 | `AMT_CREDIT_SUM_DEBT` | SUM | 14.3 | 0.0 | 87,584 | 548,942 | 6.53M | 0.007 |
| `BUREAU_AMT_CREDIT_SUM_LIMIT_MAX` | **✔** | 185 | `AMT_CREDIT_SUM_LIMIT` | MAX | 21.2 | 21.2 | 0 | 22,281 | 424,353 | -0.011 |
| `BUREAU_ACTIVE_RATE` | · | 223 | `CREDIT_ACTIVE` | DOMAIN | 14.3 | 0.0 | 0.33 | 0.35 | 1 | 0.077 |
| `BUREAU_BB_MONTHS_COUNT_MEAN` | · | 230 | `BB_MONTHS_COUNT` | MEAN | 70.0 | 70.0 | 24.75 | 27.48 | 82 | -0.080 |
| `BUREAU_DAYS_CREDIT_UPDATE_SUM` | · | 239 | `DAYS_CREDIT_UPDATE` | SUM | 14.3 | 0.0 | -1561 | -2858.05 | 0 | 0.041 |
| `BUREAU_BB_CLOSED_MONTHS_MEAN` | · | 244 | `BB_CLOSED_MONTHS` | MEAN | 70.0 | 70.0 | 9 | 12.97 | 63.50 | -0.063 |
| `BUREAU_BB_DPD_MONTHS_MEAN` | · | 248 | `BB_DPD_MONTHS` | MEAN | 70.0 | 70.0 | 0 | 0.50 | 7.43 | 0.031 |
| `BUREAU_BB_MONTHS_BALANCE_MIN_MEAN` | · | 254 | `BB_MONTHS_BALANCE_MIN` | MEAN | 70.0 | 70.0 | -33 | -34.22 | -2.50 | 0.089 |
| `BUREAU_DAYS_ENDDATE_FACT_SUM` | · | 261 | `DAYS_ENDDATE_FACT` | SUM | 14.3 | 0.0 | -1661 | -3062.04 | 0 | 0.049 |
| `BUREAU_AMT_CREDIT_MAX_OVERDUE_MIN` | · | 288 | `AMT_CREDIT_MAX_OVERDUE` | MIN | 40.2 | 40.2 | 0 | 3004.55 | 26,924 | 0.002 |
| `BUREAU_AMT_ANNUITY_MEAN` | · | 307 | `AMT_ANNUITY` | MEAN | 74.0 | 74.0 | 6494.16 | 16,052 | 110,524 | -0.001 |
| `BUREAU_BB_MONTHS_COUNT_MAX` | · | 320 | `BB_MONTHS_COUNT` | MAX | 70.0 | 70.0 | 41 | 44.29 | 95 | -0.069 |
| `BUREAU_BB_MONTHS_BALANCE_MIN_MIN` | · | 323 | `BB_MONTHS_BALANCE_MIN` | MIN | 70.0 | 70.0 | -57 | -56.49 | -3 | 0.073 |
| `BUREAU_AMT_CREDIT_SUM_LIMIT_SUM` | · | 325 | `AMT_CREDIT_SUM_LIMIT` | SUM | 14.3 | 0.0 | 0 | 20,190 | 450,000 | -0.009 |
| `BUREAU_BB_MONTHS_COUNT_MIN` | · | 327 | `BB_MONTHS_COUNT` | MIN | 70.0 | 70.0 | 9 | 14.40 | 81 | -0.048 |
| `BUREAU_AMT_ANNUITY_MAX` | · | 329 | `AMT_ANNUITY` | MAX | 74.0 | 74.0 | 12,491 | 34,838 | 263,421 | 0.001 |
| `BUREAU_BB_MONTHS_COUNT_SUM` | · | 341 | `BB_MONTHS_COUNT` | SUM | 14.3 | 0.0 | 0 | 47.81 | 540 | -0.014 |
| `BUREAU_BB_MONTHS_BALANCE_MIN_MAX` | · | 354 | `BB_MONTHS_BALANCE_MIN` | MAX | 70.0 | 70.0 | -9 | -15.05 | 0 | 0.050 |
| `BUREAU_AMT_CREDIT_SUM_LIMIT_MIN` | · | 380 | `AMT_CREDIT_SUM_LIMIT` | MIN | 21.2 | 21.2 | 0 | 1074.30 | 5392.40 | -0.005 |
| `BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM` | · | 387 | `AMT_CREDIT_SUM_OVERDUE` | SUM | 14.3 | 0.0 | 0 | 191.11 | 4.50 | 0.013 |
| `BUREAU_BB_CLOSED_MONTHS_MAX` | · | 392 | `BB_CLOSED_MONTHS` | MAX | 70.0 | 70.0 | 24 | 28.69 | 86 | -0.061 |
| `BUREAU_BB_CLOSED_MONTHS_SUM` | · | 403 | `BB_CLOSED_MONTHS` | SUM | 14.3 | 0.0 | 0 | 22.85 | 320 | -0.019 |
| `BUREAU_BB_MONTHS_BALANCE_MIN_SUM` | · | 415 | `BB_MONTHS_BALANCE_MIN` | SUM | 14.3 | 0.0 | 0 | -61.35 | 0 | 0.012 |
| `BUREAU_COUNT` | · | 433 | `–` | COUNT | 14.3 | 0.0 | 4 | 4.77 | 20 | 0.004 |
| `BUREAU_CLOSED_COUNT` | · | 434 | `CREDIT_ACTIVE` | DOMAIN | 14.3 | 0.0 | 2 | 2.98 | 15 | -0.031 |
| `BUREAU_AMT_ANNUITY_SUM` | · | 438 | `AMT_ANNUITY` | SUM | 14.3 | 0.0 | 0 | 17,188 | 220,779 | 0.002 |
| `BUREAU_CREDIT_ACTIVE_Sold_RATE` | · | 441 | `CREDIT_ACTIVE` | RATE of "Sold" | 14.3 | 0.0 | 0 | 0.00 | 0.11 | 0.017 |
| `BUREAU_AMT_ANNUITY_MIN` | · | 443 | `AMT_ANNUITY` | MIN | 74.0 | 74.0 | 0 | 7227.64 | 72,000 | -0.003 |
| `BUREAU_BB_CLOSED_MONTHS_MIN` | · | 452 | `BB_CLOSED_MONTHS` | MIN | 70.0 | 70.0 | 0 | 3.63 | 58 | -0.023 |
| `BUREAU_BB_DPD_MONTHS_MAX` | · | 467 | `BB_DPD_MONTHS` | MAX | 70.0 | 70.0 | 0 | 1.33 | 19 | 0.024 |
| `BUREAU_BB_DPD_MONTHS_MIN` | · | 470 | `BB_DPD_MONTHS` | MIN | 70.0 | 70.0 | 0 | 0.15 | 3 | 0.024 |
| `BUREAU_BB_DPD_MONTHS_SUM` | · | 561 | `BB_DPD_MONTHS` | SUM | 14.3 | 0.0 | 0 | 0.72 | 16 | 0.017 |
| `BUREAU_CREDIT_TYPE_NUNIQUE` | · | 599 | `CREDIT_TYPE` | NUNIQUE | 14.3 | 0.0 | 2 | 1.49 | 3 | 0.005 |
| `BUREAU_CNT_CREDIT_PROLONG_MEAN` | · | 621 | `CNT_CREDIT_PROLONG` | MEAN | 14.3 | 14.3 | 0 | 0.01 | 0.25 | 0.003 |
| `BUREAU_CREDIT_DAY_OVERDUE_MEAN` | · | 645 | `CREDIT_DAY_OVERDUE` | MEAN | 14.3 | 14.3 | 0 | 1.04 | 2.50 | 0.008 |
| `BUREAU_CREDIT_DAY_OVERDUE_MAX` | · | 700 | `CREDIT_DAY_OVERDUE` | MAX | 14.3 | 14.3 | 0 | 4.77 | 17 | 0.005 |
| `BUREAU_CREDIT_ACTIVE_NUNIQUE` | · | 733 | `CREDIT_ACTIVE` | NUNIQUE | 14.3 | 0.0 | 2 | 1.47 | 3 | -6.80e-04 |
| `BUREAU_CNT_CREDIT_PROLONG_MAX` | · | 766 | `CNT_CREDIT_PROLONG` | MAX | 14.3 | 14.3 | 0 | 0.04 | 1 | 0.004 |
| `BUREAU_CREDIT_DAY_OVERDUE_SUM` | · | 802 | `CREDIT_DAY_OVERDUE` | SUM | 14.3 | 0.0 | 0 | 4.24 | 9 | 0.006 |
| `BUREAU_CNT_CREDIT_PROLONG_SUM` | · | 873 | `CNT_CREDIT_PROLONG` | SUM | 14.3 | 0.0 | 0 | 0.03 | 1 | 0.004 |
| `BUREAU_CREDIT_DAY_OVERDUE_MIN` | · | 935 | `CREDIT_DAY_OVERDUE` | MIN | 14.3 | 14.3 | 0 | 0.16 | 0 | 0.008 |
| `BUREAU_CNT_CREDIT_PROLONG_MIN` | · | 936 | `CNT_CREDIT_PROLONG` | MIN | 14.3 | 14.3 | 0 | 0.00 | 0 | -1.80e-04 |
| `BUREAU_AMT_CREDIT_SUM_OVERDUE_MIN` | · | 937 | `AMT_CREDIT_SUM_OVERDUE` | MIN | 14.3 | 14.3 | 0 | 13.19 | 0 | 0 |
| `BUREAU_CREDIT_ACTIVE_Bad_debt_RATE` | · | 938 | `CREDIT_ACTIVE` | RATE of "Bad debt" | 14.3 | 0.0 | 0 | 1.03e-05 | 0 | 0.005 |
| `BUREAU_CREDIT_CURRENCY_NUNIQUE` | · | 939 | `CREDIT_CURRENCY` | NUNIQUE | 14.3 | 0.0 | 1 | 0.86 | 1 | -0.005 |
| `BUREAU_CREDIT_CURRENCY_currency_1_RATE` | · | 940 | `CREDIT_CURRENCY` | RATE of "currency 1" | 14.3 | 0.0 | 1 | 0.86 | 1 | 0.006 |
| `BUREAU_CREDIT_CURRENCY_currency_2_RATE` | · | 941 | `CREDIT_CURRENCY` | RATE of "currency 2" | 14.3 | 0.0 | 0 | 6.29e-04 | 0 | -0.007 |
| `BUREAU_CREDIT_CURRENCY_currency_3_RATE` | · | 942 | `CREDIT_CURRENCY` | RATE of "currency 3" | 14.3 | 0.0 | 0 | 6.32e-05 | 0 | 0.003 |
| `BUREAU_CREDIT_CURRENCY_currency_4_RATE` | · | 943 | `CREDIT_CURRENCY` | RATE of "currency 4" | 14.3 | 0.0 | 0 | 5.94e-06 | 0 | -0.001 |
| `BUREAU_CREDIT_ACTIVE_Active_RATE` | · | — | `CREDIT_ACTIVE` | RATE of "Active" | 14.3 | 0.0 | 0.33 | 0.35 | 1 | 0.077 |

### 7.8 Recent window — credit card, last 12 months (`CC_L12M_*`)

*16 features · 4 selected · 0.8% of model gain*

**Recent-activity windows.** Aggregated **directly to `SK_ID_CURR`**, deliberately bypassing the previous-application hub: recency is a customer-level activity signal, and keeping it flat keeps the block small. A lean mean/max/sum agg set over a curated column list, plus a record `COUNT`.

**Window** — `MONTHS_BALANCE >= -12` (months; these offsets are negative, counting back from the application date).

**Definition template** — `CC_L12M_<COL>_<AGG>` = the *<agg>* of `<COL>` over `credit_card_balance` rows inside the window.

<details><summary><b>Source-column glossary</b> (5 columns)</summary>

| Source column | What it measures |
|---|---|
| `AMT_BALANCE` | Balance during the month of previous credit |
| `AMT_CREDIT_LIMIT_ACTUAL` | Credit card limit during the month of the previous credit |
| `AMT_DRAWINGS_CURRENT` | Amount drawing during the month of the previous credit |
| `AMT_PAYMENT_CURRENT` | How much did the client pay during the month on the previous credit |
| `SK_DPD` | DPD (Days past due) during the month on the previous credit |

</details>

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `CC_L12M_AMT_DRAWINGS_CURRENT_MEAN` | **✔** | 127 | `AMT_DRAWINGS_CURRENT` | MEAN | 71.7 | 71.7 | 0 | 11,895 | 115,972 | 0.054 |
| `CC_L12M_AMT_BALANCE_MEAN` | **✔** | 134 | `AMT_BALANCE` | MEAN | 71.7 | 71.7 | 0 | 66,949 | 579,681 | 0.085 |
| `CC_L12M_AMT_BALANCE_MAX` | **✔** | 149 | `AMT_BALANCE` | MAX | 71.7 | 71.7 | 0 | 101,402 | 793,832 | 0.084 |
| `CC_L12M_AMT_DRAWINGS_CURRENT_MAX` | **✔** | 179 | `AMT_DRAWINGS_CURRENT` | MAX | 71.7 | 71.7 | 0 | 49,521 | 450,000 | 0.069 |
| `CC_L12M_AMT_DRAWINGS_CURRENT_SUM` | · | 330 | `AMT_DRAWINGS_CURRENT` | SUM | 71.7 | 0.0 | 0 | 32,168 | 651,230 | 0.047 |
| `CC_L12M_AMT_PAYMENT_CURRENT_MAX` | · | 375 | `AMT_PAYMENT_CURRENT` | MAX | 80.1 | 80.1 | 13,500 | 64,938 | 681,470 | 0.025 |
| `CC_L12M_AMT_BALANCE_SUM` | · | 394 | `AMT_BALANCE` | SUM | 71.7 | 0.0 | 0 | 202,992 | 4.69M | 0.075 |
| `CC_L12M_AMT_CREDIT_LIMIT_ACTUAL_SUM` | · | 414 | `AMT_CREDIT_LIMIT_ACTUAL` | SUM | 71.7 | 0.0 | 0 | 532,207 | 8.41M | 0.001 |
| `CC_L12M_AMT_CREDIT_LIMIT_ACTUAL_MEAN` | · | 420 | `AMT_CREDIT_LIMIT_ACTUAL` | MEAN | 71.7 | 71.7 | 120,000 | 183,726 | 900,000 | 8.90e-04 |
| `CC_L12M_AMT_PAYMENT_CURRENT_MEAN` | · | 444 | `AMT_PAYMENT_CURRENT` | MEAN | 80.1 | 80.1 | 7264.29 | 16,122 | 124,588 | 0.012 |
| `CC_L12M_SK_DPD_MEAN` | · | 519 | `SK_DPD` | MEAN | 71.7 | 71.7 | 0 | 9.52 | 12.42 | -0.004 |
| `CC_L12M_AMT_PAYMENT_CURRENT_SUM` | · | 521 | `AMT_PAYMENT_CURRENT` | SUM | 71.7 | 0.0 | 0 | 30,961 | 603,813 | 0.027 |
| `CC_L12M_AMT_CREDIT_LIMIT_ACTUAL_MAX` | · | 559 | `AMT_CREDIT_LIMIT_ACTUAL` | MAX | 71.7 | 71.7 | 135,000 | 205,751 | 900,000 | 0.002 |
| `CC_L12M_SK_DPD_SUM` | · | 676 | `SK_DPD` | SUM | 71.7 | 0.0 | 0 | 29.13 | 1 | -0.003 |
| `CC_L12M_COUNT` | · | 707 | `–` | COUNT | 71.7 | 0.0 | 0 | 2.88 | 12 | -0.019 |
| `CC_L12M_SK_DPD_MAX` | · | 716 | `SK_DPD` | MAX | 71.7 | 71.7 | 0 | 12.11 | 69 | -0.003 |

### 7.9 Recent window — POS/cash, last 12 months (`POS_L12M_*`)

*10 features · 7 selected · 1.4% of model gain*

**Recent-activity windows.** Aggregated **directly to `SK_ID_CURR`**, deliberately bypassing the previous-application hub: recency is a customer-level activity signal, and keeping it flat keeps the block small. A lean mean/max/sum agg set over a curated column list, plus a record `COUNT`.

**Window** — `MONTHS_BALANCE >= -12` (months; these offsets are negative, counting back from the application date).

**Definition template** — `POS_L12M_<COL>_<AGG>` = the *<agg>* of `<COL>` over `POS_CASH_balance` rows inside the window.

<details><summary><b>Source-column glossary</b> (3 columns)</summary>

| Source column | What it measures |
|---|---|
| `CNT_INSTALMENT_FUTURE` | Installments left to pay on the previous credit |
| `SK_DPD` | DPD (days past due) during the month of previous credit |
| `SK_DPD_DEF` | DPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit |

</details>

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `POS_L12M_CNT_INSTALMENT_FUTURE_MEAN` | **✔** | 34 | `CNT_INSTALMENT_FUTURE` | MEAN | 32.8 | 32.8 | 7.50 | 11.24 | 49.38 | 0.025 |
| `POS_L12M_CNT_INSTALMENT_FUTURE_MAX` | **✔** | 70 | `CNT_INSTALMENT_FUTURE` | MAX | 32.8 | 32.8 | 12 | 17.17 | 60 | 0.023 |
| `POS_L12M_SK_DPD_DEF_MEAN` | **✔** | 103 | `SK_DPD_DEF` | MEAN | 32.7 | 32.7 | 0 | 0.31 | 0.73 | 0.007 |
| `POS_L12M_COUNT` | **✔** | 119 | `–` | COUNT | 32.7 | 0.0 | 6 | 6.56 | 27 | -0.019 |
| `POS_L12M_SK_DPD_MEAN` | **✔** | 138 | `SK_DPD` | MEAN | 32.7 | 32.7 | 0 | 6.41 | 2.25 | 0.007 |
| `POS_L12M_SK_DPD_DEF_MAX` | **✔** | 157 | `SK_DPD_DEF` | MAX | 32.7 | 32.7 | 0 | 0.62 | 5 | 0.009 |
| `POS_L12M_CNT_INSTALMENT_FUTURE_SUM` | **✔** | 162 | `CNT_INSTALMENT_FUTURE` | SUM | 32.7 | 0.0 | 27 | 83.64 | 605 | 0.014 |
| `POS_L12M_SK_DPD_MAX` | · | 275 | `SK_DPD` | MAX | 32.7 | 32.7 | 0 | 9.30 | 14 | 0.008 |
| `POS_L12M_SK_DPD_SUM` | · | 290 | `SK_DPD` | SUM | 32.7 | 0.0 | 0 | 53.76 | 12 | 0.006 |
| `POS_L12M_SK_DPD_DEF_SUM` | · | 305 | `SK_DPD_DEF` | SUM | 32.7 | 0.0 | 0 | 3.38 | 5 | 0.006 |

### 7.10 Recent window — installments, last 365 days (`INST_L365D_*`)

*9 features · 8 selected · 1.9% of model gain*

**Recent-activity windows.** Aggregated **directly to `SK_ID_CURR`**, deliberately bypassing the previous-application hub: recency is a customer-level activity signal, and keeping it flat keeps the block small. A lean mean/max/sum agg set over a curated column list, plus a record `COUNT`.

**Window** — `DAYS_INSTALMENT >= -365` (days; these offsets are negative, counting back from the application date).

**Definition template** — `INST_L365D_<COL>_<AGG>` = the *<agg>* of `<COL>` over `installments_payments` rows inside the window.

<details><summary><b>Source-column glossary</b> (2 columns)</summary>

| Source column | What it measures |
|---|---|
| `AMT_INSTALMENT` | What was the prescribed installment amount of previous credit on this installment |
| `AMT_PAYMENT` | What the client actually paid on previous credit on this installment |

</details>

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `INST_L365D_LATE_RATE` | **✔** | 13 | `–` | DOMAIN | 29.3 | 0.0 | 0 | 0.04 | 0.50 | 0.083 |
| `INST_L365D_UNDERPAY_RATE` | **✔** | 87 | `–` | DOMAIN | 29.3 | 0.0 | 0 | 0.04 | 0.67 | 0.072 |
| `INST_L365D_AMT_INSTALMENT_MAX` | **✔** | 98 | `AMT_INSTALMENT` | MAX | 29.3 | 29.3 | 25,996 | 108,045 | 1.10M | 0.006 |
| `INST_L365D_AMT_PAYMENT_MAX` | **✔** | 100 | `AMT_PAYMENT` | MAX | 29.3 | 29.3 | 26,342 | 109,618 | 1.12M | 0.005 |
| `INST_L365D_AMT_PAYMENT_SUM` | **✔** | 120 | `AMT_PAYMENT` | SUM | 29.3 | 0.0 | 83,957 | 231,074 | 1.97M | -0.015 |
| `INST_L365D_AMT_PAYMENT_MEAN` | **✔** | 126 | `AMT_PAYMENT` | MEAN | 29.3 | 29.3 | 14,825 | 31,340 | 292,163 | -0.013 |
| `INST_L365D_AMT_INSTALMENT_MEAN` | **✔** | 151 | `AMT_INSTALMENT` | MEAN | 29.3 | 29.3 | 15,063 | 30,165 | 275,321 | -0.010 |
| `INST_L365D_COUNT` | **✔** | 188 | `–` | COUNT | 29.3 | 0.0 | 7 | 9.60 | 43 | 0.032 |
| `INST_L365D_AMT_INSTALMENT_SUM` | · | 287 | `AMT_INSTALMENT` | SUM | 29.3 | 0.0 | 85,500 | 221,829 | 1.69M | -0.006 |

### 7.11 Recent window — applications, last 365 days (`PREV_L365D_*`)

*14 features · 3 selected · 0.8% of model gain*

**Recent-activity windows.** Aggregated **directly to `SK_ID_CURR`**, deliberately bypassing the previous-application hub: recency is a customer-level activity signal, and keeping it flat keeps the block small. A lean mean/max/sum agg set over a curated column list, plus a record `COUNT`.

**Window** — `DAYS_DECISION >= -365` (days; these offsets are negative, counting back from the application date).

**Definition template** — `PREV_L365D_<COL>_<AGG>` = the *<agg>* of `<COL>` over `previous_application` rows inside the window.

<details><summary><b>Source-column glossary</b> (4 columns)</summary>

| Source column | What it measures |
|---|---|
| `AMT_ANNUITY` | Annuity of previous application |
| `AMT_APPLICATION` | For how much credit did client ask on the previous application |
| `AMT_CREDIT` | Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client initially applied for, but during our approval p… |
| `CNT_PAYMENT` | Term of previous credit at application of the previous application |

</details>

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `PREV_L365D_CNT_PAYMENT_MEAN` | **✔** | 147 | `CNT_PAYMENT` | MEAN | 54.1 | 54.1 | 12 | 18.70 | 60 | 0.022 |
| `PREV_L365D_AMT_ANNUITY_MEAN` | **✔** | 193 | `AMT_ANNUITY` | MEAN | 54.1 | 54.1 | 13,500 | 17,579 | 66,406 | -0.041 |
| `PREV_L365D_AMT_ANNUITY_MAX` | **✔** | 200 | `AMT_ANNUITY` | MAX | 54.1 | 54.1 | 15,885 | 20,845 | 78,112 | -0.034 |
| `PREV_L365D_AMT_CREDIT_MEAN` | · | 221 | `AMT_CREDIT` | MEAN | 45.0 | 45.0 | 88,265 | 155,840 | 1.05M | -0.012 |
| `PREV_L365D_AMT_APPLICATION_MEAN` | · | 225 | `AMT_APPLICATION` | MEAN | 45.0 | 45.0 | 84,402 | 145,796 | 959,625 | -0.016 |
| `PREV_L365D_AMT_CREDIT_MAX` | · | 262 | `AMT_CREDIT` | MAX | 45.0 | 45.0 | 145,075 | 328,686 | 1.97M | -0.004 |
| `PREV_L365D_CNT_PAYMENT_SUM` | · | 331 | `CNT_PAYMENT` | SUM | 45.0 | 0.0 | 0 | 16.86 | 192 | 0.031 |
| `PREV_L365D_AMT_APPLICATION_MAX` | · | 353 | `AMT_APPLICATION` | MAX | 45.0 | 45.0 | 135,000 | 302,100 | 1.80M | -0.008 |
| `PREV_L365D_AMT_ANNUITY_SUM` | · | 355 | `AMT_ANNUITY` | SUM | 45.0 | 0.0 | 0 | 15,007 | 148,614 | -0.002 |
| `PREV_L365D_CNT_PAYMENT_MAX` | · | 372 | `CNT_PAYMENT` | MAX | 54.1 | 54.1 | 12 | 22.92 | 60 | 0.031 |
| `PREV_L365D_AMT_CREDIT_SUM` | · | 459 | `AMT_CREDIT` | SUM | 45.0 | 0.0 | 0 | 294,979 | 3.74M | 0.008 |
| `PREV_L365D_AMT_APPLICATION_SUM` | · | 460 | `AMT_APPLICATION` | SUM | 45.0 | 0.0 | 0 | 271,887 | 3.42M | 0.005 |
| `PREV_L365D_COUNT` | · | 495 | `–` | COUNT | 45.0 | 0.0 | 1 | 1.63 | 11 | 0.038 |
| `PREV_L365D_APPROVED_RATE` | · | 529 | `–` | DOMAIN | 45.0 | 0.0 | 0 | 0.26 | 1 | -0.036 |

### 7.12 Recent window — bureau, last 730 days (`BUREAU_L730D_*`)

*14 features · 10 selected · 2.7% of model gain*

**Recent-activity windows.** Aggregated **directly to `SK_ID_CURR`**, deliberately bypassing the previous-application hub: recency is a customer-level activity signal, and keeping it flat keeps the block small. A lean mean/max/sum agg set over a curated column list, plus a record `COUNT`.

**Window** — `DAYS_CREDIT >= -730` (days; these offsets are negative, counting back from the application date).

**Definition template** — `BUREAU_L730D_<COL>_<AGG>` = the *<agg>* of `<COL>` over `bureau` rows inside the window.

<details><summary><b>Source-column glossary</b> (4 columns)</summary>

| Source column | What it measures |
|---|---|
| `AMT_CREDIT_SUM` | Current credit amount for the Credit Bureau credit |
| `AMT_CREDIT_SUM_DEBT` | Current debt on Credit Bureau credit |
| `CREDIT_DAY_OVERDUE` | Number of days past due on CB credit at the time of application for related loan in our sample |
| `DAYS_CREDIT` | How many days before current application did client apply for Credit Bureau credit |

</details>

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `BUREAU_L730D_DAYS_CREDIT_MAX` | **✔** | 28 | `DAYS_CREDIT` | MAX | 31.8 | 31.8 | -228 | -266.67 | -13 | 0.066 |
| `BUREAU_L730D_AMT_CREDIT_SUM_DEBT_MEAN` | **✔** | 31 | `AMT_CREDIT_SUM_DEBT` | MEAN | 35.5 | 35.5 | 83,243 | 292,463 | 3.41M | -0.009 |
| `BUREAU_L730D_AMT_CREDIT_SUM_MAX` | **✔** | 47 | `AMT_CREDIT_SUM` | MAX | 31.8 | 31.8 | 337,500 | 797,227 | 7.50M | -0.010 |
| `BUREAU_L730D_AMT_CREDIT_SUM_DEBT_MAX` | **✔** | 51 | `AMT_CREDIT_SUM_DEBT` | MAX | 35.5 | 35.5 | 151,088 | 526,421 | 6.11M | 2.70e-04 |
| `BUREAU_L730D_AMT_CREDIT_SUM_DEBT_SUM` | **✔** | 55 | `AMT_CREDIT_SUM_DEBT` | SUM | 31.8 | 0.0 | 8383.50 | 399,054 | 5.28M | 0.011 |
| `BUREAU_L730D_AMT_CREDIT_SUM_SUM` | **✔** | 57 | `AMT_CREDIT_SUM` | SUM | 31.8 | 0.0 | 205,754 | 780,526 | 8.05M | 0.004 |
| `BUREAU_L730D_DAYS_CREDIT_MEAN` | **✔** | 62 | `DAYS_CREDIT` | MEAN | 31.8 | 31.8 | -372.33 | -370.57 | -36 | 0.054 |
| `BUREAU_L730D_COUNT` | **✔** | 69 | `–` | COUNT | 31.8 | 0.0 | 1 | 1.83 | 9 | 0.072 |
| `BUREAU_L730D_AMT_CREDIT_SUM_MEAN` | **✔** | 96 | `AMT_CREDIT_SUM` | MEAN | 31.8 | 31.8 | 218,095 | 476,459 | 4.39M | -0.022 |
| `BUREAU_L730D_DAYS_CREDIT_SUM` | **✔** | 148 | `DAYS_CREDIT` | SUM | 31.8 | 0.0 | -447 | -682.61 | 0 | -0.036 |
| `BUREAU_L730D_ACTIVE_RATE` | · | 365 | `–` | DOMAIN | 31.8 | 0.0 | 0.50 | 0.46 | 1 | 0.040 |
| `BUREAU_L730D_CREDIT_DAY_OVERDUE_MEAN` | · | 665 | `CREDIT_DAY_OVERDUE` | MEAN | 31.8 | 31.8 | 0 | 0.17 | 0 | 0.019 |
| `BUREAU_L730D_CREDIT_DAY_OVERDUE_MAX` | · | 710 | `CREDIT_DAY_OVERDUE` | MAX | 31.8 | 31.8 | 0 | 0.41 | 0 | 0.020 |
| `BUREAU_L730D_CREDIT_DAY_OVERDUE_SUM` | · | 713 | `CREDIT_DAY_OVERDUE` | SUM | 31.8 | 0.0 | 0 | 0.30 | 0 | 0.023 |

### 7.13 Top-level coverage flags (`HAS_*`)

*2 features · 0 selected · 0.0% of model gain*

Set on the driver after both joins. Structurally encoded already (0/1, never null). Neither is selected — the information is fully carried by the `*_COUNT` columns and by nulls elsewhere, which is the expected result and a useful check that gain importance is not double-counting.

| Feature | Sel | Rank | Source column | Agg | Null % | Null after enc. | Median | Mean | P99 | Corr |
|---|:---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| `HAS_PREV` | · | 944 | `–` | coverage-flag | 0.0 | 0.0 | 1 | 0.95 | 1 | 0.018 |
| `HAS_BUREAU` | · | 945 | `–` | coverage-flag | 0.0 | 0.0 | 1 | 0.86 | 1 | -0.031 |

---

## 8. Appendix — the 11 dropped zero-variance columns

`constant_feature_columns` removes any column with ≤ 1 distinct non-null value *after* structural encoding. All 11 are the same artefact: taking `MIN` of a days-past-due or installment-number counter inside a loan: nearly every loan has at least one clean month and a first installment, so the inner minimum lands on the same value for every customer.

| Feature | Constant value |
|---|---|
| `PREV_CC_SK_DPD_DEF_MIN_MAX` | 0 |
| `PREV_CC_SK_DPD_DEF_MIN_MEAN` | 0 |
| `PREV_CC_SK_DPD_DEF_MIN_MIN` | 0 |
| `PREV_CC_SK_DPD_DEF_MIN_SUM` | 0 |
| `PREV_CC_SK_DPD_MIN_MAX` | 0 |
| `PREV_CC_SK_DPD_MIN_MEAN` | 0 |
| `PREV_CC_SK_DPD_MIN_MIN` | 0 |
| `PREV_CC_SK_DPD_MIN_SUM` | 0 |
| `PREV_INST_NUM_INSTALMENT_NUMBER_MIN_MIN` | 1 |
| `PREV_POS_SK_DPD_DEF_MIN_MIN` | 0 |
| `PREV_POS_SK_DPD_MIN_MIN` | 0 |

They carry no signal for any model and are dropped before fitting, alongside 13 bit-identical duplicates (§9), which is why the model sees 945 rather than 969 features.

## 9. Appendix — exact duplicate columns

14 groups of columns are **bit-identical** after structural encoding (35 columns in total). They are harmless for a tree — gain simply splits arbitrarily among them, which is one reason a low rank does not mean a weak feature — but the NN round should collapse each group to one column before scaling.

| Identical columns | Why | Best rank in group |
|---|---|---:|
| `PREV_REFUSED_RATE`<br>`PREV_NAME_CONTRACT_STATUS_Refused_RATE` | one-hot rate reproduces the hand-written domain rate (§4.3) | 10 |
| `PREV_APPROVED_RATE`<br>`PREV_NAME_CONTRACT_STATUS_Approved_RATE` | `aggregate_categorical`'s one-hot rate reproduces the hand-written domain rate exactly (§4.3) | 143 |
| `BUREAU_ACTIVE_RATE`<br>`BUREAU_CREDIT_ACTIVE_Active_RATE` | one-hot rate reproduces the hand-written domain rate (§4.3) | 223 |
| `PREV_NAME_CONTRACT_STATUS_Unused_offer_RATE`<br>`PREV_CODE_REJECT_REASON_CLIENT_RATE` | an unused offer is always coded with reject reason CLIENT — the two categories mark the same event | 554 |
| `PREV_CC_AMT_RECIVABLE_MIN_MAX`<br>`PREV_CC_AMT_TOTAL_RECEIVABLE_MIN_MAX` | different source columns (`AMT_RECIVABLE`, `AMT_TOTAL_RECEIVABLE`) collapse to the same values once the inner aggregation is taken | 606 |
| `PREV_CC_AMT_RECIVABLE_MIN_MEAN`<br>`PREV_CC_AMT_TOTAL_RECEIVABLE_MIN_MEAN` | different source columns (`AMT_RECIVABLE`, `AMT_TOTAL_RECEIVABLE`) collapse to the same values once the inner aggregation is taken | 615 |
| `PREV_CC_MONTHS_BALANCE_MAX_MEAN`<br>`PREV_CC_MONTHS_BALANCE_MAX_MIN`<br>`PREV_CC_MONTHS_BALANCE_MAX_MAX` | the inner aggregation returns the same value on every one of the customer's loans, so the outer MEAN/MIN/MAX coincide | 709 |
| `PREV_CC_AMT_RECIVABLE_MIN_MIN`<br>`PREV_CC_AMT_TOTAL_RECEIVABLE_MIN_MIN` | different source columns (`AMT_RECIVABLE`, `AMT_TOTAL_RECEIVABLE`) collapse to the same values once the inner aggregation is taken | 712 |
| `PREV_CC_AMT_RECIVABLE_MIN_SUM`<br>`PREV_CC_AMT_TOTAL_RECEIVABLE_MIN_SUM` | different source columns (`AMT_RECIVABLE`, `AMT_TOTAL_RECEIVABLE`) collapse to the same values once the inner aggregation is taken | 764 |
| `PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MIN_MEAN`<br>`PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MIN_MIN`<br>`PREV_CC_AMT_DRAWINGS_OTHER_CURRENT_MIN_MAX` | the inner aggregation returns the same value on every one of the customer's loans, so the outer MEAN/MIN/MAX coincide | 894 |
| `PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MIN_MEAN`<br>`PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MIN_MIN`<br>`PREV_CC_CNT_DRAWINGS_OTHER_CURRENT_MIN_MAX` | the inner aggregation returns the same value on every one of the customer's loans, so the outer MEAN/MIN/MAX coincide | 901 |
| `PREV_CC_SK_DPD_MIN_MEAN`<br>`PREV_CC_SK_DPD_MIN_MIN`<br>`PREV_CC_SK_DPD_MIN_MAX`<br>`PREV_CC_SK_DPD_DEF_MIN_MEAN`<br>`PREV_CC_SK_DPD_DEF_MIN_MIN`<br>`PREV_CC_SK_DPD_DEF_MIN_MAX` | different source columns (`SK_DPD`, `SK_DPD_DEF`) collapse to the same values once the inner aggregation is taken | dropped (constant) |
| `PREV_CC_SK_DPD_MIN_SUM`<br>`PREV_CC_SK_DPD_DEF_MIN_SUM` | different source columns (`SK_DPD`, `SK_DPD_DEF`) collapse to the same values once the inner aggregation is taken | dropped (constant) |
| `PREV_POS_SK_DPD_MIN_MIN`<br>`PREV_POS_SK_DPD_DEF_MIN_MIN` | different source columns (`SK_DPD`, `SK_DPD_DEF`) collapse to the same values once the inner aggregation is taken | dropped (constant) |

## 10. Reproducing this document

```python
app = pl.read_csv(datasets / 'application_train.csv', infer_schema_length=10000)
fm  = build_feature_matrix(app, datasets)          # 969 candidates + TARGET + SK_ID_CURR
enc = apply_structural_encoding(fm)                # statistics below are on this
feature_cols, y = split_xy(enc)
constants = constant_feature_columns(enc, feature_cols)   # §8
dupes     = duplicate_feature_columns(enc, feature_cols)   # §9
```

Gain ranks come from `gbt_gain_importance.csv` and the shortlist from `gbt_shortlist.txt`, both written by [`baseline_trees.ipynb`](../notebooks/modeling/baseline_trees.ipynb) §5 to the dataset directory (outside the repo).
