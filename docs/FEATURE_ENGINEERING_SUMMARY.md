# Feature Engineering Summary — Customer-level Features for TARGET

Feature engineering for predicting `TARGET` (probability a customer defaults). The
pipeline aggregates every auxiliary Home Credit table to one row per `SK_ID_CURR` and
joins it onto the application driver. Iteration 1 built numeric aggregations + curated
domain features; **iteration 2 added categorical encoding and time-windowed
aggregations** (this document reflects both).

- **Code:** [`src/home_credit_risk/features.py`](../src/home_credit_risk/features.py)
- **EDA notebook:**
  [`notebooks/feature_engineering/feature_matrix_eda.ipynb`](../notebooks/feature_engineering/feature_matrix_eda.ipynb)
- **Tests:** [`tests/test_features.py`](../tests/test_features.py)

## Pipeline

Two lineages, both aggregated to `SK_ID_CURR` and left-joined onto the driver
(`build_feature_matrix`). **No imputation is applied** — unmatched left joins leave nulls
for this analysis to interpret.

1. **Previous-application hub** (`SK_ID_PREV` as the consolidation grain). The three
   ledger tables (`credit_card_balance`, `installments_payments`, `POS_CASH_balance`) are
   consolidated **per `SK_ID_PREV`** (per prior loan), left-joined onto
   `previous_application` so each prior loan carries its own repayment behaviour, then the
   enriched frame is rolled up **to `SK_ID_CURR`**. This equal-weights each prior loan
   (a direct ledger→customer aggregation would weight customers by their number of monthly
   rows). The `365243` "never/NA" sentinel in `previous_application`'s `DAYS_*` columns is
   nulled out before aggregating.
2. **Bureau lineage** (`SK_ID_BUREAU`). `bureau_balance` aggregated per `SK_ID_BUREAU`
   (months observed, DPD-month count where `STATUS in {1..5}`, closed months), bridged via
   `bureau`, rolled up to `SK_ID_CURR`.

Each block is generic numeric aggregations (`mean`/`min`/`max`/`sum` + a record `COUNT`,
`SK_ID*` excluded) plus curated domain features. Built identically for train and test.

**Iteration 2 additions** (toggled by `include_categorical` / `include_windows`, default on):
3. **Categorical encoding** (`aggregate_categorical`) — for `previous_application` and
   `bureau`, a per-customer distinct-count (`*_NUNIQUE`) for every categorical, plus a
   one-hot **rate** per category (`*_{VALUE}_RATE`) for low-cardinality columns
   (`n_unique ≤ 10`). High-cardinality columns (`NAME_GOODS_CATEGORY`, `CREDIT_TYPE`, …)
   get the distinct-count only.
4. **Time-windowed aggregations** (`aggregate_recent`) — recent-window blocks aggregated
   **directly to `SK_ID_CURR`** over curated high-signal columns: `CC_L12M` / `POS_L12M`
   (last 12 `MONTHS_BALANCE`), `INST_L365D` / `PREV_L365D` (last 365 `DAYS_*`), `BUREAU_L730D`
   (last 730 `DAYS_CREDIT`). This block deliberately bypasses the loan-weighted hub —
   recency is a customer-level activity signal — keeping it flat and bounded.

**Iteration-3 additions** — two treatments applied to the **driver frame itself**, before
any join, covering what the aggregation pipeline structurally cannot produce:

5. **Sentinel cleaning** (`clean_application_sentinels`) — `DAYS_EMPLOYED` uses the same
   `365243` "never" sentinel as `previous_application`, in **18% of rows**, and it had
   only ever been cleaned on the auxiliary side. It drags the column's mean from a median
   of ≈-1,200 days to ≈+63,800. A null-native tree isolates the spike with one split, but
   median-impute + standard-scale (the NN / transformer rounds) is destroyed by it. The
   value is nulled and the group preserved as `DAYS_EMPLOYED_ANOMALY` — it is real signal,
   defaulting at **5.4%** against **8.7%** for everyone else.
6. **Affordability ratios** (`add_ratio_features`) — a tree splits on one column at a time
   and cannot express a quotient of two of its own inputs, no matter how deep it grows:
   `RATIO_PAYMENT_RATE` (annuity/credit), `RATIO_CREDIT_TO_INCOME`,
   `RATIO_ANNUITY_TO_INCOME`, `RATIO_CREDIT_TO_GOODS`, `RATIO_INCOME_PER_PERSON`,
   `RATIO_EMPLOYED_TO_AGE`. A zero or null denominator yields null, never an infinity.

## Result

- **Feature matrix:** `application_train` 307,511 × **971** (122 driver + **7
  driver-derived** + **842 engineered**); `application_test` 48,744 × 970 (identical
  columns minus `TARGET`). Driver grain preserved (one row per `SK_ID_CURR`).
- **Feature blocks:** iteration-1 numeric/domain ~684 (`PREV_*` incl. `PREV_CC_*` /
  `PREV_INST_*` / `PREV_POS_*`, `BUREAU_*`, `HAS_*`); iteration-2 **categorical ~99**
  (~80 `*_RATE` + ~19 `*_NUNIQUE`) and **windowed ~59** (`*_L12M` / `*_L365D` / `*_L730D`);
  iteration-3 **driver-derived 7** (6 `RATIO_*` + `DAYS_EMPLOYED_ANOMALY`).
- Per-feature definitions, statistics and selection status for every column:
  [`FEATURE_DICTIONARY.md`](FEATURE_DICTIONARY.md).

## Coverage

| Flag | Meaning | Rate |
|---|---|---|
| `HAS_PREV` | customer has ≥1 previous application | **94.6%** |
| `HAS_BUREAU` | customer has ≥1 bureau record | **85.7%** |
| `PREV_HAS_INST_RATE` | fraction of prior loans with installments | 70.7% |
| `PREV_HAS_POS_RATE` | fraction of prior loans with POS | 66.9% |
| `PREV_HAS_CC_RATE` | fraction of prior loans with a credit card | **5.7%** |
| `DAYS_EMPLOYED_ANOMALY` | driver rows carrying the `365243` sentinel | **18.0%** |

Credit cards are rare among prior loans, which is why the `PREV_CC_*` block is mostly null.

## Missingness by block (imputation input)

| Block | Cols | Mean null-rate | Why null |
|---|---|---|---|
| `PREV_` (all) | 620 | 35.6% | ~5% have no prior app; CC block dominates |
| `PREV_CC` | 332 | **58.9%** | most prior loans aren't credit cards |
| `PREV_INST` | 116 | 5.8% | most prior loans have installments |
| `PREV_POS` | 88 | 6.4% | most prior loans have POS records |
| `BUREAU_` | 68 | 28.8% | ~14% have no bureau + some `bureau_balance` gaps |
| categorical `*_RATE` | 80 | **9.1%** | null only when a customer has no prior app / bureau |
| categorical `*_NUNIQUE` | 19 | 6.8% | as above |
| windowed `*_L730D` | 14 | 32.4% | no bureau credit in the last 2 years |
| windowed `*_L365D` | 23 | 40.4% | no prev app / installments in the last year |
| windowed `*_L12M` | 26 | 57.4% | no credit-card / POS months in the last year |

## Signal — top features correlated with TARGET

Individual aggregates are weak (expected for this problem), but **the iteration-2
categorical and windowed features rank among the strongest engineered signals** — several
beat every iteration-1 numeric aggregate:

| corr | feature |
|---|---|
| +0.082 | `BUREAU_DAYS_CREDIT_MEAN` (older average bureau history) |
| +0.078 | `PREV_CC_UTILIZATION_MEAN_*` (credit-card utilization) |
| +0.078 | `PREV_REFUSED_RATE` = `PREV_NAME_CONTRACT_STATUS_Refused_RATE` (categorical) |
| +0.075 | `PREV_INST_LATE_RATE_MEAN` |
| −0.074 | `PREV_CODE_REJECT_REASON_XAP_RATE` (categorical; XAP = not rejected) |
| −0.070 | `BUREAU_CREDIT_ACTIVE_Closed_RATE` (categorical) |
| +0.069 | `INST_L365D_LATE_RATE` (**windowed** — recent late payments) |
| +0.069 | `BUREAU_ACTIVE_RATE` |
| +0.061 | `BUREAU_L730D_COUNT` (**windowed** — recent bureau activity) |

More refused prior applications, higher card utilization, recent late installments, and
more active (vs closed) bureau credit all associate with higher default risk.

## Imputation plan (for the modelling step)

Nulls here are almost always **"no such history"**, not measurement gaps — so impute with
intent, and always keep the `HAS_*` / `*_RATE` flags as missing-indicators.

- **Coverage flags** (`HAS_PREV`, `HAS_BUREAU`, `PREV_HAS_*_RATE`) — keep as-is; set the
  `*_RATE` to `0` for customers with no prior loans.
- **Counts & sums** (`*_COUNT`, `*_SUM`, `BUREAU_ACTIVE_COUNT`, …) — impute **0** (no history
  ⇒ zero records / zero total).
- **Means, rates, min/max behavioural features** (`PREV_INST_LATE_RATE*`, `BUREAU_*_MEAN`,
  `PREV_CC_*`) — genuinely unknown when absent. For **gradient-boosted trees**
  (the recommended baseline model — they split on nulls natively) leave them null. For a
  **linear / neural model**, median-impute and add the `HAS_*` indicator.
- **`PREV_CC_*` (59% null)** — dominated by "no credit card"; treat counts/sums as 0, leave
  means null-or-median. `PREV_HAS_CC_RATE` already encodes card prevalence.
- **Categorical `*_RATE` / `*_NUNIQUE` (low null)** — null only when the customer has no
  prior app / bureau; impute the rate/count to `0` (keep the `HAS_*` flag).
- **Windowed `*_L12M` / `*_L365D` / `*_L730D` (32–57% null)** — null means no activity in the
  window; treat the recent `COUNT` as `0`, leave the recent means null (trees) or median-fill
  (linear/NN) with the count as the missing-indicator.

## Next steps

- **Done:** GBT baseline + null-aware feature selection — see
  [`docs/MODELING_SUMMARY.md`](MODELING_SUMMARY.md). Constant zero-variance and
  bit-identical columns are dropped, structural nulls are encoded to `0`, and features are
  ranked by LightGBM gain importance. Every run is tracked in MLflow.
- **Remaining:** GBT hyperparameter search, then the **neural-net round** on the
  tree-selected shortlist (per-fold `SimpleImputer(median) → StandardScaler → MLP`, with
  missingness indicators and embeddings for the high-cardinality categoricals), then
  **tabular transformers**.

### Feature ideas not yet built

Ordered by expected value, for the next feature iteration:

- **More ratios, on the aggregated blocks** — the iteration-3 ratios only touch the driver.
  The same argument applies further down: `PREV_AMT_CREDIT_SUM / AMT_CREDIT` (how this loan
  compares to prior ones), `BUREAU_AMT_CREDIT_SUM_DEBT_SUM / AMT_INCOME_TOTAL` (external
  debt burden), `INST_L365D_AMT_PAYMENT_SUM / AMT_INCOME_TOTAL` (recent cash outflow).
- **Trends, not just levels** — every aggregate here is a level or an extreme. A slope over
  `MONTHS_BALANCE` (balance rising vs falling, DPD worsening vs recovering) is a different
  signal and is invisible to mean/min/max.
- **Prune the `PREV_CC_*` block** — 332 features for 4.3% of gain; the four-agg cross
  product over a table covering 25% of customers is mostly redundancy that costs fit time
  and dilutes `feature_fraction` sampling.
