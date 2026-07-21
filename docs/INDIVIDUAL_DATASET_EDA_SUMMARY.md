# EDA Summary — Home Credit Default Risk

Detailed summary of the exploratory data analysis completed for each Home Credit
dataset. Every notebook shares the same skeleton — load (polars) → basic info →
missing data → numeric → categorical → generic data-quality → a table-specific
**Functional / Business-Logic Quality Checks** section — and reuses the helpers in
[`src/home_credit_risk/eda_utils.py`](../src/home_credit_risk/eda_utils.py).

Datasets are loaded from a local copy of the Kaggle Home Credit Default Risk data
(not stored in the repo). All counts below were produced against that data.

## Datasets at a glance

All notebooks live in
[`notebooks/individual_dataset_eda/`](../notebooks/individual_dataset_eda).

| Dataset | Notebook | Rows × Cols | Grain | Has `TARGET` |
|---|---|---|---|---|
| application_train | [application_eda.ipynb](../notebooks/individual_dataset_eda/application_eda.ipynb) | 307,511 × 122 | 1 per `SK_ID_CURR` | ✅ |
| bureau | [bureau_eda.ipynb](../notebooks/individual_dataset_eda/bureau_eda.ipynb) | 1,716,428 × 17 | 1 per `SK_ID_BUREAU` | — |
| bureau_balance | [bureau_eda.ipynb](../notebooks/individual_dataset_eda/bureau_eda.ipynb) | 27,299,925 × 3 | monthly per `SK_ID_BUREAU` | — |
| previous_application | [previous_application_eda.ipynb](../notebooks/individual_dataset_eda/previous_application_eda.ipynb) | 1,670,214 × 37 | 1 per `SK_ID_PREV` | — |
| credit_card_balance | [credit_card_balance_eda.ipynb](../notebooks/individual_dataset_eda/credit_card_balance_eda.ipynb) | 3,840,312 × 23 | monthly per `SK_ID_PREV` | — |
| installments_payments | [installments_payments_eda.ipynb](../notebooks/individual_dataset_eda/installments_payments_eda.ipynb) | 13,605,401 × 8 | payment per installment | — |
| POS_CASH_balance | [pos_cash_balance_eda.ipynb](../notebooks/individual_dataset_eda/pos_cash_balance_eda.ipynb) | 10,001,358 × 8 | monthly per `SK_ID_PREV` | — |

---

## 1. `application_train.csv` — `application_eda.ipynb`

**Shape:** 307,511 × 122 · **Grain:** 1 row per `SK_ID_CURR` · **the only table with `TARGET`.**

**Standard sections:** basic info, describe, missing patterns, duplicates (on
`SK_ID_CURR`), target distribution, numeric features, outliers, categorical features,
correlations with `TARGET`, generic data quality, feature-engineering opportunities.

**Key findings:**
- **Missing data:** 67 columns have nulls, several ~70% (the `COMMONAREA_*`,
  `NONLIVINGAPARTMENTS_*` building-info block).
- **Correlation with `TARGET`** (all weak, as expected): `DAYS_BIRTH` +0.078,
  `REGION_RATING_CLIENT_W_CITY` +0.061, `DAYS_LAST_PHONE_CHANGE` +0.055, …
- **Application-specific business checks:** `AMT_CREDIT > AMT_GOODS_PRICE` in
  **64.6%** (198,763 — expected, credit bundles fees/insurance); no negative amounts;
  `DAYS_BIRTH` range −25,229..−7,489, zero positives.
- **Generic quality:** no fully-null, no constant columns, 0 duplicate rows.
- **Feature-engineering candidates:** `credit_to_goods_ratio`,
  `annuity_to_income_ratio`, `credit_to_income_ratio`, `age_years`.

---

## 2. `bureau.csv` + `bureau_balance.csv` — `bureau_eda.ipynb`

**Shapes:** bureau 1,716,428 × 17 · bureau_balance **27,299,925 × 3** ·
**merged (left join on `SK_ID_BUREAU`): 25,121,815 rows.**

This is the notebook that drove the **pandas → lazy-polars migration** (`scan_csv`,
dtype overrides, `STATUS` as Categorical).

**Key finding — Bureau Balance join coverage** (custom Section 5). Of 305,811
customers with bureau records:
- **56.00%** (171,269) have **no** `bureau_balance` history for *any* of their credits
- **43.85%** (134,108) have history for **all** their credits
- only **0.14%** (434) partial → coverage is essentially **all-or-nothing per
  customer**, making "has any balance history" a clean candidate feature.

---

## 3. `previous_application.csv` — `previous_application_eda.ipynb`

**Shape:** 1,670,214 × 37 · **Grain:** 1 row per `SK_ID_PREV` (verified unique).

**Functional checks & findings:**
- **A. Status consistency:** Approved missing annuity/term → **8**; Approved with
  credit null/0 → **1,552**; status ↔ `CODE_REJECT_REASON` mismatch → **9**.
- **B. Amounts:** `AMT_APPLICATION` vs `AMT_CREDIT` → 21.4% >, 39.4% <, and
  **23.5% (392,402)** have `AMT_APPLICATION == 0` (cash loans). *This disproved the
  initial "applied ≥ credit" assumption.* Down-payment out of bounds → 2.
- **C. Sentinel/placeholder:** `365243` sentinel in the DAYS columns — **55.95%** of
  `DAYS_FIRST_DRAWING`, ~13% of `DAYS_TERMINATION`/`DAYS_LAST_DUE`. `XNA`/`XAP`
  placeholders dominate several categoricals (81% of `CODE_REJECT_REASON`, 63.7% of
  `NAME_PRODUCT_TYPE`).
- **D. Temporal ordering:** `DAYS_LAST_DUE > DAYS_TERMINATION` in **389** rows.
- **E. Duplicate flags:** `FLAG_LAST_APPL_PER_CONTRACT == N` → 8,475;
  `NFLAG_LAST_APPL_IN_DAY == 0` → 5,900.

---

## 4. `credit_card_balance.csv` — `credit_card_balance_eda.ipynb`

**Shape:** 3,840,312 × 23 · **Grain:** `(SK_ID_PREV, MONTHS_BALANCE)` unique ·
monthly panel, `MONTHS_BALANCE` −96..−1.

**Functional checks & findings:**
- **B. Decomposition:** drawings **amount** ≠ ATM+POS+OTHER in **7,150 (0.19%)**,
  while the **count** decomposition is **perfect (0)** — a telling asymmetry.
- **C. Orderings:** `AMT_RECEIVABLE_PRINCIPAL > AMT_RECIVABLE` in **107,994 (2.81%)**;
  balance over credit limit 342,354 (8.9%, informational).
- **D. DPD:** `SK_DPD_DEF ≤ SK_DPD` always holds.
- **E. Non-negativity:** `AMT_BALANCE < 0` → 2,345;
  **`AMT_TOTAL_RECEIVABLE < 0` → 109,330 (2.85%)**.
- **F. Count ↔ amount:** ATM perfectly consistent (0), POS 1, OTHER 475.
- **G. Status:** Completed with balance > 0 → 135. Sentinel/placeholder clean.

---

## 5. `installments_payments.csv` — `installments_payments_eda.ipynb`

**Shape:** 13,605,401 × 8 (largest) · **Grain:** key
`(SK_ID_PREV, VERSION, NUMBER)` **not** unique (partial payments) · all-numeric table.

**Functional checks & findings — this is where repayment *behavior* lives:**
- **A.** 653,483 rows (4.8%) share an installment key (multi-row payments).
- **B. Presence:** `DAYS_ENTRY_PAYMENT` & `AMT_PAYMENT` null together perfectly
  (0 half-null); **2,905 never-paid** installments.
- **C. Timeliness:** **8.43% late** (1,146,669), mean **12.3 days** late, max 2,884.
- **D. Completeness:** **9.52% underpaid** (1,295,493), 1.32% overpaid,
  1,438 zero-on-due.
- **E.** Non-negativity clean. Sentinel clean (this is where the
  `SK_ID_CURR == 365243` false-positive was found, prompting the auto-skip of
  `SK_ID*` identifier columns in `detect_sentinel_values`).

---

## 6. `POS_CASH_balance.csv` — `pos_cash_balance_eda.ipynb`

**Shape:** 10,001,358 × 8 · **Grain:** `(SK_ID_PREV, MONTHS_BALANCE)` unique ·
monthly panel, `MONTHS_BALANCE` −96..−1.

**Functional checks & findings:**
- **B. Bounds:** `CNT_INSTALMENT_FUTURE > CNT_INSTALMENT` (owe more than the term) →
  **7,420 (0.074%)**.
- **C. DPD:** `SK_DPD_DEF ≤ SK_DPD` always holds.
- **E. Status:** Completed with installments left → 4;
  **Active with 0 installments left → 432,501 (4.3%)** (near-completion signal).
- **F.** Sentinel clean; **`XNA` × 2** in `NAME_CONTRACT_STATUS` (caught by the
  placeholder helper).

---

## Cross-cutting themes

- **Sentinels & placeholders** matter a lot in `previous_application` (the `365243`
  DAYS sentinel, pervasive `XNA`/`XAP`); the balance/ledger tables use genuine nulls
  instead. Handled by the shared `detect_sentinel_values` (auto-skips `SK_ID*`) and
  `check_categorical_placeholders` helpers.
- **Behavioral risk signals** are concentrated in the ledger tables — late-payment
  rate, underpayment rate, DPD, over-limit months, remaining-installment behavior —
  all at `SK_ID_PREV`/monthly grain, i.e. they need aggregating to `SK_ID_CURR` to
  become model features.
- **Join coverage / "has-history" flags** are themselves signals (56% of bureau
  customers lack `bureau_balance`; 4.3% of POS Active contracts fully paid down).

## Shared infrastructure

- `eda_utils.py` (polars) provides the shared helpers used by every notebook:
  `display_basic_info`, `show_missing_patterns`, `analyze_target_distribution`,
  `check_duplicates`, `analyze_numeric_features`, `identify_outliers`,
  `analyze_categorical_features`, `analyze_correlations`, `check_data_quality`
  (generic: fully-null / constant / duplicate-row checks),
  `identify_feature_engineering_opportunities`, `detect_sentinel_values`,
  `check_categorical_placeholders`.
- Numeric analysis helpers auto-exclude `SK_ID*` identifier columns (meaningless in
  describe / correlation / outlier detection) while the null-analysis helpers keep
  them. Covered by `tests/test_eda_utils.py`.

## Not yet done

- **Combined / customer-level dataset EDA.** The auxiliary tables are at finer grains
  than `application_train`, so the meaningful "combined" view is an
  *aggregate-to-`SK_ID_CURR`-then-join* feature matrix (not a naive flat join),
  followed by coverage/missingness analysis and correlation of the aggregated
  behavioral features with `TARGET`. This is the natural bridge from EDA to modeling.
