"""
Customer-level feature engineering for the Home Credit default-risk model.

Aggregates the auxiliary tables to one row per ``SK_ID_CURR`` and joins them onto the
application driver table. Two lineages are assembled (see
``docs/INDIVIDUAL_DATASET_EDA_SUMMARY.md`` for the data model):

1. **Previous-application hub** — the three ledger tables (credit card, installments,
   POS) are consolidated per ``SK_ID_PREV`` (a prior loan), joined onto
   ``previous_application`` so each loan carries its repayment behaviour, then rolled up
   to ``SK_ID_CURR``.
2. **Bureau lineage** — ``bureau_balance`` is aggregated per ``SK_ID_BUREAU`` and
   bridged through ``bureau`` to ``SK_ID_CURR``.

The driver frame itself gets two treatments before the joins
(:func:`clean_application_sentinels`, :func:`add_ratio_features`) — the ``365243``
"never" sentinel is nulled and flagged, and the classic affordability ratios are added,
since a tree cannot construct a quotient of two of its own inputs.

Report-only / no imputation: nulls from unmatched left joins are left in place for the
EDA to analyse.
"""

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import polars as pl
import polars.selectors as cs

from home_credit_risk.eda_utils import _measurement_numeric_columns

# The four generic aggregations, as explicit expression builders (keeps mypy strict
# happy and the aggregation set easy to extend).
_AGG_FUNCS: Dict[str, Callable[[pl.Expr], pl.Expr]] = {
    "mean": lambda e: e.mean(),
    "min": lambda e: e.min(),
    "max": lambda e: e.max(),
    "sum": lambda e: e.sum(),
}

_DEFAULT_AGGS: Sequence[str] = ("mean", "min", "max", "sum")


def _prefixed(prefix: str, *parts: str) -> str:
    """Join a prefix and name parts with underscores, skipping empty pieces."""
    return "_".join(p for p in (prefix, *parts) if p)


def aggregate_numeric(
    df: pl.DataFrame,
    group_col: str = "SK_ID_CURR",
    prefix: str = "",
    exclude: Optional[List[str]] = None,
    aggs: Sequence[str] = _DEFAULT_AGGS,
) -> pl.DataFrame:
    """
    Aggregate every numeric measurement column of ``df`` by ``group_col``.

    Produces one row per ``group_col`` with columns named ``{prefix}_{col}_{AGG}`` plus
    a ``{prefix}_COUNT`` record count. ``SK_ID*`` identifier columns are excluded
    automatically (via :func:`eda_utils._measurement_numeric_columns`).

    Parameters
    ----------
    df : pl.DataFrame
        The table to aggregate.
    group_col : str
        Grouping key (e.g. ``SK_ID_CURR`` or ``SK_ID_PREV``).
    prefix : str
        Prefix for the generated column names.
    exclude : List[str], optional
        Additional columns to exclude from aggregation.
    aggs : Sequence[str]
        Which aggregations to compute (subset of mean/min/max/sum).
    """
    unknown = [a for a in aggs if a not in _AGG_FUNCS]
    if unknown:
        raise ValueError(f"Unsupported aggregations: {unknown}")

    drop = list(exclude or []) + [group_col]
    num_cols = _measurement_numeric_columns(df, exclude=drop)

    exprs: List[pl.Expr] = [pl.len().alias(_prefixed(prefix, "COUNT"))]
    for col in num_cols:
        for agg in aggs:
            exprs.append(
                _AGG_FUNCS[agg](pl.col(col)).alias(_prefixed(prefix, col, agg.upper()))
            )

    return df.group_by(group_col).agg(exprs)


def _slug(value: str) -> str:
    """Make a category value safe for use in a column name (alphanumerics + ``_``)."""
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")


def aggregate_categorical(
    df: pl.DataFrame,
    group_col: str = "SK_ID_CURR",
    prefix: str = "",
    columns: Optional[List[str]] = None,
    max_cardinality: int = 10,
) -> pl.DataFrame:
    """
    Aggregate categorical (string) columns of ``df`` by ``group_col``.

    For every categorical column, emits a per-group distinct-count
    ``{prefix}_{col}_NUNIQUE``. For a low-cardinality column (non-null distinct values
    ≤ ``max_cardinality``) also emits, per category value, the per-group **rate**
    ``{prefix}_{col}_{VALUE}_RATE`` (the mean of ``col == value``, i.e. the proportion
    of the group's records in that category). High-cardinality columns get only the
    distinct-count, keeping the output bounded.

    Only string/categorical columns actually present in ``df`` are used (robust to
    minimal frames); pass ``columns`` to restrict to a specific subset.

    Parameters
    ----------
    df : pl.DataFrame
        The table to aggregate.
    group_col : str
        Grouping key.
    prefix : str
        Prefix for the generated column names.
    columns : List[str], optional
        Restrict to these categorical columns (intersected with what's present).
    max_cardinality : int
        One-hot rates are only produced for columns with ≤ this many distinct values.
    """
    cat_cols = df.select(cs.string() | cs.categorical()).columns
    if columns is not None:
        cat_cols = [c for c in cat_cols if c in columns]

    exprs: List[pl.Expr] = []
    for col in cat_cols:
        exprs.append(pl.col(col).n_unique().alias(_prefixed(prefix, col, "NUNIQUE")))
        values = df[col].drop_nulls().unique().to_list()
        if len(values) <= max_cardinality:
            for value in sorted(values):
                exprs.append(
                    (pl.col(col) == value)
                    .mean()
                    .alias(_prefixed(prefix, col, _slug(str(value)), "RATE"))
                )

    if not exprs:
        return df.select(group_col).unique()

    return df.group_by(group_col).agg(exprs)


def aggregate_recent(
    df: pl.DataFrame,
    time_col: str,
    window: int,
    group_col: str = "SK_ID_CURR",
    prefix: str = "",
    columns: Optional[List[str]] = None,
    aggs: Sequence[str] = ("mean", "max", "sum"),
    domain: Optional[List[pl.Expr]] = None,
) -> pl.DataFrame:
    """
    Aggregate a recent time window of ``df`` by ``group_col``.

    Keeps rows within the recent window (``time_col >= -window`` — ``MONTHS_BALANCE`` /
    ``DAYS_*`` are negative offsets from the application date), then aggregates the
    curated ``columns`` (intersected with those present) with a lean ``aggs`` set, a
    ``{prefix}_COUNT`` of in-window records, and any extra ``domain`` expressions.

    Parameters
    ----------
    df : pl.DataFrame
        The table to window and aggregate.
    time_col : str
        Negative time-offset column (``MONTHS_BALANCE`` or a ``DAYS_*`` column).
    window : int
        Window size; rows with ``time_col >= -window`` are kept.
    group_col, prefix, columns, aggs :
        As in :func:`aggregate_numeric` (``aggs`` defaults to a lean mean/max/sum).
    domain : List[pl.Expr], optional
        Extra aggregation expressions (already aliased) to include, e.g. a late-rate.
    """
    if time_col not in df.columns:
        return df.select(group_col).unique()

    recent = df.filter(pl.col(time_col) >= -window)
    cols = [c for c in (columns or []) if c in recent.columns]

    exprs: List[pl.Expr] = [pl.len().alias(_prefixed(prefix, "COUNT"))]
    for col in cols:
        for agg in aggs:
            exprs.append(
                _AGG_FUNCS[agg](pl.col(col)).alias(_prefixed(prefix, col, agg.upper()))
            )
    if domain:
        exprs.extend(domain)

    return recent.group_by(group_col).agg(exprs)


# --------------------------------------------------------------------------------------
# Per-loan ledger aggregators (grain: SK_ID_PREV)
# --------------------------------------------------------------------------------------


def aggregate_credit_card_by_prev(cc: pl.DataFrame) -> pl.DataFrame:
    """Consolidate ``credit_card_balance`` per ``SK_ID_PREV`` (prefix ``CC``)."""
    base = aggregate_numeric(cc, group_col="SK_ID_PREV", prefix="CC")
    domain = cc.group_by("SK_ID_PREV").agg(
        pl.when(pl.col("AMT_CREDIT_LIMIT_ACTUAL") > 0)
        .then(pl.col("AMT_BALANCE") / pl.col("AMT_CREDIT_LIMIT_ACTUAL"))
        .otherwise(None)
        .mean()
        .alias("CC_UTILIZATION_MEAN"),
        (pl.col("AMT_BALANCE") > pl.col("AMT_CREDIT_LIMIT_ACTUAL"))
        .sum()
        .alias("CC_OVERLIMIT_MONTHS"),
    )
    return base.join(domain, on="SK_ID_PREV", how="left")


def aggregate_installments_by_prev(inst: pl.DataFrame) -> pl.DataFrame:
    """Consolidate ``installments_payments`` per ``SK_ID_PREV`` (prefix ``INST``)."""
    base = aggregate_numeric(inst, group_col="SK_ID_PREV", prefix="INST")
    domain = inst.group_by("SK_ID_PREV").agg(
        # Rates are computed over paid installments (comparisons with a null payment
        # date/amount yield null, which the mean skips).
        (pl.col("DAYS_ENTRY_PAYMENT") > pl.col("DAYS_INSTALMENT"))
        .mean()
        .alias("INST_LATE_RATE"),
        (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT"))
        .mean()
        .alias("INST_DAYS_LATE_MEAN"),
        (pl.col("AMT_PAYMENT") < pl.col("AMT_INSTALMENT"))
        .mean()
        .alias("INST_UNDERPAY_RATE"),
        (pl.col("AMT_INSTALMENT") - pl.col("AMT_PAYMENT"))
        .mean()
        .alias("INST_SHORTFALL_MEAN"),
    )
    return base.join(domain, on="SK_ID_PREV", how="left")


def aggregate_pos_cash_by_prev(pos: pl.DataFrame) -> pl.DataFrame:
    """Consolidate ``POS_CASH_balance`` per ``SK_ID_PREV`` (prefix ``POS``)."""
    base = aggregate_numeric(pos, group_col="SK_ID_PREV", prefix="POS")
    domain = pos.group_by("SK_ID_PREV").agg(
        (pl.col("NAME_CONTRACT_STATUS") == "Completed")
        .sum()
        .alias("POS_COMPLETED_MONTHS"),
    )
    return base.join(domain, on="SK_ID_PREV", how="left")


# --------------------------------------------------------------------------------------
# Previous-application hub -> customer
# --------------------------------------------------------------------------------------

_PREV_SENTINEL = 365243


def build_previous_application_features(
    prev: pl.DataFrame,
    cc: pl.DataFrame,
    inst: pl.DataFrame,
    pos: pl.DataFrame,
    include_categorical: bool = True,
) -> pl.DataFrame:
    """
    Build customer-level features from ``previous_application`` used as a hub.

    Each ledger is consolidated per ``SK_ID_PREV`` and joined onto
    ``previous_application`` so a prior loan carries both its terms and its repayment
    behaviour; the enriched frame is then rolled up to ``SK_ID_CURR``. Returns one row
    per ``SK_ID_CURR`` with ``PREV_*`` columns. When ``include_categorical`` is set, the
    previous-application categorical columns are one-hot-rate encoded too.
    """
    # 1. Null out the 365243 "never / not applicable" sentinel in the DAYS_* columns so
    #    the aggregated means are meaningful (validated 56% in DAYS_FIRST_DRAWING).
    days_cols = [c for c in prev.columns if c.startswith("DAYS_")]
    prev = prev.with_columns(
        pl.when(pl.col(c) == _PREV_SENTINEL).then(None).otherwise(pl.col(c)).alias(c)
        for c in days_cols
    )

    # 2. Enrich each prior application with its ledger behaviour + per-loan flags.
    enriched = (
        prev.join(aggregate_credit_card_by_prev(cc), on="SK_ID_PREV", how="left")
        .join(aggregate_installments_by_prev(inst), on="SK_ID_PREV", how="left")
        .join(aggregate_pos_cash_by_prev(pos), on="SK_ID_PREV", how="left")
        .with_columns(
            pl.col("CC_COUNT").is_not_null().cast(pl.Int8).alias("HAS_CC"),
            pl.col("INST_COUNT").is_not_null().cast(pl.Int8).alias("HAS_INST"),
            pl.col("POS_COUNT").is_not_null().cast(pl.Int8).alias("HAS_POS"),
        )
    )

    # 3. Roll the enriched loans up to the customer. Generic numeric aggregations over
    #    all columns (loan terms + rolled-up ledger features), excluding the boolean
    #    HAS_* flags which are summarised explicitly as rates below.
    flags = ["HAS_CC", "HAS_INST", "HAS_POS"]
    base = aggregate_numeric(
        enriched, group_col="SK_ID_CURR", prefix="PREV", exclude=flags
    )

    domain = enriched.group_by("SK_ID_CURR").agg(
        (pl.col("NAME_CONTRACT_STATUS") == "Approved")
        .sum()
        .alias("PREV_APPROVED_COUNT"),
        (pl.col("NAME_CONTRACT_STATUS") == "Refused").sum().alias("PREV_REFUSED_COUNT"),
        (pl.col("NAME_CONTRACT_STATUS") == "Approved")
        .mean()
        .alias("PREV_APPROVED_RATE"),
        (pl.col("NAME_CONTRACT_STATUS") == "Refused").mean().alias("PREV_REFUSED_RATE"),
        pl.col("HAS_CC").mean().alias("PREV_HAS_CC_RATE"),
        pl.col("HAS_INST").mean().alias("PREV_HAS_INST_RATE"),
        pl.col("HAS_POS").mean().alias("PREV_HAS_POS_RATE"),
    )

    result = base.join(domain, on="SK_ID_CURR", how="left")
    if include_categorical:
        result = result.join(
            aggregate_categorical(enriched, "SK_ID_CURR", "PREV"),
            on="SK_ID_CURR",
            how="left",
        )
    return result


# --------------------------------------------------------------------------------------
# Bureau lineage -> customer
# --------------------------------------------------------------------------------------


def aggregate_bureau(
    bureau: pl.DataFrame,
    bureau_balance: pl.DataFrame,
    include_categorical: bool = True,
) -> pl.DataFrame:
    """
    Build customer-level bureau features (prefix ``BUREAU``).

    ``bureau_balance`` is aggregated per ``SK_ID_BUREAU`` (months observed, count of
    days-past-due months where ``STATUS in {1..5}``, count of closed months), bridged
    onto ``bureau`` via ``SK_ID_BUREAU``, then rolled up to ``SK_ID_CURR``. When
    ``include_categorical`` is set, the bureau categorical columns are one-hot-rate
    encoded too.
    """
    # Cast STATUS to string: it holds "0".."5"/"C"/"X", but a CSV chunk with only
    # digits can be inferred as an integer column.
    status = pl.col("STATUS").cast(pl.String)
    bb = bureau_balance.group_by("SK_ID_BUREAU").agg(
        pl.len().alias("BB_MONTHS_COUNT"),
        status.is_in(["1", "2", "3", "4", "5"]).sum().alias("BB_DPD_MONTHS"),
        (status == "C").sum().alias("BB_CLOSED_MONTHS"),
        pl.col("MONTHS_BALANCE").min().alias("BB_MONTHS_BALANCE_MIN"),
    )

    enriched = bureau.join(bb, on="SK_ID_BUREAU", how="left")

    base = aggregate_numeric(enriched, group_col="SK_ID_CURR", prefix="BUREAU")
    domain = enriched.group_by("SK_ID_CURR").agg(
        (pl.col("CREDIT_ACTIVE") == "Active").sum().alias("BUREAU_ACTIVE_COUNT"),
        (pl.col("CREDIT_ACTIVE") == "Closed").sum().alias("BUREAU_CLOSED_COUNT"),
        (pl.col("CREDIT_ACTIVE") == "Active").mean().alias("BUREAU_ACTIVE_RATE"),
    )
    result = base.join(domain, on="SK_ID_CURR", how="left")
    if include_categorical:
        result = result.join(
            aggregate_categorical(enriched, "SK_ID_CURR", "BUREAU"),
            on="SK_ID_CURR",
            how="left",
        )
    return result


# --------------------------------------------------------------------------------------
# Recent time-window blocks (aggregated directly to SK_ID_CURR)
# --------------------------------------------------------------------------------------


def _window_credit_card(cc: pl.DataFrame) -> pl.DataFrame:
    return aggregate_recent(
        cc,
        "MONTHS_BALANCE",
        12,
        prefix="CC_L12M",
        columns=[
            "AMT_BALANCE",
            "AMT_DRAWINGS_CURRENT",
            "AMT_PAYMENT_CURRENT",
            "SK_DPD",
            "AMT_CREDIT_LIMIT_ACTUAL",
        ],
    )


def _window_pos(pos: pl.DataFrame) -> pl.DataFrame:
    return aggregate_recent(
        pos,
        "MONTHS_BALANCE",
        12,
        prefix="POS_L12M",
        columns=["SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT_FUTURE"],
    )


def _window_installments(inst: pl.DataFrame) -> pl.DataFrame:
    domain: List[pl.Expr] = []
    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(inst.columns):
        domain.append(
            (pl.col("DAYS_ENTRY_PAYMENT") > pl.col("DAYS_INSTALMENT"))
            .mean()
            .alias("INST_L365D_LATE_RATE")
        )
    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(inst.columns):
        domain.append(
            (pl.col("AMT_PAYMENT") < pl.col("AMT_INSTALMENT"))
            .mean()
            .alias("INST_L365D_UNDERPAY_RATE")
        )
    return aggregate_recent(
        inst,
        "DAYS_INSTALMENT",
        365,
        prefix="INST_L365D",
        columns=["AMT_INSTALMENT", "AMT_PAYMENT"],
        domain=domain,
    )


def _window_prev(prev: pl.DataFrame) -> pl.DataFrame:
    domain: List[pl.Expr] = []
    if "NAME_CONTRACT_STATUS" in prev.columns:
        domain.append(
            (pl.col("NAME_CONTRACT_STATUS") == "Approved")
            .mean()
            .alias("PREV_L365D_APPROVED_RATE")
        )
    return aggregate_recent(
        prev,
        "DAYS_DECISION",
        365,
        prefix="PREV_L365D",
        columns=["AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY", "CNT_PAYMENT"],
        domain=domain,
    )


def _window_bureau(bureau: pl.DataFrame) -> pl.DataFrame:
    domain: List[pl.Expr] = []
    if "CREDIT_ACTIVE" in bureau.columns:
        domain.append(
            (pl.col("CREDIT_ACTIVE") == "Active")
            .mean()
            .alias("BUREAU_L730D_ACTIVE_RATE")
        )
    return aggregate_recent(
        bureau,
        "DAYS_CREDIT",
        730,
        prefix="BUREAU_L730D",
        columns=[
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "CREDIT_DAY_OVERDUE",
            "DAYS_CREDIT",
        ],
        domain=domain,
    )


# --------------------------------------------------------------------------------------
# Driver-frame treatments (applied before the joins)
# --------------------------------------------------------------------------------------

# Driver columns carrying the same ``365243`` "never / not applicable" sentinel that
# ``previous_application`` uses. Listed explicitly rather than detected from the data so
# the output schema is identical for train and test even if one of them happens to have
# no sentinel rows.
_APP_SENTINEL_COLUMNS: Tuple[str, ...] = ("DAYS_EMPLOYED",)


def clean_application_sentinels(app: pl.DataFrame) -> pl.DataFrame:
    """
    Null the ``365243`` sentinel in the driver's ``DAYS_*`` columns, flagging each.

    ``DAYS_EMPLOYED`` uses ``365243`` (≈1000 years) to mean "not employed" — it is 18%
    of ``application_train`` and drags the column's mean from a median of ≈-1,200 days
    to ≈+63,800. A null-native tree isolates the spike with a single split, but any
    model that imputes and scales (the NN / transformer rounds) has its mean and
    variance destroyed by it, so the sentinel is replaced with a null here.

    The sentinel group is not noise — it defaults at ~5.4% against ~8.7% for everyone
    else — so its membership is preserved as ``{COL}_ANOMALY`` (1/0, never null) rather
    than being thrown away with the value.
    """
    cols = [c for c in _APP_SENTINEL_COLUMNS if c in app.columns]
    if not cols:
        return app
    return app.with_columns(
        [
            (pl.col(c) == _PREV_SENTINEL)
            .fill_null(False)
            .cast(pl.Int8)
            .alias(f"{c}_ANOMALY")
            for c in cols
        ]
        + [
            pl.when(pl.col(c) == _PREV_SENTINEL)
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in cols
        ]
    )


# (name, numerator, denominator) — the affordability ratios. Trees split on one column
# at a time and cannot express a quotient, so these have to be built explicitly; every
# input is already a high-gain driver column.
_RATIOS: Tuple[Tuple[str, str, str], ...] = (
    ("RATIO_CREDIT_TO_INCOME", "AMT_CREDIT", "AMT_INCOME_TOTAL"),
    ("RATIO_ANNUITY_TO_INCOME", "AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    ("RATIO_PAYMENT_RATE", "AMT_ANNUITY", "AMT_CREDIT"),
    ("RATIO_CREDIT_TO_GOODS", "AMT_CREDIT", "AMT_GOODS_PRICE"),
    ("RATIO_INCOME_PER_PERSON", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
    ("RATIO_EMPLOYED_TO_AGE", "DAYS_EMPLOYED", "DAYS_BIRTH"),
)


def add_ratio_features(app: pl.DataFrame) -> pl.DataFrame:
    """
    Add the affordability ratios (``RATIO_*``) to the driver frame.

    ``RATIO_PAYMENT_RATE`` (annuity / credit) is the effective repayment speed,
    ``RATIO_CREDIT_TO_INCOME`` and ``RATIO_ANNUITY_TO_INCOME`` are leverage and
    debt-service burden, ``RATIO_CREDIT_TO_GOODS`` is how much was borrowed above the
    financed item's price, and ``RATIO_EMPLOYED_TO_AGE`` is the fraction of life spent
    in the current job (both inputs are negative day offsets, so the quotient is
    positive).

    A zero or null denominator yields a null rather than an infinity. Call *after*
    :func:`clean_application_sentinels` so ``RATIO_EMPLOYED_TO_AGE`` is not computed
    from the ``365243`` sentinel.
    """
    exprs = [
        pl.when(pl.col(den) != 0)
        .then(pl.col(num) / pl.col(den))
        .otherwise(None)
        .alias(name)
        for name, num, den in _RATIOS
        if num in app.columns and den in app.columns
    ]
    return app.with_columns(exprs) if exprs else app


# --------------------------------------------------------------------------------------
# Assembler
# --------------------------------------------------------------------------------------


def build_feature_matrix(
    app: pl.DataFrame,
    datasets: Path,
    include_categorical: bool = True,
    include_windows: bool = True,
) -> pl.DataFrame:
    """
    Assemble the customer-level feature matrix for a driver application frame.

    Cleans the driver's ``365243`` sentinels and adds the ``RATIO_*`` affordability
    features, loads the auxiliary tables from ``datasets``, builds the
    previous-application-hub and bureau feature blocks, and left-joins both onto ``app``
    on ``SK_ID_CURR``. Adds top-level coverage flags ``HAS_PREV`` / ``HAS_BUREAU``. The
    driver's own columns (and ``TARGET`` if present) are preserved; no imputation is
    applied.

    Parameters
    ----------
    app : pl.DataFrame
        Driver frame (``application_train`` or ``application_test``), one row per
        ``SK_ID_CURR``.
    datasets : Path
        Directory containing the raw Home Credit CSV files.
    include_categorical : bool
        Add one-hot-rate encodings of the previous-application / bureau categoricals.
    include_windows : bool
        Add recent time-window blocks (``*_L12M`` / ``*_L365D`` / ``*_L730D``)
        aggregated directly to ``SK_ID_CURR``.
    """
    app = add_ratio_features(clean_application_sentinels(app))

    prev = pl.read_csv(datasets / "previous_application.csv", infer_schema_length=10000)
    cc = pl.read_csv(datasets / "credit_card_balance.csv", infer_schema_length=100000)
    inst = pl.read_csv(
        datasets / "installments_payments.csv", infer_schema_length=100000
    )
    pos = pl.read_csv(datasets / "POS_CASH_balance.csv", infer_schema_length=100000)
    bureau = pl.read_csv(datasets / "bureau.csv", infer_schema_length=10000)
    bureau_balance = pl.read_csv(
        datasets / "bureau_balance.csv", infer_schema_length=10000
    )

    prev_feats = build_previous_application_features(
        prev, cc, inst, pos, include_categorical=include_categorical
    )
    bureau_feats = aggregate_bureau(
        bureau, bureau_balance, include_categorical=include_categorical
    )

    out = app.join(prev_feats, on="SK_ID_CURR", how="left").join(
        bureau_feats, on="SK_ID_CURR", how="left"
    )

    if include_windows:
        for block in (
            _window_credit_card(cc),
            _window_pos(pos),
            _window_installments(inst),
            _window_prev(prev),
            _window_bureau(bureau),
        ):
            out = out.join(block, on="SK_ID_CURR", how="left")

    return out.with_columns(
        pl.col("PREV_COUNT").is_not_null().cast(pl.Int8).alias("HAS_PREV"),
        pl.col("BUREAU_COUNT").is_not_null().cast(pl.Int8).alias("HAS_BUREAU"),
    )
