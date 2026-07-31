from pathlib import Path

import polars as pl

from home_credit_risk.features import (
    add_ratio_features,
    aggregate_bureau,
    aggregate_categorical,
    aggregate_installments_by_prev,
    aggregate_numeric,
    aggregate_recent,
    build_feature_matrix,
    build_previous_application_features,
    clean_application_sentinels,
)


def _cell(df: pl.DataFrame, key: int, col: str, key_col: str = "SK_ID_CURR") -> object:
    return df.filter(pl.col(key_col) == key)[col].item()


def test_aggregate_numeric_basic() -> None:
    df = pl.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_PREV": [5, 6, 7],  # identifier -> must be excluded
            "AMT": [10.0, 20.0, 100.0],
        }
    )
    out = aggregate_numeric(df, group_col="SK_ID_CURR", prefix="T")

    assert out.height == 2
    # SK_ID_PREV is not aggregated
    assert not any(c.startswith("T_SK_ID") for c in out.columns)
    assert _cell(out, 1, "T_COUNT") == 2
    assert _cell(out, 1, "T_AMT_MEAN") == 15.0
    assert _cell(out, 1, "T_AMT_SUM") == 30.0
    assert _cell(out, 2, "T_AMT_MAX") == 100.0


def test_installments_rates_over_paid_rows() -> None:
    inst = pl.DataFrame(
        {
            "SK_ID_PREV": [10, 10, 10],
            "SK_ID_CURR": [1, 1, 1],
            "DAYS_INSTALMENT": [-100.0, -50.0, -30.0],
            # late, on-time, unpaid(null) -> late rate over paid = 1/2
            "DAYS_ENTRY_PAYMENT": [-90.0, -60.0, None],
            "AMT_INSTALMENT": [100.0, 100.0, 100.0],
            # full, under, unpaid(null) -> underpay rate over paid = 1/2
            "AMT_PAYMENT": [100.0, 80.0, None],
        }
    )
    out = aggregate_installments_by_prev(inst)
    assert out.height == 1
    assert _cell(out, 10, "INST_LATE_RATE", key_col="SK_ID_PREV") == 0.5
    assert _cell(out, 10, "INST_UNDERPAY_RATE", key_col="SK_ID_PREV") == 0.5


def test_previous_application_hub_rollup() -> None:
    prev = pl.DataFrame(
        {
            "SK_ID_PREV": [10, 11, 20],
            "SK_ID_CURR": [1, 1, 2],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
            "DAYS_DECISION": [-100.0, -200.0, -50.0],
            # sentinel in loans 10 and 20; real value only in loan 11
            "DAYS_FIRST_DRAWING": [365243.0, -10.0, 365243.0],
            "AMT_APPLICATION": [1000.0, 2000.0, 500.0],
        }
    )
    cc = pl.DataFrame(
        {
            "SK_ID_PREV": [10, 10],
            "SK_ID_CURR": [1, 1],
            "MONTHS_BALANCE": [-1, -2],
            "AMT_BALANCE": [50.0, 60.0],
            "AMT_CREDIT_LIMIT_ACTUAL": [100.0, 100.0],
            "SK_DPD": [0, 0],
        }
    )
    inst = pl.DataFrame(
        {
            "SK_ID_PREV": [10, 20],
            "SK_ID_CURR": [1, 2],
            "DAYS_INSTALMENT": [-100.0, -50.0],
            "DAYS_ENTRY_PAYMENT": [-90.0, -60.0],
            "AMT_INSTALMENT": [100.0, 100.0],
            "AMT_PAYMENT": [100.0, 100.0],
        }
    )
    pos = pl.DataFrame(
        {
            "SK_ID_PREV": [11],
            "SK_ID_CURR": [1],
            "NAME_CONTRACT_STATUS": ["Active"],
            "MONTHS_BALANCE": [-1],
            "SK_DPD": [0],
            "CNT_INSTALMENT": [10.0],
            "CNT_INSTALMENT_FUTURE": [5.0],
        }
    )

    out = build_previous_application_features(prev, cc, inst, pos)

    assert out.height == 2  # one row per SK_ID_CURR
    assert _cell(out, 1, "PREV_COUNT") == 2
    assert _cell(out, 2, "PREV_COUNT") == 1
    # coverage rates: customer 1 has loans {10:cc,inst} {11:pos}
    assert _cell(out, 1, "PREV_HAS_CC_RATE") == 0.5
    assert _cell(out, 1, "PREV_HAS_INST_RATE") == 0.5
    assert _cell(out, 1, "PREV_HAS_POS_RATE") == 0.5
    assert _cell(out, 1, "PREV_APPROVED_RATE") == 0.5
    assert _cell(out, 2, "PREV_APPROVED_RATE") == 1.0
    # sentinel was nulled before aggregating: mean uses only loan 11's -10
    assert _cell(out, 1, "PREV_DAYS_FIRST_DRAWING_MEAN") == -10.0


def test_aggregate_bureau_bridges_balance() -> None:
    bureau = pl.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [100, 101, 200],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "AMT_CREDIT_SUM": [500.0, 600.0, 700.0],
        }
    )
    bureau_balance = pl.DataFrame(
        {
            "SK_ID_BUREAU": [100, 100, 100, 200, 200],
            "MONTHS_BALANCE": [0, -1, -2, 0, -1],
            "STATUS": ["0", "1", "C", "5", "5"],
        }
    )
    out = aggregate_bureau(bureau, bureau_balance)

    assert out.height == 2
    assert _cell(out, 1, "BUREAU_COUNT") == 2
    assert _cell(out, 1, "BUREAU_ACTIVE_COUNT") == 1
    assert _cell(out, 1, "BUREAU_ACTIVE_RATE") == 0.5
    # bureau 100 had one DPD month ("1"); bureau 101 has no balance (null, skipped)
    assert _cell(out, 1, "BUREAU_BB_DPD_MONTHS_SUM") == 1
    assert _cell(out, 2, "BUREAU_BB_DPD_MONTHS_SUM") == 2


def test_aggregate_categorical_rates_and_nunique() -> None:
    df = pl.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2],
            "STATUS": ["A", "A", "B", "A"],  # low cardinality -> rates
            "HIGH": ["v0", "v1", "v2", "v3"],  # exceeds max_cardinality -> nunique only
        }
    )
    out = aggregate_categorical(df, "SK_ID_CURR", "T", max_cardinality=2)

    # Low-cardinality column: per-category rates (proportions sum to 1)
    assert _cell(out, 1, "T_STATUS_A_RATE") == 2 / 3
    assert _cell(out, 1, "T_STATUS_B_RATE") == 1 / 3
    assert _cell(out, 1, "T_STATUS_NUNIQUE") == 2
    # High-cardinality column: distinct-count only, no one-hot rate columns
    assert _cell(out, 1, "T_HIGH_NUNIQUE") == 3
    assert not any(c.startswith("T_HIGH_v") for c in out.columns)


def test_aggregate_categorical_slugifies_values() -> None:
    df = pl.DataFrame(
        {"SK_ID_CURR": [1, 1], "X": ["Consumer credit", "Consumer credit"]}
    )
    out = aggregate_categorical(df, "SK_ID_CURR", "T")
    assert "T_X_Consumer_credit_RATE" in out.columns
    assert _cell(out, 1, "T_X_Consumer_credit_RATE") == 1.0


def test_aggregate_recent_keeps_only_in_window() -> None:
    df = pl.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1],
            "MONTHS_BALANCE": [-1, -6, -20],  # window 12 keeps -1 and -6
            "AMT": [10.0, 20.0, 999.0],
        }
    )
    out = aggregate_recent(
        df,
        "MONTHS_BALANCE",
        12,
        "SK_ID_CURR",
        "W",
        columns=["AMT"],
        aggs=("mean", "sum"),
    )
    assert _cell(out, 1, "W_COUNT") == 2
    assert _cell(out, 1, "W_AMT_MEAN") == 15.0
    assert _cell(out, 1, "W_AMT_SUM") == 30.0


def test_aggregate_recent_missing_time_col_is_safe() -> None:
    df = pl.DataFrame({"SK_ID_CURR": [1, 2], "AMT": [1.0, 2.0]})
    out = aggregate_recent(df, "MONTHS_BALANCE", 12, "SK_ID_CURR", "W", columns=["AMT"])
    assert out.columns == ["SK_ID_CURR"]  # no feature columns, no error
    assert out.height == 2


def test_clean_application_sentinels_nulls_and_flags() -> None:
    app = pl.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "DAYS_EMPLOYED": [365243, -1200, None],
        }
    )
    out = clean_application_sentinels(app)

    # The sentinel becomes null, real values and genuine nulls are untouched.
    assert out["DAYS_EMPLOYED"].to_list() == [None, -1200, None]
    # Membership of the sentinel group survives as a flag, and is never null.
    assert out["DAYS_EMPLOYED_ANOMALY"].to_list() == [1, 0, 0]
    assert out["DAYS_EMPLOYED_ANOMALY"].null_count() == 0


def test_clean_application_sentinels_emits_flag_even_without_sentinel_rows() -> None:
    # Train/test schema parity: the flag must exist whether or not the sentinel occurs.
    out = clean_application_sentinels(pl.DataFrame({"DAYS_EMPLOYED": [-1, -2]}))
    assert out["DAYS_EMPLOYED_ANOMALY"].to_list() == [0, 0]


def test_add_ratio_features_computes_quotients() -> None:
    app = pl.DataFrame(
        {
            "AMT_CREDIT": [200000.0],
            "AMT_INCOME_TOTAL": [100000.0],
            "AMT_ANNUITY": [20000.0],
            "AMT_GOODS_PRICE": [180000.0],
            "CNT_FAM_MEMBERS": [2.0],
            "DAYS_EMPLOYED": [-2000.0],
            "DAYS_BIRTH": [-10000.0],
        }
    )
    out = add_ratio_features(app)

    assert out["RATIO_CREDIT_TO_INCOME"].item() == 2.0
    assert out["RATIO_ANNUITY_TO_INCOME"].item() == 0.2
    assert out["RATIO_PAYMENT_RATE"].item() == 0.1
    assert out["RATIO_INCOME_PER_PERSON"].item() == 50000.0
    # Both day offsets are negative, so the fraction of life employed is positive.
    assert out["RATIO_EMPLOYED_TO_AGE"].item() == 0.2


def test_add_ratio_features_guards_zero_denominator() -> None:
    app = pl.DataFrame({"AMT_CREDIT": [100.0, 100.0], "AMT_INCOME_TOTAL": [0.0, None]})
    out = add_ratio_features(app)
    # A zero or null denominator yields null, never an infinity.
    assert out["RATIO_CREDIT_TO_INCOME"].to_list() == [None, None]


def test_add_ratio_features_skips_absent_inputs() -> None:
    out = add_ratio_features(pl.DataFrame({"SK_ID_CURR": [1]}))
    assert out.columns == ["SK_ID_CURR"]


def _write(df: pl.DataFrame, path: Path) -> None:
    df.write_csv(path)


def _make_synthetic(tmp_path: Path) -> None:
    """Write a minimal synthetic copy of every raw table into ``tmp_path``."""
    _write(
        pl.DataFrame(
            {
                "SK_ID_PREV": [10, 20],
                "SK_ID_CURR": [1, 2],
                "NAME_CONTRACT_STATUS": ["Approved", "Refused"],
                "DAYS_DECISION": [-100, -50],
                "AMT_APPLICATION": [1000.0, 500.0],
            }
        ),
        tmp_path / "previous_application.csv",
    )
    _write(
        pl.DataFrame(
            {
                "SK_ID_PREV": [10],
                "SK_ID_CURR": [1],
                "MONTHS_BALANCE": [-1],
                "AMT_BALANCE": [50.0],
                "AMT_CREDIT_LIMIT_ACTUAL": [100.0],
                "SK_DPD": [0],
            }
        ),
        tmp_path / "credit_card_balance.csv",
    )
    _write(
        pl.DataFrame(
            {
                "SK_ID_PREV": [10],
                "SK_ID_CURR": [1],
                "DAYS_INSTALMENT": [-100.0],
                "DAYS_ENTRY_PAYMENT": [-90.0],
                "AMT_INSTALMENT": [100.0],
                "AMT_PAYMENT": [100.0],
            }
        ),
        tmp_path / "installments_payments.csv",
    )
    _write(
        pl.DataFrame(
            {
                "SK_ID_PREV": [20],
                "SK_ID_CURR": [2],
                "NAME_CONTRACT_STATUS": ["Completed"],
                "MONTHS_BALANCE": [-1],
                "SK_DPD": [0],
                "CNT_INSTALMENT": [10.0],
                "CNT_INSTALMENT_FUTURE": [0.0],
            }
        ),
        tmp_path / "POS_CASH_balance.csv",
    )
    _write(
        pl.DataFrame(
            {
                "SK_ID_CURR": [1],
                "SK_ID_BUREAU": [100],
                "CREDIT_ACTIVE": ["Active"],
                "DAYS_CREDIT": [-200],
                "AMT_CREDIT_SUM": [500.0],
            }
        ),
        tmp_path / "bureau.csv",
    )
    _write(
        pl.DataFrame(
            {
                "SK_ID_BUREAU": [100, 100],
                "MONTHS_BALANCE": [0, -1],
                "STATUS": ["0", "1"],
            }
        ),
        tmp_path / "bureau_balance.csv",
    )


def test_build_feature_matrix_end_to_end(tmp_path: Path) -> None:
    _make_synthetic(tmp_path)
    # Driver with a customer (3) that has no auxiliary history.
    app = pl.DataFrame({"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0]})
    out = build_feature_matrix(app, tmp_path)

    assert out.height == 3  # driver grain preserved
    assert out["SK_ID_CURR"].n_unique() == 3
    assert "TARGET" in out.columns
    assert _cell(out, 1, "HAS_PREV") == 1
    assert _cell(out, 1, "HAS_BUREAU") == 1
    assert _cell(out, 3, "HAS_PREV") == 0
    assert _cell(out, 3, "HAS_BUREAU") == 0


def test_build_feature_matrix_treats_the_driver(tmp_path: Path) -> None:
    _make_synthetic(tmp_path)
    app = pl.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "DAYS_EMPLOYED": [365243, -1000, -2000],
            "DAYS_BIRTH": [-10000, -10000, -10000],
            "AMT_CREDIT": [200.0, 200.0, 200.0],
            "AMT_INCOME_TOTAL": [100.0, 100.0, 100.0],
        }
    )
    out = build_feature_matrix(app, tmp_path)

    # The sentinel is cleaned and flagged on the way through the assembler...
    assert _cell(out, 1, "DAYS_EMPLOYED") is None
    assert _cell(out, 1, "DAYS_EMPLOYED_ANOMALY") == 1
    assert _cell(out, 2, "DAYS_EMPLOYED_ANOMALY") == 0
    # ...and the ratios are built from the cleaned values, not from 365243.
    assert _cell(out, 1, "RATIO_EMPLOYED_TO_AGE") is None
    assert _cell(out, 2, "RATIO_EMPLOYED_TO_AGE") == 0.1
    assert _cell(out, 1, "RATIO_CREDIT_TO_INCOME") == 2.0


def test_build_feature_matrix_toggles(tmp_path: Path) -> None:
    _make_synthetic(tmp_path)
    app = pl.DataFrame({"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0]})

    base = build_feature_matrix(
        app, tmp_path, include_categorical=False, include_windows=False
    )
    full = build_feature_matrix(app, tmp_path)  # defaults: both on

    assert full.width > base.width
    # all-history blocks + coverage flags exist in both
    assert {"PREV_COUNT", "BUREAU_COUNT", "HAS_PREV"}.issubset(base.columns)
    # categorical + windowed blocks appear only with the toggles on
    assert any(c.startswith("PREV_NAME_CONTRACT_STATUS") for c in full.columns)
    assert not any(c.startswith("PREV_NAME_CONTRACT_STATUS") for c in base.columns)
    assert any(c.startswith("CC_L12M") for c in full.columns)
    assert not any(c.startswith("CC_L12M") for c in base.columns)
