import matplotlib

matplotlib.use("Agg")  # headless: no display needed for the plotting helpers

import polars as pl  # noqa: E402

from home_credit_risk.eda_utils import (  # noqa: E402
    _measurement_numeric_columns,
    analyze_numeric_features,
    check_data_quality,
    detect_sentinel_values,
    identify_outliers,
)


def test_check_data_quality_runs_generically(capsys) -> None:  # type: ignore[no-untyped-def]
    # Generic checks must not depend on any application-specific column names.
    df = pl.DataFrame(
        {
            "all_null": [None, None, None],
            "constant": [7, 7, 7],
            "varying": [1, 2, 3],
        }
    )
    check_data_quality(df)
    out = capsys.readouterr().out
    assert "all_null" in out  # reported as fully-null
    assert "constant" in out  # reported as constant
    assert "varying" not in out  # a normal column is not flagged


def test_measurement_columns_excludes_sk_id() -> None:
    df = pl.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "SK_ID_PREV": [3, 4],
            "SK_ID_BUREAU": [5, 6],
            "AMT_CREDIT": [100.0, 200.0],
            "MONTHS_BALANCE": [-1, -2],
            "NAME_STATUS": ["A", "B"],  # non-numeric, ignored
        }
    )
    cols = _measurement_numeric_columns(df)
    assert cols == ["AMT_CREDIT", "MONTHS_BALANCE"]


def test_measurement_columns_respects_exclude() -> None:
    df = pl.DataFrame({"AMT_CREDIT": [1.0], "MONTHS_BALANCE": [-1]})
    assert _measurement_numeric_columns(df, exclude=["MONTHS_BALANCE"]) == [
        "AMT_CREDIT"
    ]


def test_analyze_numeric_features_drops_ids() -> None:
    df = pl.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "SK_ID_PREV": [4, 5, 6],
            "AMT_CREDIT": [100.0, 200.0, 300.0],
        }
    )
    returned = analyze_numeric_features(df, show_summary=False)
    assert "SK_ID_CURR" not in returned.columns
    assert "SK_ID_PREV" not in returned.columns
    assert "AMT_CREDIT" in returned.columns


def test_identify_outliers_drops_ids() -> None:
    df = pl.DataFrame(
        {
            "SK_ID_CURR": list(range(10)),
            "AMT_CREDIT": [1.0] * 9 + [1000.0],
        }
    )
    summary = identify_outliers(df, z_threshold=2.0)
    assert "SK_ID_CURR" not in summary["column"].to_list()
    assert "AMT_CREDIT" in summary["column"].to_list()


def test_detect_sentinel_flags_measurement_column() -> None:
    df = pl.DataFrame(
        {
            "SK_ID_CURR": [365243, 1, 2],
            "DAYS_FIRST_DRAWING": [365243, -10, -20],
        }
    )
    summary = detect_sentinel_values(df)
    flagged = summary["column"].to_list()

    # The measurement column is flagged...
    assert "DAYS_FIRST_DRAWING" in flagged
    # ...but the SK_ID identifier is auto-skipped even though it equals the sentinel.
    assert "SK_ID_CURR" not in flagged


def test_detect_sentinel_respects_exclude() -> None:
    df = pl.DataFrame(
        {
            "DAYS_A": [365243, -1, -2],
            "DAYS_B": [365243, -3, -4],
        }
    )
    summary = detect_sentinel_values(df, exclude=["DAYS_B"])
    flagged = summary["column"].to_list()

    assert flagged == ["DAYS_A"]


def test_detect_sentinel_no_occurrences_returns_empty() -> None:
    df = pl.DataFrame({"DAYS_A": [-1, -2, -3]})
    summary = detect_sentinel_values(df)

    assert summary.height == 0
    assert summary.columns == ["column", "Sentinel_Count", "Sentinel_Percentage"]
