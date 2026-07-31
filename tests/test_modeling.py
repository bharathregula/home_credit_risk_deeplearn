import numpy as np
import polars as pl

from home_credit_risk.modeling import (
    apply_structural_encoding,
    category_mapping,
    constant_feature_columns,
    duplicate_feature_columns,
    split_xy,
    to_tree_matrix,
)


def _matrix() -> pl.DataFrame:
    # One row per customer, mixing structural columns (nulls => "no history") with
    # behavioural means (nulls genuinely unknown) plus a constant and an id/target.
    return pl.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "PREV_COUNT": [2, None, 5],
            "BUREAU_AMT_CREDIT_SUM": [100.0, None, 50.0],
            "PREV_REFUSED_RATE": [0.5, None, 0.0],
            "PREV_NAME_NUNIQUE": [3, None, 1],
            "HAS_PREV": [1, 0, 1],
            "PREV_CNT_PAYMENT_MEAN": [12.0, None, 6.0],
            "POS_SK_DPD_MIN": [0, 0, 0],  # zero-variance / constant
        }
    )


def test_structural_encoding_zeros_history_columns() -> None:
    out = apply_structural_encoding(_matrix())

    # Counts, sums, rates, distinct-counts and flags => 0 where they were null.
    assert out["PREV_COUNT"].null_count() == 0
    assert out.filter(pl.col("SK_ID_CURR") == 2)["PREV_COUNT"].item() == 0
    assert out.filter(pl.col("SK_ID_CURR") == 2)["BUREAU_AMT_CREDIT_SUM"].item() == 0.0
    assert out.filter(pl.col("SK_ID_CURR") == 2)["PREV_REFUSED_RATE"].item() == 0.0
    assert out.filter(pl.col("SK_ID_CURR") == 2)["PREV_NAME_NUNIQUE"].item() == 0


def test_structural_encoding_leaves_behavioural_means_null() -> None:
    out = apply_structural_encoding(_matrix())

    # A *_MEAN is genuinely unknown when absent -> must stay null.
    assert out["PREV_CNT_PAYMENT_MEAN"].null_count() == 1
    assert out.filter(pl.col("SK_ID_CURR") == 2)["PREV_CNT_PAYMENT_MEAN"].item() is None


def test_constant_feature_columns_flags_zero_variance_only() -> None:
    fm = apply_structural_encoding(_matrix())
    feature_cols, _ = split_xy(fm)

    constants = constant_feature_columns(fm, feature_cols)

    assert "POS_SK_DPD_MIN" in constants  # always 0
    assert "PREV_COUNT" not in constants  # varies
    assert "BUREAU_AMT_CREDIT_SUM" not in constants


def test_constant_feature_columns_treats_all_null_as_constant() -> None:
    fm = pl.DataFrame({"A_MEAN": [None, None, None], "B_COUNT": [1, 2, 3]})
    assert constant_feature_columns(fm, ["A_MEAN", "B_COUNT"]) == ["A_MEAN"]


def test_duplicate_feature_columns_keeps_the_first() -> None:
    fm = pl.DataFrame(
        {
            "A_RATE": [0.5, 0.0, 1.0],
            "B_RATE": [0.5, 0.0, 1.0],  # identical to A_RATE
            "C_RATE": [0.5, 0.0, 0.9],  # nearly identical -> kept
            "D_MEAN": [1.0, None, 3.0],
            "E_MEAN": [1.0, None, 3.0],  # identical including the null position
        }
    )
    dupes = duplicate_feature_columns(fm, list(fm.columns))

    assert dupes == ["B_RATE", "E_MEAN"]  # keep-first, input order preserved
    assert "A_RATE" not in dupes
    assert "C_RATE" not in dupes


def test_duplicate_feature_columns_treats_nulls_as_equal_but_positional() -> None:
    fm = pl.DataFrame({"X_MEAN": [1.0, None], "Y_MEAN": [None, 1.0]})
    # Same values, different null positions -> not duplicates.
    assert duplicate_feature_columns(fm, ["X_MEAN", "Y_MEAN"]) == []


def test_duplicate_feature_columns_handles_strings() -> None:
    fm = pl.DataFrame({"S1": ["a", "b"], "S2": ["a", "b"], "S3": ["a", "c"]})
    assert duplicate_feature_columns(fm, ["S1", "S2", "S3"]) == ["S2"]


def test_split_xy_excludes_target_and_ids() -> None:
    fm = apply_structural_encoding(_matrix())
    feature_cols, y = split_xy(fm)

    assert "TARGET" not in feature_cols
    assert "SK_ID_CURR" not in feature_cols
    assert "PREV_COUNT" in feature_cols
    assert isinstance(y, np.ndarray)
    assert y.tolist() == [0, 1, 0]
    # X built from the returned names lines up with y.
    assert fm.select(feature_cols).to_numpy().shape == (3, len(feature_cols))


def test_split_xy_honours_extra_drop() -> None:
    fm = apply_structural_encoding(_matrix())
    all_cols, _ = split_xy(fm)
    dropped, _ = split_xy(fm, drop=["POS_SK_DPD_MIN"])

    assert "POS_SK_DPD_MIN" in all_cols
    assert "POS_SK_DPD_MIN" not in dropped


def test_to_tree_matrix_codes_strings_and_reports_indices() -> None:
    fm = pl.DataFrame(
        {
            "NAME_TYPE": ["Cash", "Revolving", "Cash", None],
            "AMT": [1.0, 2.0, None, 4.0],
        }
    )
    X, cat_idx = to_tree_matrix(fm, ["NAME_TYPE", "AMT"])

    assert X.shape == (4, 2)
    assert cat_idx == [0]  # the string column
    # "Cash" appears twice -> identical integer code; distinct from "Revolving".
    assert X[0, 0] == X[2, 0]
    assert X[0, 0] != X[1, 0]
    # A null string becomes NaN (missing), not a real category.
    assert np.isnan(X[3, 0])
    # Numeric column and its null pass through untouched.
    assert X[0, 1] == 1.0
    assert np.isnan(X[2, 1])


def test_category_mapping_is_sorted_and_string_only() -> None:
    fm = pl.DataFrame({"NAME": ["b", "a", "b", None], "AMT": [1.0, 2.0, 3.0, 4.0]})
    mapping = category_mapping(fm, ["NAME", "AMT"])

    assert mapping == {"NAME": ["a", "b"]}  # sorted, nulls excluded, numeric skipped


def test_to_tree_matrix_codes_are_stable_across_frames() -> None:
    # The bug this guards: encoding two frames independently assigns codes by each
    # frame's own value order, so the same string can land on different integers.
    train = pl.DataFrame({"NAME": ["Cash", "Revolving", "Cash"]})
    test = pl.DataFrame({"NAME": ["Revolving", "Cash"]})
    mapping = category_mapping(train, ["NAME"])

    X_train, _ = to_tree_matrix(train, ["NAME"], categories=mapping)
    X_test, _ = to_tree_matrix(test, ["NAME"], categories=mapping)

    cash, revolving = X_train[0, 0], X_train[1, 0]
    assert cash != revolving
    assert X_test[0, 0] == revolving
    assert X_test[1, 0] == cash


def test_to_tree_matrix_encodes_unseen_category_as_missing() -> None:
    mapping = {"NAME": ["Cash", "Revolving"]}
    frame = pl.DataFrame({"NAME": ["Cash", "Barter"]})

    X, cat_idx = to_tree_matrix(frame, ["NAME"], categories=mapping)

    assert cat_idx == [0]
    assert X[0, 0] == 0.0  # index of "Cash" in the mapping
    assert np.isnan(X[1, 0])  # unseen value -> missing, not a new category
