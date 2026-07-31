import numpy as np
import polars as pl

from home_credit_risk.linear import (
    build_pipeline,
    cross_validate_linear,
    split_feature_types,
)

TINY = {"max_iter": 200, "C": 1.0}


def _frame(n: int = 300, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray, list[str]]:
    """Numeric signal + a null-carrying column + a string categorical."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)
    gappy = signal.copy()
    gappy[rng.random(n) < 0.25] = np.nan
    fm = pl.DataFrame(
        {
            "SIGNAL": signal,
            "GAPPY_MEAN": gappy,
            "NAME_KIND": rng.choice(["a", "b", "c"], size=n).tolist(),
            "TARGET": y,
        }
    )
    return fm, y, ["SIGNAL", "GAPPY_MEAN", "NAME_KIND"]


def test_split_feature_types_partitions_by_dtype() -> None:
    fm, _, cols = _frame()
    numeric, categorical = split_feature_types(fm, cols)

    assert numeric == ["SIGNAL", "GAPPY_MEAN"]
    assert categorical == ["NAME_KIND"]


def test_pipeline_adds_a_missingness_indicator() -> None:
    # The load-bearing design choice: a null must survive imputation as its own column,
    # or the linear model starts with less information than the null-native tree.
    fm, y, cols = _frame()
    numeric, categorical = split_feature_types(fm, cols)
    pipe = build_pipeline(numeric, categorical, TINY)

    out = pipe.named_steps["pre"].fit_transform(fm.select(cols).to_pandas())

    # 2 numeric + 1 indicator (GAPPY_MEAN only) + 3 one-hot levels
    assert out.shape[1] == 6


def test_pipeline_one_hot_encodes_rather_than_integer_codes() -> None:
    fm, y, cols = _frame()
    numeric, categorical = split_feature_types(fm, cols)
    pipe = build_pipeline(numeric, categorical, TINY, min_frequency=1)

    pre = pipe.named_steps["pre"].fit(fm.select(cols).to_pandas())
    names = pre.get_feature_names_out()
    onehot = [n for n in names if n.startswith("cat__")]

    # One column per level, not a single ordinal code column.
    assert len(onehot) == 3
    assert all("NAME_KIND" in n for n in onehot)


def test_cross_validate_linear_produces_out_of_fold_predictions() -> None:
    fm, y, cols = _frame()
    result = cross_validate_linear(fm, cols, y, TINY, n_splits=3)

    assert len(result.fold_scores) == 3
    assert result.oof_pred.shape == (len(y),)
    assert not np.any(result.oof_pred == 0.0)  # every row scored exactly once
    assert 0.0 <= result.oof_roc_auc <= 1.0
    # A separable-ish problem: the linear model must find real signal.
    assert result.oof_roc_auc > 0.75


def test_on_fold_callback_fires_per_fold() -> None:
    fm, y, cols = _frame()
    seen: list[int] = []

    cross_validate_linear(
        fm, cols, y, TINY, n_splits=3, on_fold=lambda i, auc: seen.append(i)
    )

    assert seen == [1, 2, 3]


def test_preprocessing_is_fitted_per_fold_not_globally() -> None:
    # Leakage guard: the imputer's median must come from the training split only. If
    # the pipeline were fitted once on the whole frame, a fold whose training rows have
    # a very different median would still score identically -- so the two must differ.
    fm, y, cols = _frame()
    shifted = fm.with_columns(
        pl.when(pl.arange(0, fm.height) < fm.height // 2)
        .then(pl.col("GAPPY_MEAN") + 50.0)
        .otherwise(pl.col("GAPPY_MEAN"))
        .alias("GAPPY_MEAN")
    )

    a = cross_validate_linear(fm, cols, y, TINY, n_splits=3)
    b = cross_validate_linear(shifted, cols, y, TINY, n_splits=3)

    assert a.oof_roc_auc != b.oof_roc_auc


def test_same_seed_is_deterministic() -> None:
    fm, y, cols = _frame()
    a = cross_validate_linear(fm, cols, y, TINY, n_splits=3, seed=7)
    b = cross_validate_linear(fm, cols, y, TINY, n_splits=3, seed=7)

    assert a.fold_scores == b.fold_scores
    np.testing.assert_array_equal(a.oof_pred, b.oof_pred)
