import numpy as np
import pytest

from home_credit_risk.evaluation import CVResult, TimeBudgetExceeded
from home_credit_risk.gbt import DEFAULT_PARAMS, cross_validate, curve_flatness

# Tiny, fast params — these tests check the loop's mechanics, not model quality.
TINY = {"n_estimators": 12, "num_leaves": 4, "min_child_samples": 5, "verbose": -1}


def _data(n: int = 240, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """A separable-ish binary problem with a null column and a categorical column."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)
    noise = rng.normal(size=n)
    cat = rng.integers(0, 3, size=n).astype(float)
    gappy = signal.copy()
    gappy[rng.random(n) < 0.2] = np.nan  # nulls must survive to LightGBM untouched
    X = np.column_stack([signal, noise, cat, gappy])
    return X, y, [2]


def test_cross_validate_returns_one_score_per_fold() -> None:
    X, y, cat = _data()
    result = cross_validate(X, y, cat, TINY, n_splits=3)

    assert isinstance(result, CVResult)
    assert len(result.fold_scores) == 3
    assert len(result.best_iterations) == 3
    assert all(0.0 <= s <= 1.0 for s in result.fold_scores)
    assert 0.0 <= result.oof_roc_auc <= 1.0
    assert 0.0 <= result.oof_pr_auc <= 1.0


def test_every_row_gets_exactly_one_out_of_fold_prediction() -> None:
    # The whole point of OOF: each row is predicted by a model that never trained on it.
    X, y, cat = _data()
    result = cross_validate(X, y, cat, TINY, n_splits=3)

    assert result.oof_pred.shape == (len(y),)
    assert not np.any(result.oof_pred == 0.0)  # 0.0 is the untouched initial value
    assert np.all((result.oof_pred > 0.0) & (result.oof_pred < 1.0))


def test_importances_align_with_feature_columns() -> None:
    X, y, cat = _data()
    result = cross_validate(X, y, cat, TINY, n_splits=3)

    assert result.importances.shape == (X.shape[1],)
    assert result.importances.sum() > 0  # gain accumulated across folds


def test_on_fold_is_called_once_per_fold_with_the_fold_score() -> None:
    X, y, cat = _data()
    seen: list[tuple[int, float]] = []

    result = cross_validate(
        X, y, cat, TINY, n_splits=3, on_fold=lambda i, auc: seen.append((i, auc))
    )

    assert [i for i, _ in seen] == [1, 2, 3]  # 1-indexed, in order
    assert [auc for _, auc in seen] == result.fold_scores


def test_on_fold_exception_aborts_the_run() -> None:
    # This is the pruning seam: Optuna raises TrialPruned from on_fold to stop early.
    X, y, cat = _data()
    calls: list[int] = []

    class Pruned(Exception):
        pass

    def prune_after_second(fold: int, auc: float) -> None:
        calls.append(fold)
        if fold == 2:
            raise Pruned

    with pytest.raises(Pruned):
        cross_validate(X, y, cat, TINY, n_splits=5, on_fold=prune_after_second)

    assert calls == [1, 2]  # folds 3-5 never ran


def test_same_seed_gives_identical_results() -> None:
    X, y, cat = _data()
    a = cross_validate(X, y, cat, TINY, n_splits=3, seed=7)
    b = cross_validate(X, y, cat, TINY, n_splits=3, seed=7)

    assert a.fold_scores == b.fold_scores
    np.testing.assert_array_equal(a.oof_pred, b.oof_pred)


def test_different_fold_counts_are_not_comparable() -> None:
    # Guard for the trap in the tuning plan: 3-fold and 5-fold at the same seed are
    # different partitions, so their scores must never be compared to each other.
    X, y, cat = _data()
    three = cross_validate(X, y, cat, TINY, n_splits=3, seed=42)
    five = cross_validate(X, y, cat, TINY, n_splits=5, seed=42)

    assert three.oof_roc_auc != five.oof_roc_auc


def test_params_are_merged_over_the_defaults() -> None:
    X, y, cat = _data()
    # Only n_estimators is overridden, so the rest of DEFAULT_PARAMS must still apply.
    result = cross_validate(X, y, cat, {"n_estimators": 7}, n_splits=2)

    assert all(b <= 7 for b in result.best_iterations)
    assert DEFAULT_PARAMS["learning_rate"] == 0.02  # defaults untouched by the merge


def test_without_early_stopping_all_trees_are_used() -> None:
    X, y, cat = _data()
    result = cross_validate(X, y, cat, {**TINY, "n_estimators": 9}, n_splits=2)

    # best_iteration_ is None with early stopping off -> falls back to the full
    # budget, which is what makes the final unbiased re-fit possible.
    assert result.best_iterations == [9, 9]
    assert result.median_best_iteration == 9


def test_budget_was_binding_detects_a_truncated_fit() -> None:
    X, y, cat = _data()
    result = cross_validate(
        X, y, cat, {**TINY, "n_estimators": 200}, n_splits=2, early_stopping_rounds=5
    )

    # A trivial problem converges long before 200 trees, so the cap cannot be binding.
    assert not result.budget_was_binding(200, 5)
    # ...but judged against a cap just above where it stopped, it would be.
    assert result.budget_was_binding(max(result.best_iterations) + 1, 5)


def test_curve_flatness_measures_loss_away_from_the_peak() -> None:
    X, y, cat = _data()
    result = cross_validate(
        X, y, cat, {**TINY, "n_estimators": 60}, n_splits=3, early_stopping_rounds=20
    )

    flat = curve_flatness(result, fractions=(0.5, 1.5))

    assert flat["peak_iteration"] >= 1
    assert 0.0 <= flat["peak_auc"] <= 1.0
    # Moving away from the peak can only lose AUC, never gain it.
    assert flat["auc_loss_at_0.5x"] >= 0.0
    assert flat["auc_loss_at_1.5x"] >= 0.0


def test_curve_flatness_is_empty_without_curves() -> None:
    empty = CVResult(
        fold_scores=[0.5],
        best_iterations=[1],
        oof_pred=np.zeros(1),
        oof_roc_auc=0.5,
        oof_pr_auc=0.5,
        importances=np.zeros(1),
    )
    assert curve_flatness(empty) == {}


def test_max_seconds_aborts_mid_training() -> None:
    # The between-folds `on_fold` hook cannot bound cost when a *single* fold overruns:
    # that is how one search trial reached 34 minutes against a 10-minute budget. The
    # guard therefore runs every boosting iteration.
    X, y, cat = _data()

    with pytest.raises(TimeBudgetExceeded):
        cross_validate(
            X, y, cat, {**TINY, "n_estimators": 500}, n_splits=2, max_seconds=0.0
        )


def test_max_seconds_none_does_not_interfere() -> None:
    X, y, cat = _data()
    result = cross_validate(X, y, cat, TINY, n_splits=2, max_seconds=None)

    assert len(result.fold_scores) == 2


def test_generous_max_seconds_completes_normally() -> None:
    X, y, cat = _data()
    a = cross_validate(X, y, cat, TINY, n_splits=2, seed=3, max_seconds=600)
    b = cross_validate(X, y, cat, TINY, n_splits=2, seed=3)

    # The guard must be inert when it never fires.
    assert a.fold_scores == b.fold_scores
