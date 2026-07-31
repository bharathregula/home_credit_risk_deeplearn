"""
Gradient-boosted-tree cross-validation for the Home Credit default-risk model.

This is the one module that imports an estimator. ``modeling.py`` deliberately stays
free of LightGBM / scikit-learn so it remains a pure prep layer; the fitting loop lives
here so that the tuning search, the neural-net round and the notebooks all evaluate on
**identical folds** rather than each re-implementing the loop.

Two design points worth knowing:

- **One CV run, one score.** :func:`cross_validate` returns the out-of-fold prediction
  vector and its ROC-AUC / PR-AUC. Per-fold scores come back too, but the OOF number is
  the only one comparable across configurations.
- **``on_fold`` is a framework-agnostic pruning seam.** Optuna is *not* imported here.
  The caller passes a callback that receives ``(fold_index, fold_auc)`` after every fold
  and may raise to abandon the run — which is exactly what
  ``optuna.TrialPruned`` needs, without this module depending on Optuna. See
  ``scripts/tune_gbt.py``.

``n_estimators`` is not a hyperparameter to search. With ``early_stopping_rounds`` set
it is *derived*: each fold fits until its validation AUC stops improving and records
``best_iteration_``. Pass a cap generous enough that it never binds — if
``best_iteration_`` lands within ``early_stopping_rounds`` of ``n_estimators`` then the
budget is the constraint, not convergence, and the score is a floor rather than the
configuration's ceiling.
"""

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from home_credit_risk.evaluation import (
    N_SPLITS,
    SEED,
    CVResult,
    TimeBudgetExceeded,
)


def _time_guard(deadline: float) -> Any:
    """
    A LightGBM callback that aborts training once ``deadline`` passes.

    Checked **every boosting iteration**, which is the only granularity that actually
    bounds cost. A between-folds check cannot: in this project a single fold once ran 34
    minutes against a 10-minute trial budget, so the guard had no opportunity to fire
    until the damage was done. Trial cost varies by orders of magnitude across a search
    space — configurations that never trigger early stopping run to the full tree cap —
    and quality-based pruning is no defence, since a slow trial may be scoring well.
    """

    def _callback(env: Any) -> None:
        if time.time() > deadline:
            raise TimeBudgetExceeded(
                f"exceeded the time budget at iteration {env.iteration}"
            )

    # LightGBM inspects these attributes: run after the built-in callbacks, and after
    # each iteration rather than before it.
    _callback.order = 40  # type: ignore[attr-defined]
    _callback.before_iteration = False  # type: ignore[attr-defined]
    return _callback


# The reference configuration (docs/MODELING_SUMMARY.md). Kept here as the single source
# of truth so the notebook, the tuning search and the final re-fit cannot drift apart.
DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "num_leaves": 34,
    "feature_fraction": 0.7,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "min_child_samples": 100,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    cat_idx: Sequence[int],
    params: Optional[Mapping[str, Any]] = None,
    *,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    early_stopping_rounds: Optional[int] = None,
    on_fold: Optional[Callable[[int, float], None]] = None,
    max_seconds: Optional[float] = None,
) -> CVResult:
    """
    Stratified k-fold cross-validation of a LightGBM classifier.

    Parameters
    ----------
    X, y :
        Feature matrix and target from :func:`modeling.to_tree_matrix` /
        :func:`modeling.split_xy`. ``X`` keeps its nulls — LightGBM learns a default
        split direction for missing, so no imputation happens anywhere in this path.
    cat_idx :
        Column indices of the integer-coded categoricals, for LightGBM's native
        categorical handling (from :func:`modeling.to_tree_matrix`).
    params :
        LightGBM keyword arguments; defaults to :data:`DEFAULT_PARAMS`. Merged *over*
        the defaults, so a partial override only changes what it names.
    n_splits, seed :
        Fold count and shuffle seed. Leave at the module defaults unless you intend the
        result to be incomparable with every other run.
    early_stopping_rounds :
        Patience for early stopping against each fold's own validation split. ``None``
        disables it, which is how the final unbiased re-fit is done: with no early
        stopping there is no best-iteration chosen on the scored fold, so the
        selection optimism disappears.
    on_fold :
        Called as ``on_fold(fold_index, fold_auc)`` after each fold, 1-indexed. Any
        exception it raises propagates out of this function — the mechanism a pruner
        uses to abandon an unpromising run partway through.
    max_seconds :
        Wall-clock budget for the whole run, enforced **every boosting iteration** and
        raising :class:`TimeBudgetExceeded`. Necessary because trial cost varies by
        orders of magnitude across a hyperparameter space, and a config that never
        triggers early stopping runs to the full tree cap. ``on_fold`` cannot bound this
        on its own: it only fires between folds.

    Notes
    -----
    Predictions use ``best_iteration_`` when early stopping is enabled. That iteration
    is chosen on the fold being scored, so the OOF figure is mildly optimistic; the bias
    is constant across runs (fine for comparisons) but a final number should come from a
    re-fit with ``early_stopping_rounds=None``.
    """
    merged: Dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
    cat = list(cat_idx)
    deadline = time.time() + max_seconds if max_seconds is not None else None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    importances = np.zeros(X.shape[1], dtype=float)
    fold_scores: List[float] = []
    best_iterations: List[int] = []
    fold_curves: List[List[float]] = []

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model = LGBMClassifier(**merged)
        # Annotated as the heterogeneous list `fit` expects: the two callback classes
        # have unrelated types, so an inferred list would be invariant and rejected.
        callbacks: List[Callable[..., Any]] = [log_evaluation(0)]
        if deadline is not None:
            callbacks.append(_time_guard(deadline))
        if early_stopping_rounds is not None:
            # verbose=False: the callback's own per-fold banner is noise, and whether
            # stopping fired is reported from best_iterations instead.
            callbacks.insert(0, early_stopping(early_stopping_rounds, verbose=False))

        model.fit(
            X[tr],
            y[tr],
            # eval_X/eval_y replace the deprecated eval_set: one array each (or a
            # *tuple* for several sets) — not the list-of-pairs eval_set took.
            eval_X=X[va],
            eval_y=y[va],
            eval_metric="auc",
            categorical_feature=cat,
            callbacks=callbacks,
        )

        # asarray: predict_proba is typed as a union including sparse matrices, which is
        # unreachable for a dense ndarray input.
        oof[va] = np.asarray(model.predict_proba(X[va]))[:, 1]
        importances += model.booster_.feature_importance(importance_type="gain")
        score = float(roc_auc_score(y[va], oof[va]))
        fold_scores.append(score)
        # Without early stopping best_iteration_ is None; the fit used every tree.
        best_iterations.append(int(model.best_iteration_ or merged["n_estimators"]))
        curve = model.evals_result_.get("valid_0", {}).get("auc", [])
        fold_curves.append([float(v) for v in curve])

        if on_fold is not None:
            on_fold(fold, score)

    return CVResult(
        fold_scores=fold_scores,
        best_iterations=best_iterations,
        oof_pred=oof,
        oof_roc_auc=float(roc_auc_score(y, oof)),
        oof_pr_auc=float(average_precision_score(y, oof)),
        importances=importances,
        fold_curves=fold_curves,
    )


def curve_flatness(
    result: CVResult, fractions: Sequence[float] = (0.7, 1.3)
) -> Dict[str, float]:
    """
    How much AUC is given up by using the wrong number of trees.

    Averages each fold's validation curve, then reports the loss at
    ``fraction × best_iteration`` relative to the peak. When those losses are small
    (order 1e-4) the optimum is flat, the exact tree count is immaterial, and it can be
    fixed rather than tuned — which is the question the budget run exists to answer.
    """
    if not result.fold_curves or not result.fold_curves[0]:
        return {}
    width = min(len(c) for c in result.fold_curves)
    mean_curve = np.mean([c[:width] for c in result.fold_curves], axis=0)
    peak = int(np.argmax(mean_curve))
    out: Dict[str, float] = {
        "peak_iteration": float(peak + 1),
        "peak_auc": float(mean_curve[peak]),
    }
    for frac in fractions:
        idx = min(max(int(round((peak + 1) * frac)) - 1, 0), width - 1)
        out[f"auc_loss_at_{frac:g}x"] = float(mean_curve[peak] - mean_curve[idx])
    return out
