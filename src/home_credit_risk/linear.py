"""
Logistic-regression challenger, and the dense preprocessing it needs.

The gradient-boosted tree (:mod:`home_credit_risk.gbt`) consumes nulls and categoricals
natively. Every other model family needs them removed, and *how* they are removed
decides whether a comparison is fair. This module owns that path for the challenger.

:mod:`home_credit_risk.nn` deliberately does **not** reuse this pipeline: embeddings
need integer codes rather than one-hot blocks, and torch needs per-fold tensors rather
than a fitted sklearn transformer. The two share the three principles below, not code.

Three choices make the comparison honest rather than rigged:

- **Missingness indicators are added.** 183 of the 200 shortlisted features contain
  nulls and the nulls are predictive — customers with a null
  ``BUREAU_DAYS_ENDDATE_FACT_MEAN`` default at 10.5% against an 8.07% base rate.
  Median-filling alone would hand the linear model strictly less information than the
  tree had, and any "trees beat linear" conclusion drawn from that would be an artifact
  of the preprocessing.
- **Everything is fitted inside the fold.** Imputation medians, scaler statistics and
  the one-hot vocabulary are learned on the training split only. Fitting them on the
  full frame is the classic leak: the validation fold's own median is not available at
  scoring time in production.
- **Categoricals are one-hot encoded, not integer-coded.** A linear model reads an
  integer code as a magnitude, so ``ORGANIZATION_TYPE == 47`` would be "more" than
  ``== 3``. The tree partitions category *sets* and is immune; the linear model is not.

The challenger exists to answer a question the GBT alone cannot: whether the tree's
complexity is *justified*. A GBT that beats a properly-built linear model by a wide
margin has earned its interpretability cost; one that beats a handicapped baseline has
proved nothing.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from home_credit_risk.evaluation import N_SPLITS, SEED, CVResult

# Regularised by default: after one-hot expansion the design matrix carries strongly
# correlated columns, and an unpenalised fit on 307k rows would produce unstable
# coefficients without improving discrimination.
# `l1_ratio=0.0` is pure L2 in scikit-learn's current API; the older `penalty="l2"`
# spelling is deprecated as of 1.8, and `n_jobs` no longer has any effect.
DEFAULT_PARAMS: Dict[str, Any] = {
    "C": 0.1,
    "l1_ratio": 0.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": SEED,
}


def split_feature_types(
    fm: pl.DataFrame, feature_cols: Sequence[str]
) -> Tuple[List[str], List[str]]:
    """Partition ``feature_cols`` into (numeric, categorical) by dtype."""
    numeric, categorical = [], []
    for col in feature_cols:
        if col not in fm.columns:
            continue
        if fm[col].dtype in (pl.String, pl.Categorical):
            categorical.append(col)
        else:
            numeric.append(col)
    return numeric, categorical


def build_pipeline(
    numeric: Sequence[str],
    categorical: Sequence[str],
    params: Optional[Mapping[str, Any]] = None,
    min_frequency: int = 50,
) -> Pipeline:
    """
    Median-impute + indicate + scale numerics, one-hot categoricals, then fit logistic
    regression.

    ``add_indicator=True`` is the load-bearing argument: it appends a binary column per
    numeric feature that had missing values in training, so the information the tree
    reads from a null survives imputation. ``min_frequency`` folds rare categories into
    an "infrequent" bucket, which keeps the design matrix bounded and avoids fitting a
    coefficient to a handful of rows.
    """
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric_pipe, list(numeric)),
            ("cat", categorical_pipe, list(categorical)),
        ],
        remainder="drop",
    )
    merged: Dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
    return Pipeline([("pre", pre), ("model", LogisticRegression(**merged))])


def cross_validate_linear(
    fm: pl.DataFrame,
    feature_cols: Sequence[str],
    y: np.ndarray,
    params: Optional[Mapping[str, Any]] = None,
    *,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    on_fold: Optional[Callable[[int, float], None]] = None,
) -> CVResult:
    """
    Cross-validate the logistic challenger on the **same folds** as the GBT.

    ``n_splits`` and ``seed`` default to the canonical split so the resulting OOF score
    is directly comparable to :func:`gbt.cross_validate`; changing either makes the
    number incomparable with every other run in this project.

    The whole pipeline — imputer, scaler, one-hot vocabulary, model — is fitted per
    fold, so nothing about the validation split informs the transform applied to it.

    ``CVResult.importances`` is returned as zeros: coefficients live in the
    post-one-hot, post-indicator space and do not map back to ``feature_cols``
    one-for-one. The challenger is a benchmark, not a feature-selection tool.
    """
    pdf = fm.select(list(feature_cols)).to_pandas()
    numeric, categorical = split_feature_types(fm, feature_cols)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    fold_scores: List[float] = []

    for fold, (tr, va) in enumerate(skf.split(pdf, y), 1):
        pipe = build_pipeline(numeric, categorical, params)
        pipe.fit(pdf.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(pdf.iloc[va])[:, 1]
        score = float(roc_auc_score(y[va], oof[va]))
        fold_scores.append(score)
        if on_fold is not None:
            on_fold(fold, score)

    return CVResult(
        fold_scores=fold_scores,
        best_iterations=[0] * n_splits,
        oof_pred=oof,
        oof_roc_auc=float(roc_auc_score(y, oof)),
        oof_pr_auc=float(average_precision_score(y, oof)),
        importances=np.zeros(len(feature_cols), dtype=float),
    )
