"""
Modeling helpers for the Home Credit default-risk model.

Pure polars / numpy utilities that prepare the engineered feature matrix
(``features.build_feature_matrix``) for a classifier. Deliberately free of any
estimator dependency (scikit-learn / LightGBM) so the module stays cleanly typed under
mypy-strict; the actual model fitting lives in ``notebooks/modeling/``.

Two ideas drive this module (see ``docs/MODELING_SUMMARY.md``):

- **Structural encoding is not imputation.** A null ``*_COUNT`` / ``*_SUM`` /
  ``*_RATE`` / ``HAS_*`` means "no such history", whose correct value is ``0`` — so
  :func:`apply_structural_encoding` fills those with ``0``. Behavioural ``*_MEAN`` /
  ``*_MIN`` / ``*_MAX`` are genuinely unknown when absent and are left null (a
  null-native tree consumes them directly; a later linear/NN model imputes them
  per-fold inside cross-validation).
- **Selection should be null-aware.** Feature pruning is done downstream from
  gradient-boosted-tree gain importance, not from missingness or median-filled
  correlations; :func:`constant_feature_columns` and :func:`duplicate_feature_columns`
  only remove genuinely useless zero-variance and bit-identical columns.
"""

import hashlib
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

# Columns whose null means "no such history" and whose correct fill is 0. Counts and
# sums of absent records are zero; one-hot rates and distinct-counts are zero; the HAS_*
# coverage flags are already 0/1 but are normalised for safety.
_STRUCTURAL_SUFFIXES: Tuple[str, ...] = ("_COUNT", "_SUM", "_NUNIQUE", "_RATE")
_STRUCTURAL_PREFIXES: Tuple[str, ...] = ("HAS_",)

# Never used as model inputs: the target and the customer identifier.
_ALWAYS_DROP: Tuple[str, ...] = ("TARGET",)


def _is_structural(col: str) -> bool:
    """Whether ``col`` is a count/sum/rate/flag whose null should become 0."""
    return col.startswith(_STRUCTURAL_PREFIXES) or col.endswith(_STRUCTURAL_SUFFIXES)


def apply_structural_encoding(fm: pl.DataFrame) -> pl.DataFrame:
    """
    Fill structural nulls with ``0`` (feature construction, not statistical imputation).

    For every ``*_COUNT`` / ``*_SUM`` / ``*_NUNIQUE`` / ``*_RATE`` column and every
    ``HAS_*`` flag, a null means the customer has no such history — the correct value
    is ``0``, not "unknown" — so it is filled. Behavioural ``*_MEAN`` / ``*_MIN`` /
    ``*_MAX`` columns are left null on purpose (genuinely unknown when absent).
    """
    structural = [c for c in fm.columns if _is_structural(c)]
    if not structural:
        return fm
    return fm.with_columns(pl.col(c).fill_null(0) for c in structural)


def constant_feature_columns(
    fm: pl.DataFrame, feature_cols: Sequence[str]
) -> List[str]:
    """
    Return the zero-variance columns among ``feature_cols`` (safe to drop pre-fit).

    A column is zero-variance if it has at most one distinct non-null value — i.e. it
    is fully null or constant (e.g. the ``*_SK_DPD*_MIN_*`` columns that are always 0).
    Such columns carry no signal for any model. Call this *after*
    :func:`apply_structural_encoding` so a formerly all-null structural column is judged
    on its encoded (constant-0) values.
    """
    cols = [c for c in feature_cols if c in fm.columns]
    if not cols:
        return []
    distinct = fm.select(pl.col(c).drop_nulls().n_unique().alias(c) for c in cols)
    return [c for c in cols if int(distinct[c].item()) <= 1]


def duplicate_feature_columns(
    fm: pl.DataFrame, feature_cols: Sequence[str]
) -> List[str]:
    """
    Return columns that exactly duplicate an earlier column (safe to drop pre-fit).

    Keep-first: for each group of bit-identical columns only the *first* name in
    ``feature_cols`` order is retained and the rest are returned. Nulls compare equal,
    so this is run *after* :func:`apply_structural_encoding` on the values the model
    will actually see.

    Redundant aggregations produce a surprising number of these — an inner ``MIN`` that
    is constant within every loan makes the outer mean/min/max coincide, and
    ``aggregate_categorical``'s one-hot rate can reproduce a hand-written domain rate
    exactly. Duplicates are harmless for a tree (gain just splits arbitrarily between
    them) but are perfectly collinear inputs for a linear / neural model.
    """
    cols = [c for c in feature_cols if c in fm.columns]
    buckets: Dict[bytes, List[str]] = {}
    for col in cols:
        # Hash first so string and numeric columns are handled uniformly, then verify
        # exact equality within the bucket — a hash collision must not drop a column.
        digest = hashlib.blake2b(
            fm[col].hash(seed=0).to_numpy().tobytes(), digest_size=16
        ).digest()
        buckets.setdefault(digest, []).append(col)

    dupes: List[str] = []
    for group in buckets.values():
        if len(group) == 1:
            continue
        keepers: List[str] = []
        for col in group:
            if any(fm[col].equals(fm[k]) for k in keepers):
                dupes.append(col)
            else:
                keepers.append(col)
    return [c for c in cols if c in set(dupes)]


def split_xy(
    fm: pl.DataFrame, drop: Sequence[str] = ()
) -> Tuple[List[str], np.ndarray]:
    """
    Split ``fm`` into the model feature-column list and the ``y`` target vector.

    Excludes ``TARGET``, any ``SK_ID*`` identifier, and any names in ``drop`` (e.g. the
    zero-variance columns from :func:`constant_feature_columns`). Returns the ordered
    feature-column names (build ``X`` with ``fm.select(cols).to_numpy()`` — keeping the
    names lets you map gain importances back to features) and ``y`` as a numpy array.
    """
    if "TARGET" not in fm.columns:
        raise ValueError("fm must contain a 'TARGET' column")
    drop_set = set(_ALWAYS_DROP) | set(drop)
    feature_cols = [
        c for c in fm.columns if c not in drop_set and not c.startswith("SK_ID")
    ]
    y: np.ndarray = fm["TARGET"].to_numpy()
    return feature_cols, y


def category_mapping(
    fm: pl.DataFrame, feature_cols: Sequence[str]
) -> Dict[str, List[str]]:
    """
    Fit the value→code mapping for the string columns among ``feature_cols``.

    Returns ``{column: sorted distinct non-null values}``; a value's integer code is its
    index in that list. Fit this on the **training** frame and hand it to
    :func:`to_tree_matrix` for every other frame, so a category maps to the same code
    everywhere — encoding two frames independently would otherwise assign codes by each
    frame's own value set and silently scramble the mapping at scoring time.
    """
    return {
        col: sorted(fm[col].drop_nulls().unique().to_list())
        for col in feature_cols
        if col in fm.columns and fm[col].dtype in (pl.String, pl.Categorical)
    }


def to_tree_matrix(
    fm: pl.DataFrame,
    feature_cols: Sequence[str],
    categories: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Build a numeric ``X`` for gradient-boosted trees from ``feature_cols``.

    String / categorical columns (the driver's own ``NAME_*`` etc.) are integer-coded
    so a numpy matrix can be formed, and their positions are returned as
    ``categorical_feature`` indices for LightGBM's native categorical handling — nulls
    stay null (missing) rather than becoming a spurious category. Numeric columns pass
    through unchanged, keeping their nulls for the tree to split on.

    Codes come from ``categories`` (see :func:`category_mapping`) when given, and from
    this frame's own sorted values otherwise. A value absent from the mapping encodes
    as null — an unseen category is genuinely unknown, and LightGBM already has a
    learned direction for missing.
    """
    mapping = (
        dict(categories)
        if categories is not None
        else category_mapping(fm, feature_cols)
    )
    exprs: List[pl.Expr] = []
    categorical_idx: List[int] = []
    for i, col in enumerate(feature_cols):
        if fm[col].dtype in (pl.String, pl.Categorical):
            levels = [str(v) for v in mapping.get(col, [])]
            exprs.append(
                pl.col(col)
                .cast(pl.String)
                .cast(pl.Enum(levels), strict=False)
                .to_physical()
                .alias(col)
            )
            categorical_idx.append(i)
        else:
            exprs.append(pl.col(col))
    X: np.ndarray = fm.select(exprs).to_numpy()
    return X, categorical_idx
