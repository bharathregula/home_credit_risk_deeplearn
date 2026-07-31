"""
The shared cross-validation contract: fold definition, result type, cost guard.

Every model family in this project — trees (:mod:`home_credit_risk.gbt`), the logistic
challenger (:mod:`home_credit_risk.linear`) and networks (:mod:`home_credit_risk.nn`) —
reports through the same :class:`CVResult` on the same folds, which is what makes their
scores comparable at all.

This module is **deliberately free of any estimator import**. It used to live in
``gbt.py``, which meant importing the CV contract also imported LightGBM: a network-only
run pulled in LightGBM's OpenMP runtime alongside PyTorch's, and on macOS the two thread
pools collide and segfault the process inside a torch kernel. Splitting the contract out
lets each model family load exactly one numeric backend.

A process that genuinely needs two backends at once (a future blend) should either set
``OMP_NUM_THREADS=1`` or, better, consume persisted out-of-fold prediction vectors
instead of re-fitting both families in one interpreter.
"""

import time
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np

# Canonical split. Every run must use these or its score is not comparable: a different
# `n_splits` produces different partitions even at the same seed.
N_SPLITS = 5
SEED = 42


class TimeBudgetExceeded(RuntimeError):
    """Raised when a cross-validation run outruns its wall-clock budget."""


@dataclass(frozen=True)
class CVResult:
    """Outcome of one cross-validation run, whatever the model family."""

    fold_scores: List[float]
    #: Trees for a GBT, epochs for a network, unused (zeros) for a linear model.
    best_iterations: List[int]
    oof_pred: np.ndarray
    oof_roc_auc: float
    oof_pr_auc: float
    importances: np.ndarray
    #: Per-fold validation metric at every boosting iteration — the curve used to judge
    #: how flat the optimum is. Tree-specific; empty for other families.
    fold_curves: List[List[float]] = field(default_factory=list)

    @property
    def mean_fold_score(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std_fold_score(self) -> float:
        return float(np.std(self.fold_scores))

    @property
    def median_best_iteration(self) -> int:
        """The iteration count to fix for a final re-fit without early stopping."""
        return int(np.median(self.best_iterations))

    def budget_was_binding(self, n_estimators: int, patience: int) -> bool:
        """
        Whether the iteration cap, rather than convergence, ended the fits.

        True when any fold's best iteration sits within ``patience`` of the cap: early
        stopping cannot have fired, so the model was still improving when it ran out of
        budget and the score understates the configuration.
        """
        return any(b > n_estimators - patience for b in self.best_iterations)


def deadline_from(max_seconds: Any) -> Any:
    """Absolute deadline for a wall-clock budget, or ``None`` when unbounded."""
    return time.time() + max_seconds if max_seconds is not None else None
