"""
Logistic-regression challenger on the tree-selected shortlist.

Answers the question the GBT cannot answer about itself: **is the tree's complexity
justified?** Without a properly-built linear baseline, "the GBT scores 0.795" has no
reference point, and a later "the neural net beat the tree" is a two-point comparison
with no floor under it.

Run after `scripts/tune_gbt.py final`, which writes the shortlist this consumes::

    uv run python scripts/challenger.py                 # top-200 shortlist
    uv run python scripts/challenger.py --all-features  # all 945 (slow, memory-heavy)

The challenger is deliberately built to be *strong*, not to be beaten: it receives
missingness indicators, per-fold-fitted imputation and scaling, and one-hot
categoricals (see :mod:`home_credit_risk.linear`). A baseline handicapped by its
preprocessing proves nothing about the model that beats it.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import List, Optional

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from home_credit_risk.evaluation import N_SPLITS, SEED  # noqa: E402
from home_credit_risk.linear import (  # noqa: E402
    DEFAULT_PARAMS,
    cross_validate_linear,
    split_feature_types,
)
from home_credit_risk.tracking import log_cv_metrics, start_run  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gbtscript", Path(__file__).resolve().parent / "gbt.py"
)
assert _spec is not None and _spec.loader is not None
_gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gs)

SHORTLIST = Path("~/code/datasets/gbt_shortlist.txt").expanduser()

# The GBT's headline, for the comparison printed at the end.
GBT_OOF_ROC_AUC = 0.79478


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="use all 945 features instead of the top-200 shortlist",
    )
    parser.add_argument("--C", type=float, default=DEFAULT_PARAMS["C"])
    args = parser.parse_args(argv)

    t0 = time.time()
    app = pl.read_csv(_gs.DATASETS / "application_train.csv", infer_schema_length=10000)
    from home_credit_risk.features import build_feature_matrix
    from home_credit_risk.modeling import apply_structural_encoding, split_xy

    fm = apply_structural_encoding(build_feature_matrix(app, _gs.DATASETS))
    all_cols, y = split_xy(fm)

    if args.all_features:
        feature_cols = all_cols
        label = "all-features"
    else:
        shortlist = [f for f in SHORTLIST.read_text().split("\n") if f]
        feature_cols = [c for c in shortlist if c in fm.columns]
        label = "shortlist"

    numeric, categorical = split_feature_types(fm, feature_cols)
    print(
        f"logistic challenger on {len(feature_cols)} features "
        f"({len(numeric)} numeric, {len(categorical)} categorical), "
        f"prepared in {time.time() - t0:.1f}s",
        flush=True,
    )

    params = {**DEFAULT_PARAMS, "C": args.C}
    with start_run(
        f"logistic-{label}",
        params={
            **params,
            "n_features": len(feature_cols),
            "n_numeric": len(numeric),
            "n_categorical": len(categorical),
            "cv_seed": SEED,
            "n_splits": N_SPLITS,
            "feature_set": label,
        },
        tags={"model": "logistic-regression", "round": "challenger"},
    ):
        result = cross_validate_linear(
            fm,
            feature_cols,
            y,
            params,
            on_fold=lambda i, auc: print(f"  fold {i}: AUC {auc:.5f}", flush=True),
        )
        log_cv_metrics(
            result.fold_scores,
            {
                "oof_roc_auc": result.oof_roc_auc,
                "oof_pr_auc": result.oof_pr_auc,
                "runtime_seconds": time.time() - t0,
            },
        )

    print(f"\nlogistic regression ({label})")
    print(f"  OOF ROC-AUC {result.oof_roc_auc:.5f}   PR-AUC {result.oof_pr_auc:.5f}")
    print(f"  mean fold {result.mean_fold_score:.5f} +/- {result.std_fold_score:.5f}")
    gap = GBT_OOF_ROC_AUC - result.oof_roc_auc
    print(f"\n  GBT (LightGBM, 945 features): {GBT_OOF_ROC_AUC:.5f}")
    print(f"  gap: {gap:+.5f} in the GBT's favour")
    print(
        "  the tree's complexity is "
        + ("justified" if gap > 0.01 else "only marginally justified")
        + " on discrimination alone"
    )
    print(f"  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
