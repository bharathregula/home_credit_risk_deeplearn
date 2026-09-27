"""
Neural-network round: reference architecture, preprocessing ablations, architectures.

Stage A — the reference embedding MLP::

    uv run python scripts/run_nn.py --arch mlp

Stage B — preprocessing ablations, each an isolated change on the canonical folds.
These decide the preprocessing every later architecture inherits, which is why they run
*before* any architecture comparison::

    uv run python scripts/run_nn.py --quantile      # B1  rank-gauss instead (-0.0018)
    uv run python scripts/run_nn.py --no-indicators # B2  drop the flags     (-0.0012)
    uv run python scripts/run_nn.py --embeddings    # B3  embeddings instead (-0.0009)
    uv run python scripts/run_nn.py --all-features  # B4  945 features       (-0.0027)

Defaults are the measured winners from those runs, not assumptions: standard scaling,
indicators on, one-hot categoricals, 200-feature shortlist.

Stage C — architectures, with Stage-B preprocessing frozen::

    uv run python scripts/run_nn.py --arch resnet
    uv run python scripts/run_nn.py --arch mlp --seeds 5   # seed ensemble (ranks)

Stage D — the two directions the evidence actually favours. Seed-ensembling was worth
more than every architecture and preprocessing change combined, and numeric encoding is
reported to matter more than architecture, so both are tried before any transformer::

    uv run python scripts/run_nn.py --arch tabm            # implicit 32-member ensemble
    uv run python scripts/run_nn.py --arch tabm --periodic   # + periodic numeric embeds

Every run logs to MLflow tagged ``round=nn`` so it lands on the same leaderboard as the
tree and logistic runs, and prints its position against both.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from home_credit_risk.evaluation import N_SPLITS, SEED  # noqa: E402
from home_credit_risk.nn import (  # noqa: E402
    NNConfig,
    cross_validate_nn,
    resolve_device,
)
from home_credit_risk.tracking import (  # noqa: E402
    log_cv_metrics,
    log_oof_predictions,
    start_run,
)

# Deliberately NOT importing scripts/tune_gbt.py: it pulls in LightGBM, whose OpenMP
# runtime collides with PyTorch's and segfaults the process. A network run must load
# exactly one numeric backend.
DATASETS = Path("~/code/datasets/home-credit-default-risk").expanduser()
SHORTLIST = Path("~/code/datasets/gbt_shortlist.txt").expanduser()
OOF_DIR = DATASETS.parent / "oof"

# Reference points from the completed tree round, for the comparison printed at the end.
GBT_OOF = 0.79478
GBT_OOF_SHORTLIST = 0.79456
LOGISTIC_OOF = 0.77741


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["mlp", "resnet", "tabm"], default="mlp")
    # TabM: implicit ensembling inside one model. Motivated by this round's own
    # finding that 5-seed rank-averaging (+0.00283) beat every other change.
    parser.add_argument("--k", type=int, default=32, help="TabM members")
    parser.add_argument(
        "--periodic", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--d-embedding", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=0.05)
    # Defaults are the Stage-B measured winner: standard scaling, indicators on,
    # one-hot categoricals. Both directions are available for re-ablation.
    parser.add_argument(
        "--quantile", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--indicators", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--embeddings", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--all-features", action="store_true")
    parser.add_argument("--seeds", type=int, default=1, help="rank-average N seeds")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--patience", type=int, default=30, help="default = --epochs, i.e. full cycle"
    )
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--label", default=None, help="MLflow run name override")
    args = parser.parse_args(argv)

    t0 = time.time()
    from home_credit_risk.features import build_feature_matrix
    from home_credit_risk.modeling import apply_structural_encoding, split_xy

    app = pl.read_csv(DATASETS / "application_train.csv", infer_schema_length=10000)
    fm = apply_structural_encoding(build_feature_matrix(app, DATASETS))
    all_cols, y = split_xy(fm)

    if args.all_features:
        feature_cols, feature_set = all_cols, "all-945"
    else:
        shortlist = [f for f in SHORTLIST.read_text().split("\n") if f]
        feature_cols = [c for c in shortlist if c in fm.columns]
        feature_set = "shortlist-200"

    device = resolve_device(args.device)
    label = args.label or "-".join(
        filter(
            None,
            [
                f"nn-{args.arch}",
                feature_set,
                "rankgauss" if args.quantile else None,
                None if args.indicators else "noind",
                "emb" if args.embeddings else None,
                f"k{args.k}" if args.arch == "tabm" else None,
                f"plr{args.d_embedding}" if args.periodic else None,
                f"x{args.seeds}" if args.seeds > 1 else None,
            ],
        )
    )
    print(
        f"{label}: {len(feature_cols)} features, device={device}, "
        f"prepared in {time.time() - t0:.1f}s",
        flush=True,
    )

    oofs, fold_matrix, all_epochs = [], [], []
    for s in range(args.seeds):
        config = NNConfig(
            arch=args.arch,
            dropout=args.dropout,
            lr=args.lr,
            max_epochs=args.epochs,
            patience=args.patience,
            quantile=args.quantile,
            indicators=args.indicators,
            embeddings=args.embeddings,
            k=args.k,
            periodic=args.periodic,
            d_embedding=args.d_embedding,
            sigma=args.sigma,
            device=device,
            seed=SEED + s,
        )
        tag = f"  seed {s + 1}/{args.seeds}" if args.seeds > 1 else ""
        result = cross_validate_nn(
            fm,
            feature_cols,
            y,
            config,
            on_fold=lambda i, auc: print(f"{tag}  fold {i}: AUC {auc:.5f}", flush=True),
        )
        oofs.append(result.oof_pred)
        fold_matrix.append(result.fold_scores)
        all_epochs.append(result.best_iterations)
        if args.seeds > 1:
            print(f"{tag}  OOF {result.oof_roc_auc:.5f}", flush=True)

    # Rank-average across seeds: AUC depends only on ordering, and averaging ranks is
    # robust to seeds whose probability scales differ.
    from sklearn.metrics import average_precision_score, roc_auc_score

    blended = np.mean([rankdata(o) / len(o) for o in oofs], axis=0)
    oof_auc = float(roc_auc_score(y, blended))
    oof_pr = float(average_precision_score(y, blended))
    mean_folds = np.mean(fold_matrix, axis=0).tolist()

    with start_run(
        label,
        params={
            "arch": args.arch,
            "feature_set": feature_set,
            "n_features": len(feature_cols),
            "quantile": args.quantile,
            "indicators": args.indicators,
            "embeddings": args.embeddings,
            "k": args.k if args.arch == "tabm" else None,
            "periodic": args.periodic,
            "d_embedding": args.d_embedding if args.periodic else None,
            "sigma": args.sigma if args.periodic else None,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "dropout": args.dropout,
            "lr": args.lr,
            "device": device,
            "cv_seed": SEED,
            "n_splits": N_SPLITS,
        },
        tags={"model": "neural-net", "round": "nn"},
    ):
        log_cv_metrics(
            mean_folds,
            {
                "oof_roc_auc": oof_auc,
                "oof_pr_auc": oof_pr,
                "mean_best_epoch": float(np.mean(all_epochs)),
                "runtime_seconds": time.time() - t0,
            },
        )
        # The blend stage consumes this vector rather than re-fitting both families
        # in one interpreter, which would load two OpenMP runtimes and segfault.
        log_oof_predictions(blended, y, out_dir=OOF_DIR, filename=f"{label}.npz")

    print(f"\n{label}")
    print(f"  OOF ROC-AUC {oof_auc:.5f}   PR-AUC {oof_pr:.5f}")
    print(f"  mean epochs to best: {np.mean(all_epochs):.1f}")
    print("\n  reference points (identical folds):")
    print(f"    GBT / 945 features    {GBT_OOF:.5f}   ({oof_auc - GBT_OOF:+.5f})")
    print(
        f"    GBT / 200 shortlist   {GBT_OOF_SHORTLIST:.5f}   "
        f"({oof_auc - GBT_OOF_SHORTLIST:+.5f})"
    )
    delta_lr = oof_auc - LOGISTIC_OOF
    print(f"    Logistic / 200        {LOGISTIC_OOF:.5f}   ({delta_lr:+.5f})")
    if oof_auc < LOGISTIC_OOF:
        print(
            "\n  *** BELOW THE LOGISTIC CHALLENGER. Treat as a harness bug, not a "
            "result: check input scaling and the embedding index before reporting."
        )
    print(f"  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
