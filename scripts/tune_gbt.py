"""
GBT tuning driver: budget probe, Optuna search, transfer, and the final clean re-fit.

Run in this order — each mode consumes the previous one's answer::

    uv run python scripts/tune_gbt.py budget                  # where does it converge?
    uv run python scripts/tune_gbt.py search --minutes 55  # joint regularisation search
    uv run python scripts/tune_gbt.py transfer             # best config at target lr
    uv run python scripts/tune_gbt.py final --n-estimators N  # unbiased + shortlist

Why a script rather than a notebook: the search runs for the better part of an hour, has
to survive a disconnect, and must be resumable. The Optuna study is persisted to
``sqlite:///optuna.db``, so re-running ``search`` extends the existing history rather
than starting over. The cross-validation loop itself lives in
:mod:`home_credit_risk.gbt` so it is unit-tested and shared with the later NN round.

The feature matrix is built **once** per invocation and reused across every trial: there
is no parquet cache, and rebuilding costs ~16s of CSV reading that must not be repeated
for each of a dozen-plus trials.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from home_credit_risk.evaluation import (  # noqa: E402
    N_SPLITS,
    SEED,
    CVResult,
    TimeBudgetExceeded,
)
from home_credit_risk.features import build_feature_matrix  # noqa: E402
from home_credit_risk.gbt import (  # noqa: E402
    DEFAULT_PARAMS,
    cross_validate,
    curve_flatness,
)
from home_credit_risk.modeling import (  # noqa: E402
    apply_structural_encoding,
    category_mapping,
    constant_feature_columns,
    duplicate_feature_columns,
    split_xy,
    to_tree_matrix,
)
from home_credit_risk.tracking import (  # noqa: E402
    log_cv_metrics,
    log_feature_importance,
    log_text_artifact,
    start_run,
)

DATASETS = Path("~/code/datasets/home-credit-default-risk").expanduser()
OUT_DIR = DATASETS.parent
STUDY_STORAGE = "sqlite:///optuna.db"
STUDY_NAME = "gbt-screening"

# The search runs at a higher learning rate than the target so each trial converges in
# ~0.4x the trees (~7 min instead of ~18). The winner is transferred back down by the
# `transfer` mode; regularisation optima are close enough across nearby learning rates
# for *ranking* purposes, which is all the search needs to do.
SEARCH_LR = 0.05
TARGET_LR = 0.02


def prepare() -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """Build and encode the matrix, drop dead columns, return the model inputs."""
    t0 = time.time()
    app = pl.read_csv(DATASETS / "application_train.csv", infer_schema_length=10000)
    fm = apply_structural_encoding(build_feature_matrix(app, DATASETS))

    feature_cols, y = split_xy(fm)
    constants = set(constant_feature_columns(fm, feature_cols))
    feature_cols = [c for c in feature_cols if c not in constants]
    dupes = set(duplicate_feature_columns(fm, feature_cols))
    feature_cols = [c for c in feature_cols if c not in dupes]

    cats = category_mapping(fm, feature_cols)
    X, cat_idx = to_tree_matrix(fm, feature_cols, categories=cats)
    print(
        f"matrix {fm.shape} -> {len(feature_cols)} features "
        f"({len(constants)} constant, {len(dupes)} duplicate dropped), "
        f"{len(cat_idx)} categorical, in {time.time() - t0:.1f}s",
        flush=True,
    )
    return X, y, feature_cols, cat_idx


def _report(result: CVResult, label: str) -> None:
    print(f"\n{label}")
    for i, (s, b) in enumerate(zip(result.fold_scores, result.best_iterations), 1):
        print(f"  fold {i}: AUC {s:.5f}  best_iter {b}")
    print(f"  OOF ROC-AUC {result.oof_roc_auc:.5f}   PR-AUC {result.oof_pr_auc:.5f}")
    print(
        f"  mean fold {result.mean_fold_score:.5f} +/- {result.std_fold_score:.5f}"
        f"   median best_iter {result.median_best_iteration}",
        flush=True,
    )


def mode_budget(args: argparse.Namespace) -> None:
    """
    Find where the model actually converges, and re-establish the reference there.

    One full 5-fold run at a large cap with wide patience answers three things at once:
    the peak tree count, how flat the optimum is, and a new reference OOF directly
    comparable to the old one. Patience of 50 was too tight at lr=0.02, where late
    gains are ~1e-5/round; 300 will not stop on noise.
    """
    X, y, feature_cols, cat_idx = prepare()
    params = {**DEFAULT_PARAMS, "n_estimators": args.n_estimators}

    t0 = time.time()
    with start_run(
        f"gbt-budget-{args.n_estimators}",
        params={
            **params,
            "early_stopping_rounds": args.patience,
            "n_features": len(feature_cols),
            "cv_seed": SEED,
            "n_splits": N_SPLITS,
        },
        tags={"model": "lightgbm", "round": "budget-probe"},
    ) as run:
        result = cross_validate(
            X, y, cat_idx, params, early_stopping_rounds=args.patience
        )
        flat = curve_flatness(result)
        binding = result.budget_was_binding(args.n_estimators, args.patience)
        log_cv_metrics(
            result.fold_scores,
            {
                "oof_roc_auc": result.oof_roc_auc,
                "oof_pr_auc": result.oof_pr_auc,
                "median_best_iteration": float(result.median_best_iteration),
                "runtime_seconds": time.time() - t0,
                **flat,
            },
        )
        log_feature_importance(feature_cols, result.importances.tolist())
        run_id = run.info.run_id

    _report(result, f"budget run (cap {args.n_estimators}, patience {args.patience})")
    print(f"  curve: {', '.join(f'{k}={v:.5f}' for k, v in flat.items())}")
    if binding:
        print(
            "\n  *** THE CAP IS STILL BINDING: a fold stopped within `patience` of "
            f"{args.n_estimators} trees. Raise --n-estimators and re-run before tuning "
            "anything else — the score is a floor, not this config's ceiling."
        )
    else:
        print(
            f"\n  early stopping fired on every fold; the budget is no longer the "
            f"constraint. Use --n-estimators {int(result.median_best_iteration * 1.4)} "
            f"for subsequent lr={TARGET_LR} runs."
        )
    print(f"  mlflow run {run_id}")


# The parameters this pass searches. Every one is present in DEFAULT_PARAMS, so the
# incumbent can be enqueued as trial 0 by looking its values up here.
SEARCH_KEYS = ("num_leaves", "min_child_samples", "feature_fraction", "reg_lambda")


def _suggest(trial: optuna.Trial) -> Dict[str, Any]:
    """
    The screening space: four parameters that interact, searched jointly.

    Deliberately narrow. A ~1-hour budget buys 12-18 trials and TPE needs 6-8 random
    startup trials before it models anything, so a wide space would leave every
    dimension unresolved. `subsample`, `reg_alpha` and `cat_smooth` are held fixed and
    only enter a second search if this pass shows headroom.
    """
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
    }


def mode_search(args: argparse.Namespace) -> None:
    """Joint TPE search over the regularisation parameters, with fold-level pruning."""
    X, y, feature_cols, cat_idx = prepare()
    deadline = time.time() + args.minutes * 60

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STUDY_STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        # Prune only after two folds: a single fold's AUC is noise at this fold spread
        # (+/-0.004), so pruning on fold 1 would discard good configurations.
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        load_if_exists=True,
    )

    # A seeded sampler replays its random sequence from the beginning every time it is
    # constructed, so resuming a study re-evaluates the trials it already holds — this
    # was observed: a resumed run reproduced trial 1's parameters exactly as trial 4.
    # Offsetting the seed by the number of existing trials keeps a given resume point
    # reproducible while advancing into unexplored territory. Only matters below
    # `n_startup_trials`, where sampling is random; TPE conditions on history instead.
    if study.trials:
        study.sampler = optuna.samplers.TPESampler(seed=SEED + len(study.trials))
        print(
            f"resuming with {len(study.trials)} existing trials "
            f"(sampler re-seeded to {SEED + len(study.trials)})",
            flush=True,
        )

    # Score the incumbent inside the same study, on the same folds, at the same learning
    # rate. Without it a trial's AUC at lr=0.05 cannot be interpreted at all: it is not
    # comparable to the lr=0.02 reference, only to other trials in this study.
    if not study.trials:
        study.enqueue_trial({k: DEFAULT_PARAMS[k] for k in SEARCH_KEYS})

    def objective(trial: optuna.Trial) -> float:
        params = {
            **DEFAULT_PARAMS,
            **_suggest(trial),
            "learning_rate": SEARCH_LR,
            "n_estimators": args.n_estimators,
        }
        t0 = time.time()

        def on_fold(fold: int, auc: float) -> None:
            trial.report(auc, step=fold)
            if trial.should_prune():
                trial.set_user_attr("stopped", "median-pruner")
                raise optuna.TrialPruned()
            # Cost guard. MedianPruner only prunes on *quality*, so a config that scores
            # well but converges slowly runs to the cap and can swallow the whole search
            # budget: in the first screening pass one trial took 79 min against ~5 for
            # its neighbours, because heavy regularisation (reg_lambda=71,
            # feature_fraction=0.31) meant early stopping never fired at 2500 trees.
            # Trial cost is *not* constant across the search space.
            elapsed = time.time() - t0
            if elapsed > args.trial_minutes * 60:
                trial.set_user_attr("stopped", "time-budget")
                print(
                    f"  trial {trial.number} abandoned after {elapsed / 60:.1f} min "
                    f"(> --trial-minutes {args.trial_minutes})",
                    flush=True,
                )
                raise optuna.TrialPruned()

        with start_run(
            f"trial-{trial.number}",
            params={**params, "cv_seed": SEED, "n_splits": N_SPLITS},
            tags={"model": "lightgbm", "round": "optuna-screening"},
        ):
            try:
                result = cross_validate(
                    X,
                    y,
                    cat_idx,
                    params,
                    early_stopping_rounds=args.patience,
                    on_fold=on_fold,
                    max_seconds=args.trial_minutes * 60,
                )
            except TimeBudgetExceeded:
                # Bounded per boosting iteration, so this fires even when a *single*
                # fold overruns -- the case the between-folds check in `on_fold` misses.
                trial.set_user_attr("stopped", "time-budget")
                print(
                    f"  trial {trial.number} abandoned after "
                    f"{(time.time() - t0) / 60:.1f} min "
                    f"(> --trial-minutes {args.trial_minutes})",
                    flush=True,
                )
                raise optuna.TrialPruned() from None
            log_cv_metrics(
                result.fold_scores,
                {
                    "oof_roc_auc": result.oof_roc_auc,
                    "oof_pr_auc": result.oof_pr_auc,
                    "median_best_iteration": float(result.median_best_iteration),
                    "runtime_seconds": time.time() - t0,
                },
            )
        print(
            f"trial {trial.number}: OOF {result.oof_roc_auc:.5f} "
            f"({time.time() - t0:.0f}s) {trial.params}",
            flush=True,
        )
        return result.oof_roc_auc

    def stop_at_deadline(st: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        if time.time() > deadline:
            print("time budget reached — stopping the study", flush=True)
            st.stop()

    study.optimize(objective, callbacks=[stop_at_deadline], catch=())

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    on_time = sum(1 for t in pruned if t.user_attrs.get("stopped") == "time-budget")
    print(
        f"\ncompleted {len(done)} trials, pruned {len(pruned)} "
        f"({on_time} on the time budget, {len(pruned) - on_time} on quality)"
    )
    if len(done) < 10:
        print(
            "  NOTE: TPESampler samples randomly for its first 10 trials, so below "
            "that the search is random, not TPE-guided. Re-run `search` to extend the "
            "study — it resumes from the persisted history."
        )
    print(f"best OOF {study.best_value:.5f} with {study.best_params}")
    print("\ntop 5:")
    for t in sorted(done, key=lambda t: -(t.value or 0))[:5]:
        print(f"  {t.value:.5f}  {t.params}")
    print(
        f"\nnext: uv run python scripts/tune_gbt.py transfer  "
        f"(re-fits the best config at lr={TARGET_LR})"
    )


def _best_params() -> Dict[str, Any]:
    study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_STORAGE)
    return dict(study.best_params)


def mode_transfer(args: argparse.Namespace) -> None:
    """Re-fit the search winner at the target learning rate on the canonical folds."""
    X, y, feature_cols, cat_idx = prepare()
    best = _best_params()
    params = {
        **DEFAULT_PARAMS,
        **best,
        "learning_rate": TARGET_LR,
        "n_estimators": args.n_estimators,
    }
    print(f"transferring {best} to lr={TARGET_LR}, cap {args.n_estimators}", flush=True)

    t0 = time.time()
    with start_run(
        "gbt-tuned-transfer",
        params={**params, "cv_seed": SEED, "n_splits": N_SPLITS},
        tags={"model": "lightgbm", "round": "tuned-transfer"},
    ):
        result = cross_validate(
            X, y, cat_idx, params, early_stopping_rounds=args.patience
        )
        log_cv_metrics(
            result.fold_scores,
            {
                "oof_roc_auc": result.oof_roc_auc,
                "oof_pr_auc": result.oof_pr_auc,
                "median_best_iteration": float(result.median_best_iteration),
                "runtime_seconds": time.time() - t0,
            },
        )
    _report(result, f"transfer to lr={TARGET_LR}")
    if result.budget_was_binding(args.n_estimators, args.patience):
        print("  *** cap binding again — raise --n-estimators")
    print(
        f"\nnext: uv run python scripts/tune_gbt.py final "
        f"--n-estimators {result.median_best_iteration}"
    )


def mode_final(args: argparse.Namespace) -> None:
    """
    The quotable number: fixed tree count, **no early stopping**.

    With early stopping on, predictions come from ``best_iteration_`` — an iteration
    chosen by scoring the very fold being predicted, which flatters the OOF figure (and
    flatters it *more* at a larger cap, since there are more iterations to argmax over).
    Fixing the count removes that selection entirely, so this is the run to quote, and
    the one whose shortlist the NN round consumes.
    """
    X, y, feature_cols, cat_idx = prepare()
    best = _best_params() if args.use_study else {}
    params = {
        **DEFAULT_PARAMS,
        **best,
        "learning_rate": TARGET_LR,
        "n_estimators": args.n_estimators,
    }

    t0 = time.time()
    with start_run(
        "gbt-final",
        params={
            **params,
            "cv_seed": SEED,
            "n_splits": N_SPLITS,
            "early_stopping": False,
        },
        tags={"model": "lightgbm", "round": "final"},
    ) as run:
        result = cross_validate(X, y, cat_idx, params, early_stopping_rounds=None)
        log_cv_metrics(
            result.fold_scores,
            {
                "oof_roc_auc": result.oof_roc_auc,
                "oof_pr_auc": result.oof_pr_auc,
                "runtime_seconds": time.time() - t0,
            },
        )
        imp = log_feature_importance(feature_cols, result.importances.tolist())
        shortlist = imp.head(200)["feature"].to_list()
        log_text_artifact("\n".join(shortlist), "gbt_shortlist.txt")
        run_id = run.info.run_id

    imp.write_csv(OUT_DIR / "gbt_gain_importance.csv")
    (OUT_DIR / "gbt_shortlist.txt").write_text("\n".join(shortlist))

    _report(result, f"FINAL (no early stopping, {args.n_estimators} trees)")
    print(f"  cumulative gain of top 200: {imp.head(200)['gain_norm'].sum():.4f}")
    print(f"  wrote {OUT_DIR / 'gbt_gain_importance.csv'} and gbt_shortlist.txt")
    print(f"  mlflow run {run_id}")


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Dict[str, float]:
    """
    Calibration-in-the-large, Brier score, and equal-count ECE.

    Ranking metrics cannot see miscalibration at all — every number here is invariant to
    nothing, which is exactly why they are the right instruments for this comparison.
    ``mean_pred`` against the base rate is the headline: a reweighted model's
    predictions are pushed off the true posterior by a known monotone factor, and this
    measures how far.
    """
    order = np.argsort(p)
    edges = np.array_split(order, bins)
    ece = sum(
        len(idx) * abs(float(p[idx].mean()) - float(y[idx].mean()))
        for idx in edges
        if len(idx)
    ) / len(y)
    return {
        "mean_pred": float(p.mean()),
        "base_rate": float(y.mean()),
        "calibration_ratio": float(p.mean() / y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(ece),
    }


def mode_reweight(args: argparse.Namespace) -> None:
    """
    Measure what ``scale_pos_weight`` actually does to this model.

    The whitepaper (§3.4) ruled reweighting out by reasoning: at the population optimum
    a class weight is a monotone transform of the posterior, so it cannot move a ranking
    metric — but a *fitted* GBT is not at that optimum, so ROC-AUC can drift either way.
    This runs the exact final configuration twice, toggling only the weight, on the
    canonical folds.

    Both halves are fitted here rather than reusing the stored ``gbt-final`` scores,
    because the interesting quantity is calibration and OOF prediction vectors were
    never persisted. The model is deterministic at a fixed seed, so the paired per-fold
    comparison needs no noise-floor caveat.
    """
    X, y, feature_cols, cat_idx = prepare()
    best = _best_params() if args.use_study else {}
    base = {
        **DEFAULT_PARAMS,
        **best,
        "learning_rate": TARGET_LR,
        "n_estimators": args.n_estimators,
    }
    ratio = float((y == 0).sum()) / float((y == 1).sum())
    weight = args.weight if args.weight is not None else ratio
    print(
        f"base rate {y.mean():.4%}  neg/pos ratio {ratio:.2f}  "
        f"testing scale_pos_weight={weight:.2f}",
        flush=True,
    )

    outcomes: Dict[str, CVResult] = {}
    for label, w in [("unweighted", None), ("reweighted", weight)]:
        params = dict(base)
        if w is not None:
            params["scale_pos_weight"] = w
        t0 = time.time()
        print(f"\n=== {label} ===", flush=True)
        with start_run(
            f"gbt-reweight-{label}",
            params={
                **params,
                "cv_seed": SEED,
                "n_splits": N_SPLITS,
                "early_stopping": False,
            },
            tags={"model": "lightgbm", "round": "reweight"},
        ):
            result = cross_validate(X, y, cat_idx, params, early_stopping_rounds=None)
            cal = _calibration(y, result.oof_pred)
            log_cv_metrics(
                result.fold_scores,
                {
                    "oof_roc_auc": result.oof_roc_auc,
                    "oof_pr_auc": result.oof_pr_auc,
                    "runtime_seconds": time.time() - t0,
                    **cal,
                },
            )
        outcomes[label] = result
        _report(result, f"{label} ({args.n_estimators} trees, no early stopping)")
        print(f"  mean_pred {cal['mean_pred']:.5f} vs base {cal['base_rate']:.5f}")
        print(f"  Brier {cal['brier']:.6f}   ECE {cal['ece']:.6f}")

    a, b = outcomes["unweighted"], outcomes["reweighted"]
    deltas = [t - u for u, t in zip(a.fold_scores, b.fold_scores)]
    wins = sum(d > 0 for d in deltas)
    print("\n" + "=" * 62)
    print("PAIRED PER-FOLD COMPARISON (reweighted - unweighted)")
    for i, d in enumerate(deltas, 1):
        print(
            f"  fold {i}: {a.fold_scores[i - 1]:.5f} -> {b.fold_scores[i - 1]:.5f}"
            f"   {d:+.5f}"
        )
    print(f"  reweighting wins {wins}/{len(deltas)} folds")
    print(
        f"  OOF ROC-AUC {a.oof_roc_auc:.5f} -> {b.oof_roc_auc:.5f} "
        f"({b.oof_roc_auc - a.oof_roc_auc:+.5f})"
    )
    print(
        f"  OOF PR-AUC  {a.oof_pr_auc:.5f} -> {b.oof_pr_auc:.5f} "
        f"({b.oof_pr_auc - a.oof_pr_auc:+.5f})"
    )

    ca, cb = _calibration(y, a.oof_pred), _calibration(y, b.oof_pred)
    print("\nCALIBRATION (what reweighting actually changes)")
    print(f"  base rate            {ca['base_rate']:.5f}")
    print(f"  mean predicted PD    {ca['mean_pred']:.5f} -> {cb['mean_pred']:.5f}")
    print(
        f"  ratio to base rate   {ca['calibration_ratio']:.3f}x -> "
        f"{cb['calibration_ratio']:.3f}x"
    )
    print(f"  Brier                {ca['brier']:.6f} -> {cb['brier']:.6f}")
    print(f"  ECE (10 equal bins)  {ca['ece']:.6f} -> {cb['ece']:.6f}")

    # Rank correlation: how much of the reordering is real vs. a pure rescale?
    from scipy.stats import spearmanr

    rho = float(spearmanr(a.oof_pred, b.oof_pred).statistic)
    print(f"\n  Spearman rho between the two score vectors: {rho:.6f}")
    print("  (1.0 would mean a pure monotone rescale -- identical ranking, and")
    print("   therefore identical ROC-AUC by construction.)")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("budget", help="find the convergence point + new reference")
    p.add_argument("--n-estimators", type=int, default=6000)
    p.add_argument("--patience", type=int, default=300)
    p.set_defaults(func=mode_budget)

    p = sub.add_parser("search", help="Optuna screening search")
    p.add_argument("--minutes", type=float, default=55.0, help="total study budget")
    # 1500 rather than 2500: at lr=0.05 the reference converges by ~570 trees, so this
    # keeps ~2.6x headroom while bounding what a non-converging config can cost.
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument(
        "--trial-minutes",
        type=float,
        default=10.0,
        help="abandon a trial exceeding this (checked between folds, so a single very "
        "slow fold can still overshoot)",
    )
    p.set_defaults(func=mode_search)

    p = sub.add_parser("transfer", help="best config at the target learning rate")
    p.add_argument("--n-estimators", type=int, default=6000)
    p.add_argument("--patience", type=int, default=300)
    p.set_defaults(func=mode_transfer)

    p = sub.add_parser("final", help="fixed trees, no early stopping (quotable run)")
    p.add_argument("--n-estimators", type=int, required=True)
    p.add_argument(
        "--use-study",
        action="store_true",
        default=True,
        help="apply the Optuna best params (default); --no-use-study for the baseline",
    )
    p.add_argument("--no-use-study", dest="use_study", action="store_false")
    p.set_defaults(func=mode_final)

    p = sub.add_parser("reweight", help="does scale_pos_weight help? (paired test)")
    p.add_argument("--n-estimators", type=int, required=True)
    p.add_argument(
        "--weight",
        type=float,
        default=None,
        help="scale_pos_weight; defaults to the neg/pos ratio (~11.4)",
    )
    p.add_argument("--use-study", action="store_true", default=True)
    p.add_argument("--no-use-study", dest="use_study", action="store_false")
    p.set_defaults(func=mode_reweight)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
