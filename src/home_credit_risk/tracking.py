"""
MLflow experiment tracking for the modeling rounds.

Every model round (GBT tuning, the NN round, the tabular-transformer round) logs to the
same experiment so runs are comparable in one place. The conventions this module
enforces:

- **One MLflow run per cross-validation run, not per fold.** Fold AUCs are logged as
  steps of the same metric, so a run's headline number is always the out-of-fold score
  — the only one that is comparable across models.
- **The fold split is a logged parameter.** ``cv_seed`` / ``n_splits`` are recorded on
  every run because an AUC delta smaller than the fold spread is meaningless unless the
  folds were identical.
- **The feature set is a logged parameter**, so a run that gained AUC by adding
  features is distinguishable from one that gained it by tuning.

Tracking URI and experiment name come from :mod:`home_credit_risk.config` (override
with ``MLFLOW_TRACKING_URI`` / ``MLFLOW_EXPERIMENT``); the default is a local SQLite
store, ``sqlite:///mlflow.db`` — MLflow 3 retired the plain-directory file store, so a
database URI is required. Metrics and params live in that database; **artifacts** go to
MLflow's default artifact root, ``./mlruns/<experiment_id>/<run_id>/artifacts/`` (the
artifact root is independent of the backend store and is *not* redirected by the SQLite
URI). Both paths are gitignored.

Browse the runs with::

    PYTHONPATH=tools/compat uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

The ``PYTHONPATH`` is not optional on Python 3.14: mlflow 3.14's UI server imports
``importlib.abc.Traversable``, removed in 3.14, and fails to boot without the shim in
``tools/compat/sitecustomize.py``. Nothing in this module needs it — logging and
:func:`best_runs` are client-side only.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

# MLflow 3 reports usage to mlflow-telemetry.io on import. Opt out before the import so
# nothing about these runs leaves the machine; set MLFLOW_DISABLE_TELEMETRY=false in the
# environment to opt back in.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from home_credit_risk.config import settings  # noqa: E402
from home_credit_risk.logger import logger  # noqa: E402


def configure(
    tracking_uri: Optional[str] = None,
    experiment: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> str:
    """
    Point MLflow at the configured store and experiment, returning the tracking URI.

    Safe to call repeatedly. Pass ``tracking_uri`` / ``experiment`` to override the
    settings.

    ``artifact_location`` takes effect **only when the experiment does not yet exist** —
    MLflow fixes an experiment's artifact root at creation and it cannot be changed
    afterwards. Set it to send artifacts somewhere other than the default
    ``./mlruns/<experiment_id>/`` — a bucket, a shared volume, or a temporary directory
    in tests, which is what keeps a test run from writing into the working tree.
    """
    uri = tracking_uri or settings.mlflow_tracking_uri
    name = experiment or settings.mlflow_experiment
    mlflow.set_tracking_uri(uri)
    if artifact_location is not None and mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)
    return uri


@contextmanager
def start_run(
    run_name: str,
    params: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, str]] = None,
    tracking_uri: Optional[str] = None,
    experiment: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> Iterator[mlflow.ActiveRun]:
    """
    Open an MLflow run with ``params`` and ``tags`` already logged.

    Used as a context manager around a whole cross-validation loop::

        with start_run("lgbm-baseline", params={"learning_rate": 0.02}) as run:
            ...
            log_cv_metrics(fold_aucs, {"oof_roc_auc": auc})
    """
    configure(tracking_uri, experiment, artifact_location)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(dict(params))
        if tags:
            mlflow.set_tags(dict(tags))
        logger.info("mlflow run %s (%s) started", run_name, run.info.run_id)
        yield run


@contextmanager
def resume_run(run_id: str) -> Iterator[mlflow.ActiveRun]:
    """
    Reopen a finished run to attach more metrics or artifacts.

    Notebooks split one logical run across several cells; rather than holding a run
    open across cell boundaries (where an exception would leave it dangling), each cell
    reopens the run by id.
    """
    with mlflow.start_run(run_id=run_id) as run:
        yield run


def log_cv_metrics(
    fold_scores: Sequence[float],
    summary: Mapping[str, float],
    metric_name: str = "fold_roc_auc",
) -> None:
    """
    Log per-fold scores as steps of one metric, plus the summary scalars.

    ``summary`` carries the numbers that actually compare across models — the OOF
    ROC-AUC / PR-AUC — while the per-fold series records the spread, which is what
    tells you whether a delta between two runs is real.
    """
    for i, score in enumerate(fold_scores, 1):
        mlflow.log_metric(metric_name, float(score), step=i)
    if fold_scores:
        mlflow.log_metric(f"{metric_name}_mean", float(np.mean(fold_scores)))
        mlflow.log_metric(f"{metric_name}_std", float(np.std(fold_scores)))
    for key, value in summary.items():
        mlflow.log_metric(key, float(value))


def log_feature_importance(
    features: Sequence[str], importances: Sequence[float], top_n: int = 40
) -> pl.DataFrame:
    """
    Log the gain ranking as a table artifact, returning it as a frame.

    The full ranking is attached to the run (``feature_importance.json``) so a run's
    shortlist can be reconstructed later; the top ``top_n`` normalised gains are also
    logged as metrics for at-a-glance comparison in the UI.
    """
    imp = (
        pl.DataFrame(
            {"feature": list(features), "gain": [float(g) for g in importances]}
        )
        .with_columns((pl.col("gain") / pl.col("gain").sum()).alias("gain_norm"))
        .sort("gain", descending=True)
    )
    mlflow.log_table(imp.to_pandas(), artifact_file="feature_importance.json")
    mlflow.log_metric("features_with_gain", float((imp["gain"] > 0).sum()))
    for row in imp.head(top_n).iter_rows(named=True):
        mlflow.log_metric(f"gain_norm.{row['feature']}", float(row["gain_norm"]))
    return imp


def log_text_artifact(text: str, filename: str) -> None:
    """Attach a plain-text artifact (e.g. the selected feature shortlist) to the run."""
    mlflow.log_text(text, filename)


def log_file_artifact(path: Path) -> None:
    """Attach an existing file (e.g. a written CSV) to the run."""
    mlflow.log_artifact(str(path))


def run_history(
    experiment: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    tag: Optional[str] = None,
) -> pl.DataFrame:
    """
    Every run in the experiment, oldest first, with its params and headline metrics.

    :func:`best_runs` answers "what won"; this answers "how did we get there", which is
    the question a results notebook asks. Ordering is chronological rather than by score
    because the story of a tuning round is a sequence — a run that *lost* is evidence
    about the choice it tested.

    ``tag`` filters on the ``round`` tag (``"nn"``, ``"gbt"``) so one round's notebook
    does not have to reason about another's runs.
    """
    configure(tracking_uri, experiment)
    name = experiment or settings.mlflow_experiment
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return pl.DataFrame()
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.round = '{tag}'" if tag else "",
        order_by=["attributes.start_time ASC"],
        output_format="pandas",
    )
    if not isinstance(runs, pd.DataFrame) or runs.empty:
        return pl.DataFrame()

    keep: Dict[str, Any] = {
        "run_name": runs["tags.mlflow.runName"],
        "started": runs["start_time"],
    }
    for col in runs.columns:
        if col.startswith("metrics.") and not col.startswith("metrics.gain_norm."):
            keep[col.removeprefix("metrics.")] = runs[col]
        elif col.startswith("params."):
            keep[col.removeprefix("params.")] = runs[col].astype("object")
    return pl.DataFrame(keep, strict=False)


def fold_scores(
    run_names: Sequence[str],
    metric: str = "fold_roc_auc",
    experiment: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> pl.DataFrame:
    """
    Per-fold scores for the named runs, long-format (``run_name``, ``fold``, ``score``).

    Fold-level detail is what separates a real improvement from fold noise: two runs
    whose OOF scores differ by less than their fold spread have not been distinguished.
    Because every run here uses the identical split (``cv_seed=42``, ``n_splits=5``),
    fold *k* of one run is directly comparable to fold *k* of another — which makes a
    paired comparison valid where an unpaired one would not be.
    """
    configure(tracking_uri, experiment)
    name = experiment or settings.mlflow_experiment
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return pl.DataFrame()
    client = mlflow.tracking.MlflowClient()
    names: List[str] = []
    folds: List[int] = []
    scores: List[float] = []
    for run_name in run_names:
        found = client.search_runs(
            [exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'"
        )
        if not found:
            continue
        for point in client.get_metric_history(found[0].info.run_id, metric):
            names.append(run_name)
            folds.append(int(point.step))
            scores.append(float(point.value))
    return pl.DataFrame({"run_name": names, "fold": folds, "score": scores})


def best_runs(
    metric: str = "oof_roc_auc",
    top_n: int = 10,
    experiment: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> pl.DataFrame:
    """
    Return the experiment's best runs by ``metric``, newest tie broken arbitrarily.

    The leaderboard for the tuning phase: one row per run with its parameters, so a
    regression is traceable to the setting that caused it.
    """
    configure(tracking_uri, experiment)
    name = experiment or settings.mlflow_experiment
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return pl.DataFrame()
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n,
        output_format="pandas",
    )
    if not isinstance(runs, pd.DataFrame) or runs.empty:
        return pl.DataFrame()
    keep: Dict[str, Any] = {
        "run_name": runs["tags.mlflow.runName"],
        metric: runs[f"metrics.{metric}"],
    }
    for col in runs.columns:
        if col.startswith("params."):
            keep[col.removeprefix("params.")] = runs[col]
    return pl.DataFrame(keep)
