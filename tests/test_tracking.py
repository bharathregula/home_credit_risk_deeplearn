from pathlib import Path
from typing import TypedDict

import mlflow
import numpy as np
import polars as pl
import pytest

from home_credit_risk.tracking import (
    best_runs,
    fold_scores,
    load_oof,
    log_cv_metrics,
    log_feature_importance,
    log_oof_predictions,
    log_text_artifact,
    resume_run,
    run_history,
    start_run,
)


def _uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


class _Store(TypedDict):
    """Keyword arguments that point ``start_run`` entirely inside a temp directory."""

    tracking_uri: str
    experiment: str
    artifact_location: str


def _store(tmp_path: Path) -> _Store:
    """
    Fully isolate a test's MLflow state inside ``tmp_path``.

    The backend URI alone is not enough: an experiment's artifact root is independent of
    the backend store, so without ``artifact_location`` every ``log_text_artifact`` /
    ``log_feature_importance`` call writes into ``./mlruns`` in the working tree.

    Typed as a ``TypedDict`` rather than ``dict[str, str]`` so that ``**_store(...)``
    type-checks against ``start_run``'s heterogeneous keyword signature (``params`` is a
    ``Mapping[str, Any]``, ``tags`` a ``Mapping[str, str]``). A plain ``dict`` splat is
    an ``arg-type`` error under ``mypy .``, which is what CI runs.
    """
    return {
        "tracking_uri": _uri(tmp_path),
        "experiment": "unit-test",
        "artifact_location": str(tmp_path / "artifacts"),
    }


def test_start_run_logs_params_and_tags(tmp_path: Path) -> None:
    with start_run(
        "unit-run",
        params={"learning_rate": 0.02, "n_splits": 5},
        tags={"model": "lgbm"},
        **_store(tmp_path),
    ) as run:
        run_id = run.info.run_id

    logged = mlflow.get_run(run_id)
    assert logged.data.params["learning_rate"] == "0.02"
    assert logged.data.params["n_splits"] == "5"
    assert logged.data.tags["model"] == "lgbm"


def test_log_cv_metrics_records_folds_and_summary(tmp_path: Path) -> None:
    with start_run("cv-run", **_store(tmp_path)) as run:
        log_cv_metrics([0.78, 0.80], {"oof_roc_auc": 0.79, "oof_pr_auc": 0.28})
        run_id = run.info.run_id

    data = mlflow.get_run(run_id).data
    assert data.metrics["oof_roc_auc"] == 0.79
    assert data.metrics["oof_pr_auc"] == 0.28
    assert data.metrics["fold_roc_auc_mean"] == 0.79
    # Per-fold scores are steps of one metric, so the run keeps the whole series.
    history = mlflow.MlflowClient().get_metric_history(run_id, "fold_roc_auc")
    assert [m.value for m in history] == [0.78, 0.80]
    assert [m.step for m in history] == [1, 2]


def test_log_feature_importance_ranks_and_normalises(tmp_path: Path) -> None:
    with start_run("imp-run", **_store(tmp_path)) as run:
        imp = log_feature_importance(["a", "b", "c"], [1.0, 3.0, 0.0])
        run_id = run.info.run_id

    assert imp["feature"].to_list() == ["b", "a", "c"]  # sorted by gain
    assert imp["gain_norm"].to_list() == [0.75, 0.25, 0.0]
    assert mlflow.get_run(run_id).data.metrics["features_with_gain"] == 2


def test_log_text_artifact_and_best_runs(tmp_path: Path) -> None:
    for name, auc in (("worse", 0.70), ("better", 0.80)):
        with start_run(name, params={"tag": name}, **_store(tmp_path)):
            log_cv_metrics([auc], {"oof_roc_auc": auc})
            log_text_artifact("feat_a\nfeat_b", "shortlist.txt")

    board = best_runs(experiment="unit-test", tracking_uri=_uri(tmp_path))

    assert isinstance(board, pl.DataFrame)
    assert board["run_name"].to_list()[0] == "better"  # ordered by the metric
    assert board["oof_roc_auc"].to_list() == [0.80, 0.70]
    assert "tag" in board.columns  # params come along for traceability


def test_resume_run_attaches_to_the_same_run(tmp_path: Path) -> None:
    # A notebook logs the CV loop in one cell and the feature ranking in another; both
    # must land on the same run rather than creating a second one.
    with start_run("split-cells", **_store(tmp_path)) as run:
        run_id = run.info.run_id
        log_cv_metrics([0.79], {"oof_roc_auc": 0.79})

    with resume_run(run_id):
        log_feature_importance(["a"], [1.0])

    data = mlflow.get_run(run_id).data
    assert data.metrics["oof_roc_auc"] == 0.79
    assert data.metrics["features_with_gain"] == 1
    board = best_runs(experiment="unit-test", tracking_uri=_uri(tmp_path))
    assert board.height == 1  # one run, not two


def test_artifacts_stay_inside_the_configured_location(tmp_path: Path) -> None:
    # Regression guard: the backend URI alone does not redirect artifacts, so these
    # tests used to write run dirs into ./mlruns. Nothing may escape tmp_path.
    repo_mlruns = Path.cwd() / "mlruns"
    before = set(repo_mlruns.rglob("*")) if repo_mlruns.exists() else set()

    with start_run("artifact-run", **_store(tmp_path)) as run:
        log_text_artifact("feat_a\nfeat_b", "shortlist.txt")
        log_feature_importance(["a", "b"], [1.0, 2.0])
        run_id = run.info.run_id

    written = list((tmp_path / "artifacts").rglob("*"))
    names = {p.name for p in written}
    assert "shortlist.txt" in names
    assert "feature_importance.json" in names
    assert run_id in {p.name for p in (tmp_path / "artifacts").iterdir()}

    after = set(repo_mlruns.rglob("*")) if repo_mlruns.exists() else set()
    assert after == before, "the test wrote MLflow artifacts into the working tree"


def test_best_runs_on_unknown_experiment_is_empty(tmp_path: Path) -> None:
    assert best_runs(experiment="nope", tracking_uri=_uri(tmp_path)).is_empty()


def test_run_history_is_chronological_not_ranked(tmp_path: Path) -> None:
    # A tuning story is a sequence: the run that lost is evidence about the choice it
    # tested, so ordering must be by time even when a later run scores worse.
    for name, auc in [("first", 0.79), ("second", 0.75), ("third", 0.80)]:
        with start_run(name, params={"cv_seed": 42}, **_store(tmp_path)):
            log_cv_metrics([auc], {"oof_roc_auc": auc})

    hist = run_history(experiment="unit-test", tracking_uri=_uri(tmp_path))

    assert hist["run_name"].to_list() == ["first", "second", "third"]
    assert hist["oof_roc_auc"].to_list() == [0.79, 0.75, 0.80]
    assert "cv_seed" in hist.columns  # params come along for free


def test_run_history_filters_by_round_tag(tmp_path: Path) -> None:
    with start_run("tree", tags={"round": "gbt"}, **_store(tmp_path)):
        log_cv_metrics([0.79], {"oof_roc_auc": 0.79})
    with start_run("net", tags={"round": "nn"}, **_store(tmp_path)):
        log_cv_metrics([0.78], {"oof_roc_auc": 0.78})

    nn_only = run_history(experiment="unit-test", tracking_uri=_uri(tmp_path), tag="nn")

    assert nn_only["run_name"].to_list() == ["net"]


def test_fold_scores_returns_one_row_per_fold(tmp_path: Path) -> None:
    with start_run("five-fold", **_store(tmp_path)):
        log_cv_metrics([0.71, 0.72, 0.73, 0.74, 0.75], {"oof_roc_auc": 0.73})

    folds = fold_scores(
        ["five-fold"], experiment="unit-test", tracking_uri=_uri(tmp_path)
    )

    assert folds.height == 5
    assert folds["fold"].to_list() == [1, 2, 3, 4, 5]
    assert folds["score"].to_list() == [0.71, 0.72, 0.73, 0.74, 0.75]


def test_fold_scores_skips_unknown_runs_rather_than_raising(tmp_path: Path) -> None:
    # A notebook naming a run that was never logged should lose that series, not die
    # partway through rendering.
    with start_run("real", **_store(tmp_path)):
        log_cv_metrics([0.71, 0.72], {"oof_roc_auc": 0.715})

    folds = fold_scores(
        ["real", "never-ran"], experiment="unit-test", tracking_uri=_uri(tmp_path)
    )

    assert set(folds["run_name"].to_list()) == {"real"}


def test_oof_round_trips_through_a_stable_path(tmp_path: Path) -> None:
    oof = np.array([0.1, 0.9, 0.4, 0.6])
    y = np.array([0, 1, 0, 1])

    with start_run("with-oof", **_store(tmp_path)):
        path = log_oof_predictions(oof, y, out_dir=tmp_path / "oof")

    assert path is not None
    back_oof, back_y = load_oof(path)
    np.testing.assert_array_equal(back_oof, oof)
    np.testing.assert_array_equal(back_y, y)


def test_oof_rejects_a_label_length_mismatch(tmp_path: Path) -> None:
    # The vector is only interpretable against labels in the same row order; a length
    # mismatch is the one misalignment that is cheap to catch, so catch it loudly
    # rather than persisting a file nothing downstream can trust.
    with start_run("bad-oof", **_store(tmp_path)):
        with pytest.raises(ValueError, match="same shape"):
            log_oof_predictions(np.zeros(4), np.zeros(3), out_dir=tmp_path / "oof")


def test_oof_is_attached_to_the_run_even_without_a_stable_path(tmp_path: Path) -> None:
    with start_run("artifact-only", **_store(tmp_path)) as run:
        assert log_oof_predictions(np.zeros(3), np.zeros(3)) is None
        run_id = run.info.run_id

    files = [f.path for f in mlflow.MlflowClient().list_artifacts(run_id)]
    assert "oof.npz" in files
