import numpy as np
import polars as pl
import pytest
import torch

from home_credit_risk.evaluation import TimeBudgetExceeded
from home_credit_risk.nn import (
    NNConfig,
    build_model,
    cross_validate_nn,
    embedding_dim,
    encode_categoricals,
    fit_category_vocab,
    fit_preprocessor,
    one_hot,
)

# Tiny and CPU-pinned: these test the harness, not model quality. MPS reductions are not
# bit-exact, so determinism is only assertable on CPU.
TINY = NNConfig(
    hidden=(8,), max_epochs=3, patience=2, batch_size=64, device="cpu", dropout=0.0
)


def _frame(n: int = 300, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)
    gappy = signal.copy()
    gappy[rng.random(n) < 0.25] = np.nan
    fm = pl.DataFrame(
        {
            "SIGNAL": signal,
            "GAPPY_MEAN": gappy,
            "NAME_KIND": rng.choice(["a", "b", "c"], size=n).tolist(),
            "TARGET": y,
        }
    )
    return fm, y, ["SIGNAL", "GAPPY_MEAN", "NAME_KIND"]


def test_embedding_dim_grows_sublinearly_and_caps() -> None:
    assert embedding_dim(2) < embedding_dim(19) < embedding_dim(58)
    assert embedding_dim(58) <= 16
    # A huge column still cannot dominate the input width.
    assert embedding_dim(10_000, cap=16) == 16


def test_fit_category_vocab_reserves_index_zero() -> None:
    vocab = fit_category_vocab(["b", "a", "b", None])

    assert 0 not in vocab.values()  # index 0 belongs to unknown/null
    assert sorted(vocab) == ["a", "b"]
    assert sorted(vocab.values()) == [1, 2]


def test_unseen_and_null_categories_encode_as_the_oov_index() -> None:
    vocab = {"NAME_KIND": fit_category_vocab(["a", "b"])}
    frame = pl.DataFrame({"NAME_KIND": ["a", "zzz", None, "b"]})

    codes = encode_categoricals(frame, ["NAME_KIND"], vocab)

    assert codes[0, 0] == vocab["NAME_KIND"]["a"]
    assert codes[1, 0] == 0  # unseen at training time
    assert codes[2, 0] == 0  # null
    assert codes[3, 0] == vocab["NAME_KIND"]["b"]


def test_one_hot_width_includes_the_unknown_level() -> None:
    vocab = {"NAME_KIND": fit_category_vocab(["a", "b"])}
    codes = np.array([[0], [1], [2]])

    block = one_hot(codes, vocab, ["NAME_KIND"])

    assert block.shape == (3, 3)  # unknown + 2 levels
    assert block.sum(axis=1).tolist() == [1.0, 1.0, 1.0]


def test_indicators_add_one_column_per_gappy_feature() -> None:
    raw = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, 6.0]])

    with_ind = fit_preprocessor(raw, TINY).transform(raw)
    without = fit_preprocessor(raw, NNConfig(**{**TINY.__dict__, "indicators": False}))

    assert with_ind.shape[1] == 3  # 2 numeric + 1 indicator (only col 1 had a null)
    assert without.transform(raw).shape[1] == 2
    # The indicator marks the originally-missing row, and survives imputation.
    assert with_ind[:, 2].tolist() == [1.0, 0.0, 0.0]


def test_quantile_transform_tames_a_heavy_tail_where_standard_scaling_cannot() -> None:
    # The motivating case: one value ~1000x p99. Standard scaling leaves the bulk of the
    # distribution crushed near zero; rank-gauss spreads it across the usable range.
    rng = np.random.default_rng(0)
    raw = np.concatenate([rng.normal(size=(999, 1)), [[10_000.0]]])

    rank_gauss = fit_preprocessor(
        raw, NNConfig(**{**TINY.__dict__, "quantile": True})
    ).transform(raw)
    standard = fit_preprocessor(raw, TINY).transform(raw)  # the default since Stage B

    bulk_rg = float(np.abs(rank_gauss[:999]).mean())
    bulk_std = float(np.abs(standard[:999]).mean())
    assert bulk_rg > 5 * bulk_std


def test_every_row_gets_exactly_one_out_of_fold_prediction() -> None:
    fm, y, cols = _frame()
    result = cross_validate_nn(fm, cols, y, TINY, n_splits=3)

    assert len(result.fold_scores) == 3
    assert result.oof_pred.shape == (len(y),)
    assert np.all((result.oof_pred > 0.0) & (result.oof_pred < 1.0))
    assert 0.0 <= result.oof_roc_auc <= 1.0


def test_preprocessing_and_vocab_are_fitted_per_fold() -> None:
    # Leakage guard, matching test_linear.py: shifting half the rows must change the
    # result, which it cannot if the transforms were fitted once on the whole frame.
    fm, y, cols = _frame()
    shifted = fm.with_columns(
        pl.when(pl.arange(0, fm.height) < fm.height // 2)
        .then(pl.col("GAPPY_MEAN") + 50.0)
        .otherwise(pl.col("GAPPY_MEAN"))
        .alias("GAPPY_MEAN")
    )

    a = cross_validate_nn(fm, cols, y, TINY, n_splits=3)
    b = cross_validate_nn(shifted, cols, y, TINY, n_splits=3)

    assert a.oof_roc_auc != b.oof_roc_auc


def test_one_hot_mode_runs_without_embedding_tables() -> None:
    fm, y, cols = _frame()
    config = NNConfig(**{**TINY.__dict__, "embeddings": False})

    result = cross_validate_nn(fm, cols, y, config, n_splits=3)

    assert len(result.fold_scores) == 3
    assert np.all(result.oof_pred > 0.0)


def test_resnet_architecture_runs() -> None:
    fm, y, cols = _frame()
    config = NNConfig(**{**TINY.__dict__, "arch": "resnet"})

    result = cross_validate_nn(fm, cols, y, config, n_splits=2)

    assert len(result.fold_scores) == 2


def test_unknown_architecture_is_rejected() -> None:
    fm, y, cols = _frame()
    config = NNConfig(**{**TINY.__dict__, "arch": "nope"})

    with pytest.raises(ValueError, match="unknown arch"):
        cross_validate_nn(fm, cols, y, config, n_splits=2)


def test_same_seed_is_deterministic_on_cpu() -> None:
    fm, y, cols = _frame()
    a = cross_validate_nn(fm, cols, y, TINY, n_splits=2)
    b = cross_validate_nn(fm, cols, y, TINY, n_splits=2)

    assert a.fold_scores == b.fold_scores
    np.testing.assert_array_equal(a.oof_pred, b.oof_pred)


def test_on_fold_callback_fires_per_fold() -> None:
    fm, y, cols = _frame()
    seen: list[int] = []

    cross_validate_nn(
        fm, cols, y, TINY, n_splits=3, on_fold=lambda i, auc: seen.append(i)
    )

    assert seen == [1, 2, 3]


def test_max_seconds_aborts_training() -> None:
    fm, y, cols = _frame()

    with pytest.raises(TimeBudgetExceeded):
        cross_validate_nn(fm, cols, y, TINY, n_splits=2, max_seconds=0.0)


def test_best_iterations_records_the_early_stopping_epoch() -> None:
    fm, y, cols = _frame()
    result = cross_validate_nn(fm, cols, y, TINY, n_splits=2)

    # Epochs, not tree counts -- bounded by max_epochs.
    assert all(1 <= e <= TINY.max_epochs for e in result.best_iterations)


def test_tabm_members_are_not_identical_at_init() -> None:
    # The whole premise of BatchEnsemble is that random sign init makes members
    # functionally distinct. Initialise the adapters to ones and every member computes
    # the same function, so the "ensemble" is one model wearing 32 hats.
    from home_credit_risk.nn import BatchEnsembleLinear

    layer = BatchEnsembleLinear(6, 4, k=8)
    x = torch.randn(3, 8, 6)

    out = layer(x)

    assert out.shape == (3, 8, 4)
    spread = out.std(dim=1).mean().item()
    assert spread > 1e-3, "members collapsed to an identical function"


def test_tabm_emits_one_logit_per_member() -> None:
    fm, y, cols = _frame()
    config = NNConfig(**{**TINY.__dict__, "arch": "tabm", "k": 4})
    model = build_model(config, n_numeric=2, cardinalities=[3])

    out = model(torch.randn(5, 2), torch.zeros(5, 1, dtype=torch.long))

    assert out.shape == (5, 4)


def test_tabm_trains_and_scores_every_row() -> None:
    fm, y, cols = _frame()
    config = NNConfig(**{**TINY.__dict__, "arch": "tabm", "k": 4})

    result = cross_validate_nn(fm, cols, y, config, n_splits=3)

    assert len(result.fold_scores) == 3
    assert np.all((result.oof_pred > 0.0) & (result.oof_pred < 1.0))


def test_periodic_embedding_widens_each_numeric_feature() -> None:
    from home_credit_risk.nn import PeriodicEmbeddings

    embed = PeriodicEmbeddings(n_features=3, d_embedding=8)

    out = embed(torch.randn(5, 3))

    assert out.shape == (5, 24)  # 3 features x 8 dims
    assert embed.out_features == 24


def test_periodic_embedding_rejects_an_odd_width() -> None:
    from home_credit_risk.nn import PeriodicEmbeddings

    with pytest.raises(ValueError, match="even"):
        PeriodicEmbeddings(n_features=2, d_embedding=7)


def test_tabm_with_periodic_embeddings_runs() -> None:
    fm, y, cols = _frame()
    config = NNConfig(
        **{**TINY.__dict__, "arch": "tabm", "k": 2, "periodic": True, "d_embedding": 4}
    )

    result = cross_validate_nn(fm, cols, y, config, n_splits=2)

    assert len(result.fold_scores) == 2
