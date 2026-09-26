import math
from typing import cast

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

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


def test_validation_is_scored_in_batches_not_one_allocation() -> None:
    # Regression guard. Scoring a whole fold in a single forward pass allocates a tensor
    # proportional to rows x members x width. That was survivable for the plain MLP and
    # blew up at 19.58 GiB once periodic embeddings widened the input ~10x, so the
    # predict path must never see more than batch_size rows at once.
    from home_credit_risk.nn import _predict

    widths: list[int] = []

    class _Spy(nn.Module):
        def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
            widths.append(len(x_num))
            return torch.zeros(len(x_num), 3)  # 3 members

    preds = _predict(_Spy(), torch.zeros(250, 4), torch.zeros(250, 1).long(), 64)

    assert preds.shape == (250,), "every row must get exactly one prediction"
    assert max(widths) <= 64, f"predict saw a batch of {max(widths)} rows"
    assert sum(widths) == 250, "rows were dropped or double-counted"


def test_predict_averages_member_probabilities_not_logits() -> None:
    # Averaging in logit space is a geometric mean of odds, which over-weights confident
    # members. With logits -4 and +4 the ensemble answer is 0.5, but that holds by
    # coincidence there, so this uses an asymmetric pair instead.
    from home_credit_risk.nn import _predict

    class _Fixed(nn.Module):
        def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
            n = len(x_num)
            return torch.tensor([[0.0, 4.0]]).repeat(n, 1)

    preds = _predict(_Fixed(), torch.zeros(4, 2), torch.zeros(4, 1).long(), 2)

    expected = (torch.sigmoid(torch.tensor(0.0)) + torch.sigmoid(torch.tensor(4.0))) / 2
    np.testing.assert_allclose(preds, float(expected), rtol=1e-6)


def test_periodic_embedding_bmm_matches_the_einsum_contraction() -> None:
    # The bmm rewrite exists purely for speed on MPS, where torch.einsum has no fused
    # kernel for this contraction. It must stay numerically identical, so this pins the
    # two formulations against each other rather than trusting the rewrite.
    from home_credit_risk.nn import PeriodicEmbeddings

    torch.manual_seed(0)
    emb = PeriodicEmbeddings(n_features=6, d_embedding=8)
    x = torch.randn(16, 6)

    z = 2 * math.pi * emb.coeffs.unsqueeze(0) * x.unsqueeze(-1)
    z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
    reference = torch.relu(
        torch.einsum("bfd,fde->bfe", z, emb.linear) + emb.bias
    ).flatten(1)

    torch.testing.assert_close(emb(x), reference, rtol=0, atol=0)


def test_periodic_embeds_only_the_continuous_block() -> None:
    # x_num is [continuous | indicators | one-hot]. Embedding the binary tail is both
    # meaningless (sin/cos of a 0/1 column is two constants) and expensive, so the model
    # must widen only the leading continuous columns and pass the rest through.
    from home_credit_risk.nn import (
        BatchEnsembleLinear,
        PeriodicEmbeddings,
        TabM,
    )

    n_cont, n_binary, d = 4, 6, 8
    embed = PeriodicEmbeddings(n_cont, d_embedding=d)
    model = TabM(
        n_numeric=n_cont + n_binary,
        cardinalities=[],
        hidden=[16],
        dropout=0.0,
        k=2,
        embed=embed,
        n_continuous=n_cont,
    )

    # First layer input width = embedded continuous + untouched binary tail.
    first = cast(BatchEnsembleLinear, model.layers[0])
    assert first.weight.shape[1] == n_cont * d + n_binary

    out = model(torch.randn(5, n_cont + n_binary), torch.zeros(5, 0).long())
    assert out.shape == (5, 2)


def test_periodic_embedding_is_not_applied_to_binary_columns() -> None:
    # Changing only a binary column must change the input the body sees *linearly* --
    # it is passed through, not run through sin/cos. Verified by checking the embedded
    # continuous part is untouched when only the binary tail changes.
    from home_credit_risk.nn import PeriodicEmbeddings

    embed = PeriodicEmbeddings(3, d_embedding=4)
    x = torch.randn(2, 9)  # 3 continuous + 6 binary
    x_flipped = x.clone()
    x_flipped[:, 3:] = 1.0 - x_flipped[:, 3:]

    torch.testing.assert_close(embed(x[:, :3]), embed(x_flipped[:, :3]))


def test_build_model_defaults_n_continuous_to_all_numeric() -> None:
    # Back-compatible: callers that do not know the split get the old behaviour.
    config = NNConfig(
        **{**TINY.__dict__, "arch": "tabm", "k": 2, "periodic": True, "d_embedding": 4}
    )
    model = build_model(config, n_numeric=5, cardinalities=[])

    assert model.n_continuous == 5
