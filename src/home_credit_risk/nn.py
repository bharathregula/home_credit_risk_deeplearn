"""
Neural-network models for the Home Credit default-risk problem.

The third model family, after :mod:`home_credit_risk.gbt` (trees) and
:mod:`home_credit_risk.linear` (logistic challenger). This is the only torch-importing
module, mirroring how ``gbt.py`` isolates LightGBM, and it evaluates on the **same
folds** so its score is directly comparable to the other two.

Preprocessing is where a tabular network is won or lost, and two choices here are
load-bearing:

- **Rank-gauss, not standard scaling.** 53 of the 159 numeric shortlist features have
  ``max/p99 > 10``, reaching **2810×**. Under ``StandardScaler`` one such outlier
  compresses that feature's entire useful range into a sliver around zero — invisible
  to a tree, which only reads split order, and destructive to a network, whose first
  layer sees raw magnitudes. ``QuantileTransformer`` with a normal output distribution
  maps every feature to the same well-conditioned shape.
- **Missingness indicators.** 182 of the 200 shortlisted features contain nulls and the
  nulls predict — customers with a null ``BUREAU_DAYS_ENDDATE_FACT_MEAN`` default at
  10.5% against an 8.07% base rate. Imputation alone discards exactly the signal the
  null-native tree reads for free.

Both are toggles on :class:`NNConfig` rather than hard-coded, because both are claims
this project has asserted and neither had been measured on a network until now.

Everything that can leak is fitted **inside the fold**: imputation medians, the quantile
mapping, and the categorical vocabulary. An unseen or null category maps to a reserved
index 0 rather than crashing or silently colliding with a real level.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch import nn

from home_credit_risk.evaluation import N_SPLITS, SEED, CVResult, TimeBudgetExceeded


@dataclass(frozen=True)
class NNConfig:
    """Architecture and preprocessing for one cross-validation run."""

    arch: str = "mlp"  # "mlp" | "resnet"
    hidden: Tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 30
    #: Defaults to the full cycle, i.e. early stopping never fires. `OneCycleLR` is
    #: designed so the best model appears *after* annealing; stopping during its
    #: high-LR phase cuts training off before that. Measured directly: at lr=1e-3 the
    #: best epoch was 9.6 (right at the LR peak) and the run scored 0.7773; at lr=3e-4
    #: with the cycle allowed to complete, the best epoch moved to 17.8 -- into the
    #: annealing phase -- and the run scored 0.7801. Predictions from the best epoch are
    #: kept regardless, so running the full cycle costs time, not accuracy.
    patience: int = 30
    #: Rank-gauss (True) or standard scaling (False) for numeric features.
    #: **Measured, and the opposite of the expected answer**: standard scaling scored
    #: 0.78195 against rank-gauss's 0.78011 on identical folds. Rank transformation
    #: discards magnitude, and for credit risk the *size* of an overdue amount is
    #: signal, not just tail noise; BatchNorm also re-normalises activations, so the
    #: network is far less sensitive to input conditioning than the 2810x tails suggest.
    quantile: bool = False
    #: Append a binary indicator per numeric feature that had nulls in the train fold.
    #: Worth +0.00117 -- real, in the predicted direction, but a third the size of the
    #: feature-set effect below.
    indicators: bool = True
    #: TabM ensemble members. 32 is the paper's default; cost grows sub-linearly
    #: because the weight matrix is shared, but activations are (batch, k, width).
    k: int = 32
    #: Periodic embeddings for numeric features (Gorishniy et al., 2022) instead of
    #: feeding scaled scalars straight in. Only consumed by ``arch="tabm"`` for now.
    periodic: bool = False
    d_embedding: int = 24
    sigma: float = 0.05
    #: Entity embeddings (True) or one-hot encoding (False) for categoricals.
    #: **Measured**: one-hot scored 0.78103 against embeddings' 0.78011. Embeddings pay
    #: off at thousands of levels; the widest column here has 58, where one-hot costs
    #: ~108 columns and a 13-dim embedding is a bottleneck that loses information rather
    #: than usefully regularising.
    embeddings: bool = False
    device: Optional[str] = None
    seed: int = SEED

    def as_params(self) -> Dict[str, Any]:
        """Flat dict for experiment tracking."""
        return {
            "arch": self.arch,
            "hidden": "-".join(str(h) for h in self.hidden),
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "quantile": self.quantile,
            "indicators": self.indicators,
            "embeddings": self.embeddings,
            "seed": self.seed,
        }


def resolve_device(preference: Optional[str] = None) -> str:
    """Pick a torch device, preferring Apple MPS when present."""
    if preference:
        return preference
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embedding_dim(cardinality: int, cap: int = 16) -> int:
    """
    Embedding width for a categorical of ``cardinality`` levels.

    The usual heuristic, ``1.6 * card**0.56`` capped at ``cap``: 58 levels
    (``ORGANIZATION_TYPE``) → 13 dims, 3 levels (``CODE_GENDER``) → 2. Growing
    sub-linearly keeps a high-cardinality column from dominating the input width.
    """
    return max(1, min(cap, int(math.ceil(1.6 * cardinality**0.56))))


def fit_category_vocab(values: Sequence[Any]) -> Dict[str, int]:
    """
    Map the distinct values of one categorical column to embedding indices.

    **Index 0 is reserved** for null and for values not present in the training fold,
    so a category unseen at scoring time degrades to a learned "unknown" vector instead
    of raising or colliding with a real level. Fit on training rows only.
    """
    seen = sorted({str(v) for v in values if v is not None})
    return {value: i + 1 for i, value in enumerate(seen)}


def encode_categoricals(
    frame: pl.DataFrame, columns: Sequence[str], vocabs: Mapping[str, Mapping[str, int]]
) -> np.ndarray:
    """Integer-code ``columns`` with ``vocabs``; unknown and null both become 0."""
    if not columns:
        return np.zeros((frame.height, 0), dtype=np.int64)
    out = np.zeros((frame.height, len(columns)), dtype=np.int64)
    for j, col in enumerate(columns):
        vocab = vocabs[col]
        out[:, j] = [
            0 if v is None else vocab.get(str(v), 0) for v in frame[col].to_list()
        ]
    return out


def one_hot(
    codes: np.ndarray, vocabs: Mapping[str, Mapping[str, int]], columns: Sequence[str]
) -> np.ndarray:
    """Expand integer codes to one-hot blocks (including the index-0 unknown level)."""
    if not columns:
        return np.zeros((codes.shape[0], 0), dtype=np.float32)
    blocks = []
    for j, col in enumerate(columns):
        width = len(vocabs[col]) + 1
        block = np.zeros((codes.shape[0], width), dtype=np.float32)
        block[np.arange(codes.shape[0]), codes[:, j]] = 1.0
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


@dataclass
class FoldPreprocessor:
    """Numeric transforms fitted on one training fold."""

    imputer: SimpleImputer
    scaler: Any
    indicator_cols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def transform(self, raw: np.ndarray) -> np.ndarray:
        # Scale *before* filling. QuantileTransformer ignores NaN when fitting and
        # passes it through when transforming, so ranks come from real values only and
        # the fill lands at 0, the transformed median. Imputing first would instead put
        # a large point mass at the median before ranking -- for a 70%-null feature that
        # mass consumes the middle of the distribution and forces the few real values
        # out into the tails, distorting every distance between them.
        nan_aware = isinstance(self.scaler, QuantileTransformer)
        source = raw if nan_aware else self.imputer.transform(raw)
        scaled = self.scaler.transform(source)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        if self.indicator_cols.size:
            # Indicators are appended *after* scaling: they are already 0/1 and passing
            # them through a quantile map would be meaningless.
            flags = np.isnan(raw[:, self.indicator_cols]).astype(np.float32)
            scaled = np.concatenate([scaled, flags], axis=1)
        return np.asarray(scaled, dtype=np.float32)


def fit_preprocessor(raw_train: np.ndarray, config: NNConfig) -> FoldPreprocessor:
    """Fit imputation, scaling and the indicator column set on training rows only."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(raw_train)
    scaler: Any = (
        QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, raw_train.shape[0]),
            subsample=200_000,
            random_state=config.seed,
        )
        if config.quantile
        else StandardScaler()
    )
    # StandardScaler is not NaN-aware, so it alone needs pre-imputed input to fit.
    scaler.fit(raw_train if config.quantile else imputer.transform(raw_train))
    indicator_cols = (
        np.flatnonzero(np.isnan(raw_train).any(axis=0))
        if config.indicators
        else np.zeros(0, dtype=int)
    )
    return FoldPreprocessor(imputer, scaler, indicator_cols)


class EmbeddingMLP(nn.Module):
    """Entity embeddings concatenated to scaled numerics, then a dense body."""

    def __init__(
        self,
        n_numeric: int,
        cardinalities: Sequence[int],
        hidden: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c + 1, embedding_dim(c)) for c in cardinalities]
        )
        width = n_numeric + sum(embedding_dim(c) for c in cardinalities)
        layers: List[nn.Module] = []
        for size in hidden:
            layers += [
                nn.Linear(width, size),
                nn.BatchNorm1d(size),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            width = size
        layers.append(nn.Linear(width, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num]
        for j, emb in enumerate(self.embeddings):
            parts.append(emb(x_cat[:, j]))
        out: torch.Tensor = self.body(torch.cat(parts, dim=1)).squeeze(1)
        return out


class _ResBlock(nn.Module):
    """Pre-norm residual block (Gorishniy et al., 2021)."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(width)
        self.up = nn.Linear(width, width * 2)
        self.down = nn.Linear(width * 2, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down(self.drop(nn.functional.gelu(self.up(self.norm(x)))))
        out: torch.Tensor = x + self.drop(z)
        return out


class ResNetMLP(nn.Module):
    """Residual MLP: a stem, N pre-norm blocks at constant width, then a head."""

    def __init__(
        self,
        n_numeric: int,
        cardinalities: Sequence[int],
        hidden: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c + 1, embedding_dim(c)) for c in cardinalities]
        )
        width = hidden[0] if hidden else 256
        in_dim = n_numeric + sum(embedding_dim(c) for c in cardinalities)
        self.stem = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(
            *[_ResBlock(width, dropout) for _ in range(max(1, len(hidden)))]
        )
        self.head = nn.Sequential(nn.BatchNorm1d(width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num]
        for j, emb in enumerate(self.embeddings):
            parts.append(emb(x_cat[:, j]))
        z = self.head(self.blocks(self.stem(torch.cat(parts, dim=1))))
        out: torch.Tensor = z.squeeze(1)
        return out


class PeriodicEmbeddings(nn.Module):
    """
    Per-feature periodic embedding of numeric columns (Gorishniy et al., 2022).

    Each scalar ``x`` becomes ``[sin(2*pi*c*x), cos(2*pi*c*x)]`` over learned
    frequencies
    ``c``, then a per-feature linear + ReLU. The point is resolution: a single weight
    can
    only give a linear response to a numeric feature, whereas a bank of frequencies lets
    the first layer represent thresholds and non-monotone responses — the things a tree
    gets for free by splitting the same column twice.

    ``sigma`` sets the frequency scale and is the one sensitive hyperparameter; too
    large
    and the embedding aliases nearby values into unrelated codes.
    """

    def __init__(
        self, n_features: int, d_embedding: int = 24, sigma: float = 0.05
    ) -> None:
        super().__init__()
        if d_embedding % 2:
            raise ValueError("d_embedding must be even (half sin, half cos)")
        self.coeffs = nn.Parameter(torch.randn(n_features, d_embedding // 2) * sigma)
        self.linear = nn.Parameter(torch.empty(n_features, d_embedding, d_embedding))
        self.bias = nn.Parameter(torch.zeros(n_features, d_embedding))
        bound = d_embedding**-0.5
        nn.init.uniform_(self.linear, -bound, bound)
        self.out_features = n_features * d_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = 2 * math.pi * self.coeffs.unsqueeze(0) * x.unsqueeze(-1)
        z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        # bmm, not einsum. Mathematically identical -- per feature, multiply the
        # (batch, d) block by that feature's (d, d) matrix -- but `torch.einsum` has no
        # fused MPS kernel for this contraction and falls back to something very
        # slow:
        # a 5-fold run made 18 seconds of progress in 22 minutes of wall clock with
        # the machine otherwise idle. Reshaping to an explicit batched matmul over
        # the feature axis keeps it on the fast path.
        z = torch.bmm(z.transpose(0, 1), self.linear).transpose(0, 1) + self.bias
        out: torch.Tensor = torch.relu(z).flatten(1)
        return out


class BatchEnsembleLinear(nn.Module):
    """
    One shared weight matrix plus a cheap rank-1 adapter per ensemble member.

    This is what makes TabM affordable: ``k`` members share ``weight`` and differ only
    by
    per-member input/output scaling vectors, so the parameter count is that of one model
    plus ``k * (in + 2*out)`` rather than ``k`` times everything.

    The sign initialisation is load-bearing. Initialising ``r`` and ``s`` to random +/-1
    makes the members functionally distinct from step zero; initialise them to ones and
    every member computes the identical function and the ensemble collapses.
    """

    def __init__(self, in_features: int, out_features: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.r = nn.Parameter(torch.randint(0, 2, (k, in_features)).float() * 2 - 1)
        self.s = nn.Parameter(torch.randint(0, 2, (k, out_features)).float() * 2 - 1)
        self.bias = nn.Parameter(torch.zeros(k, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, k, in_features)
        out: torch.Tensor = (x * self.r) @ self.weight.T * self.s + self.bias
        return out


class TabM(nn.Module):
    """
    A BatchEnsemble MLP — ``k`` implicit members trained as one model (ICLR 2025).

    Motivated directly by this project's own measurement: rank-averaging 5 independently
    seeded MLPs was worth +0.00283, more than every preprocessing and architecture
    change
    in the round combined. That result says the win is *ensembling*, not any particular
    architecture — and 5 seeds cost 5 full training runs. TabM aims at the same effect
    for
    roughly the price of one.

    Every member sees the same row and produces its own logit; the loss is averaged over
    members, and prediction averages their probabilities.
    """

    def __init__(
        self,
        n_numeric: int,
        cardinalities: Sequence[int],
        hidden: Sequence[int],
        dropout: float,
        k: int = 32,
        embed: Optional[PeriodicEmbeddings] = None,
        n_continuous: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.embed = embed
        # ``x_num`` arrives as [continuous | missingness indicators | one-hot], and only
        # the leading continuous block may be periodically embedded. Embedding a 0/1
        # column is meaningless -- sin/cos of two points -- and it is not free: with 192
        # continuous columns against 290 binary ones, embedding all 482 would spend 60%
        # of the capacity on nothing and cost 2.5x the compute.
        self.n_continuous = n_numeric if n_continuous is None else n_continuous
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c + 1, embedding_dim(c)) for c in cardinalities]
        )
        numeric_width = (
            embed.out_features + (n_numeric - self.n_continuous)
            if embed is not None
            else n_numeric
        )
        width = numeric_width + sum(embedding_dim(c) for c in cardinalities)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for size in hidden:
            self.layers.append(BatchEnsembleLinear(width, size, k))
            self.norms.append(nn.BatchNorm1d(size))
            width = size
        self.head = BatchEnsembleLinear(width, 1, k)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if self.embed is None:
            parts = [x_num]
        else:
            # Embed the continuous block; pass the binary tail through untouched.
            parts = [
                self.embed(x_num[:, : self.n_continuous]),
                x_num[:, self.n_continuous :],
            ]
        for j, emb in enumerate(self.embeddings):
            parts.append(emb(x_cat[:, j]))
        z = torch.cat(parts, dim=1).unsqueeze(1).expand(-1, self.k, -1)
        for layer, norm in zip(self.layers, self.norms):
            z = layer(z)
            # BatchNorm1d takes 2-D input, so members are folded into the batch axis.
            b, k, f = z.shape
            z = norm(z.reshape(b * k, f)).reshape(b, k, f)
            z = self.drop(torch.nn.functional.gelu(z))
        out: torch.Tensor = self.head(z).squeeze(-1)  # (batch, k)
        return out


def build_model(
    config: NNConfig,
    n_numeric: int,
    cardinalities: Sequence[int],
    n_continuous: Optional[int] = None,
) -> nn.Module:
    """Instantiate the architecture named by ``config.arch``."""
    if config.arch == "mlp":
        return EmbeddingMLP(n_numeric, cardinalities, config.hidden, config.dropout)
    if config.arch == "resnet":
        return ResNetMLP(n_numeric, cardinalities, config.hidden, config.dropout)
    if config.arch == "tabm":
        n_cont = n_numeric if n_continuous is None else n_continuous
        embed = (
            PeriodicEmbeddings(n_cont, config.d_embedding, config.sigma)
            if config.periodic
            else None
        )
        return TabM(
            n_numeric,
            cardinalities,
            config.hidden,
            config.dropout,
            k=config.k,
            embed=embed,
            n_continuous=n_cont,
        )
    raise ValueError(f"unknown arch {config.arch!r}")


def _predict(
    model: nn.Module, x_num: torch.Tensor, x_cat: torch.Tensor, batch_size: int
) -> np.ndarray:
    """
    Score rows in batches, averaging TabM's per-member probabilities.

    Batching here is not an optimisation — it is a correctness requirement. Scoring a
    whole fold in one forward pass allocates a tensor proportional to
    ``rows x members x width``, and that product is only small by accident: a plain MLP
    needs ~118 MB, but TabM with periodic embeddings widens 192 numeric features to
    4,608 dims and multiplies by ``k`` members, which demanded **19.58 GiB in a single
    allocation** and aborted the run. Any architecture that widens its input
    re-triggers it.

    Member probabilities are averaged, never logits: the members are an ensemble, and
    averaging in logit space is a geometric mean of odds that weights confident members
    more heavily than an ensemble should.
    """
    out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_num), batch_size):
            stop = start + batch_size
            probs = torch.sigmoid(model(x_num[start:stop], x_cat[start:stop]))
            if probs.ndim == 2:
                probs = probs.mean(dim=1)
            out.append(probs.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=float)


def _train_fold(
    model: nn.Module,
    xn_tr: torch.Tensor,
    xc_tr: torch.Tensor,
    y_tr: torch.Tensor,
    xn_va: torch.Tensor,
    xc_va: torch.Tensor,
    y_va: np.ndarray,
    config: NNConfig,
    deadline: Optional[float],
) -> Tuple[np.ndarray, int]:
    """Train one fold with early stopping on val AUC; returns (val preds, epochs)."""
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    n_batches = max(1, math.ceil(len(y_tr) / config.batch_size))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=config.lr,
        epochs=config.max_epochs,
        steps_per_epoch=n_batches,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_preds, best_epoch, stale = -1.0, np.zeros(len(y_va)), 0, 0
    for epoch in range(1, config.max_epochs + 1):
        if deadline is not None and time.time() > deadline:
            raise TimeBudgetExceeded(f"exceeded the time budget at epoch {epoch}")
        model.train()
        order = torch.randperm(len(y_tr), device=xn_tr.device)
        for start in range(0, len(order), config.batch_size):
            idx = order[start : start + config.batch_size]
            if len(idx) < 2:  # BatchNorm needs >1 row
                continue
            optimiser.zero_grad(set_to_none=True)
            logits = model(xn_tr[idx], xc_tr[idx])
            if logits.ndim == 2:  # TabM: (batch, k) — one logit per member
                loss = loss_fn(logits, y_tr[idx].unsqueeze(1).expand_as(logits))
            else:
                loss = loss_fn(logits, y_tr[idx])
            loss.backward()
            optimiser.step()
            schedule.step()

        model.eval()
        preds = _predict(model, xn_va, xc_va, config.batch_size)
        auc = float(roc_auc_score(y_va, preds))
        if auc > best_auc:
            best_auc, best_preds, best_epoch, stale = auc, preds, epoch, 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    return best_preds, best_epoch


def cross_validate_nn(
    fm: pl.DataFrame,
    feature_cols: Sequence[str],
    y: np.ndarray,
    config: Optional[NNConfig] = None,
    *,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    on_fold: Optional[Callable[[int, float], None]] = None,
    max_seconds: Optional[float] = None,
) -> CVResult:
    """
    Cross-validate a tabular network on the canonical folds.

    Returns the same :class:`~home_credit_risk.gbt.CVResult` as the tree and logistic
    paths, computed with the same ``n_splits`` and ``seed``, so the OOF score is
    directly comparable. ``best_iterations`` carries the early-stopping epoch per fold
    rather than a tree count.

    Imputation, scaling and the categorical vocabulary are all fitted **inside** each
    training fold. ``importances`` is returned as zeros: a network has no per-input
    attribution comparable to split gain, and inventing one would mislead.
    """
    config = config or NNConfig()
    device = torch.device(resolve_device(config.device))

    numeric = [
        c for c in feature_cols if fm[c].dtype not in (pl.String, pl.Categorical)
    ]
    categorical = [c for c in feature_cols if c not in set(numeric)]
    raw_numeric = fm.select(numeric).to_numpy().astype(np.float64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    fold_scores: List[float] = []
    epochs: List[int] = []
    deadline = time.time() + max_seconds if max_seconds is not None else None

    for fold, (tr, va) in enumerate(skf.split(raw_numeric, y), 1):
        torch.manual_seed(config.seed + fold)

        pre = fit_preprocessor(raw_numeric[tr], config)
        xn_tr_np, xn_va_np = (
            pre.transform(raw_numeric[tr]),
            pre.transform(raw_numeric[va]),
        )

        vocabs = {c: fit_category_vocab(fm[c][tr].to_list()) for c in categorical}
        codes_tr = encode_categoricals(fm[tr], categorical, vocabs)
        codes_va = encode_categoricals(fm[va], categorical, vocabs)

        if config.embeddings:
            cardinalities = [len(vocabs[c]) for c in categorical]
            xc_tr = torch.as_tensor(codes_tr, dtype=torch.long, device=device)
            xc_va = torch.as_tensor(codes_va, dtype=torch.long, device=device)
        else:
            # One-hot: fold the categoricals into the numeric block and give the model
            # no embedding tables at all.
            cardinalities = []
            xn_tr_np = np.concatenate(
                [xn_tr_np, one_hot(codes_tr, vocabs, categorical)], axis=1
            )
            xn_va_np = np.concatenate(
                [xn_va_np, one_hot(codes_va, vocabs, categorical)], axis=1
            )
            xc_tr = torch.zeros((len(tr), 0), dtype=torch.long, device=device)
            xc_va = torch.zeros((len(va), 0), dtype=torch.long, device=device)

        xn_tr = torch.as_tensor(xn_tr_np, dtype=torch.float32, device=device)
        xn_va = torch.as_tensor(xn_va_np, dtype=torch.float32, device=device)
        y_tr = torch.as_tensor(y[tr], dtype=torch.float32, device=device)

        model = build_model(
            config, xn_tr.shape[1], cardinalities, n_continuous=raw_numeric.shape[1]
        ).to(device)
        preds, best_epoch = _train_fold(
            model, xn_tr, xc_tr, y_tr, xn_va, xc_va, y[va], config, deadline
        )

        oof[va] = preds
        score = float(roc_auc_score(y[va], preds))
        fold_scores.append(score)
        epochs.append(best_epoch)
        if on_fold is not None:
            on_fold(fold, score)

    return CVResult(
        fold_scores=fold_scores,
        best_iterations=epochs,
        oof_pred=oof,
        oof_roc_auc=float(roc_auc_score(y, oof)),
        oof_pr_auc=float(average_precision_score(y, oof)),
        importances=np.zeros(len(feature_cols), dtype=float),
    )
