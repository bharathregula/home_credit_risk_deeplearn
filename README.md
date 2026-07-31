# Home Credit Risk Deeplearn

This project is a small Python starter template for building and testing a credit-risk related package with linting, type checking, and automated tests.

## What you will find here

- A simple package structure under [src/home_credit_risk](src/home_credit_risk)
- Basic logging and configuration examples
- Automated tests with pytest
- Pre-commit checks for formatting, linting, and type safety

## Prerequisites

Make sure you have Python 3.14+ and uv installed.

- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- Verify your Python version:

```bash
python --version
```

## First-time setup

From the project root, run:

```bash
uv sync --group dev
```

This creates the virtual environment and installs the project and development dependencies.

## Run the example app

```bash
uv run python src/home_credit_risk/main.py
```

You should see a message printed to the terminal.

## Run the tests

```bash
uv run pytest
```

## Run quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

## Pre-commit hooks

To run the local hooks before committing:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Exploratory data analysis

EDA for each Home Credit dataset lives in
[notebooks/individual_dataset_eda](notebooks/individual_dataset_eda) and reuses the
shared helpers in [src/home_credit_risk/eda_utils.py](src/home_credit_risk/eda_utils.py).

For a detailed, dataset-by-dataset write-up of what was checked and what was found
(shapes, grains, missing data, and the table-specific business-logic quality checks),
see **[docs/INDIVIDUAL_DATASET_EDA_SUMMARY.md](docs/INDIVIDUAL_DATASET_EDA_SUMMARY.md)**.

The Kaggle Home Credit datasets are not stored in the repo; the notebooks load them
from a local path.

## Feature engineering

Customer-level features for predicting `TARGET` are built in
[src/home_credit_risk/features.py](src/home_credit_risk/features.py), which aggregates
every auxiliary table to one row per `SK_ID_CURR` (previous-application hub via
`SK_ID_PREV` + a bureau lineage) and joins them onto the application driver via
`build_feature_matrix`. The assembled matrix is explored in
[notebooks/feature_engineering/feature_matrix_eda.ipynb](notebooks/feature_engineering/feature_matrix_eda.ipynb).

See **[docs/FEATURE_ENGINEERING_SUMMARY.md](docs/FEATURE_ENGINEERING_SUMMARY.md)** for the
feature inventory, coverage/missingness, top correlations with `TARGET`, and the
imputation plan.

## Modeling

The gradient-boosted-tree baseline is built in
[src/home_credit_risk/modeling.py](src/home_credit_risk/modeling.py) (pure polars/numpy prep
helpers — structural encoding, constant/duplicate-column drop, stable category mapping,
tree matrix) and
[notebooks/modeling/baseline_trees.ipynb](notebooks/modeling/baseline_trees.ipynb) (LightGBM
5-fold CV, ROC-AUC, gain-importance shortlist).

See **[docs/MODELING_SUMMARY.md](docs/MODELING_SUMMARY.md)** for the CV setup, ROC-AUC, top
features, and a note on **imputation-vs-feature-selection ordering** (why trees need no
statistical imputation and selection is done null-natively).

**Leaderboard** — three model families, all on the identical 5-fold split (`SEED=42`,
`N_SPLITS=5`), which is the only reason the numbers can be compared at all:

| Model | Features | OOF ROC-AUC |
|---|---|---|
| **LightGBM GBT** | 945 | **0.79478** |
| LightGBM GBT | 200 shortlist | 0.79456 |
| Neural net, 5-seed ensemble | 200 | 0.78500 |
| Neural net, single seed | 200 | 0.78235 |
| ResNet-MLP | 200 | 0.77900 |
| Logistic regression | 200 | 0.77741 |

The ordering **tree > network > linear** is the expected one for heterogeneous tabular
data. The +0.01715 gap between the tree and the logistic model is attributable to *model
class* rather than feature count, since the GBT scores 0.79456 on the challenger's
identical 200 features. Note the shortlist is worth **+0.0027 to the network and nothing
(−0.0002) to the tree that produced it**.

### Hyperparameter tuning

[scripts/tune_gbt.py](scripts/tune_gbt.py) drives the search; the cross-validation loop itself lives
in [src/home_credit_risk/gbt.py](src/home_credit_risk/gbt.py) so it is unit-tested and
shared with the later NN round. Run the modes **in order** — each consumes the previous
one's answer:

```bash
uv run python scripts/tune_gbt.py budget                  # where does it actually converge?
uv run python scripts/tune_gbt.py search --minutes 55     # joint TPE search, resumable
uv run python scripts/tune_gbt.py transfer                # best config at the target lr
uv run python scripts/tune_gbt.py final --n-estimators N  # unbiased number + new shortlist
```

**Order applies to phases, not to individual hyperparameters:**

- **`n_estimators` is never searched.** With early stopping it is *derived* — each fold
  fits until its validation AUC stops improving. What you choose is a cap generous enough
  never to bind, plus patience wide enough not to stop on noise. `budget` exists because
  the original run's cap of 1000 was binding on all five folds, which made the
  configuration's ceiling unknown.
- **`learning_rate` is handled outside the search**, because it is inversely coupled to
  tree count and therefore to trial cost. `search` runs at `0.05` so trials converge in
  ~0.4× the trees; `transfer` moves the winner back to `0.02`.
- **Everything else is searched jointly.** `num_leaves`, `min_child_samples`,
  `feature_fraction` and `reg_lambda` interact — a larger `num_leaves` is only safe with a
  larger `min_child_samples`. Tuning them one at a time finds a worse optimum than a joint
  search using the same number of fits.

The Optuna study persists to `sqlite:///optuna.db` (gitignored), so re-running `search`
extends the existing history rather than starting over.

### Neural networks

[scripts/run_nn.py](scripts/run_nn.py) runs the NN round; the models live in
[src/home_credit_risk/nn.py](src/home_credit_risk/nn.py).

```bash
uv run python scripts/run_nn.py --arch mlp                # reference MLP (0.78235)
uv run python scripts/run_nn.py --quantile                # ablation: rank-gauss scaling
uv run python scripts/run_nn.py --no-indicators           # ablation: no missingness flags
uv run python scripts/run_nn.py --embeddings              # ablation: embeddings vs one-hot
uv run python scripts/run_nn.py --arch resnet             # residual MLP (0.77900)
uv run python scripts/run_nn.py --arch mlp --seeds 5      # seed ensemble (0.78500)
```

Every toggle is symmetric (`--quantile/--no-quantile` etc.) so each ablation is
re-runnable in both directions. **The defaults are the measured winners, not the intuitive
ones** — see the ablation table in [docs/MODELING_SUMMARY.md](docs/MODELING_SUMMARY.md)
before "restoring" any of them.

**Preprocessing is settled before architectures are compared.** Preprocessing choices
transfer across architectures; architecture choices do not transfer across preprocessing,
so ranking architectures on badly-scaled inputs measures robustness-to-bad-scaling rather
than architecture quality.

Three of the four predictions going into that stage were wrong, which is the round's most
useful outcome:

- **Rank-gauss lost to standard scaling** (0.78011 vs 0.78195), despite 53 of the 159
  numeric features having `max/p99 > 10`, reaching 2810×. The rank transform discards
  magnitude, and BatchNorm re-normalises the tails anyway.
- **Embeddings lost to one-hot** (0.78011 vs 0.78103): the widest categorical has only 58
  levels, where a 13-dimensional embedding is a bottleneck rather than a compression.
- **Seed-ensembling (+0.00283) beat every preprocessing and architecture change in the
  round**, and the seed-noise floor (std 0.00030 across 5 seeds) means two of the Stage-B
  effects sit under 3 SE and are recorded as directional only.

> **Note:** LightGBM and PyTorch each ship an OpenMP runtime and must not be loaded into
> one process — on macOS the two thread pools collide and segfault. The shared CV contract
> therefore lives in [src/home_credit_risk/evaluation.py](src/home_credit_risk/evaluation.py),
> which imports no estimator, so each model family loads exactly one numeric backend.

### Challenger model

[scripts/challenger.py](scripts/challenger.py) fits a logistic-regression baseline on the
tree-selected shortlist, via the dense preprocessing in
[src/home_credit_risk/linear.py](src/home_credit_risk/linear.py). (The NN round does *not*
reuse this pipeline — `nn.py` builds its own dense path, because embeddings need integer
codes and torch needs per-fold tensors. The two share intent, not code.)

```bash
uv run python scripts/challenger.py
```

It exists to answer what the GBT cannot answer about itself: **is the tree's complexity
justified?** The baseline is deliberately built to be strong — missingness indicators so
imputation doesn't destroy the signal in the nulls, per-fold-fitted imputer/scaler/one-hot
vocabulary, and one-hot rather than integer codes (a linear model reads code 47 as larger
than code 3). A baseline handicapped by its preprocessing proves nothing about the model
that beats it.

### Experiment tracking

Every modeling run logs to MLflow via
[src/home_credit_risk/tracking.py](src/home_credit_risk/tracking.py) — one run per
cross-validation, with per-fold AUCs as metric steps, the fold seed and feature count as
parameters, and the gain ranking + shortlist as artifacts. Metrics and params go to a local
`sqlite:///mlflow.db`, artifacts to `./mlruns/<experiment_id>/<run_id>/artifacts/` (both
gitignored); telemetry to MLflow's servers is disabled.

```bash
# browse runs at http://127.0.0.1:5555
PYTHONPATH=tools/compat uv run mlflow ui \
    --backend-store-uri sqlite:///mlflow.db --port 5555
```

Two gotchas, both handled above:

- **`PYTHONPATH=tools/compat` is required on Python 3.14.** mlflow 3.14's UI server
  imports `importlib.abc.Traversable`, which 3.14 removed, and dies with an `ImportError`
  before binding the port. [tools/compat/sitecustomize.py](tools/compat/sitecustomize.py)
  restores the name at interpreter startup — it has to be a `sitecustomize` module rather
  than a patch in a wrapper, because `mlflow ui` spawns the server as a subprocess. It
  self-disables once mlflow fixes the import. Only the UI is affected; logging and
  `best_runs()` work without it.
- **Avoid port 5000 on macOS** — AirPlay Receiver (`ControlCenter`) already listens there,
  so the default port fails with `[Errno 48] Address already in use`.

```python
from home_credit_risk.tracking import best_runs, fold_scores, run_history
best_runs()               # leaderboard by oof_roc_auc, with each run's parameters
run_history(tag="nn")     # every run of one round, chronological, params + metrics
fold_scores(["gbt-final", "logistic-shortlist"])   # per-fold, for paired comparison
```

### Results notebooks

Two notebooks read the completed runs back out and explain what the rounds bought:

| Notebook | Covers |
|---|---|
| [notebooks/modeling/gbt_optimization.ipynb](notebooks/modeling/gbt_optimization.ipynb) | tuning trajectory, budget probe and optimum flatness, the Optuna study, paired fold test, gain importance |
| [notebooks/modeling/nn_results.ipynb](notebooks/modeling/nn_results.ipynb) | the Stage-A harness bugs the sanity gate caught, Stage-B ablations against the noise floor, seed ensembling, cross-family comparison |

**Neither notebook fits anything.** They read `mlflow.db` and `optuna.db` only, so neither
imports LightGBM or torch — which is what lets both be open at once without the OpenMP
collision. They also `chdir` to the repo root in the setup cell, because `.env` (needed at
import time for `API_KEY`) and both `sqlite:///` URIs resolve relative to the working
directory.

Note the repo's `nbstripout` hook clears outputs on commit, so a fresh clone shows the
notebooks unrun; execute them to regenerate the figures. What they **cannot** show is ROC
curves, calibration or decile lift — those need the out-of-fold prediction vectors, and
only summary scalars were logged (`CVResult.oof_pred` is never persisted).

## Model white paper

**[docs/MODEL_WHITEPAPER.md](docs/MODEL_WHITEPAPER.md)** is the formal model-risk write-up
(SR 11-7 style): design rationale, data lineage and treatments, selection process, outcome
analysis (KS 0.446, Gini 0.590, decile lift, calibration, segment stability), code
controls, and a consolidated **gap register**.

It states plainly that the model is **not approved for production credit decisioning** and
records why — including that gender, age and marital status are direct model inputs
(ECOA/Reg B blocking), and that the source data has no date field so out-of-time
validation is impossible. Sections with no supporting evidence are marked `GAP` rather
than filled in.

## Feature dictionary

**[docs/FEATURE_DICTIONARY.md](docs/FEATURE_DICTIONARY.md)** documents all **969 candidate
features** — plain-English definition, how each is calculated, distribution statistics,
encoding logic, and whether it made the baseline's **top-200 gain shortlist**.
[docs/feature_dictionary.csv](docs/feature_dictionary.csv) is the same content one row per
feature, for filtering.

## Project layout

**`src/` is the library; `scripts/` is how you run it.** The dependency is one-directional
— `scripts/run_nn.py` imports `home_credit_risk.nn`, never the reverse.

| | `src/home_credit_risk/` | `scripts/` |
|---|---|---|
| Contains | models, CV loops, config dataclasses | argparse, dataset paths, MLflow wiring |
| Imports | nothing outside the package | the package |
| Imported by | tests, notebooks, other modules | nothing |
| Knows the CSVs' location | no | yes |

That split is what lets `uv run pytest tests/test_nn.py` exercise the real training loop on
300 synthetic rows in about a second: the loop has no idea where the data lives, so tests
and CI never need the 300k-row files. It is also how the **OpenMP constraint** is enforced
— separate entry points mean separate processes, so no run loads both LightGBM and torch.

```text
src/home_credit_risk/
├── __init__.py
├── config.py
├── logger.py
├── eda_utils.py        # shared polars EDA helpers
├── features.py         # customer-level feature engineering
├── modeling.py         # model-prep helpers (structural encoding, selection, tree matrix)
├── gbt.py              # LightGBM cross-validation (the only estimator-importing module)
├── linear.py           # logistic challenger + its own dense preprocessing
├── nn.py               # neural networks (the only torch-importing module)
├── evaluation.py       # shared CV contract (CVResult, folds) — estimator-free
├── tracking.py         # MLflow experiment tracking
├── main.py
└── trivial_test.py

notebooks/
├── individual_dataset_eda/   # one EDA notebook per dataset
├── feature_engineering/      # feature_matrix_eda.ipynb
└── modeling/                 # baseline_trees.ipynb

docs/
├── INDIVIDUAL_DATASET_EDA_SUMMARY.md
├── FEATURE_ENGINEERING_SUMMARY.md
├── MODELING_SUMMARY.md
├── FEATURE_DICTIONARY.md    # every feature: definition, stats, selected?
├── feature_dictionary.csv   # machine-readable version
├── INTERVIEW_PREP.md        # defending the feature engineering: 5 questions + drill
└── MODEL_WHITEPAPER.md      # formal model-risk white paper (SR 11-7 style template)

scripts/                     # runners: CLI + dataset paths + MLflow wiring only.
├── tune_gbt.py              # tuning driver: budget -> search -> transfer -> final
├── challenger.py            # logistic-regression baseline on the shortlist
└── run_nn.py                # neural-network round: architectures + ablations

tools/compat/
└── sitecustomize.py         # Python 3.14 shim required to launch the MLflow UI

tests/
├── conftest.py              # pins OMP_NUM_THREADS=1 (LightGBM + torch in one process)
├── test_trivial.py
├── test_eda_utils.py
├── test_features.py
├── test_modeling.py
├── test_gbt.py              # CV loop, pruning seam, budget-binding — tiny synthetic data
├── test_linear.py           # challenger pipeline + per-fold leakage guard
├── test_nn.py               # vocab/OOV, preprocessing toggles, determinism on CPU
└── test_tracking.py         # MLflow logging against a temp store
```

## Next steps

If you are new to the project, start by reading the package files in [src/home_credit_risk](src/home_credit_risk) and then add your own features and tests.
