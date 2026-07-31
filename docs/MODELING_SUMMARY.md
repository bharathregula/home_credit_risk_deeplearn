# Modeling Summary — Gradient-Boosted-Tree Baseline

LightGBM on the customer-level feature matrix: a benchmark ROC-AUC for `TARGET` and a
null-aware feature shortlist for the neural-net and tabular-transformer rounds to come.
Trees only so far.

**Current best: OOF ROC-AUC 0.79478** (PR-AUC 0.29270), from a tuned configuration at a
fixed 1946 trees with no early stopping — so the figure carries no iteration-selection
bias. Five rounds, +0.0044 from the 0.79040 starting point; see [Result](#result).

- **Code:** [`src/home_credit_risk/modeling.py`](../src/home_credit_risk/modeling.py)
  (pure polars/numpy prep helpers) — model fitting lives in the notebook.
- **Notebook:**
  [`notebooks/modeling/baseline_trees.ipynb`](../notebooks/modeling/baseline_trees.ipynb)
- **Tests:** [`tests/test_modeling.py`](../tests/test_modeling.py)
- **Features:** [`docs/FEATURE_ENGINEERING_SUMMARY.md`](FEATURE_ENGINEERING_SUMMARY.md)

## Imputation vs. feature-selection ordering

A question drove this round's design: *do we impute before selecting features, or select
first?* The worry — pruning a feature that imputation would later have made useful — is
valid, but only against a specific mistake. The resolution:

- **"Imputation" here is two different operations.**
  1. **Structural encoding** — a null `*_COUNT` / `*_SUM` / `*_NUNIQUE` / `*_RATE` / `HAS_*`
     means *"no such history"*; its correct value is `0`, not "unknown". This is feature
     construction, done up front in `apply_structural_encoding`. It changes what the feature
     *means*, so it must precede everything.
  2. **Statistical imputation** — median-filling behavioural `*_MEAN` / `*_MIN` / `*_MAX`,
     which are genuinely unknown when absent. This is the only *real* imputation.
- **Don't select on a missingness-corrupted criterion.** Dropping columns by null-rate, or
  ranking by univariate correlation on median-filled data, is the actual bug the ordering
  worry points at. Median-then-select has the *opposite* bias too: median-fill injects a
  spike that deflates the true correlation. Either way, univariate-on-filled-data is wrong.
- **Correct order:** structural-encode → select with a **null-aware** criterion (GBT gain
  importance) → *(NN round only)* statistical-impute **inside cross-validation, per fold**.
  LightGBM splits on nulls natively, so for the trees there is **no statistical imputation at
  all** and the ordering question dissolves. Only genuinely zero-variance columns are dropped
  before fitting (they carry no signal for any model).

## Pipeline

1. `build_feature_matrix(application_train)` → 307,511 × 971, then
   `apply_structural_encoding` (count/sum/rate/flag nulls → 0; behavioural means left null).
2. `split_xy` drops `TARGET` + `SK_ID*`; `constant_feature_columns` drops **11** zero-variance
   columns (e.g. `*_SK_DPD*_MIN_*`, always 0) and `duplicate_feature_columns` drops **13**
   bit-identical columns → **945 features**.
3. `category_mapping` + `to_tree_matrix` integer-code the **16** driver string categoricals
   (`ORGANIZATION_TYPE`, `OCCUPATION_TYPE`, …) from an explicit sorted mapping and return
   their indices for LightGBM's native categorical handling; numeric nulls pass through
   untouched.
4. **Stratified 5-fold CV** via `gbt.cross_validate` — the one module that imports an
   estimator, so `modeling.py` stays a pure prep layer and every round (tuning, NN,
   transformer) evaluates on identical folds. LightGBM (`learning_rate=0.02`,
   `num_leaves=34`, `feature_fraction=0.7`, `subsample=0.8`, `reg_lambda=1.0`,
   `min_child_samples=100`), early stopping on fold-validation AUC. Positives ≈ **8.1%**
   (imbalanced; AUC is rank-based so it is the primary metric, PR-AUC reported alongside).
5. Everything is logged to **MLflow** (`home_credit_risk.tracking`): one run per CV, per-fold
   AUCs as metric steps, `cv_seed` / `n_splits` / feature counts as parameters, gain ranking
   and shortlist as artifacts.

## Result

| Metric | R1 baseline | R2 fixes + ratios | R3 tree budget | R4 tuned | **R5 final (clean)** |
|---|---|---|---|---|---|
| **OOF ROC-AUC** | 0.79040 | 0.79253 | 0.79306 | 0.79488 | **0.79478** |
| OOF PR-AUC | 0.286 | 0.2893 | 0.29007 | 0.29287 | **0.29270** |
| mean fold AUC | 0.7904 ±0.0040 | 0.79256 ±0.00367 | 0.79317 ±0.00349 | 0.79491 ±0.00362 | 0.79480 ±0.00372 |
| features | 951 | 945 | 945 | 945 | 945 |
| trees | 1000 cap-bound | 1000 cap-bound | 1423 converged | 1946 converged | **1946 fixed** |
| early stopping | 50 (never fired) | 50 (never fired) | 300 | 300 | **none** |

Every round uses the same folds and seed, so each delta is attributable to the one thing
that changed: **+0.0021** from the feature work, **+0.0005** from letting the trees
converge, **+0.0018** from the tuned hyperparameters. **Total +0.0044**, and R5 is the
quotable figure — it carries no iteration-selection bias.

### Round 4-5: hyperparameter tuning

The search (`scripts/tune_gbt.py search`, Optuna TPE at `lr=0.05`, transferred to `0.02`) moved
four parameters a long way from the defaults:

| parameter | default | tuned |
|---|---:|---:|
| `num_leaves` | 34 | 24 |
| `min_child_samples` | 100 | 32 |
| `feature_fraction` | 0.70 | **0.34** |
| `reg_lambda` | 1.0 | 21.4 |

The aggressive column subsampling is the interesting one, and it corroborates the feature
dictionary: with 332 `PREV_CC_*` columns earning 4% of gain, most features are redundant,
so showing each tree a third of the columns decorrelates the ensemble more than it starves
it. Smaller trees with far weaker leaf-size constraints and much stronger L2 is a
coherent package, not a lucky draw.

**The improvement is significant on a paired test.** All five folds improved over the R3
reference — mean **+0.00174**, worst fold still **+0.00092**, exact sign test **p = 0.031**
one-sided. That matters because +0.0018 is *half* the ±0.0036 fold spread and would look
like noise in an unpaired comparison; in a paired design the fold-to-fold variance cancels.
This test was impossible for round 2 (per-fold AUCs were not recorded then) and is the
reason every run now logs them to MLflow.

### The evaluation optimism, measured

R5 re-fits at a fixed 1946 trees with **no early stopping**, so no iteration is chosen from
the fold being scored. Comparing it to R4 isolates the bias exactly:

| | OOF ROC-AUC |
|---|---:|
| R4, best-iteration chosen per fold | 0.79488 |
| R5, fixed trees, no selection | 0.79478 |
| **selection optimism** | **0.00010** |

One ten-thousandth — as the flatness measurement predicted. The bias was never material,
and since it is constant across runs, every cross-round comparison above stands. Only fold
3 lost anything meaningful (−0.00053): its own optimum was 2578 trees, so a fixed 1946
slightly undertrains it.

**One consequence for the NN round:** the top-200 shortlist now captures **80.7%** of
cumulative gain, down from 82.7%. That is the expected effect of `feature_fraction=0.34` —
gain spreads more evenly when each tree sees fewer columns — but it means the top-200 cut
is marginally less complete than before, and a larger shortlist may be worth testing.

### What round 2 changed

Three correctness fixes and six new features (see
[`FEATURE_ENGINEERING_SUMMARY.md`](FEATURE_ENGINEERING_SUMMARY.md) §5-6):

- **The `DAYS_EMPLOYED` sentinel.** `365243` in 18% of driver rows was never cleaned on the
  application side. Nulling it moved the column itself from **rank 12 → rank 7**: the tree
  had been spending splits separating a fake +1000-year value from real tenures instead of
  ranking the tenures.
- **The affordability ratios earn 5.5% of total gain**, and two of them land in the top six
  features in the model — above every engineered aggregate and every application column
  except the external credit scores. This is the clearest possible demonstration that a
  quotient is not something a tree can find for itself: both inputs to
  `RATIO_PAYMENT_RATE` were already in the matrix and already high-gain.

| rank | gain | feature |
|---:|---:|---|
| 5 | 2.12% | `RATIO_PAYMENT_RATE` (annuity / credit) |
| 6 | 1.87% | `RATIO_CREDIT_TO_GOODS` |
| 19 | 0.60% | `RATIO_ANNUITY_TO_INCOME` |
| 26 | 0.48% | `RATIO_EMPLOYED_TO_AGE` |
| 52 | 0.27% | `RATIO_CREDIT_TO_INCOME` |
| 107 | 0.15% | `RATIO_INCOME_PER_PERSON` |
| 856 | 0.00% | `DAYS_EMPLOYED_ANOMALY` |

- **`DAYS_EMPLOYED_ANOMALY` earns nothing here — and that is the expected result.** Its
  information is already carried by the null it replaced, and LightGBM splits on nulls
  natively, so the flag is pure redundancy *for a tree*. It is kept because the NN and
  transformer rounds impute those nulls away, and the flag is what survives that.
- **13 bit-identical columns dropped.** Harmless for trees; perfectly collinear inputs for
  anything with a weight matrix.

## Feature importance (gain)

**Top overall** — the external credit scores still dominate, but two ratios now sit above
every aggregate:

| rank | gain | feature |
|---:|---:|---|
| 1 | 10.60% | `EXT_SOURCE_2` |
| 2 | 9.22% | `EXT_SOURCE_3` |
| 3 | 5.02% | `ORGANIZATION_TYPE` |
| 4 | 3.67% | `EXT_SOURCE_1` |
| 5 | 2.32% | `RATIO_PAYMENT_RATE` |
| 6 | 1.71% | `RATIO_CREDIT_TO_GOODS` |
| 7 | 1.65% | `OCCUPATION_TYPE` |
| 8 | 1.65% | `DAYS_EMPLOYED` (was rank 12 before the sentinel was cleaned) |

*(Measured on the round-5 final model. Gains are flatter than in earlier rounds because
`feature_fraction=0.34` shows each tree a third of the columns, spreading attribution.)*

**Top engineered** (excluding the driver's own and driver-derived columns) — the aggregation
and the iteration-2 categorical/windowed work pays off:

| rank | gain | feature |
|---:|---:|---|
| 10 | 0.94% | `PREV_REFUSED_RATE` (categorical, iteration 2) |
| 12 | 0.93% | `PREV_INST_AMT_PAYMENT_MIN_MEAN` |
| 13 | 0.91% | `INST_L365D_LATE_RATE` (**windowed** — recent late payments) |
| 14 | 0.91% | `BUREAU_DAYS_CREDIT_MEAN` (older average bureau history) |
| 17 | 0.65% | `BUREAU_CREDIT_ACTIVE_Closed_RATE` (categorical, iteration 2) |
| 18 | 0.64% | `BUREAU_DAYS_CREDIT_MAX` |

Full per-feature ranking, definitions and statistics:
[`FEATURE_DICTIONARY.md`](FEATURE_DICTIONARY.md).

## Shortlist for the later rounds

Top **200** features by gain (all with >0 gain, **80.7%** of cumulative gain — see the note
on `feature_fraction` above) are written to `gbt_shortlist.txt`, and the full ranking to
`gbt_gain_importance.csv`, both next to the dataset directory and attached to the MLflow run.
Both are regenerated by the round-5 final model, so they reflect the tuned configuration. The NN round runs on this shortlist with a
per-fold `SimpleImputer(median) → StandardScaler → MLP` pipeline (the concrete leakage-safe
imputation), and compares AUC. Because the shortlist was chosen **null-natively**, no feature
was discarded on a missingness-corrupted criterion that imputation might have rescued.

## The tree budget (round 3)

Rounds 1-2 ran `n_estimators=1000` with `early_stopping(50)`, and **early stopping never
fired**: `best_iteration_` was 982 / 960 / 998 / 999 / 997. Every fit was ended by the
budget, not by convergence, so the configuration's ceiling was unknown and any
hyperparameter search would have been optimising a truncated model.

Re-running at `n_estimators=6000, early_stopping(300)` — patience of 50 is too tight at
`lr=0.02`, where late-stage gains are ~1e-5/round — early stopping fires on every fold:

| fold | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| `best_iteration_` | 1423 | 1164 | 2044 | 1310 | 1840 |
| fold AUC | 0.79087 | 0.79867 | 0.78950 | 0.79582 | 0.79101 |

**The budget was worth only +0.0005** (0.79253 → 0.79306) — a quarter of what the six ratio
features delivered. The model was already near-converged at 1000 trees despite early
stopping not having formally fired. Worth checking, but not the headroom the binding cap
suggested.

### The optimum is flat, which closes the question

Averaging the per-fold validation curves and reading off the loss away from the peak
(`gbt.curve_flatness`):

| trees | AUC given up |
|---|---:|
| 0.7× peak (~996) | **0.00043** |
| 1.3× peak (~1849) | **0.00003** |

Running 30% fewer trees costs four ten-thousandths of AUC. Three consequences:

1. **`n_estimators` is settled and should not be tuned** — fix it and spend the search
   budget elsewhere. This is exactly the question the flatness measurement exists to answer.
2. **The per-fold spread is not meaningful.** `best_iteration_` ranges 1164-2044 (1.75×)
   while AUC barely moves: the argmax lands almost arbitrarily on a plateau.
3. **The iteration-selection optimism below is negligible**, and now quantifiably so.

## Challenger: is the tree's complexity justified?

A GBT compared only against other GBTs proves nothing about whether the algorithm choice
was right. `scripts/challenger.py` fits the interpretable incumbent of credit scoring — a
logistic regression — on the **same folds**, via the dense preprocessing in
`home_credit_risk.linear`.

| Model | Features | OOF ROC-AUC | Gini | PR-AUC |
|---|---:|---:|---:|---:|
| **GBT (LightGBM)** | 945 | **0.79478** | 0.5896 | 0.2927 |
| GBT (LightGBM) | 200 shortlist | 0.79456 | 0.5891 | 0.2912 |
| Logistic regression | 200 shortlist | 0.77741 | 0.5548 | 0.2628 |

**Two effects, cleanly separated by the middle row:**

- **Model class: +0.01715** (GBT vs LR on *identical* 200 features). The GBT wins on 5 of
  5 folds; its worst fold beats the LR's best-matched fold by +0.0134. For scale, every
  feature-engineering and tuning change in this project combined moved the GBT +0.0044 —
  the model-class gap is **four times** that. The complexity is justified.
- **Feature count: −0.00022** (945 → 200 features). Statistically nothing; fold 2 is
  actually *higher* on the shortlist. The top-200 carries essentially all the usable
  signal despite capturing "only" 80.7% of cumulative gain.

The second finding matters for the next round: **the NN inherits the same 200 features,
and this establishes they are not a handicap.** A weak NN result cannot be attributed to
the feature budget.

The baseline was built to be *strong*, not to be beaten — missingness indicators (so
imputation does not discard the signal in the nulls the tree reads natively), per-fold
fitted imputation and scaling, one-hot rather than integer-coded categoricals. A
handicapped baseline would have proved nothing. Even so, note what the LR still lacks and
a production scorecard would have: **WOE binning with coarse classing**, which captures
non-linearity inside a linear framework and would narrow the gap.

## Neural-network round

The third model family, on the same folds. Code: `home_credit_risk.nn`, driven by
`scripts/run_nn.py`. **Blending is deliberately not attempted here** — every figure below is a
standalone network.

| Model | OOF ROC-AUC | vs GBT (200 feat.) | vs logistic |
|---|---:|---:|---:|
| GBT (LightGBM, 200 features) | 0.79456 | — | +0.01715 |
| **MLP, 5-seed rank-averaged ensemble** | **0.78500** | **−0.00956** | **+0.00759** |
| MLP, single seed | 0.78235 | −0.01221 | +0.00494 |
| ResNet-MLP, single seed | 0.77900 | −0.01556 | +0.00159 |
| Logistic regression | 0.77741 | −0.01715 | — |

**The network does not beat the tree**, closing to within **0.0096** on identical
features. That is the expected outcome for heterogeneous tabular data and was stated as
the expectation before the round began, not after.

### Preprocessing ablations — three of four predictions were wrong

Each is an isolated change on the canonical folds, measured against a 0.78011 reference:

| Change | OOF | Δ | Predicted? |
|---|---:|---:|---|
| All 945 features instead of 200 | 0.77743 | **−0.00268** | no prediction |
| Standard scaling instead of rank-gauss | 0.78195 | **+0.00184** | ✗ backwards |
| Drop the 182 missingness indicators | 0.77894 | −0.00117 | ✓ direction, overstated |
| One-hot instead of entity embeddings | 0.78103 | +0.00092 | ✗ backwards |
| **Standard scaling + one-hot together** | **0.78235** | +0.00224 | sub-additive |

- **Rank-gauss lost.** It was predicted to be the highest-value choice in the round,
  reasoning from 53 features with `max/p99 > 10` (up to 2810×). Two likely causes: rank
  transformation discards *magnitude*, and for credit risk the size of an overdue amount
  is signal rather than tail noise; and BatchNorm re-normalises activations anyway, so the
  network is far less sensitive to input conditioning than the raw tails suggest.
- **Embeddings lost.** They pay off at thousands of levels; the widest column here has 58,
  where one-hot costs ~108 columns and a 13-dim embedding is a bottleneck that discards
  information rather than usefully regularising.
- **Feature count is the largest effect, and it is architecture-specific.** Restricting to
  the 200-feature shortlist is worth **+0.0027** to the network but was worth nothing to
  the GBT (−0.00022). Trees select features implicitly through splits; a dense network
  gives all 945 inputs weights and must spend capacity learning to ignore ~745 redundant
  ones.
- The combination was **confirmed by running it**, not assumed: 0.78235 against the
  0.78287 that naive addition predicts.

### How large is seed noise? (and which conclusions survive it)

Five seeds of the identical configuration: **std 0.00030**, range 0.00077. The standard
error of a difference between two single-seed runs is therefore **0.00043**.

| Effect | Δ | vs noise | Verdict |
|---|---:|---:|---|
| Feature set 945 → 200 | 0.00268 | 6.3 SE | solid |
| Standard scaling | 0.00184 | 4.3 SE | solid |
| Missingness indicators | 0.00117 | 2.8 SE | **borderline** |
| One-hot vs embeddings | 0.00092 | 2.2 SE | **borderline** |

The two smaller conclusions are close enough to seed noise that they should be treated as
directional rather than established. Repeating them across seeds would settle it; that is
cheap and worth doing before either is quoted as a finding.

**This also tempers a claim made elsewhere in these docs.** Informative missingness is
described as a central design thesis. On the network it is worth **+0.0012 at 2.8 SE** —
enough to keep the indicators, not enough to carry the weight of "central". It remains
structurally important for the *tree* (which reads nulls natively and needs no indicators
at all), and the whitepaper's fair-lending and data sections are unaffected.

### Seed ensembling is worth more than any architecture change

Rank-averaging 5 seeds: **0.78500**, **+0.00283** over the mean single seed and +0.00243
over the best. That is larger than every preprocessing effect and roughly 4× the spread
between the two architectures tried — for no design risk at all.

**ResNet-MLP is not carried forward.** It scored 0.77900 against the plain MLP's 0.78235
*and* cost 648s against 134s. One caveat stated plainly: `lr=3e-4` / `dropout=0.4` were
tuned for the plain MLP, and residual networks generally prefer different settings, so
this measures "ResNet under MLP-tuned hyperparameters" rather than a clean architecture
comparison. Given ensembling returned 4× more than the architecture gap for a fraction of
the effort, further architecture variation was judged the weaker use of the budget.

**TabM was not attempted.** It was the plan's most implementation-risky item, gated on the
first two ladder rungs landing cleanly; ResNet-MLP did not.

## Known evaluation optimism — resolved

`predict_proba` uses `best_iteration_`, and that iteration is the argmax of validation AUC
on the same fold the predictions are then scored against — a selection made on the scored
data, which flatters the OOF figure. Note that **early stopping never fired in rounds 1-2**,
so no *stopping* decision came from the validation fold; only the iteration choice did.

The flatness above predicted the bias would be a few ten-thousandths at most. Round 5
measured it directly at **0.00010** (see the results table). Negligible, constant across
runs, and now eliminated from the headline figure: the reported 0.79478 comes from a re-fit
with `n_estimators` fixed and `early_stopping_rounds=None`, so predictions use every tree
and no iteration is chosen from the scored fold (`scripts/tune_gbt.py final`).

## Next steps

1. **NN round — the pivot.** GBT optimisation is closed; see below for why. Run on the
   shortlist with per-fold impute + scale (`home_credit_risk.linear` already provides the
   path), **missingness indicators** (183 of the 200 shortlisted features have nulls, and
   the missingness is itself predictive: `BUREAU_DAYS_ENDDATE_FACT_*` is null for
   customers defaulting at 10.5% vs the 8.1% base rate), and **embeddings** for
   `ORGANIZATION_TYPE` (58 levels) / `OCCUPATION_TYPE`. The shortlist is not a handicap —
   the GBT scores 0.79456 on those same 200 features versus 0.79478 on all 945.

   *Why GBT tuning is closed:* three attempts to extend the Optuna study produced 5
   completed trials against a planned ~12, with TPE never engaging. Two harness defects
   were found and fixed (unbounded trial cost; a seeded sampler replaying its sequence on
   resume), but the marginal return no longer justifies the time — the five trials span
   only ~0.001 in fold-1 AUC, and a GBT+NN blend is worth more than further tuning.
   The study persists in `optuna.db` and resumes if you return to it.
2. **NN round** on the shortlist — per-fold impute + scale, **missingness indicators** (183
   of the 200 shortlisted features have nulls, and the missingness is itself predictive:
   `BUREAU_DAYS_ENDDATE_FACT_*` is null for customers defaulting at 10.5% vs the 8.1% base
   rate), and **embeddings** for `ORGANIZATION_TYPE` (58 levels) / `OCCUPATION_TYPE`.
3. **Tabular transformers** (FT-Transformer / TabPFN-style) on the same folds.
4. **Blend** — the NN will most likely score below the GBT on its own, which is the normal
   tabular result; the payoff is a rank-average of decorrelated errors.
5. Probability calibration and test-set scoring for a submission. Note that scoring
   `application_test` **requires** passing the training `category_mapping` to
   `to_tree_matrix`.
