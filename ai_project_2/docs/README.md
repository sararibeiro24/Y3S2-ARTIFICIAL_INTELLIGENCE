# Fraud Detection — Internal Design Notes

Internal write-up of every decision made in this project. Not a tutorial.
The audience is us (and whoever inherits this), so it skips the obvious and
records the things we'd otherwise forget in two weeks.

The notebook at `data/catching-the-0-1-semi-supervised-fraud-detection.ipynb`
is the reference we worked from. This document tracks where we diverged from
it, what we tried, and what we kept.

---

## 1. Problem framing

Binary classification on a transaction log. Target `isFraud ∈ {0, 1}`. Severe
class imbalance (~0.13% positives). The dataset is PaySim, a synthetic
mobile-money simulator — every transaction has source/dest balances before
and after the operation, a `type`, an `amount`, and a `step` (hour-of-sim).

The problem is framed as a **scoring task** (rank by fraud probability), not
a hard yes/no — the consumer is a review queue with finite analyst capacity,
so the only thresholds that matter are the ones that keep precision above
the level at which analysts can keep up.

---

## 2. Data flow at a glance

```
raw CSV
  └─ data.load.load_data
        └─ data.clean.clean              (drop NA, dup, identifier cols)
            └─ data.encode.encode_transaction_type   (str → int8)
                └─ data.split.make_splits             (temporal, val + test)
                    └─ features.engineer.engineer    (balance errors, time, flags)
                        └─ persisted as parquet under data/processed/
                            └─ features.preprocess.build_preprocessor   (OHE+Scaler)
                                └─ training.model.fit_supervised        (XGBoost)
                                    └─ evaluation.*                     (metrics, SHAP, gain)
```

Each stage is its own module so we can run or replace it in isolation.

---

## 3. Module layout

```
model/
  main.py                          CLI: eda | prep | train | eval | all
  src/
    config.py                      single source of truth for constants
    data/
      load.py                      csv reader
      clean.py                     dropna, drop_duplicates, drop id columns
      encode.py                    type string → int8 (parquet-safe)
      split.py                     temporal split + val + class-balanced labeled
    features/
      engineer.py                  balance errors, zero flags, time, amount, type flags
      preprocess.py                OHE(type) + StandardScaler(numerics) Pipeline
    training/
      model.py                     XGBoost wrapper, derives scale_pos_weight from data
      model_io.py                  joblib save/load
    evaluation/
      metrics.py                   PR-AUC, AP, ROC-AUC, recall@precision, conf-counts
      report.py                    JSON dump + comparison table
      explain.py                   SHAP summary + gain importance
    analysis/
      distributions.py             EDA plots (class balance, type counts, etc.)
    pipeline.py                    orchestrator (prepare → train → evaluate_all)
docs/
  presentation.pptx                final deck (built from P2 template + chart slides)
  README.md                        this file
scripts/
  update_charts.py                 generates chart slides, injects into presentation.pptx
```

`model/api/` and `web/` are placeholders for downstream components we are
not building in this project.

---

## 4. What came from the notebook (and what didn't)

The reference notebook (`data/catching-the-0-1-semi-supervised-fraud-detection.ipynb`)
is a Kaggle-style walkthrough. We did not start from a blank slate — the
following choices are reused, and the rest of this section explains why
they survive the rewrite.

### Kept directly from the notebook

| choice                                      | why we kept it                                                                                  |
|---------------------------------------------|-------------------------------------------------------------------------------------------------|
| **PaySim dataset, target = `isFraud`**      | The assignment dataset; not a decision we get to make.                                          |
| **Drop `isFlaggedFraud`, `nameOrig`, `nameDest`** | The flag is a legacy rule with poor coverage; names are identifiers, not signal.                |
| **Limited-label regime, 1k per class**      | Matches the notebook's semi-supervised framing and keeps results comparable.                    |
| **Balance-equation residuals (`errorBalanceOrig/Dest`)** | The notebook's strongest single feature; falls out of basic accounting (in + out should balance). |
| **Zero-balance flags** (`orig_zero_after`, `dest_zero_before`, `dest_zero_after`) | Same intuition — accounts emptied right before/after a transfer are the canonical mule fingerprint. |
| **`log1p(amount)`**                         | Compresses the heavy right tail so trees don't burn splits on outliers.                         |
| **`is_high_amount` (quantile flag)**        | Captures the "very large transaction" prior cheaply, mirrors the notebook's `isHighAmount`.     |
| **XGBoost as the classifier**               | Notebook's choice; remains the strongest off-the-shelf model for tabular + imbalance.           |
| **`scale_pos_weight` for class imbalance**  | Standard XGBoost knob for skewed targets. We compute it from data instead of hardcoding 775 (see 5.8). |
| **PR-AUC / Average Precision as headline metric** | Right metric under ~0.1% positive rate — accuracy is meaningless here.                          |
| **SHAP for explanation**                    | Confirms the notebook's narrative about which features carry the signal.                        |

### Added or changed

| change                                            | rationale (full detail in §5)                                                          |
|---------------------------------------------------|----------------------------------------------------------------------------------------|
| Temporal split on `step` (vs stratified random)   | Honest evaluation under drift (5.1)                                                    |
| Validation set carved between train and test      | Stops the notebook's threshold-tuning-on-test leak (5.2)                               |
| No 1:773 downsampling of the unlabeled pool       | Keeps the marginal honest for any future semi-supervised work (5.3)                    |
| Time features (`hour_of_day`, `day_of_week`)      | `step` carries circadian structure the notebook never uses                             |
| `is_transfer_or_cashout` explicit flag            | Tightens early splits, helps in the small-labeled regime                               |
| `type` cast to `int8`, OHE inside sklearn Pipeline | Parquet-safe and inference-ready, vs the notebook's inline `replace` (5.6, 5.7)        |
| `scale_pos_weight` derived from labels at fit time | Tracks the data instead of being hardcoded (5.8)                                       |
| Self-training, IF stacking, calibration, cost thresholds — **all tried and dropped** | Documented in 5.10 — the simpler model wins and we want teammates to know why.         |
| `recall@precision=0.9` reported alongside PR-AUC  | Maps directly to analyst capacity — the metric the business actually optimises         |
| Module layout (data / features / training / evaluation / analysis) | The notebook is one linear script; modular code is reproducible and testable           |
| `Pipeline` + `ColumnTransformer` for preprocessing | Single fit/transform contract across labeled, unlabeled, val, test, and future inference |
| CLI with stages (`prep | train | eval | eda`)     | Re-run individual phases without redoing data prep                                     |

If something in §5 doesn't appear in either table above, treat it as new
work and read the rationale carefully — it's not in the notebook.

---

## 5. Every decision, with rationale

### 5.1 Split: temporal, not stratified-random

The notebook does a 20% stratified random split. We use a **temporal split
on `step`** instead:

- last 20% of `step` → test
- next 10% (before that) → val
- everything earlier → train pool

`USE_TEMPORAL_SPLIT = True` in `config.py`. The stratified path is still
implemented in `_stratified_random_split` for ablation.

Why: fraud patterns drift in time. A random split lets the model see future
behaviour during training, which inflates test metrics. The temporal split
matches deployment reality: train on the past, predict the future.

### 5.2 Validation set added

Notebook has only `labeled_train / unlabeled_train / test`. We add a `val`
set. Anything tuned on `val` (thresholds, calibrators, hyperparameters)
never touches `test`. The notebook tunes its operating threshold on the test
set — that's a label leak. We don't.

### 5.3 Unlabeled pool kept at natural class ratio

The notebook downsamples non-fraud in the unlabeled pool to a 1:773 ratio.
We don't. That choice bakes the true fraud prior into the supposedly
"unlabeled" data, which is the exact thing semi-supervised methods are
supposed to discover from the marginal distribution. Even though we don't
ship semi-supervised in the end (see 4.10), leaving the unlabeled pool
honest means we can revisit it later without redoing data prep.

### 5.4 Labeled-per-class sampling

`LABELED_PER_CLASS = 1000`. Two thousand labels total, balanced by class.
Mirrors the notebook's regime. Realistic for a freshly-launched product
where confirmed fraud cases are scarce.

### 5.5 Feature engineering

All from `features/engineer.py`:

| feature                  | rationale                                              |
|--------------------------|--------------------------------------------------------|
| `errorBalanceOrig/Dest`  | balance-equation residuals — the notebook's #1 signal  |
| `orig_zero_after`        | account emptied right after the transaction            |
| `dest_zero_before/after` | destination account state — mule-account fingerprint   |
| `hour_of_day`            | `step % 24` — captures circadian fraud patterns        |
| `day_of_week`            | `(step // 24) % 7`                                     |
| `amount_log`             | `log1p(amount)` — squashes the heavy tail              |
| `is_high_amount`         | flag at p99 of amount distribution                     |
| `is_transfer_or_cashout` | in PaySim, fraud occurs only in these two types        |

`is_transfer_or_cashout` is a cheap shortcut for the model — trees would
discover it anyway, but providing it explicitly tightens early splits and
reduces variance in the small-labeled regime.

### 5.6 Encoding `type`

`encode.py` maps the string `type` to `int8` (not just `int`).

- Casting to `int8` explicitly is important — without `.astype("int8")` the
  result of `Series.replace({str: int})` stays `object` dtype with int
  *values*. Round-tripping through parquet may then promote to `int64`, but
  the OneHotEncoder fitted on the original `object` dtype rejects the new
  numeric input. We hit this bug, fixed it permanently in `encode.py`.

OHE itself happens in the sklearn Pipeline (see 4.7), not in `encode.py`.

### 5.7 Preprocessing pipeline

`features/preprocess.py` builds a `ColumnTransformer`:

- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` on `type`
- `StandardScaler` on the raw + engineered numerics
- `passthrough` for the engineered binary flags
- `verbose_feature_names_out=False`, `.set_output(transform="pandas")`

Fit on `pd.concat([labeled, unlabeled])` so the scaler sees the true
marginal distribution rather than just the balanced 2k labeled set.

Why a Pipeline for an XGBoost model that doesn't need scaling? Because the
contract (`fit_preprocessor` / `transform`) is the only place that needs to
know about column types, and having a single object covers val + test +
future inference. Replacing the model with anything that *does* care about
scale (logistic regression, MLP, calibration head) becomes a no-op change.

### 5.8 XGBoost model

`training/model.py` wraps `XGBClassifier` with:

- `XGB_PARAMS` from config (n_estimators=400, max_depth=6, lr=0.08, etc.)
- `scale_pos_weight` **derived from the labeled class ratio at fit time**.

The notebook hardcodes `scale_pos_weight = 775`. We compute
`neg / pos` from the actual labels passed in. Tracks the data rather than
the dataset version.

We pick `eval_metric="aucpr"` — the right ranking metric under heavy
imbalance — even though we never actually pass an eval set; it's the metric
XGBoost will use internally if early stopping is ever enabled.

### 5.9 Evaluation

`evaluation/metrics.py` returns a `Report` dataclass with:

- `pr_auc`, `average_precision`, `roc_auc`
- precision / recall / f1 at the configured threshold
- `recall_at_p90` — the "what fraction of fraud do we catch while keeping
  precision ≥ 0.9" number, which maps directly to analyst capacity
- confusion counts (TN, FP, FN, TP)

The two we lean on: **PR-AUC** (overall ranking quality under imbalance) and
**recall@P=0.9** (operational reality). Accuracy/ROC are reported for
completeness but not load-bearing here.

### 5.10 Things we tried and dropped

These are documented because someone will try them again otherwise.

| attempt                              | result                                       | verdict |
|--------------------------------------|----------------------------------------------|---------|
| Iterative self-training with class-balanced top-q quantile pseudo-labels | PR-AUC 0.995, +400 FP, no recall gain        | drop    |
| Isotonic calibration on val          | useful only if a downstream stage needs calibrated probs; none does | drop |
| Cost-based threshold via `C_FN·FN + C_FP·FP` on val | optimal point converged to t≈0.5; baseline already did this well | drop |
| Isolation-Forest stacking feature    | `if_score` ranked #6 by gain; removing it dropped PR-AUC by ~0.0001 | drop |

The notebook positioned this as a semi-supervised problem; in practice the
engineered features make it solvable with 2k labels and plain XGBoost.
Self-training was net-negative because pseudo-labels added noise faster
than they added signal. We kept the modules just long enough to confirm
this, then deleted them.

### 5.11 Persistence

- Splits → `data/processed/{labeled_train,unlabeled_train,val,test}.parquet`
- Model artifacts → `model/artifacts/model.joblib`
  - `TrainedArtifacts` bundles: `preprocessor`, `model`, `threshold`,
    `feature_names`. No anomaly scorer anymore.
- Figures → `model/reports/figures/`
- Metrics → `model/reports/metrics/evaluation.json`

### 5.12 CLI

`python -m model.main {eda|prep|train|eval|all}`. Default (no subcommand)
runs `all`. Subcommands exist so we can re-run individual stages without
re-doing data prep, which takes a minute or two on the full CSV.

---

## 6. Current results

Trained on 2,000 labeled rows (1k per class) from the temporal train pool.
Evaluated on the temporal tail (`step` > p80, ~1.25M transactions).

| metric             | value     |
|--------------------|-----------|
| PR-AUC             | 0.9998    |
| Average Precision  | 0.9998    |
| ROC-AUC            | 1.0000    |
| Recall @ P=0.9     | 0.9995    |
| Precision @ t=0.5  | 0.9955    |
| Recall    @ t=0.5  | 0.9988    |
| F1        @ t=0.5  | 0.9972    |
| TP / FN / FP / TN  | 4245 / 5 / 19 / 1,244,467 |

Caveat: PaySim is synthetic and the balance-equation residuals
(`errorBalanceOrig`) are an almost-perfect leakage of the labelling rule.
Real-world fraud detection should not expect these numbers. They are
useful as a sanity ceiling and to show the pipeline behaves; they are not
useful as a prediction of production behaviour.

Top features by gain (latest run):

```
orig_zero_after           77.92
errorBalanceOrig          32.71
newbalanceOrig            18.53
is_transfer_or_cashout     8.85
type_3                     4.32
oldbalanceOrg              1.88
amount                     1.70
type_4                     1.68
amount_log                 1.23
dest_zero_after            1.01
```

---

## 7. How to run

```powershell
# end-to-end
.venv/Scripts/python.exe -m model.main all

# stage-by-stage
.venv/Scripts/python.exe -m model.main prep
.venv/Scripts/python.exe -m model.main train
.venv/Scripts/python.exe -m model.main eval
.venv/Scripts/python.exe -m model.main eda

# refresh the chart slides in docs/presentation.pptx
.venv/Scripts/python.exe scripts/update_charts.py
```

`update_charts.py` is idempotent on title prefix `[chart]`. If you rename a
chart slide manually it survives re-runs but the script can't replace it
either, so you'll get duplicates. Either keep the prefix or accept the
manual maintenance.

---

## 8. Presentation deck

`docs/presentation.pptx` is built from the teacher's P2 template. We kept
all 12 original sections (customer / problem / data / solution / model /
setup / results / stakeholders / challenges / unavailable data / ethical
risks) and added chart slides after `results` (confusion matrix, PR curve,
ROC curve, feature importance, score distribution). One extra slide
"model interpretation" sits between the chart block and the stakeholders
section.

The chart slides are regenerated from the latest model state by running
`scripts/update_charts.py` — do this whenever you retrain.

---

## 9. Open ends

- `model/api/` and `web/` are not wired. If we ever build them, the model
  artifact at `model/artifacts/model.joblib` plus
  `features/preprocess.transform` is the full inference contract.
- Probability calibration is not in the pipeline. If a downstream consumer
  needs calibrated scores (e.g. expected-value computations), drop an
  `IsotonicRegression` between `model.predict_proba` and the threshold.
- Concept drift is not monitored. In production we'd compare incoming
  feature distributions against the training-time distributions and
  retrain when they diverge — out of scope here.

---

## 10. Known quirks

- `BOXPLOT_COLUMNS` in `config.py` overlaps with `NUMERIC_COLUMNS` minus
  `step`. Kept separate because the EDA boxplots intentionally exclude
  `step` (it's not a magnitude).
- `analysis/distributions.py` still uses the legacy `TYPE_ENCODING` to
  label boxplots — it was written before the OHE Pipeline existed. Works
  fine; we just haven't reunited the two code paths.
- The dataclass on `Splits` returns frames already engineered. Anyone
  loading the parquets directly gets the engineered columns for free.
