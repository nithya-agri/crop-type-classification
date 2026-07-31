# Crop-Type Classification on PASTIS

A pixel-level crop-type classifier trained on the **PASTIS** Sentinel-2 dataset (a French
agricultural benchmark). The pipeline compresses each pixel's 46-step × 10-band Sentinel-2
time series into phenology-aware tabular features, then trains a **Random Forest** to predict
one of 18 land-cover / crop classes per pixel.

This README is the **reproduction guide** — it walks through every stage in order, from raw
data to final prediction maps. For the full methodology, design rationale, and results
discussion, see **[report.md](report.md)**.

---

## Results at a glance

Numbers from the latest run (held-out test set via `evaluate.py`):

| Metric | Value |
| :--- | :--- |
| Overall accuracy | **0.8217** |
| Macro-F1 | **0.4514** |
| Balanced accuracy | 0.4857 |
| Cohen's κ | 0.7774 |
| Mean IoU (mIoU) | 0.3872 |

Reference readouts: single-fold validation macro-F1 **0.4957**, aggregated 5-fold CV macro-F1
**0.4488**. The model is strong on abundant, phenologically-distinct crops (rapeseed, wheat,
corn, barley, soybean — per-class F1 0.85–0.94) and weak on rare / spectrally-ambiguous
classes. See [report.md §9](report.md) for the per-class breakdown and confusion analysis.

Confusion matrices and example prediction maps live in `outputs/figures/`.

---

## Repository layout

```
.
├── src/
│   ├── feature_engineering.py   # raw .npy patches → per-patch Parquet feature tables
│   ├── data_split.py            # patch-level train/val/test split from PASTIS folds
│   ├── data_loading.py          # load Parquet, filter, impute, assemble X/y (imported, not run directly)
│   ├── train.py                 # tune + train + evaluate RF (val + 5-fold CV); saves model
│   ├── recreate_model.py        # rebuild the saved model from known best params (skips the search)
│   ├── evaluate.py              # ONE-TIME held-out test evaluation of the saved model
│   └── visualize.py             # per-patch prediction maps (RGB · truth · pred · agreement)
├── outputs/
│   ├── splits/                  # {train,val,test}_patch_ids.txt          (shipped)
│   ├── processed_data/          # patch_*.parquet feature tables          (generated — NOT shipped)
│   ├── models/                  # rf_model.joblib (Git LFS), label_encoder.joblib, imputer.joblib (shipped)
│   └── figures/                 # confusion matrices + predictions/       (shipped)
├── exploration.ipynb            # EDA that motivated the feature choices
├── report.md                    # full technical report
└── requirements.txt
```

**What ships in the repo vs. what you generate:** the trained model, encoder, imputer, split
files, and figures are committed. The engineered `processed_data/*.parquet` files are **not**
(they are large and derived) — you regenerate them from the raw dataset in Step 2 below.

---

## Prerequisites

1. **Python 3.11** (developed on 3.11.7).
2. **Git LFS** — the trained model (`outputs/models/rf_model.joblib`) is stored via Git Large
   File Storage. Without LFS you'll clone a small text pointer instead of the real model.
3. **The PASTIS dataset subset.** The scripts expect a `PASTIS_subset/` directory with this
   layout (obtain PASTIS from the official benchmark — [VSainteuf/pastis-benchmark](https://github.com/VSainteuf/pastis-benchmark)):

   ```
   PASTIS_subset/
   ├── metadata.geojson          # patch metadata incl. Fold and acquisition dates
   ├── DATA_S2/
   │   └── S2_<patch_id>.npy      # one array per patch, shape (timesteps, 10 bands, H, W)
   └── ANNOTATIONS/
       └── <...patch_id...>.npy   # one (H, W) class map per patch (filename contains the patch id)
   ```

---

## Setup

```bash
# 1. Install Git LFS once on your machine, then clone (or pull LFS objects after cloning)
git lfs install
git clone <this-repo-url>
cd crop-type-classification
git lfs pull                      # fetches the real rf_model.joblib

# 2. Create an environment and install dependencies
python3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Point the code at your dataset

The dataset path is currently **hard-coded**. Before running, set it to wherever your
`PASTIS_subset/` lives in these three files:

| File | What to edit |
| :--- | :--- |
| `src/data_split.py` | `DATA_PATH = '.../PASTIS_subset/'` (near the top) |
| `src/feature_engineering.py` | `DATA_PATH = '.../PASTIS_subset/'` (inside `if __name__ == "__main__"`) |
| `src/visualize.py` | `DATA_PATH = '.../PASTIS_subset/'` (near the top) |

> `data_loading.py`, `train.py`, `recreate_model.py`, and `evaluate.py` only use relative
> `../outputs/...` paths, so they need no editing.

All scripts are run **from inside the `src/` directory** (their output paths are relative to it):

```bash
cd src
```

---

## Reproduce the results, step by step

Run these in order. Steps 1–2 require the raw dataset; steps 3–5 operate on the generated
Parquet features and the saved model.

### Step 1 — Create the train/val/test split

```bash
python data_split.py
```

Reads `metadata.geojson` and splits patches by PASTIS's own `Fold` field
(folds 1/2/3 → train, fold 4 → val, fold 5 → test — a spatially-disjoint, patch-level split
to prevent leakage). Writes:

```
../outputs/splits/train_patch_ids.txt   (66 patches)
../outputs/splits/val_patch_ids.txt     (18 patches)
../outputs/splits/test_patch_ids.txt    (18 patches)
```

*(These are already committed; running this reproduces them identically.)*

### Step 2 — Engineer features (raw patches → Parquet)

```bash
python feature_engineering.py
```

For every patch: filters the time series to a single crop season
(Sep 2018 → Aug 2019), applies the cloud/shadow heuristic, gap-fills, and computes the
**140-column** feature table (19 phenology/index features + 120 monthly raw-band means +
`cloud_frac`). Writes one Parquet file per patch:

```
../outputs/processed_data/patch_<id>.parquet
```

This is the step that produces the (un-shipped) `processed_data/` directory that every
later step depends on.

### Step 3 — Train the model

You have two options.

**Option A — full pipeline (tuning + training).** Runs a `RandomizedSearchCV`
(20 candidates × 3 grouped CV folds), fits the best Random Forest, evaluates on the
validation fold and via aggregated 5-fold CV, and saves everything.

```bash
python train.py
```

> ⚠️ **The hyperparameter search is slow** — the logged reference run took ~8.8 hours of
> wall-clock (`outputs/Crop_classification_model_training_output.txt`). Runtime is very
> hardware-dependent. XGBoost is kept as an optional challenger, gated off by default; to
> include it, change the last line of `train.py` to `main(run_xgb=True)`.

**Option B — skip the search (fast).** Rebuilds the exact saved artifacts using the best
hyperparameters already found (`n_estimators=100, min_samples_leaf=10, max_features=0.3,
max_depth=20`) with a single fit, and sanity-checks against the reported validation macro-F1:

```bash
python recreate_model.py
```

Either option writes:

```
../outputs/models/rf_model.joblib        # (Git LFS in the repo)
../outputs/models/label_encoder.joblib
../outputs/models/imputer.joblib
```

Option A additionally saves the validation and 5-fold CV confusion matrices to
`../outputs/figures/`.

> The imputer is fit on the **training split only** and saved, so evaluation and inference
> reuse the exact same fill values — no val/test leakage.

### Step 4 — Final held-out test evaluation

```bash
python evaluate.py
```

Loads the saved model + encoder + imputer and scores the **test fold once**. Prints overall
accuracy, per-class precision/recall/F1, per-class + mean IoU, Cohen's κ, balanced accuracy,
and the most-confused class pairs; saves the test confusion matrix to `../outputs/figures/`.
This reproduces the "Results at a glance" numbers above.

### Step 5 — Visualize predictions

```bash
python visualize.py
```

Auto-selects the best / worst / a few random test patches (by per-patch macro-F1) and writes a
4-panel figure for each — Sentinel-2 RGB · ground truth · RF prediction · agreement map — to
`../outputs/figures/predictions/`. To target specific patches instead:

```bash
python visualize.py --patch-ids 30003 30273 30549
```

---

## Notes & caveats

- **Determinism:** a fixed seed (`RANDOM_STATE = 42`) is used throughout. Exact metrics can
  still shift slightly with library versions / thread counts.
- **Reproducing without retraining:** the repo ships the trained model, so after Steps 1–2
  (which you need for the Parquet features) you can jump straight to Steps 4–5 using the
  committed model — no need to re-run the multi-hour search.
- **Cloud masking is a heuristic**, not the official Scene Classification Layer — see
  [report.md §4](report.md).
- **Rare classes** (e.g. Grapevine, Potatoes, Orchard) have very few pixels and are essentially
  unlearnable at this sample size; some fall entirely in the training split and can't be scored
  on test. This is expected — see [report.md §8–§9](report.md).

For the complete write-up (feature design, cloud heuristic, leakage controls, error analysis,
and recommended next steps), read **[report.md](report.md)**.
