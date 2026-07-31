# **Crop-Type Classification on PASTIS — Technical Report**

## **1. Objective & Scope**

The goal is a pixel-level crop-type classifier trained on the PASTIS Sentinel-2 dataset (a French agricultural benchmark). The pipeline turns raw Sentinel-2 image time series into a per-pixel tabular feature matrix, then trains tree-ensemble classifiers to predict one of 18 land-cover/crop classes per pixel.

**Pipeline stages (in src/):**
*   `exploration.ipynb` — EDA that motivated the feature choices.
*   `feature_engineering.py` — raw .npy patches → per-patch Parquet feature tables.
*   `data_split.py` — reproducible patch-level train/val/test split from PASTIS folds.
*   `data_loading.py` — load Parquet, filter, impute, assemble X/y arrays.
*   `train.py` — tune, train, and evaluate RF/XGBoost (validation + aggregated 5-fold CV).
*   `recreate_model.py` — one-shot refit of the RF/label-encoder/imputer from the already-found best hyperparameters (skips the multi-hour search; used to regenerate the saved artifacts).
*   `evaluate.py` — **final held-out test-set evaluation** of the saved tuned model, with extended metrics (accuracy, balanced accuracy, Cohen's kappa, per-class + mean IoU, top confused pairs).
*   `visualize.py` — per-patch 4-panel prediction maps (RGB · ground truth · prediction · agreement) with auto-selection of best/worst/typical test patches.

**Model artifacts & git LFS:** the fitted model is persisted to `outputs/models/rf_model.joblib` alongside its `label_encoder.joblib` and `imputer.joblib`. Because the serialized forest is a large binary (well beyond a comfortable plain-git blob), it is tracked with **Git Large File Storage** — `.gitattributes` declares `outputs/models/rf_model.joblib filter=lfs diff=lfs merge=lfs -text`, so git stores a lightweight text pointer in history and the binary payload in LFS. This keeps the repository lightweight and diffable while still versioning the trained model. `recreate_model.py` exists as the deterministic fallback path to regenerate the same artifacts from the saved best hyperparameters if the LFS object is unavailable.

---

## **2. Area of Interest (AOI) & Assumptions**

**AOI:** France. This drives the single most important temporal decision in the pipeline.
*   Crop season = September 2018 → August 2019. `filter_to_a_single_season()` keeps only timesteps in [2018-09-01, 2019-08-31] and drops the trailing Sept–Oct 2019 acquisitions. Those later dates belong to the next growing season, during which a farmer may plant a different crop on the same parcel — including them would "pollute" a pixel's signal with two different crops under one label.

**Grid:** 102 patches, each 128×128 px, 46 timesteps, 10 Sentinel-2 bands (confirmed in EDA). 
*   PASTIS Sentinel-2 is at 10 m spatial resolution.
*   Band mapping (`BAND_NAME_TO_IDX`): B2 Blue, B3 Green, B4 Red, B5/B6/B7 Red-Edge 1/2/3, B8 NIR, B8A Narrow-NIR, B11 SWIR1, B12 SWIR2.
*   Label set: 20 nominal classes (0 Background, 1 Meadow, 2 Soft winter wheat, 3 Corn, 4 Winter barley, 5 Winter rapeseed, 6 Spring barley, 7 Sunflower, 8 Grapevine, 9 Beet, 10 Winter triticale, 11 Winter durum wheat, 12 Fruits/veg/flowers, 13 Potatoes, 14 Leguminous fodder, 15 Soybeans, 16 Orchard, 17 Mixed cereal, 18 Sorghum, 19 Void).

**Key data assumptions baked into code:**
a. PASTIS ships no cloud mask (no SCL / cloud-probability layer), so cloud/shadow contamination is estimated with a home-grown heuristic (§4).
b. Class 9 (Beet) is absent from the entire subset (train/val/test). Labels are therefore label-encoded to a contiguous range so the model only ever sees the 18 present classes.
c. Void pixels (class 19, ~8.6% of the grid) are flagged (`is_void=True`), not deleted, so the full 128×128 grid survives to Parquet and predicted-map visualizations aren't missing pixels. They are filtered out at train/eval time.

---

## **3. Feature Engineering (feature_engineering.py)**

The design philosophy, is to compress a 46-step × 10-band time series into phenology-aware summary statistics rather than feed raw time series to a tabular model. Feature choices were made based on visual interpretation of the spectral signatures and index patterns of different classes in the notebook.

### **3.1 Feature set (140 columns total)**

**A. 19 seasonal (phenology) features derived from spectral indices:**

| Index | Features | Rationale (what it captures) |
| :--- | :--- | :--- |
| **EVI** | min, max, AUC | Canopy vigor magnitude & season-integrated greenness (EVI resists saturation/soil vs NDVI). |
| **RECI (Red-Edge Chlorophyll)** | AUC, std, max | Chlorophyll content & its seasonal variability — good for separating cereals. |
| **BR (Blue/Red ratio)** | n_spikes, max | Transient brightness events; spike count is a soil/residue/flowering signal. |
| **NDVI** | AUC, month-of-max (cyclic sin/cos) | Total greenness + timing of peak growth — the strongest phenological discriminator. |
| **NDBI** | min | Built-up / bare-soil contrast (non-vegetation separation). |
| **NDRE** | max, AUC | Red-edge greenness, sensitive to canopy nitrogen. |
| **SWIR/NIR** | slope, days from max→min, month-of-max (cyclic), std | Moisture/senescence dynamics and their timing. |

**B. 120 monthly raw-band means** — mean of each of the 10 raw bands within each of the 12 season months (`{band}_{mon}_mean`). These were a later addition (commit "added monthly band values") giving the model a coarse but complete monthly spectral trajectory alongside the hand-crafted indices.

**C. 1 QA feature** — `cloud_frac`, the fraction of a pixel's season flagged as cloud/shadow. Intentionally included so the model can learn to distrust unreliable pixels rather than be blind to data quality (can be dropped from `FEATURE_COLS` if undesired).

### **3.2 Design decisions worth highlighting (from comments)**
*   **Fixed month ordering** (`SEASON_MONTHS = [9,10,…,8]`): guarantees every patch's Parquet has the identical 120 monthly columns even if a patch lacks an acquisition in some month (that column is simply NaN). Prevents schema drift across patches.
*   **Cyclic month encoding** (sin/cos of month-of-max): month 12 and month 1 are adjacent in reality; encoding them as raw integers would falsely place December and January maximally far apart. Sin/cos preserves the circular distance.
*   **Gap-filling before AUC/slope/spikes** (`_fill_gaps`): cloud-masked observations become NaN; these are linearly interpolated along the time axis using actual day-offsets (not row position) so masked gaps don't zero out or distort integral/slope/peak calculations. Pixels NaN at every timestep stay NaN.
*   **AUC via `np.trapezoid` over real day offsets** — season-integrated index value, robust to irregular revisit spacing.
*   **Safe `nanargmax`/`nanargmin` wrappers** avoid exceptions on fully-masked (all-NaN) pixels and return an explicit "all-NaN" mask so those pixels are set to NaN rather than a bogus index.
*   **`EPS = 1e-8` guard** on every ratio/normalized-difference denominator to avoid divide-by-zero; residual Inf that escapes this guard is treated as corruption and dropped downstream.

---

## **4. Cloud / Shadow Masking (the biggest data-quality lever)**

Because PASTIS has no official cloud mask, `compute_cloud_shadow_mask()` implements a lightweight spectral + temporal heuristic on raw reflectance (0–10000 scale). The comments are explicit that this is "an approximation, not a substitute for the official Scene Classification Layer" and should be tuned against visually-inspected patches.

Two complementary checks (OR'd together):

*   **Spectral heuristic**
    *   **Cloud:** visible brightness `mean(B2,B3,B4) > 2000` (~0.20 reflectance) AND `B11 < 1.5·B4`. The SWIR condition is the clever part: true clouds are spectrally flat, whereas bright soil/sand/rooftops get proportionally brighter in SWIR — so the SWIR/Red ratio separates cloud from other bright surfaces.
    *   **Shadow:** overall brightness `mean(B2,B3,B4,B8) < 400` AND `B8` (NIR) `< 500`. The NIR condition rules out dense healthy canopy (dark in visible but bright in NIR).
*   **Rolling-window NDVI drop** — a sharp single-timestep NDVI dip (>0.45) relative to a local rolling median (window = 5 timesteps), not the whole-season median. The documented reasoning: crop NDVI follows a smooth green-up → peak → senescence curve, so comparing against a season-long median would misflag legitimate low-NDVI stages (early bare soil, late senescence/harvest) as cloud. A true cloud/haze event is a transient dip a local comparison isolates cleanly.

**Measured impact:** mean `cloud_frac` over non-void pixels is ~0.18 — i.e. roughly 18% of observations are being masked on average. Flagged observations are set to NaN and excluded from all aggregations.

---

## **5. Train / Val / Test Split (data_split.py)**

*   **Patch-level split**, using PASTIS's own Fold field: folds 1/2/3 → train (66 patches), fold 4 → val (18), fold 5 → test (18).
*   **Why patch-level, not pixel-level (documented):** pixels within a patch are spatially autocorrelated; a random pixel split would leak near-identical neighboring pixels across train/test and inflate accuracy. Splitting whole patches — and reusing PASTIS folds, which were constructed to be spatially disjoint — prevents this.
*   **Split IDs** are persisted to text files for reproducibility; sanity-check asserts confirm no overlap and full patch accounting.
*   **Consistency:** the same grouped-CV discipline is carried into hyperparameter search (`StratifiedGroupKFold` on `patch_id`), so no patch spans both sides of any CV fold either.

---

## **6. Data Loading, Imputation & Leakage Control (data_loading.py)**

Void filtered, Inf dropped, NaN imputed — three distinct treatments:
*   **Inf** = genuine numerical corruption (a division that escaped the EPS guard). Rare, meaningless → rows dropped.
*   **NaN** = expected structural missingness (a pixel with no valid acquisition in a month, or fully cloud-masked all season). Common across 120 monthly columns — in the current run **~13–15% of rows carry at least one NaN feature** (train 14.96%, val 14.63%, test 13.50%). Dropping every NaN row would non-randomly gut cloudier patches, so NaN is median-imputed instead.
*   **Void** (class 19) filtered via `~is_void`.

**Leakage-safe imputation:** the median `SimpleImputer` is fit on the training split only and applied to val/test — no val/test statistics leak into imputation. The fitted imputer is returned alongside the data so inference later uses identical fill values.
`groups` (`patch_id`) is threaded through for grouped CV.

---

## **7. Modeling & Why Random Forest Was Chosen (train.py)**

### **7.1 Models**
*   **Random Forest** (primary; always runs, saved to disk).
*   **XGBoost** (challenger; gated behind `run_xgb=False` — currently off by default).

### **7.2 Why RF is the selected model**
The code structure and defaults make RF the operational choice, and it is well-justified for this problem:
*   **Tabular, mixed-scale features** (indices, AUCs, cyclic terms, raw band means) — tree ensembles handle heterogeneous scales without normalization and need no feature standardization.
*   **Robust** to the ~18% imputed / noisy features and to the cloud-heuristic imperfections; splits are threshold-based and tolerant of outliers.
*   **Native multiclass** + `class_weight="balanced"` handles the severe imbalance directly (see §8).
*   **Strong precedent** — the cited literature (Tufail et al.; Tuğaç et al.) uses RF/SVM on Sentinel-2 time series for crop mapping; RF is the field-standard baseline.
*   **Fewer, more forgiving hyperparameters** than XGBoost and no early-stopping/learning-rate tuning, so it reaches a solid result with less search. XGBoost is kept as an optional upgrade path rather than the default.
*   **Interpretability & low operational risk** — feature importances, deterministic with a fixed seed, trivial to persist (joblib) with its label encoder and imputer.
*   **Infrastructure requirement: RF doesn't require GPU for training, unlike deep learning models.

The empirical comparison (§9) also does not show XGBoost decisively beating RF on the hard minority classes, so the simpler model is retained.

### **7.3 Tuning protocol**
*   `RandomizedSearchCV` (20 iterations) over sensible grids:
    *   **RF:** `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
    *   **XGB:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.
*   **Scoring = f1_macro** — deliberately optimizes unweighted per-class F1 so the dominant Background/Meadow classes don't drown out rare crops.
*   **CV = StratifiedGroupKFold** grouped by patch — the same anti-leakage principle as the main split.
*   **Core-oversubscription guard:** base estimators use `n_jobs=1` while the search uses `n_jobs=-1`; `n_jobs` is restored to `-1` on the final fitted estimator.

**Selected hyperparameters (from the latest search):** `n_estimators=100`, `min_samples_leaf=10`, `max_features=0.3`, `max_depth=20`, with best in-search CV macro-F1 = **0.4588**. The randomized search over 20 candidates × 3 folds (60 fits) took ~8.8 h wall-clock (31,782 s); `recreate_model.py` hard-codes these params so the model can be rebuilt with a single fit instead of re-running the search.

### **7.4 Evaluation protocol (two-tier, with a documented rationale)**
*   **Single fixed-fold validation** (fold 4). Fast, but has a known artifact: classes 8, 13, 16, 18 have zero support in the val fold (and 8/13/16 in test), so their rows are empty — you literally cannot measure them there.
*   **Aggregated 5-fold CV** over the "dev" set (train+val combined) via `cross_val_predict` + `StratifiedGroupKFold`. Every dev patch serves as validation exactly once, so every class present anywhere in dev gets evaluated, curing the missing-class artifact. This re-fits the fixed tuned hyperparameters per fold (it does not re-tune).
*   **Final held-out test evaluation** now runs *once*, in a separate `evaluate.py` (not in `train.py`), after tuning/model-selection was locked in — exactly the discipline the comments prescribe. It loads the already-saved model + encoder + imputer (reusing the exact training-time impute statistics rather than refitting), and reports overall accuracy, per-class precision/recall/F1, a row-normalized confusion matrix, **per-class + mean IoU (mIoU)**, **Cohen's kappa**, **balanced accuracy**, and the **top confused (true→predicted) class pairs**. Classes appearing in test but never in training would raise rather than be silently mishandled (a guard on the label encoder).
*   **Confusion matrices** are row-normalized (per-class recall), chosen for readability under imbalance. Saved to `outputs/figures/` as `rf_val_confusion_matrix_v3.png`, `rf_cv_confusion_matrix_v3.png`, and `rf_test_confusion_matrix_v3.png`.
*   **Qualitative prediction maps** (`visualize.py`) reconstruct each patch's spatial grid and render a 4-panel figure — Sentinel-2 RGB (least-cloudy acquisition) · ground truth · RF prediction · agreement map — with per-patch accuracy and macro-F1 in the title. Left to auto-select, it ranks test patches by per-patch macro-F1 and renders the best, worst, and a random-sampled spread (saved under `outputs/figures/predictions/`).

---

## **8. Class Imbalance, Sample Size & Label Quality**

Measured non-void class distribution (1,527,081 pixels):

| Class                   | %      | Class                    | %                                   |
|:------------------------|:-------|:-------------------------|:------------------------------------|
| **0 Background**        | 33.1   | **4 Winter barley**      | 3.8                                 |
| **1 Meadow**            | 19.4   | **10 Winter triticale**  | 1.0                                 |
| **2 Soft winter wheat** | 13.0   | **14 Leguminous fodder** | 0.83                                |
| **3 Corn**              | 12.9   | **7 Sunflower**          | 0.63                                |
| **15 Soybeans**         | 8.0    | **17 Mixed cereal**      | 0.34                                |
| **5 Winter rapeseed**   | 6.1    | **18 Sorghum**           | 0.33                                
| **6 Spring barley**     | 0.23   | **12 F.V.F.**            | 0.21                                |
| **8 Grapevine**         | 0.09   | **11 W.durum**           | 0.05                                |
| **13 Potatoes**         | 0.01   | **16 Orchard**           | 0.005                               |

*   **Extreme imbalance** (>6000:1 top-to-bottom). The rarest classes are decimal-fraction slivers: Orchard = 75 pixels, Potatoes = 158, Winter durum wheat = 725, Grapevine = 1,345 across the entire dataset.
*   **Handling in code:** RF `class_weight="balanced"`; XGB `compute_sample_weight("balanced")`; macro-F1 scoring. These raise minority recall but cannot manufacture signal from ~100 pixels concentrated in one or two patches.
*   **Label-quality / structural caveats:**
    *   "Background" (33%) and "Meadow" (19%) are not crops — they dominate and act as attractor classes for anything ambiguous (see §9).
    *   **Spatial concentration:** because splits are patch-level, a rare class living in one patch may be entirely in one split — its evaluation is fragile regardless of model quality. In the current split, **Grapevine (8), Potatoes (13), and Orchard (16) fall entirely inside the training patches (zero support in the test fold)**, so the final test evaluation literally cannot score them; they are only measurable in the aggregated CV (where each is 0.00). Their whole-dataset pixel counts (75 / 158 / 1,345 for orchard / potatoes / grapevine) confirm how little there is to learn from.
    *   **Void** ~8.6% of the grid is unlabeled and correctly excluded.
    *   **Composite/ambiguous label definitions** ("Mixed cereal", "Fruits/vegetables/flowers", "Leguminous fodder") are inherently spectrally heterogeneous — noisy targets by construction.

---

## **9. Results & Interpretation**
*(Recalls read from the row-normalized confusion matrices; diagonal = recall. Test numbers are from the one-time held-out `evaluate.py` run.)*

### **9.0 Headline numbers**

| Readout | Macro-F1 | Notes |
| :--- | :--- | :--- |
| **Final held-out test** (`evaluate.py`) | **0.4514** | accuracy **0.8217**, balanced accuracy **0.4857**, Cohen's κ **0.7774**, **mIoU 0.3872** (over 16/18 classes with support) |
| Single-fold validation (fold 4) | 0.4957 | overall accuracy 0.83; under-covers rare classes (8/13/16/18 empty) |
| Aggregated 5-fold CV (dev set) | 0.4488 | most reliable per-class readout; every present class scored |
| In-search best CV (tuning) | 0.4588 | RandomizedSearchCV selection score |

The high overall accuracy (0.82) alongside a much lower macro-F1 (0.45) is the signature of severe imbalance: the model is excellent on the abundant, phenologically-distinct classes and near-useless on the rare ones, and macro-F1 / balanced accuracy / mIoU deliberately refuse to let the big classes hide that. **Cohen's κ = 0.78 confirms the accuracy is genuine agreement, not an artifact of predicting the majority class.**

### **9.1 Random Forest — where it performs (5-fold CV & test recall)**

**Performed well (recall — CV / test):**
*   **Winter rapeseed (5):** 0.96 / 0.95 — best class. Rapeseed's uniquely bright yellow-flowering signal (visible-reflectance spike in spring) is spectrally unmistakable (test IoU 0.88, F1 0.94).
*   **Soft winter wheat (2):** 0.91 / 0.94, **Corn (3):** 0.90 / 0.91, **Winter barley (4):** 0.88 / 0.92 — abundant, well-timed phenology (test IoU 0.84 / 0.81 / 0.82).
*   **Soybeans (15):** 0.85 / 0.92 — well-sampled summer crop with a strong NDVI peak (test IoU 0.74).
*   **Meadow (1):** 0.81 / 0.86, **Background (0):** 0.78 / 0.75 — the two dominant non-crop classes.
*   **Sunflower (7):** 0.61 / 0.70 — moderate; distinctive but brief flowering (test IoU 0.57).

This is a marked improvement over the earlier model version (e.g. wheat/corn/barley now 0.9+ vs. ~0.65–0.69 before, rapeseed 0.96 vs 0.84), reflecting the tuned hyperparameters and the full monthly-band feature set.

**Performed poorly / collapsed (CV recall):**
*   **Grapevine (8), Winter durum wheat (11), Potatoes (13), Orchard (16):** all 0.00 — the smallest classes are essentially never recovered (8/13/16 also have zero test support, so test can't score them at all).
*   **Fruits/veg/flowers (12):** 0.00, **Mixed cereal (17):** 0.00, **Sorghum (18):** 0.00, **Leguminous fodder (14):** 0.32, **Winter triticale (10):** 0.45, **Spring barley (6):** 0.40 — partial-to-total failure on mid/low-support classes.

### **9.2 Specific confusions and their likely causes** *(fractions from the 5-fold CV matrix)*

| True class | Confused with | CV recall | Probable reason |
| :--- | :--- | :--- | :--- |
| **Winter triticale (10)** | Soft winter wheat (2) 0.46 | 0.45 | Triticale is a wheat×rye hybrid — near-identical winter-cereal phenology; same sowing/harvest window, same NDVI shape. |
| **Winter durum wheat (11)** | Soft winter wheat (2) 0.93 | 0.00 | Two wheat varieties — spectrally almost the same species; only ~725 px total. |
| **Mixed cereal (17)** | wheat (2) 0.52, rapeseed (5) 0.18, triticale (10) 0.16 | 0.00 | Composite label — literally a mix of cereals, so it scatters into its constituents. |
| **Spring barley (6)** | wheat (2) 0.40, Background (0) 0.11 | 0.40 | Winter/spring cereal overlap; tiny sample (test support 153, missed entirely). |
| **Sorghum (18)** | Corn (3) 0.53, Soybeans (15) 0.30 | 0.00 | Both tall C4 summer crops with matching green-up/senescence timing. |
| **Sunflower (7)** | Corn (3) 0.15, Soybeans (15) 0.10, Background (0) 0.10 | 0.61 | Overlapping summer season; distinctive flowering is brief and easily missed at ~10-day revisit. |
| **Leguminous fodder (14)** | Meadow (1) 0.44, Background (0) 0.13 | 0.32 | Grassy/leguminous forage looks like managed grassland (meadow). |
| **Grapevine (8)** | Background (0) 0.88, Meadow (1) 0.12 | 0.00 | Sparse woody rows → mixed pixels dominated by bare soil/background at 10 m; ~1,345 px. |
| **Orchard (16)** | Meadow (1) 0.65, Background (0) 0.35 | 0.00 | Trees over grass/soil — pixel signal is the understory; 75 px total. |
| **Potatoes (13)** | Sunflower (7) 0.51, Soybeans (15) 0.32, Corn (3) 0.10 | 0.00 | 158 px; no learnable signal, absorbed by other summer crops. |
| **Fruits/veg/flowers (12)** | Background (0) 0.62, Corn (3) 0.12, Soybeans (15) 0.11, rapeseed (5) 0.09 | 0.00 | Heterogeneous horticulture, tiny sample. |

The **largest raw-count test confusions** (`evaluate.py`, true→predicted) confirm this at scale: Background→Meadow (18,582) and Meadow→Background (6,320) dominate simply because those classes are huge; then Corn→Soybeans (1,735), Triticale→wheat (1,303), FVF→Meadow (1,244), Leguminous→Meadow (862).

**Two systematic patterns emerge:**
*   **Phenological near-twins collapse into each other** — the winter-cereal cluster (wheat / durum wheat / triticale / mixed cereal / spring barley) and the summer-C4 cluster (corn / sorghum) are the dominant *crop-vs-crop* error structure. This is a feature-separability ceiling, not just imbalance: even a perfect classifier struggles when two classes share a growth calendar and canopy structure. Durum wheat → soft wheat at 0.93 is the extreme case.
*   **Rare classes leak into Background/Meadow** — when a class has too few pixels to define a decision region (grapevine, orchard, FVF), balanced weighting still can't stop it being absorbed by the two huge non-crop attractor classes.

### **9.3 Single-fold validation vs CV vs test**
The RF validation matrix shows classes 8, 13, 16, 18 as empty rows (zero support) — the exact artifact the CV evaluation was built to fix. Val-fold recall is broadly consistent with the final test for the common classes (e.g. rapeseed 0.95 val / 0.95 test, wheat 0.93 / 0.94, corn 0.91 / 0.91), and the test macro-F1 (0.4514) lands between the optimistic single-fold val (0.4957) and the conservative aggregated CV (0.4488) — i.e. the CV number was an honest, slightly-pessimistic predictor of held-out performance, confirming the evaluation design is trustworthy. Note classes 8/13/16 are unmeasurable on test (zero support), so CV remains the definitive readout for the rare classes.

### **9.4 RF vs XGBoost**
XGBoost remained gated off (`run_xgb=False`) for this run, so there are no fresh XGB numbers here. In earlier exploratory comparisons XGBoost was broadly comparable — strong on rapeseed, the same minority-class collapse, and the same cereal↔cereal / corn↔sorghum confusions — without a decisive win on the classes that matter, so RF is retained as the simpler operational model. XGBoost stays as an optional upgrade path rather than the default.

### **9.5 Qualitative prediction maps (`visualize.py`)**
Per-patch 4-panel figures were generated for a spread of best/worst/typical test patches (saved under `outputs/figures/predictions/`, e.g. patches 30003, 30052, 30273, 30335, 30371, 30527, 30549). They make the quantitative story visible spatially: field interiors of the well-sampled crops are predicted cleanly (large green regions in the agreement panel), while errors concentrate at field edges (mixed pixels) and on whole parcels belonging to a confusable/rare class — matching the confusion structure above rather than appearing as random salt-and-pepper noise.

---

## **10. Strengths & Limitations of the Approach**

### **Strengths**
*   **Leakage discipline is exemplary:** patch-level split from spatially-disjoint folds, grouped CV in tuning, train-only imputation statistics. Reported accuracy is honest.
*   **Phenology-aware features** (timing-of-peak, AUCs, cyclic month encoding) capture the temporal structure that actually separates crops — better than naive per-date reflectance.
*   **Explicit, well-reasoned handling of missing data** (Inf vs NaN vs void, local-median cloud detection, gap interpolation on real day offsets).
*   **Imbalance handled honestly** via macro-F1 + balanced weights, and evaluation designed to surface rare-class failure rather than hide it behind overall accuracy.
*   **Reproducible & operationalizable** — fixed seeds, saved model + encoder + imputer, schema-stable Parquet.

### **Limitations**
*   **Cloud masking is a heuristic**, not SCL-validated; both misses (residual haze) and false positives (misflagged bare soil/senescence) propagate into features. ~18% masked is substantial.
*   **Pixel-level model ignores spatial context** — no texture, no field geometry, no neighbor smoothing; mixed pixels at field edges and for row crops (vines/orchards) are systematically misassigned.
*   **Hand-crafted temporal summaries discard fine temporal detail** a sequence model (LSTM/Transformer/TempCNN) could exploit.
*   **Rare classes are effectively unlearnable** at current sample sizes; balanced weighting cannot fix ~100-pixel classes concentrated in single patches.
*   **Small AOI (102 patches)** limits both training signal and evaluation stability for minority classes.
*   **10 m resolution** is a hard floor for narrow/sparse land covers (vineyards, orchards).

---

## **11. Effects of Resolution, Clouds, Missing Data & Seasonal Timing**

*   **Spatial resolution (10 m):** fine for large annual field crops (corn, wheat, rapeseed, soybean — which score best), but too coarse for row/tree crops (grapevine, orchard) whose pixels are dominated by soil/understory → their collapse into Background/Meadow is partly a resolution artifact.
*   **Cloud contamination:** ~18% of observations masked; the heuristic's imperfections inject noise into every index and monthly-mean feature. Cloudier pixels lose more observations, making their interpolated features less reliable — which is exactly why `cloud_frac` is exposed to the model.
*   **Missing observations:** structural NaNs (no acquisition in a month, or fully-masked pixels) are median-imputed. Imputation biases those pixels toward the population median, weakening discrimination — heavier for cloudy patches.
*   **Seasonal timing:** the deliberate Sept 2018–Aug 2019 window is essential and correct — it isolates one French cropping cycle and avoids mixing the next season's crop into a pixel's label. But it also means the discriminative window for each crop is short; crops sharing a calendar (winter cereals; corn/sorghum) become hard to separate, which dominates the error structure.

---

## **12. Recommended Next Steps**

### **Data / labels**
1. Adopt or approximate a real cloud mask — use s2cloudless / a Sentinel-2 cloud-probability product, or validate the current heuristic against a few hand-labeled patches and tune thresholds; log precision/recall of the mask itself.
2. Expand the AOI / patch count to give minority classes enough pixels and enough patches to survive a patch-level split (fixing the missing-class-in-fold problem at the source).
3. Audit composite labels (Mixed cereal, Fruits/veg/flowers, Leguminous fodder) — consider merging or re-defining; they are noisy by construction.

### **Features**
4. Add discriminators for the confused clusters: senescence-timing and harvest-date features to separate winter cereals; explicit flowering-detection features for rapeseed/sunflower; red-edge inflection (S2REP/MTCI, already prototyped in EDA) for cereal species separation.
5. Add texture / spatial context (GLCM, neighborhood statistics) to help row/tree crops.
6. Run feature-importance / selection to prune redundant monthly-mean columns (140 features, many collinear).

### **Modeling & evaluation**
7. Class-imbalance remedies beyond weighting: targeted oversampling/SMOTE for mid-tier classes, or a two-stage hierarchy (crop vs non-crop → crop family → species) matching the natural confusion structure.
8. Try a temporal deep model (TempCNN / LSTM / Transformer, or PSE+TAE as in the PASTIS paper) that consumes the full sequence — likely the biggest single accuracy lever, and a fair benchmark against the RF baseline.
9. Enable and complete the XGBoost path, and add post-hoc spatial smoothing (majority filter / CRF) on predicted maps to remove salt-and-pepper noise.
10. Re-split so the rare classes (grapevine/potatoes/orchard) actually appear in test, since they are currently unmeasurable there.
11. Report calibrated confidence and consider abstaining (predict "uncertain") on high-cloud_frac or low-margin pixels rather than forcing a low-quality label.

> **Bottom line:** The pipeline is methodologically sound and leakage-safe, and RF is a well-justified, appropriately simple choice. On the held-out test set it reaches **82% overall accuracy (κ = 0.78)** with **macro-F1 0.45 / mIoU 0.39** — performing strongly on abundant, phenologically-distinct crops (rapeseed, wheat, corn, barley, soybean; per-class F1 0.85–0.94) and failing predictably on (a) rare classes with too few/too-concentrated pixels and (b) phenological near-twins (winter-cereal varieties; corn/sorghum). The gap between high accuracy and modest macro-F1 is the imbalance signature, honestly surfaced rather than hidden. The largest gains will come from better cloud handling, more (and better-distributed) data for minority classes, spatial/temporal context in the features/model, and a hierarchical treatment of the confusable crop families.
> 
>
> 
### References:
1. Assessing feature extraction, selection, and classification combinations for crop mapping using Sentinel-2 time series: A case study in northern Italy. Rahat Tufail , Patrizia Tassinari , Daniele Torreggiani Department of Agricultural and Food Sciences, University of Bologna, Viale Fanin 48, 40127, Bologna, Italy. 
2. https://dibyendudeb.com/random-forest-crop-classification-sentinel-2-python/#Part_2_%E2%80%94_Feature_Engineering_for_Crop_Classification
3. https://github.com/Surv-Lukmon/Crop-Classification
4. Classification of Agricultural Crops with Random Forest and Support Vector Machine Algorithms Using Sentinel-2 and Landsat-8 Images Murat Güven Tuğaç1 * Fatih Fehmi Şimşek2 * Harun Torunlar1
1 Ministry of Agriculture and Forestry, Soil, Fertilizer and Water Resources Central Research Institute, Ankara/Türkiye. 2 Ministry of Agriculture and Forestry, General Directorate of Agricultural Reform, Ankara/Türkiye

---
