# ---------------------------------------------------------------
# Usage later, when loading the feature Parquet dataset:
#
# train_ids = load_patch_ids("splits/train_patch_ids.txt")
# val_ids = load_patch_ids("splits/val_patch_ids.txt")
# test_ids = load_patch_ids("splits/test_patch_ids.txt")
#
# import pyarrow.dataset as ds
# dataset = ds.dataset("path/to/parquet_features", format="parquet")
# df = dataset.to_table().to_pandas()
#
# train_df = df[df["patch_id"].isin(train_ids) & ~df["is_void"]]
# val_df   = df[df["patch_id"].isin(val_ids) & ~df["is_void"]]
# test_df  = df[df["patch_id"].isin(test_ids) & ~df["is_void"]]
# ---------------------------------------------------------------

"""
Load the engineered feature Parquet dataset, filter by train/val/test patch
split, exclude void pixels, impute missing values, and return clean X/y
arrays ready for model training.

Pulls together:
    - feature Parquet files written by feature_engineering.py
    - patch id split files written by data_split.py

NaN handling
------------
Two different things can produce non-finite values in the feature columns,
and they're handled differently:

  - Inf: genuine numerical corruption (e.g. a division that escaped the
    EPS guard). Rare, and not meaningful data -- rows with Inf are dropped.
  - NaN: *expected*, structural missing data -- a pixel with no valid
    (unmasked) acquisition in a given calendar month, or a pixel that was
    cloud/shadow-masked for the entire season (see feature_engineering.py's
    compute_cloud_shadow_mask). This is common (esp. across 120 monthly
    columns) and dropping every row that hits it would silently and
    non-randomly gut the dataset -- patches with more cloud cover would
    lose disproportionately more pixels. Instead these are imputed with
    per-column medians computed from the TRAINING split only, then applied
    to train/val/test alike, to avoid leaking val/test statistics into
    the imputation.
"""

import os
import numpy as np
from sklearn.impute import SimpleImputer
import pyarrow.dataset as ds

from feature_engineering import BAND_NAME_TO_IDX, SEASON_MONTHS, MONTH_ABBR

PARQUET_DIR = "../outputs/processed_data"
SPLIT_DIR = "../outputs/splits"

# ---- seasonal features (unchanged) ----
SEASONAL_FEATURE_COLS = [
    "EVI_min", "EVI_max", "EVI_auc",
    "RECI_auc", "RECI_std", "RECI_max",
    "BR_n_spikes", "BR_max",
    "NDVI_auc", "NDVI_month_max_sin", "NDVI_month_max_cos",
    "NDBI_min",
    "NDRE_max", "NDRE_auc",
    "SWIR_NIR_slope", "SWIR_NIR_days_max_to_min",
    "SWIR_NIR_month_max_sin", "SWIR_NIR_month_max_cos", "SWIR_NIR_std",
]

# ---- monthly raw-band means (10 bands x 12 months = 120 cols) ----
# Built from feature_engineering.py's own constants so this list can't
# silently drift out of sync with what that script actually writes.
MONTHLY_BAND_MEAN_COLS = [
    f"{band_name}_{MONTH_ABBR[month]}_mean"
    for band_name in BAND_NAME_TO_IDX
    for month in SEASON_MONTHS
]

# ---- cloud/shadow QA feature ----
# Included by default -- lets the model learn "less reliable when
# cloud_frac is high" rather than being blind to data quality. Drop it
# from FEATURE_COLS below if you'd rather it not influence predictions.
CLOUD_QA_COLS = ["cloud_frac"]

FEATURE_COLS = SEASONAL_FEATURE_COLS + CLOUD_QA_COLS + MONTHLY_BAND_MEAN_COLS

LABEL_COL = "class"


def load_patch_ids(path):
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_full_dataset(parquet_dir):
    """Read all per-patch parquet files into a single DataFrame."""
    dataset = ds.dataset(parquet_dir, format="parquet")
    df = dataset.to_table().to_pandas()
    return df


def report_bad_values(df, feature_cols):
    """Flag rows with Inf in feature columns (genuine numerical corruption,
    e.g. a division that escaped the EPS guard). These are dropped -- NOT
    the same as NaN, which is expected missing data handled by imputation
    in prepare_splits_with_imputation() below."""
    vals = df[feature_cols].to_numpy(dtype=np.float64)
    inf_mask = np.isinf(vals).any(axis=1)
    n_bad = inf_mask.sum()
    if n_bad:
        print(f"  Warning: {n_bad} rows ({100 * n_bad / len(df):.2f}%) have "
              f"Inf in feature columns (numerical corruption) and will be dropped.")
    return inf_mask


def prepare_split(df, patch_ids, feature_cols, label_col):
    """Filter a dataframe down to a given patch split, drop void pixels and
    any rows with Inf in feature values (NaN is left in place for
    imputation), and split into X/y. Returns raw (pre-imputation) X."""
    split_df = df[df["patch_id"].isin(patch_ids)]
    split_df = split_df[~split_df["is_void"]]

    inf_mask = report_bad_values(split_df, feature_cols)
    split_df = split_df[~inf_mask]

    X = split_df[feature_cols].to_numpy(dtype=np.float32)
    y = split_df[label_col].to_numpy()
    groups = split_df["patch_id"].to_numpy()  # for grouped CV later

    return X, y, groups


def report_missing(X, split_name, feature_cols):
    n_nan = np.isnan(X).any(axis=1).sum()
    if n_nan:
        print(f"  {split_name}: {n_nan:,} rows ({100 * n_nan / len(X):.2f}%) "
              f"have at least one NaN feature (expected -- missing months / "
              f"fully cloud-masked pixels); will be median-imputed.")


def load_train_val_test(parquet_dir=PARQUET_DIR, split_dir=SPLIT_DIR,
                         feature_cols=FEATURE_COLS, label_col=LABEL_COL):
    train_ids = load_patch_ids(os.path.join(split_dir, "train_patch_ids.txt"))
    val_ids = load_patch_ids(os.path.join(split_dir, "val_patch_ids.txt"))
    test_ids = load_patch_ids(os.path.join(split_dir, "test_patch_ids.txt"))

    print("Loading full feature dataset from parquet...")
    df = load_full_dataset(parquet_dir)
    print(f"  Total rows (all patches, incl. void): {len(df):,}")

    X_train, y_train, g_train = prepare_split(df, train_ids, feature_cols, label_col)
    X_val, y_val, g_val = prepare_split(df, val_ids, feature_cols, label_col)
    X_test, y_test, g_test = prepare_split(df, test_ids, feature_cols, label_col)

    print(f"\nTrain: {X_train.shape[0]:,} pixels from {len(train_ids)} patches")
    print(f"Val:   {X_val.shape[0]:,} pixels from {len(val_ids)} patches")
    print(f"Test:  {X_test.shape[0]:,} pixels from {len(test_ids)} patches")

    report_missing(X_train, "Train", feature_cols)
    report_missing(X_val, "Val", feature_cols)
    report_missing(X_test, "Test", feature_cols)

    # ---- impute NaN using TRAIN statistics only, applied to all splits ----
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    print("\nTrain class distribution:")
    vals, counts = np.unique(y_train, return_counts=True)
    for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
        print(f"  class {v}: {c:,} ({100*c/len(y_train):.1f}%)")

    return {
        "train": (X_train, y_train, g_train),
        "val": (X_val, y_val, g_val),
        "test": (X_test, y_test, g_test),
        "imputer": imputer,  # save alongside the model for consistent inference later
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    load_train_val_test()