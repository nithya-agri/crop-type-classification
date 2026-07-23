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
split, exclude void pixels, and return clean X/y arrays ready for model
training.

Pulls together:
    - feature Parquet files written by feature_engineering.py
    - patch id split files written by data_split.py
"""

import os
import numpy as np
import pyarrow.dataset as ds

PARQUET_DIR = "../outputs/processed_data"
SPLIT_DIR = "../outputs/splits"

# Must match the feature columns produced by feature_engineering.py
FEATURE_COLS = [
    "EVI_min", "EVI_max", "EVI_auc",
    "RECI_auc", "RECI_std", "RECI_max",
    "BR_n_spikes", "BR_max",
    "NDVI_auc", "NDVI_month_max_sin", "NDVI_month_max_cos",
    "NDBI_min",
    "NDRE_max", "NDRE_auc",
    "SWIR_NIR_slope", "SWIR_NIR_days_max_to_min",
    "SWIR_NIR_month_max_sin", "SWIR_NIR_month_max_cos", "SWIR_NIR_std",
]

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
    """Check for NaN/Inf in feature columns (can arise from divisions by
    near-zero denominators, e.g. NDBI/NDVI/etc. on flat or missing pixels)."""
    bad_mask = ~np.isfinite(df[feature_cols].values).all(axis=1)
    n_bad = bad_mask.sum()
    if n_bad:
        print(f"  Warning: {n_bad} rows ({100*n_bad/len(df):.2f}%) have NaN/Inf "
              f"in feature columns and will be dropped.")
    return bad_mask


def prepare_split(df, patch_ids, feature_cols, label_col):
    """Filter a dataframe down to a given patch split, drop void pixels and
    any rows with non-finite feature values, and split into X/y."""
    split_df = df[df["patch_id"].isin(patch_ids)]
    split_df = split_df[~split_df["is_void"]]

    bad_mask = report_bad_values(split_df, feature_cols)
    split_df = split_df[~bad_mask]

    X = split_df[feature_cols].to_numpy(dtype=np.float32)
    y = split_df[label_col].to_numpy()
    groups = split_df["patch_id"].to_numpy()  # for grouped CV later

    return X, y, groups


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

    print("\nTrain class distribution:")
    vals, counts = np.unique(y_train, return_counts=True)
    for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
        print(f"  class {v}: {c:,} ({100*c/len(y_train):.1f}%)")

    return {
        "train": (X_train, y_train, g_train),
        "val": (X_val, y_val, g_val),
        "test": (X_test, y_test, g_test),
    }


if __name__ == "__main__":
    load_train_val_test()
