"""
Create a reproducible train/val/test split at the PATCH level, using the
predefined `Fold` field from PASTIS's metadata.geojson.

Why patch-level (not pixel-level): pixels within a patch are spatially
correlated, so splitting individual pixels randomly leaks information
between train/test and inflates accuracy. Splitting whole patches avoids
this, and using PASTIS's own folds ensures the split is spatially disjoint
by design (that's what the folds were built for).

Split used here:
    Folds 1, 2, 3  -> train
    Fold 4         -> val   (for hyperparameter / feature tuning)
    Fold 5         -> test  (touched only once, at the very end)

Patch IDs for each split are saved to text files so the split is
reproducible across runs/sessions without re-deriving it.
"""

import json
import os
DATA_PATH = '/Users/nithya/Desktop/Personal/solafume/Agri-Assignment_01/PASTIS_subset/'
METADATA_PATH = os.path.join(DATA_PATH, 'metadata.geojson')
OUT_DIR = "../outputs/splits"

TRAIN_FOLDS = [1, 2, 3]
VAL_FOLDS = [4]
TEST_FOLDS = [5]


def load_fold_patch_map(metadata_path):
    with open(metadata_path) as f:
        data = json.load(f)

    fold_to_patches = {}
    for feat in data["features"]:
        fold = feat["properties"]["Fold"]
        patch_id = feat["properties"]["ID_PATCH"]
        fold_to_patches.setdefault(fold, []).append(patch_id)

    return fold_to_patches


def build_split(fold_to_patches, folds):
    patch_ids = []
    for fold in folds:
        patch_ids.extend(fold_to_patches.get(fold, []))
    return sorted(patch_ids)


def save_patch_ids(patch_ids, path):
    with open(path, "w") as f:
        for pid in patch_ids:
            f.write(f"{pid}\n")


def load_patch_ids(path):
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def split_dataset():
    os.makedirs(OUT_DIR, exist_ok=True)

    fold_to_patches = load_fold_patch_map(METADATA_PATH)

    train_ids = build_split(fold_to_patches, TRAIN_FOLDS)
    val_ids = build_split(fold_to_patches, VAL_FOLDS)
    test_ids = build_split(fold_to_patches, TEST_FOLDS)

    # sanity check: no overlap, all patches accounted for
    assert not (set(train_ids) & set(val_ids)), "train/val overlap!"
    assert not (set(train_ids) & set(test_ids)), "train/test overlap!"
    assert not (set(val_ids) & set(test_ids)), "val/test overlap!"

    total_expected = sum(len(v) for v in fold_to_patches.values())
    total_actual = len(train_ids) + len(val_ids) + len(test_ids)
    assert total_actual == total_expected, (
        f"patch count mismatch: {total_actual} != {total_expected}"
    )

    save_patch_ids(train_ids, os.path.join(OUT_DIR, "train_patch_ids.txt"))
    save_patch_ids(val_ids, os.path.join(OUT_DIR, "val_patch_ids.txt"))
    save_patch_ids(test_ids, os.path.join(OUT_DIR, "test_patch_ids.txt"))

    print(f"Train: {len(train_ids)} patches (folds {TRAIN_FOLDS})")
    print(f"Val:   {len(val_ids)} patches (fold {VAL_FOLDS})")
    print(f"Test:  {len(test_ids)} patches (fold {TEST_FOLDS})")
    print(f"\nSaved to {OUT_DIR}/train_patch_ids.txt, val_patch_ids.txt, test_patch_ids.txt")


if __name__ == "__main__":
    split_dataset()
