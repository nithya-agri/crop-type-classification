"""
Visualize model predictions on selected test patches.

For each selected patch, produces a 4-panel figure:
    Sentinel-2 RGB composite | Ground-truth label map | Model prediction | Agreement map
plus per-patch pixel accuracy and macro F1 in the title.

Usage:
    python visualize_predictions.py
        # no --patch-ids given: auto-picks best/worst/random test patches
        # by per-patch macro F1, so you get a spread of good and bad examples

    python visualize_predictions.py --patch-ids 12 45 88
        # visualize specific patch IDs instead
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import joblib

from feature_engineering import (
    BAND_NAME_TO_IDX, compute_cloud_shadow_mask, load_patch_dates,
    filter_to_a_single_season,
)
from data_loading import load_train_val_test, FEATURE_COLS, PARQUET_DIR

# ---- paths --
DATA_PATH = '/Users/nithya/Desktop/Personal/solafume/Agri-Assignment_01/PASTIS_subset/'
S2_DIR = os.path.join(DATA_PATH, 'DATA_S2')
METADATA_PATH = os.path.join(DATA_PATH, 'metadata.geojson')
RF_MODEL_PATH = "../outputs/models/rf_model.joblib"
LABEL_ENCODER_PATH = "../outputs/models/label_encoder.joblib"
IMPUTER_PATH = "../outputs/models/imputer.joblib"
OUT_DIR = "../outputs/figures/predictions"

VOID_CLASS_ID = 19
NO_PRED_SENTINEL = -1  # marks a pixel where no prediction was made (void)


# =====================================================================
# grid reconstruction / color mapping
# =====================================================================
def reconstruct_grid(pixel_ids, values, h, w, fill_value=np.nan):
    """Scatter a flat (pixel_id, value) series back into its (h, w) spatial
    grid. pixel_id follows the same row-major flatten order used when the
    patch was originally flattened in feature_engineering.compute_features."""
    grid = np.full(h * w, fill_value, dtype=np.float32)
    grid[np.asarray(pixel_ids, dtype=int)] = values
    return grid.reshape(h, w)


def rgb_stretch(band_2d, low_pct=2, high_pct=98):
    """Simple percentile contrast stretch for display (raw reflectance
    values are usually far too dark/flat to view directly)."""
    lo, hi = np.percentile(band_2d, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((band_2d - lo) / (hi - lo), 0, 1)


def build_class_colormap(all_classes):
    """Fixed, distinguishable color per class, plus dedicated colors for
    void pixels and for 'no prediction made' (so both are visually
    distinct from any real class)."""
    base_colors = plt.cm.tab20.colors  # 20 distinct discrete RGB colors
    color_map = {}
    for i, c in enumerate(sorted(all_classes)):
        r, g, b = base_colors[i % len(base_colors)]
        color_map[c] = (r, g, b, 1.0)
    color_map[VOID_CLASS_ID] = (0.85, 0.85, 0.85, 1.0)  # light gray
    color_map[NO_PRED_SENTINEL] = (1.0, 1.0, 1.0, 1.0)  # white
    return color_map


def grid_to_rgb(class_grid, color_map):
    h, w = class_grid.shape
    out = np.ones((h, w, 4))
    for c, color in color_map.items():
        out[class_grid == c] = color
    return out


# =====================================================================
# Sentinel-2 RGB composite: pick the least cloud/shadow-affected
# available acquisition, reusing the same cloud detector from
# feature_engineering.py so "least cloudy" means the same thing here as
# it did during feature computation.
# =====================================================================
def select_least_cloudy_timestep(s2_data, band_idx=BAND_NAME_TO_IDX):
    no_of_ts, no_of_bands, h, w = s2_data.shape
    flat = s2_data.reshape(no_of_ts, no_of_bands, h * w).astype(np.float32)
    cloud_mask = compute_cloud_shadow_mask(flat, band_idx)  # (ts, pixels)
    frac_per_ts = cloud_mask.mean(axis=1)
    best_ts = int(np.argmin(frac_per_ts))
    return best_ts, float(frac_per_ts[best_ts])


def build_rgb_composite(s2_data, ts_idx, band_idx=BAND_NAME_TO_IDX):
    r = s2_data[ts_idx, band_idx["B4"]].astype(np.float32)
    g = s2_data[ts_idx, band_idx["B3"]].astype(np.float32)
    b = s2_data[ts_idx, band_idx["B2"]].astype(np.float32)
    return np.stack([rgb_stretch(r), rgb_stretch(g), rgb_stretch(b)], axis=-1)


# =====================================================================
# per-patch metrics + auto-selection of interesting patches
# =====================================================================
def compute_per_patch_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1


def select_patches_for_visualization(g_test, y_test_enc, y_pred_enc,
                                      n_best=2, n_worst=2, n_random=2,
                                      random_state=42):
    """Rank test patches by per-patch macro F1, using bulk predictions
    already computed over the whole test set (no re-predicting per patch),
    so you get a spread of best/worst/typical examples automatically."""
    df = pd.DataFrame({"patch_id": g_test, "y_true": y_test_enc, "y_pred": y_pred_enc})
    rows = []
    for pid, group in df.groupby("patch_id"):
        acc, f1 = compute_per_patch_metrics(group["y_true"], group["y_pred"])
        rows.append((pid, acc, f1, len(group)))
    metrics_df = pd.DataFrame(rows, columns=["patch_id", "accuracy", "macro_f1", "n_pixels"])
    metrics_df = metrics_df.sort_values("macro_f1").reset_index(drop=True)

    worst = metrics_df.head(n_worst)["patch_id"].tolist()
    best = metrics_df.tail(n_best)["patch_id"].tolist()
    remaining = metrics_df[~metrics_df["patch_id"].isin(worst + best)]
    n_random = min(n_random, len(remaining))
    random_sample = (
        remaining.sample(n=n_random, random_state=random_state)["patch_id"].tolist()
        if n_random > 0 else []
    )

    return {"best": best, "worst": worst, "random": random_sample}, metrics_df


# =====================================================================
# main per-patch visualization
# =====================================================================
def visualize_patch(patch_id, s2_dir, patch_dates, rf, imputer, label_encoder,
                     feature_cols, parquet_dir, out_dir, tag=""):
    patch_parquet = os.path.join(parquet_dir, f"patch_{patch_id}.parquet")
    df = pd.read_parquet(patch_parquet)

    n_pixels = int(df["pixel_id"].max()) + 1
    side = int(round(np.sqrt(n_pixels)))
    h = w = side

    # ---- ground truth (full grid, void included as its own class) ----
    gt_grid = reconstruct_grid(df["pixel_id"], df["class"], h, w, fill_value=VOID_CLASS_ID)

    # ---- predict only the labeled (non-void) pixels ----
    non_void = ~df["is_void"].to_numpy()
    X = df.loc[non_void, feature_cols].to_numpy(dtype=np.float32)
    X = np.where(np.isinf(X), np.nan, X)  # treat rare Inf same as missing data
    if imputer is not None:
        X = imputer.transform(X)
    y_pred_enc = rf.predict(X)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    pred_values_full = np.full(len(df), NO_PRED_SENTINEL, dtype=np.float32)
    pred_values_full[non_void] = y_pred
    pred_grid = reconstruct_grid(df["pixel_id"], pred_values_full, h, w, fill_value=NO_PRED_SENTINEL)

    y_true_non_void = df.loc[non_void, "class"].to_numpy()
    acc, f1 = compute_per_patch_metrics(y_true_non_void, y_pred)

    # ---- Sentinel-2 RGB composite from the least-cloudy acquisition ----
    s2_path = os.path.join(s2_dir, f"S2_{patch_id}.npy")
    s2_data = np.load(s2_path)
    dates_dict = patch_dates.get(patch_id, {})
    no_of_ts = s2_data.shape[0]
    dates = [dates_dict[t] for t in range(no_of_ts) if t in dates_dict]
    if len(dates) == no_of_ts:
        s2_season, dates_season = filter_to_a_single_season(s2_data, dates)
    else:
        s2_season, dates_season = s2_data, dates

    ts_idx, cloud_frac_ts = select_least_cloudy_timestep(s2_season)
    rgb = build_rgb_composite(s2_season, ts_idx)
    acq_date = dates_season[ts_idx].strftime("%Y-%m-%d") if dates_season else "unknown"

    # ---- color-coded label grids ----
    all_classes = set(df["class"].unique()) | set(np.unique(y_pred))
    color_map = build_class_colormap(all_classes)
    gt_rgb = grid_to_rgb(gt_grid, color_map)
    pred_rgb = grid_to_rgb(pred_grid, color_map)

    # ---- agreement / error map ----
    flat_gt, flat_pred = gt_grid.flatten(), pred_grid.flatten()
    labeled = flat_gt != VOID_CLASS_ID
    error_flat = np.full(h * w, -1, dtype=np.int8)  # -1 = void
    error_flat[labeled & (flat_gt == flat_pred)] = 1   # correct
    error_flat[labeled & (flat_gt != flat_pred)] = 0   # incorrect
    error_grid = error_flat.reshape(h, w)
    error_colors = {1: (0.20, 0.65, 0.20, 1.0),   # green = correct
                    0: (0.80, 0.10, 0.10, 1.0),   # red   = incorrect
                    -1: (0.85, 0.85, 0.85, 1.0)}  # gray  = void
    error_rgb = grid_to_rgb(error_grid, error_colors)

    # ---- overall cloud burden for this patch (context, not per-timestep) ----
    mean_cloud_frac = df.loc[non_void, "cloud_frac"].mean() if "cloud_frac" in df.columns else float("nan")

    # ---- plot ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Sentinel-2 RGB\n({acq_date}, {cloud_frac_ts*100:.1f}% masked that date)")
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground truth\n(gray = void)")
    axes[2].imshow(pred_rgb)
    axes[2].set_title("RF prediction\n(white = no prediction / void)")
    axes[3].imshow(error_rgb)
    axes[3].set_title("Agreement\n(green=correct, red=wrong, gray=void)")
    for ax in axes:
        ax.axis("off")

    tag_str = f" - {tag}" if tag else ""
    fig.suptitle(
        f"Patch {patch_id}{tag_str}  |  pixel accuracy: {acc:.3f}  |  macro F1: {f1:.3f}  |  "
        f"mean season cloud_frac: {mean_cloud_frac:.3f}\n\n"
    )
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"patch_{patch_id}_prediction.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}  (acc={acc:.3f}, macro_f1={f1:.3f})")
    return acc, f1


# =====================================================================
def main(patch_ids=None, n_best=2, n_worst=2, n_random=2):
    rf = joblib.load(RF_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    imputer = joblib.load(IMPUTER_PATH) if os.path.exists(IMPUTER_PATH) else None
    if imputer is None:
        print(f"WARNING: no saved imputer found at {IMPUTER_PATH}; will fall back to "
              f"refitting one from the current train split if needed (see earlier caveat "
              f"about drift if parquet/splits have changed since training).")

    patch_dates = load_patch_dates(METADATA_PATH)

    tags = {}
    if patch_ids is None:
        print("No --patch-ids given; auto-selecting best/worst/random test patches "
              "by per-patch macro F1...")
        data = load_train_val_test(imputer=imputer)
        X_test, y_test, g_test = data["test"]
        if imputer is None:
            imputer = data["imputer"]
        y_test_enc = label_encoder.transform(y_test)
        y_pred_enc = rf.predict(X_test)

        selections, metrics_df = select_patches_for_visualization(
            g_test, y_test_enc, y_pred_enc, n_best=n_best, n_worst=n_worst, n_random=n_random,
        )
        print("\nPer-patch macro F1 summary (test set):")
        print(metrics_df.sort_values("macro_f1").to_string(index=False))

        for pid in selections["worst"]:
            tags[pid] = "worst performing"
        for pid in selections["best"]:
            tags[pid] = "best performing"
        for pid in selections["random"]:
            tags[pid] = "random sample"
        patch_ids = list(tags.keys())
    else:
        tags = {pid: "" for pid in patch_ids}

    print(f"\nVisualizing {len(patch_ids)} patch(es): {patch_ids}")
    for pid in patch_ids:
        visualize_patch(
            pid, S2_DIR, patch_dates, rf, imputer, label_encoder,
            FEATURE_COLS, PARQUET_DIR, OUT_DIR, tag=tags.get(pid, ""),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-ids", type=int, nargs="+", default=None,
                        help="Specific patch IDs to visualize. If omitted, "
                             "auto-selects best/worst/random test patches.")
    parser.add_argument("--n-best", type=int, default=2)
    parser.add_argument("--n-worst", type=int, default=2)
    parser.add_argument("--n-random", type=int, default=2)
    args = parser.parse_args()
    main(patch_ids=args.patch_ids, n_best=args.n_best, n_worst=args.n_worst, n_random=args.n_random)