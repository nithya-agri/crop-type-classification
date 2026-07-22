"""
Convert Sentinel-2 npy patches directly into a pixel-level feature matrix
(Parquet), without ever materializing a long-format table.

Parquet format is chosen to reduce the file size and optimise data loading.

Assumes, per patch:
    S2 array shape:    (no_of_ts, no_of_bands, H, W)
    class array shape: (1, H, W)

Output: one Parquet file per patch under `outputs`, each with columns:
    patch_id, pixel_id, class, <17 feature columns>
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

EPS = 1e-8


# =====================================================================
# 1) map band name -> index in the S2 array's band axis.
# =====================================================================
BAND_NAME_TO_IDX = {
    "B2": 0,   # Blue
    "B3": 1,    # Green
    "B4": 2,   # Red
    "B5": 3,   # Red Edge 1
    "B6": 4,   # Red Edge 2
    "B7": 5,   # Red Edge 3
    "B8": 6,   # NIR
    "B8A": 7,  # Narrow NIR
    "B11": 8,  # SWIR1
    "B12": 9   # SWIR2
}


def load_patch_dates(geojson_path):
    """
    Read the metadata.geojson file and build a lookup:
    ID_PATCH -> {observation_index: datetime object}

    The geojson stores dates as YYYYMMDD integers under
    properties['dates-S2'], keyed by observation index (as strings).
    """
    with open(geojson_path) as f:
        meta = json.load(f)

    patch_dates = {}
    for feature in meta['features']:
        props = feature['properties']
        patch_id = props['ID_PATCH']
        dates_raw = props['dates-S2']  # e.g. {"0": 20180920, "1": 20180930, ...}

        dates_parsed = {
            int(obs_idx): datetime.strptime(str(date_int), '%Y%m%d')
            for obs_idx, date_int in dates_raw.items()
        }
        patch_dates[patch_id] = dates_parsed

    return patch_dates


def compute_features(data, dates, band_idx=BAND_NAME_TO_IDX):
    """
    data: np.ndarray, shape (no_of_ts, no_of_bands, H, W)
    dates: array-like of length no_of_ts, datetime-like, in the SAME order
           as the ts axis of `data`.

    Returns: pd.DataFrame, one row per pixel, columns = 17 features.
    """
    no_of_ts, no_of_bands, h, w = data.shape
    n_pixels = h * w

    # reshape spatial dims -> pixel axis: (ts, band, pixel)
    flat = data.reshape(no_of_ts, no_of_bands, n_pixels).astype(np.float32)

    b2 = flat[:, band_idx["B2"], :]
    b4 = flat[:, band_idx["B4"], :]
    b5 = flat[:, band_idx["B5"], :]
    b8 = flat[:, band_idx["B8"], :]
    b11 = flat[:, band_idx["B11"], :]

    # ---- day offsets from first acquisition (for AUC / slope / timing) ----
    dates = pd.to_datetime(pd.Series(list(dates)))
    day_offsets = (dates - dates.iloc[0]).dt.days.values.astype(np.float64)  # (ts,)
    months = dates.dt.month.values  # (ts,)

    # ---- spectral indices, all shape (ts, pixels) ----
    EVI = 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1 + EPS)
    RECI = (b8 / (b5 + EPS)) - 1
    BR = b2 / (b4 + EPS)
    NDVI = (b8 - b4) / (b8 + b4 + EPS)
    NDBI = (b11 - b8) / (b11 + b8 + EPS)
    NDRE = (b8 - b5) / (b8 + b5 + EPS)
    SWIR_NIR = b11 / (b8 + EPS)

    def auc(y):
        return np.trapz(y, x=day_offsets, axis=0)

    def cyclic_month(argmax_idx):
        m = months[argmax_idx]
        sin = np.sin(2 * np.pi * m / 12)
        cos = np.cos(2 * np.pi * m / 12)
        return sin, cos

    def linreg_slope(y):
        # closed-form slope, vectorized across pixels
        x = day_offsets
        x_mean = x.mean()
        y_mean = y.mean(axis=0)
        Sxy = ((x[:, None] - x_mean) * (y - y_mean)).sum(axis=0)
        Sxx = ((x - x_mean) ** 2).sum()
        return Sxy / (Sxx + EPS)

    def n_spikes(y, k=1.0):
        # local maxima (strictly greater than both neighbors) above mean + k*std
        thresh = y.mean(axis=0) + k * y.std(axis=0)
        is_peak = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > thresh[None, :])
        return is_peak.sum(axis=0)

    feats = {}

    # --- EVI: min, max, auc ---
    feats["EVI_min"] = np.nanmin(EVI, axis=0)
    feats["EVI_max"] = np.nanmax(EVI, axis=0)
    feats["EVI_auc"] = auc(EVI)

    # --- RECI: auc, std, max ---
    feats["RECI_auc"] = auc(RECI)
    feats["RECI_std"] = np.nanstd(RECI, axis=0)
    feats["RECI_max"] = np.nanmax(RECI, axis=0)

    # --- BR: n_spikes, max ---
    feats["BR_n_spikes"] = n_spikes(BR)
    feats["BR_max"] = np.nanmax(BR, axis=0)

    # --- NDVI: auc, month_of_max (cyclic) ---
    ndvi_argmax = np.nanargmax(NDVI, axis=0)
    feats["NDVI_auc"] = auc(NDVI)
    feats["NDVI_month_max_sin"], feats["NDVI_month_max_cos"] = cyclic_month(ndvi_argmax)

    # --- NDBI: min ---
    feats["NDBI_min"] = np.nanmin(NDBI, axis=0)

    # --- NDRE: max, auc ---
    feats["NDRE_max"] = np.nanmax(NDRE, axis=0)
    feats["NDRE_auc"] = auc(NDRE)

    # --- SWIR_NIR: slope, days_max_to_min, month_of_max (cyclic), std ---
    swir_argmax = np.nanargmax(SWIR_NIR, axis=0)
    swir_argmin = np.nanargmin(SWIR_NIR, axis=0)
    feats["SWIR_NIR_slope"] = linreg_slope(SWIR_NIR)
    feats["SWIR_NIR_days_max_to_min"] = day_offsets[swir_argmin] - day_offsets[swir_argmax]
    feats["SWIR_NIR_month_max_sin"], feats["SWIR_NIR_month_max_cos"] = cyclic_month(swir_argmax)
    feats["SWIR_NIR_std"] = np.nanstd(SWIR_NIR, axis=0)

    feat_df = pd.DataFrame(feats)
    feat_df.insert(0, "pixel_id", np.arange(n_pixels))

    return feat_df


def filter_to_a_single_season(data, dates):
    # ---- Filter to a single crop season: Sept 2018 - Aug 2019 ----
    # Keeps only timesteps within the season window; drops the trailing
    # Sept-Oct 2019 dates that belong to the *next* season.
    SEASON_START = pd.Timestamp("2018-09-01")
    SEASON_END = pd.Timestamp("2019-08-31")

    dates_ts = pd.to_datetime(pd.Series(dates), format="%Y%m%d")
    keep_mask = (dates_ts >= SEASON_START) & (dates_ts <= SEASON_END)
    keep_idx = np.where(keep_mask.values)[0]

    data = data[keep_idx]  # subset the ts axis: (ts, bands, H, W) -> (ts_filtered, bands, H, W)
    dates = [dates[i] for i in keep_idx]

    return data, dates


def flag_void(feat_df):
    # ---- Flag void/unlabeled pixels instead of dropping them ----
    # Keeping them (with is_void=True) preserves the full 128x128 grid
    # per patch, so predicted-map visualizations aren't missing pixels.
    # Filter on is_void==False at train/eval time instead.
    VOID_CLASS_ID = 19
    feat_df["is_void"] = feat_df["class"] == VOID_CLASS_ID
    n_void = feat_df["is_void"].sum()
    if n_void:
        tqdm.write(f"patch {feat_df['patch_id'].values[0]}: {round(n_void/len(feat_df)*100.0, 2)} % void pixels flagged")
    return feat_df


def process_all_patches(s2_dir, class_dir, patch_dates, out_dir):
    """
    s2_dir: directory of npy files, one per patch, shape (ts, bands, H, W)
    class_dir: directory of npy files, one per patch, shape (H, W)
    patch_dates: dict {patch_id: {ts_index: date}}
    out_dir: where to write per-patch parquet files
    """
    os.makedirs(out_dir, exist_ok=True)
    s2_files = os.listdir(s2_dir)

    for f in tqdm(s2_files):
        patch_id = int(f.split("_")[-1].split(".")[0])

        s2_path = os.path.join(s2_dir, f)
        data = np.load(s2_path)  # (ts, bands, H, W)

        dates_dict = patch_dates.get(patch_id)
        no_of_ts = data.shape[0]
        dates = [dates_dict[t] for t in range(no_of_ts)]

        data, dates = filter_to_a_single_season(data, dates)

        feat_df = compute_features(data, dates)

        # match class file for this patch
        class_files = [cf for cf in os.listdir(class_dir) if str(patch_id) in cf]
        class_path = os.path.join(class_dir, class_files[0])
        class_arr = np.load(class_path).flatten()  # (H*W,)

        feat_df["class"] = class_arr
        feat_df.insert(0, "patch_id", patch_id)

        feat_df = flag_void(feat_df)

        out_path = os.path.join(out_dir, f"patch_{patch_id}.parquet")
        feat_df.to_parquet(out_path, index=False)


# ---------------------------------------------------------------
# Usage:

DATA_PATH = '/Users/nithya/Desktop/Personal/solafume/Agri-Assignment_01/PASTIS_subset/'
metadata_path = os.path.join(DATA_PATH, 'metadata.geojson')
s2_folder_path = os.path.join(DATA_PATH, 'DATA_S2')
anno_folder_path = os.path.join(DATA_PATH, 'ANNOTATIONS')

patch_dates = load_patch_dates(metadata_path)

process_all_patches(
    s2_dir=s2_folder_path,
    class_dir=anno_folder_path,
    patch_dates=patch_dates,   # your existing dict
    out_dir="../outputs/processed_data",
)
