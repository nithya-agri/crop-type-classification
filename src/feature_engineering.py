"""
Convert Sentinel-2 npy patches directly into a pixel-level feature matrix
(Parquet), without ever materializing a long-format table.

Parquet format is chosen to reduce the file size and optimise data loading.

Assumes, per patch:
    S2 array shape:    (no_of_ts, no_of_bands, H, W)
    class array shape: (1, H, W)

Cloud/shadow-impacted observations are detected with a lightweight
spectral + temporal-outlier heuristic (PASTIS ships no cloud mask) and
excluded from all aggregations; see compute_cloud_shadow_mask().

Output: one Parquet file per patch under `outputs`, each with columns:
    patch_id, pixel_id, class, cloud_frac, <19 seasonal feature columns>,
    <monthly raw-band mean columns: 10 bands x 12 months = 120 columns>
"""
import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from tqdm import tqdm

EPS = 1e-8

# ---- Cloud / shadow masking thresholds ----
# PASTIS ships no cloud mask (no SCL / cloud-probability layer), so this is
# a lightweight heuristic run directly on the raw reflectance bands (0-10000
# scale). Tune against a few visually-inspected patches before trusting it
# at scale -- it's an approximation, not a substitute for the official
# Scene Classification Layer.
CLOUD_BRIGHTNESS_THRESH = 2000  # 0.20 reflectance; mean(B2,B3,B4) above this -> bright-suspect
SHADOW_DARKNESS_THRESH = 400    # 0.04 reflectance; mean(B2,B3,B4,B8) below this -> dark-suspect
NIR_WATER_THRESH = 500          # 0.05 reflectance; low NIR characteristic of water/shadow
NDVI_DROP_THRESH = 0.45         # one-sided drop (vs. local rolling median) flagging cloud/haze interference
NDVI_ROLLING_WINDOW = 5         # timesteps in the rolling median window (per pixel, along time axis)


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

# Fixed month ordering for the Sept 2018 - Aug 2019 crop season. Using a
# fixed list (rather than whatever months happen to appear in a given
# patch's dates) guarantees every patch's output parquet has the exact
# same set of monthly-mean columns, even if a patch is missing an
# acquisition in some month (that column is just NaN for that patch).
SEASON_MONTHS = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
MONTH_ABBR = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
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


def compute_rolling_ndvi_drop(ndvi, window_size=NDVI_ROLLING_WINDOW, drop_thresh=NDVI_DROP_THRESH):
    """
    ndvi: np.ndarray, shape (ts, pixels)

    Flags a sharp, single-timestep NDVI drop relative to that pixel's own
    *local* temporal neighborhood (rolling median), rather than its whole-
    season median. This matters because crop NDVI follows a smooth
    green-up -> peak -> senescence curve: comparing against a season-long
    median would misflag genuine low-NDVI stages (early bare soil, late
    senescence/harvest) as cloud. A true cloud/haze event is a transient
    dip that the next clear observation recovers from, so it stands out
    sharply against local neighbors even while riding on a normal seasonal
    trend -- comparing locally is what makes this distinguish cloud from
    real phenology.

    Returns: boolean array, shape (ts, pixels).
    """
    # median_filter's window only spans the time axis (size 1 on the pixel
    # axis keeps every pixel independent); mode='nearest' repeats the edge
    # value so the first/last timesteps still get a sensible local median.
    rolling_median = median_filter(ndvi, size=(window_size, 1), mode="nearest")
    return (rolling_median - ndvi) > drop_thresh


def compute_cloud_shadow_mask(flat, band_idx):
    """
    flat: np.ndarray, shape (ts, bands, pixels) -- raw reflectance, as built
          at the top of compute_features.

    Returns: boolean array, shape (ts, pixels). True marks a
    (timestep, pixel) observation as cloud-, shadow-, or haze-impacted and
    a candidate for exclusion from feature aggregation.

    Combines two checks:
      1. Spectral heuristic: bright visible bands with a relatively flat
         (not SWIR-elevated) spectrum -> cloud; uniformly dark visible +
         low NIR -> shadow. The SWIR ratio (B11 vs B4) is what separates
         true cloud from other bright surfaces like soil/sand/rooftops,
         which get proportionally brighter in SWIR rather than flatter.
      2. Rolling-window NDVI drop: cloud/haze suppresses NDVI toward 0 as
         a transient, single-timestep dip. Comparing against a local
         temporal median (not the whole-season median) catches that dip
         without misflagging genuine, gradual green-up/senescence stages.
    """
    b2 = flat[:, band_idx["B2"], :]
    b3 = flat[:, band_idx["B3"], :]
    b4 = flat[:, band_idx["B4"], :]
    b8 = flat[:, band_idx["B8"], :]
    b11 = flat[:, band_idx["B11"], :]

    # ---- 1) spectral heuristic ----
    visible_brightness = (b2 + b3 + b4) / 3.0
    # cloud: bright in visible, and SWIR not disproportionately elevated
    # relative to red (rules out bright soil/urban, which get brighter in SWIR)
    is_cloud = (visible_brightness > CLOUD_BRIGHTNESS_THRESH) & (b11 < b4 * 1.5)

    overall_brightness = (b2 + b3 + b4 + b8) / 4.0
    # shadow: uniformly dark visible+NIR, and NIR specifically low
    # (rules out dense healthy canopy, which is dark in visible but bright in NIR)
    is_shadow = (overall_brightness < SHADOW_DARKNESS_THRESH) & (b8 < NIR_WATER_THRESH)

    spectral_mask = is_cloud | is_shadow

    # ---- 2) rolling-window NDVI drop ----
    ndvi = (b8 - b4) / (b8 + b4 + EPS)
    ndvi_drop = compute_rolling_ndvi_drop(ndvi)

    return spectral_mask | ndvi_drop


def _safe_nanargmax(arr, axis=0):
    """np.nanargmax that doesn't raise on all-NaN columns (fully-masked
    pixels). Returns (index_array, all_nan_mask)."""
    all_nan = np.all(np.isnan(arr), axis=axis)
    filled = np.where(np.isnan(arr), -np.inf, arr)
    idx = np.argmax(filled, axis=axis)
    return idx, all_nan


def _safe_nanargmin(arr, axis=0):
    """np.nanargmin that doesn't raise on all-NaN columns. Returns
    (index_array, all_nan_mask)."""
    all_nan = np.all(np.isnan(arr), axis=axis)
    filled = np.where(np.isnan(arr), np.inf, arr)
    idx = np.argmin(filled, axis=axis)
    return idx, all_nan


def compute_features(data, dates, band_idx=BAND_NAME_TO_IDX):
    """
    data: np.ndarray, shape (no_of_ts, no_of_bands, H, W)
    dates: array-like of length no_of_ts, datetime-like, in the SAME order
           as the ts axis of `data`.

    Returns: pd.DataFrame, one row per pixel, columns = 19 seasonal
             features + 120 monthly raw-band mean features (10 bands x
             12 months).
    """
    no_of_ts, no_of_bands, h, w = data.shape
    n_pixels = h * w

    # reshape spatial dims -> pixel axis: (ts, band, pixel)
    flat = data.reshape(no_of_ts, no_of_bands, n_pixels).astype(np.float32)

    # ---- cloud / shadow masking (see compute_cloud_shadow_mask) ----
    cloud_mask = compute_cloud_shadow_mask(flat, band_idx)  # (ts, pixels)
    n_flagged = cloud_mask.sum()
    if n_flagged:
        pct = 100.0 * n_flagged / cloud_mask.size
        tqdm.write(f"  cloud/shadow masked {pct:.2f}% of (timestep, pixel) observations")
    flat[np.broadcast_to(cloud_mask[:, None, :], flat.shape)] = np.nan

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

    def _fill_gaps(y):
        # Linearly interpolate NaN gaps along the time axis, per pixel
        # (using actual day offsets, not just row position), so cloud-
        # masked observations don't zero out or distort AUC/slope/peak
        # calculations. Pixels that are NaN at every timestep stay NaN.
        df = pd.DataFrame(y, index=day_offsets)
        return df.interpolate(method="index", limit_direction="both").values

    def auc(y):
        filled = _fill_gaps(y)
        result = np.trapezoid(filled, x=day_offsets, axis=0)
        result[np.all(np.isnan(y), axis=0)] = np.nan
        return result

    def cyclic_month(argmax_idx):
        m = months[argmax_idx]
        sin = np.sin(2 * np.pi * m / 12)
        cos = np.cos(2 * np.pi * m / 12)
        return sin, cos

    def linreg_slope(y):
        # closed-form slope, vectorized across pixels
        filled = _fill_gaps(y)
        x = day_offsets
        x_mean = x.mean()
        y_mean = filled.mean(axis=0)
        Sxy = ((x[:, None] - x_mean) * (filled - y_mean)).sum(axis=0)
        Sxx = ((x - x_mean) ** 2).sum()
        slope = Sxy / (Sxx + EPS)
        slope[np.all(np.isnan(y), axis=0)] = np.nan
        return slope

    def n_spikes(y, k=1.0):
        # local maxima (strictly greater than both neighbors) above mean + k*std
        filled = _fill_gaps(y)
        thresh = filled.mean(axis=0) + k * filled.std(axis=0)
        is_peak = (filled[1:-1] > filled[:-2]) & (filled[1:-1] > filled[2:]) & (filled[1:-1] > thresh[None, :])
        counts = is_peak.sum(axis=0).astype(np.float64)
        counts[np.all(np.isnan(y), axis=0)] = np.nan
        return counts

    feats = {}

    # QA feature: fraction of this pixel's season flagged cloud/shadow.
    # Useful for filtering unreliable pixels at train/eval time
    feats["cloud_frac"] = cloud_mask.mean(axis=0)

    with warnings.catch_warnings():
        # nanmin/nanmax/nanstd legitimately hit all-NaN slices for pixels
        # that were cloud-masked at every timestep -- expected, not a bug.
        warnings.simplefilter("ignore", category=RuntimeWarning)

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
        ndvi_argmax, ndvi_all_nan = _safe_nanargmax(NDVI)
        feats["NDVI_auc"] = auc(NDVI)
        feats["NDVI_month_max_sin"], feats["NDVI_month_max_cos"] = cyclic_month(ndvi_argmax)
        feats["NDVI_month_max_sin"][ndvi_all_nan] = np.nan
        feats["NDVI_month_max_cos"][ndvi_all_nan] = np.nan

        # --- NDBI: min ---
        feats["NDBI_min"] = np.nanmin(NDBI, axis=0)

        # --- NDRE: max, auc ---
        feats["NDRE_max"] = np.nanmax(NDRE, axis=0)
        feats["NDRE_auc"] = auc(NDRE)

        # --- SWIR_NIR: slope, days_max_to_min, month_of_max (cyclic), std ---
        swir_argmax, swir_max_all_nan = _safe_nanargmax(SWIR_NIR)
        swir_argmin, swir_min_all_nan = _safe_nanargmin(SWIR_NIR)
        feats["SWIR_NIR_slope"] = linreg_slope(SWIR_NIR)
        feats["SWIR_NIR_days_max_to_min"] = day_offsets[swir_argmin] - day_offsets[swir_argmax]
        feats["SWIR_NIR_days_max_to_min"][swir_max_all_nan | swir_min_all_nan] = np.nan
        feats["SWIR_NIR_month_max_sin"], feats["SWIR_NIR_month_max_cos"] = cyclic_month(swir_argmax)
        feats["SWIR_NIR_month_max_sin"][swir_max_all_nan] = np.nan
        feats["SWIR_NIR_month_max_cos"][swir_max_all_nan] = np.nan
        feats["SWIR_NIR_std"] = np.nanstd(SWIR_NIR, axis=0)

        # --- Monthly raw-band means: for every band and every month in the
        #     fixed season ordering, mean of that band's raw (non-index)
        #     values over timesteps falling in that month, ignoring any
        #     cloud/shadow-masked observations. NaN if the patch has no
        #     valid (unmasked) acquisition in that month. ---
        for band_name, b_idx in band_idx.items():
            band_data = flat[:, b_idx, :]  # (ts, pixels)
            for month in SEASON_MONTHS:
                col = f"{band_name}_{MONTH_ABBR[month]}_mean"
                month_mask = months == month
                if month_mask.any():
                    feats[col] = np.nanmean(band_data[month_mask], axis=0)
                else:
                    feats[col] = np.full(n_pixels, np.nan, dtype=np.float32)

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


if __name__ == "__main__":
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