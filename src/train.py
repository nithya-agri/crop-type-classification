"""
Train Random Forest and XGBoost classifiers on the engineered feature set,
with class-imbalance handling, and evaluate on the validation set.

Class 9 is absent from the entire dataset (train/val/test) and is excluded
via label encoding below -- the model will only ever see/predict the
18 classes actually present (0-8, 10-18), plus void already dropped upstream.
"""

import time
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb

from data_loading import load_train_val_test

RANDOM_STATE = 42


def evaluate(model, X, y_true_encoded, label_encoder, split_name):
    y_pred_encoded = model.predict(X)

    macro_f1 = f1_score(y_true_encoded, y_pred_encoded, average="macro")
    print(f"\n=== {split_name} results ===")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nPer-class report:")
    all_labels = list(range(len(label_encoder.classes_)))
    print(classification_report(
        y_true_encoded, y_pred_encoded,
        labels=all_labels,
        target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0,
    ))

    cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=all_labels)
    return macro_f1, cm


def plot_confusion_matrix(cm, class_names, title, out_path, normalize=True):
    """Plot and save a confusion matrix heatmap.

    normalize=True shows row-wise percentages (recall per true class),
    which is usually more readable than raw counts given the class imbalance.
    """
    if normalize:
        with np.errstate(all="ignore"):
            cm_display = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)  # classes with 0 support -> 0 row
        fmt_suffix = " (row-normalized)"
        vmax = 1.0
    else:
        cm_display = cm
        fmt_suffix = " (counts)"
        vmax = cm.max()

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(6, n * 0.5)))
    im = ax.imshow(cm_display, cmap="Blues", vmin=0, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title + fmt_suffix)

    # annotate cells
    thresh = vmax / 2
    for i in range(n):
        for j in range(n):
            val = cm_display[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                     fontsize=6, color="white" if val > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix heatmap to {out_path}")


def main():
    data = load_train_val_test()
    X_train, y_train, g_train = data["train"]
    X_val, y_val, g_val = data["val"]
    X_test, y_test, g_test = data["test"]

    # ---------------------------------------------------------------
    # Encode labels to a contiguous 0..N-1 range (class 9 is absent, so raw
    # labels aren't contiguous -- XGBoost in particular requires this).
    # ---------------------------------------------------------------
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    n_classes = len(label_encoder.classes_)
    print(f"Classes present: {list(label_encoder.classes_)} ({n_classes} total)")

    # =================================================================
    # Random Forest
    # =================================================================
    print("\n" + "=" * 60)
    print("Training Random Forest...")
    print("=" * 60)

    N_ESTIMATORS = 300
    BATCH_SIZE = 25  # trees added per iteration -- keeps warm_start overhead low

    rf = RandomForestClassifier(
        n_estimators=0,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        warm_start=True,
        verbose=0,
    )

    start = time.time()
    n_batches = N_ESTIMATORS // BATCH_SIZE
    for i in tqdm(range(n_batches), desc="RF training"):
        rf.n_estimators = (i + 1) * BATCH_SIZE
        rf.fit(X_train, y_train_enc)
    print(f"RF training took {time.time() - start:.1f}s")

    rf_val_f1, rf_val_cm = evaluate(rf, X_val, y_val_enc, label_encoder, "RF - Validation")
    plot_confusion_matrix(
        rf_val_cm, label_encoder.classes_,
        title="RF Validation Confusion Matrix",
        out_path="../outputs/figures/rf_val_confusion_matrix.png",
    )

    joblib.dump(rf, "../outputs/models/rf_model.joblib")
    joblib.dump(label_encoder, "../outputs/models/label_encoder.joblib")

    # =================================================================
    # XGBoost
    # =================================================================
    print("\n" + "=" * 60)
    print("Training XGBoost...")
    print("=" * 60)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_enc)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=n_classes,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )
    start = time.time()
    xgb_model.fit(
        X_train, y_train_enc,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val_enc)],
        verbose=10,
    )
    print(f"XGBoost training took {time.time() - start:.1f}s")

    xgb_val_f1, xgb_val_cm = evaluate(xgb_model, X_val, y_val_enc, label_encoder, "XGBoost - Validation")
    plot_confusion_matrix(
        xgb_val_cm, label_encoder.classes_,
        title="XGBoost Validation Confusion Matrix",
        out_path="../outputs/figures/xgb_val_confusion_matrix.png",
    )

    xgb_model.save_model("../outputs/models/xgb_model.json")

    # =================================================================
    # Compare
    # =================================================================
    print("\n" + "=" * 60)
    print(f"RF      validation macro F1: {rf_val_f1:.4f}")
    print(f"XGBoost validation macro F1: {xgb_val_f1:.4f}")
    print("=" * 60)

    # ---------------------------------------------------------------
    # NOTE: test set is intentionally not touched here. Only run the
    # winning model on X_test/y_test once, at the very end, after all
    # tuning decisions are finalized based on validation results.
    # ---------------------------------------------------------------


if __name__ == "__main__":
    main()