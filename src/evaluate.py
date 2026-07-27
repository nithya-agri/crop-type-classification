"""
Final held-out test-set evaluation for the tuned RF model.

Run this ONCE, after all tuning/model-selection decisions are locked in
based on validation results. Loads the already-saved model + label encoder
from train.py's run rather than retraining anything.

Reports: overall accuracy, per-class precision/recall/F1 (via
train.evaluate's classification_report), confusion matrix, per-class IoU
and mean IoU (mIoU), Cohen's kappa, balanced accuracy, and the most
confused class pairs.
"""
import os

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, cohen_kappa_score, balanced_accuracy_score

from data_loading import load_train_val_test
from train import evaluate, plot_confusion_matrix

RF_MODEL_PATH = "../outputs/models/rf_model.joblib"
LABEL_ENCODER_PATH = "../outputs/models/label_encoder.joblib"
IMPUTER_PATH = "../outputs/models/imputer.joblib"


def compute_iou_per_class(cm):
    """
    cm: confusion matrix, rows=true, cols=predicted, shape (n_classes, n_classes)

    IoU per class = TP / (TP + FP + FN), the standard segmentation metric,
    applied here per-class since this is dense pixel-level classification.
    Classes with zero union (never present as true or predicted label) get
    NaN rather than 0, so they don't distort the mean -- consistent with
    how zero_division is handled in the classification_report.
    """
    n = cm.shape[0]
    ious = np.full(n, np.nan)
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        union = tp + fp + fn
        if union > 0:
            ious[i] = tp / union
    return ious


def top_confused_pairs(cm, class_names, top_k=10):
    """
    Returns the top_k largest off-diagonal (true, predicted) confusion
    counts, as (true_class, predicted_class, count) -- i.e. the specific
    class pairs the model mixes up most, which the aggregate metrics above
    don't surface directly.
    """
    n = cm.shape[0]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], cm[i, j]))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_k]


def main():
    saved_imputer = None
    if os.path.exists(IMPUTER_PATH):
        print(f"Found saved imputer at {IMPUTER_PATH} -- reusing exact training-time statistics.")
        saved_imputer = joblib.load(IMPUTER_PATH)
    else:
        print(f"WARNING: no saved imputer found at {IMPUTER_PATH}. Refitting a new one "
              f"from the current train split -- only safe if the parquet files and "
              f"split files haven't changed since your original training run.")


    print("Loading held-out test set (train/val also loaded internally, "
          "but only test is used here)...")
    data = load_train_val_test()
    X_test, y_test, g_test = data["test"]

    rf = joblib.load(RF_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    # Guard against a class appearing in test but never seen in train --
    # LabelEncoder.transform() raises on unseen labels rather than
    # silently mishandling them, so surface that clearly if it happens.
    unseen = set(y_test) - set(label_encoder.classes_)
    if unseen:
        raise ValueError(
            f"Test set contains classes never seen during training: {unseen}. "
            f"Known classes: {list(label_encoder.classes_)}"
        )

    y_test_enc = label_encoder.transform(y_test)

    # ---- precision/recall/F1 (per-class + macro/weighted) + confusion matrix ----
    test_f1, test_cm = evaluate(rf, X_test, y_test_enc, label_encoder, "RF - Test (FINAL)")
    plot_confusion_matrix(
        test_cm, label_encoder.classes_,
        title="RF Test Confusion Matrix (FINAL)",
        out_path="../outputs/figures/rf_test_confusion_matrix.png",
    )

    y_pred_enc = rf.predict(X_test)

    # ---- overall accuracy ----
    acc = accuracy_score(y_test_enc, y_pred_enc)

    # ---- per-class IoU / mIoU ----
    ious = compute_iou_per_class(test_cm)
    valid_ious = ious[~np.isnan(ious)]
    miou = valid_ious.mean() if len(valid_ious) else float("nan")

    # ---- Cohen's kappa: agreement beyond what chance would predict, given
    # the class distribution. More informative than raw accuracy when
    # classes are as imbalanced as this dataset is. ----
    kappa = cohen_kappa_score(y_test_enc, y_pred_enc)

    # ---- balanced accuracy: mean per-class recall. Complements macro F1 --
    # tells you recall-only performance, unaffected by precision. ----
    bal_acc = balanced_accuracy_score(y_test_enc, y_pred_enc)

    print("\n" + "=" * 60)
    print("RF - Test (FINAL) - extended metrics")
    print("=" * 60)
    print(f"Overall accuracy:      {acc:.4f}")
    print(f"Balanced accuracy:     {bal_acc:.4f}  (mean per-class recall)")
    print(f"Cohen's kappa:         {kappa:.4f}  (agreement beyond chance)")
    print(f"Mean IoU (mIoU):       {miou:.4f}  (over {len(valid_ious)}/{len(ious)} classes with any true or predicted support)")

    print("\nPer-class IoU:")
    for cls, iou in zip(label_encoder.classes_, ious):
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A (no true or predicted samples)"
        print(f"  class {cls}: {iou_str}")

    print(f"\nTop confused class pairs (true -> predicted, count):")
    for true_c, pred_c, count in top_confused_pairs(test_cm, label_encoder.classes_, top_k=10):
        print(f"  {true_c} -> {pred_c}: {count:,}")

    print("\n" + "=" * 60)
    print(f"RF FINAL test macro F1: {test_f1:.4f}")
    print(f"RF FINAL test mIoU:     {miou:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()