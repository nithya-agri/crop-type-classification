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
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict
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


def tune_rf(X_train, y_train_enc, groups, n_iter=20, cv_splits=3):
    """Randomized hyperparameter search for RF using patch-grouped CV, so no
    patch's pixels ever span both the search-train and search-val side of a
    fold (same leakage concern as the main train/val/test split)."""
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    }

    base_model = RandomForestClassifier(
        n_jobs=1,  # avoid oversubscribing cores: search itself uses n_jobs=-1
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    sgkf = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=sgkf,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=2,
    )
    start = time.time()
    search.fit(X_train, y_train_enc, groups=groups)
    print(f"RF search took {time.time() - start:.1f}s")
    print(f"Best RF params: {search.best_params_}")
    print(f"Best RF CV macro F1: {search.best_score_:.4f}")

    return search.best_estimator_


def tune_xgb(X_train, y_train_enc, groups, sample_weights, n_classes, n_iter=20, cv_splits=3):
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    base_model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=n_classes,
        n_jobs=1,  # avoid oversubscribing cores: search itself uses n_jobs=-1
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

    sgkf = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=sgkf,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=2,
    )

    start = time.time()
    # sample_weight is sliced automatically per-fold to match each split's indices
    search.fit(X_train, y_train_enc, groups=groups, sample_weight=sample_weights)
    print(f"XGBoost search took {time.time() - start:.1f}s")
    print(f"Best XGBoost params: {search.best_params_}")
    print(f"Best XGBoost CV macro F1: {search.best_score_:.4f}")

    return search.best_estimator_


def cross_validated_evaluate(model, X, y_encoded, groups, label_encoder,
                              model_name, sample_weight=None, cv_splits=5):
    """Evaluate a (already-tuned) model via out-of-fold predictions across
    ALL provided data, using StratifiedGroupKFold. Every patch is used as
    validation exactly once across the rotation, so the aggregated report
    covers every class present anywhere in this data -- avoiding the
    "class missing from a single fixed val fold" artifact.

    Note: this evaluates the *fixed hyperparameters* of `model` (cloned and
    refit per fold internally by cross_val_predict) -- it does not re-tune.
    """
    sgkf = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    fit_params = {}
    if sample_weight is not None:
        fit_params["sample_weight"] = sample_weight

    print(f"\nRunning {cv_splits}-fold cross_val_predict for {model_name}...")
    start = time.time()
    try:
        # newer sklearn (metadata routing): 'params'
        y_pred_oof = cross_val_predict(
            model, X, y_encoded, groups=groups, cv=sgkf, n_jobs=-1,
            params=fit_params if fit_params else None,
        )
    except TypeError:
        # older sklearn: 'fit_params'
        y_pred_oof = cross_val_predict(
            model, X, y_encoded, groups=groups, cv=sgkf, n_jobs=-1,
            fit_params=fit_params if fit_params else None,
        )
    print(f"{model_name} cross_val_predict took {time.time() - start:.1f}s")

    all_labels = list(range(len(label_encoder.classes_)))
    macro_f1 = f1_score(y_encoded, y_pred_oof, average="macro")
    print(f"\n=== {model_name} - Aggregated {cv_splits}-fold CV results ===")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(
        y_encoded, y_pred_oof,
        labels=all_labels,
        target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0,
    ))

    cm = confusion_matrix(y_encoded, y_pred_oof, labels=all_labels)
    plot_confusion_matrix(
        cm, label_encoder.classes_,
        title=f"{model_name} {cv_splits}-Fold CV Confusion Matrix",
        out_path=f"{model_name.lower()}_cv_confusion_matrix.png",
    )

    return macro_f1, cm


def main(run_xgb=False):
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
    print("Tuning Random Forest (GroupKFold randomized search)...")
    print("=" * 60)

    rf = tune_rf(X_train, y_train_enc, g_train, n_iter=20, cv_splits=3)
    rf.n_jobs = -1  # restore full parallelism for the final fitted estimator

    rf_val_f1, rf_val_cm = evaluate(rf, X_val, y_val_enc, label_encoder, "RF - Validation")
    plot_confusion_matrix(
        rf_val_cm, label_encoder.classes_,
        title="RF Validation Confusion Matrix",
        out_path="../outputs/figures/rf_val_confusion_matrix.png",
    )

    # ---------------------------------------------------------------
    # Aggregate 5-fold CV evaluation over train+val combined ("dev" set).
    # Every dev patch is used as validation exactly once across the 5
    # rotations, so classes missing from a single fixed val fold (like
    # classes 8/13/16/18 ) still get evaluated here. Test stays untouched.
    # ---------------------------------------------------------------
    X_dev = np.concatenate([X_train, X_val])
    y_dev_enc = np.concatenate([y_train_enc, y_val_enc])
    g_dev = np.concatenate([g_train, g_val])

    rf_cv_f1, rf_cv_cm = cross_validated_evaluate(
        rf, X_dev, y_dev_enc, g_dev, label_encoder,
        model_name="RF", cv_splits=5,
    )

    joblib.dump(rf, "../outputs/models/rf_model.joblib")
    joblib.dump(label_encoder, "../outputs/models/label_encoder.joblib")

    # =================================================================
    # XGBoost
    # =================================================================
    if run_xgb:
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_enc)

        print("\n" + "=" * 60)
        print("Tuning XGBoost (GroupKFold randomized search)...")
        print("=" * 60)

        xgb_model = tune_xgb(
            X_train, y_train_enc, g_train, sample_weights, n_classes,
            n_iter=20, cv_splits=3,
        )
        xgb_model.n_jobs = -1  # restore full parallelism for the final fitted estimator

        xgb_val_f1, xgb_val_cm = evaluate(xgb_model, X_val, y_val_enc, label_encoder, "XGBoost - Validation")
        plot_confusion_matrix(
            xgb_val_cm, label_encoder.classes_,
            title="XGBoost Validation Confusion Matrix",
            out_path="../outputs/figures/xgb_val_confusion_matrix.png",
        )

        sample_weights_dev = compute_sample_weight(class_weight="balanced", y=y_dev_enc)
        xgb_cv_f1, xgb_cv_cm = cross_validated_evaluate(
            xgb_model, X_dev, y_dev_enc, g_dev, label_encoder,
            model_name="XGBoost", sample_weight=sample_weights_dev, cv_splits=5,
        )

        xgb_model.save_model("xgb_model.json")

    # =================================================================
    # Compare
    # =================================================================
    print("\n" + "=" * 60)
    print(f"RF      single-fold validation macro F1: {rf_val_f1:.4f}")
    print(f"RF      5-fold CV (dev set)  macro F1:    {rf_cv_f1:.4f}")
    # print(f"XGBoost single-fold validation macro F1: {xgb_val_f1:.4f}")
    # print(f"XGBoost 5-fold CV (dev set)  macro F1:    {xgb_cv_f1:.4f}")
    print("=" * 60)

    # ---------------------------------------------------------------
    # NOTE: test set is intentionally not touched here. Only run the
    # winning model on X_test/y_test once, at the very end, after all
    # tuning decisions are finalized based on validation results.
    # ---------------------------------------------------------------


if __name__ == "__main__":
    main()