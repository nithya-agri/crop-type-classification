"""
Recreate rf_model.joblib, label_encoder.joblib, and imputer.joblib using
the best hyperparameters already found by tune_rf's RandomizedSearchCV in
train.py -- skips the (many-hour) search itself and just does one fit.

IMPORTANT: this only reproduces an equivalent model to your original run
if the underlying parquet feature files and train/val/test split files
are UNCHANGED since that run finished. Both the imputer and label encoder
are refit here from X_train/y_train (since those were the files that got
deleted) -- if the data has moved at all since your 6-hour run, you'll
get a new imputer/label encoder that don't exactly match what was
implicitly validated during that run. The sanity check against your
reported validation macro F1 at the end is meant to catch that.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from data_loading import load_train_val_test

RANDOM_STATE = 42  # must match train.py's RANDOM_STATE

# From your RandomizedSearchCV run -- copied here to skip re-searching.
BEST_RF_PARAMS = {
    "n_estimators": 100,
    "min_samples_leaf": 10,
    "max_features": 0.3,
    "max_depth": 20,
}

# Reported from your original run, for the sanity check below.
EXPECTED_VAL_MACRO_F1 = 0.4957


def main():
    print("Loading train split (val is also loaded, used only for the sanity "
          "check at the end; test is untouched)...")
    data = load_train_val_test()  # imputer=None here -> fits a fresh one from X_train
    X_train, y_train, g_train = data["train"]
    X_val, y_val, g_val = data["val"]
    imputer = data["imputer"]

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    print(f"Classes present: {list(label_encoder.classes_)} ({len(label_encoder.classes_)} total)")

    print(f"\nFitting RandomForestClassifier with best params: {BEST_RF_PARAMS}")
    rf = RandomForestClassifier(
        **BEST_RF_PARAMS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train_enc)

    joblib.dump(rf, "../outputs/models/rf_model.joblib")
    joblib.dump(label_encoder, "../outputs/models/label_encoder.joblib")
    joblib.dump(imputer, "../outputs/models/imputer.joblib")
    print("\nSaved rf_model.joblib, label_encoder.joblib, imputer.joblib")

    # ---- sanity check: does this land near your original reported number? ----
    y_val_enc = label_encoder.transform(y_val)
    val_pred = rf.predict(X_val)
    val_f1 = f1_score(y_val_enc, val_pred, average="macro")
    print(f"\nSanity check -- validation macro F1: {val_f1:.4f} "
          f"(originally reported: {EXPECTED_VAL_MACRO_F1:.4f})")
    if abs(val_f1 - EXPECTED_VAL_MACRO_F1) > 0.01:
        print("WARNING: this differs from your original run by more than a small "
              "margin -- the parquet files or split files may have changed since "
              "then, or randomness wasn't fully controlled somewhere. Worth "
              "double-checking before trusting these recreated files.")
    else:
        print("Close match to your original run -- recreated files look consistent.")


if __name__ == "__main__":
    main()