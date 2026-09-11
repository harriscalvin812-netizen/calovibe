"""
train_model.py
--------------
Trains the calorie-burn prediction model used by app.py's Workouts tab.

ML task    : Supervised regression
Target     : Calories_Burned for one workout session
Features   : Age, Gender, Weight, Height, BMI, Session_Duration, Workout_Type
             (restricted to fields the Streamlit UI actually collects from a
             live user -- see model_utils.py for why)
Algorithm  : RandomForestRegressor (scikit-learn), inside a Pipeline with a
             ColumnTransformer that scales numeric columns and one-hot
             encodes the categorical ones.
Baseline   : Historical mean calories-per-minute-per-kg per workout type
             (this was the ONLY estimation method in the original app.py).
             We report it side-by-side with the ML model so the improvement
             from actually training a model is visible and explainable.

Run:
    python train_model.py

Requires fitness_app.db / data_raw/ to exist -> run prepare_data.py first
(this script will also auto-run the raw-file staging step if data_raw/ is
missing, so `python train_model.py` alone works too).

Outputs:
    models/calorie_burn_model.pkl   (joblib-pickled sklearn Pipeline)
    models/model_metrics.json       (MAE / RMSE / R2 for model + baseline,
                                      plus feature importances)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import (
    CATEGORICAL_FEATURES,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET,
)
from prepare_data import RAW_DIR, find_file, stage_raw_files


def load_training_data():
    if not os.path.isdir(RAW_DIR):
        stage_raw_files()
    path = find_file("gym_members_exercise_tracking")
    if not path:
        raise FileNotFoundError(
            "gym_members_exercise_tracking.csv not found. Make sure "
            "Calories_Burned_During_Exercise_Dataset.zip (or the loose csv) "
            "is in this folder, then run: python prepare_data.py"
        )
    df = pd.read_csv(path)
    df["BMI"] = df["Weight (kg)"] / (df["Height (m)"] ** 2)
    return df


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])


def baseline_predict(train_df, test_df):
    """Reproduces the ORIGINAL app's approach: mean calories-per-minute-per-kg
    per workout type, learned only from the training split (fit on train,
    scored on test -- same discipline as the ML model, so the comparison
    is fair)."""
    minutes = train_df["Session_Duration (hours)"] * 60
    train_df = train_df.copy()
    train_df["rate"] = train_df["Calories_Burned"] / minutes / train_df["Weight (kg)"]
    rate_by_type = train_df.groupby("Workout_Type")["rate"].mean()
    overall_rate = train_df["rate"].mean()

    test_minutes = test_df["Session_Duration (hours)"] * 60
    rates = test_df["Workout_Type"].map(rate_by_type).fillna(overall_rate)
    return rates.values * test_minutes.values * test_df["Weight (kg)"].values


def main():
    print("Loading training data...")
    df = load_training_data()
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    print(f"  -> {len(df):,} session rows")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42
    )

    print("Training RandomForestRegressor...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("Evaluating on held-out test set (20%)...")
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)

    base_preds = baseline_predict(df_train, df_test)
    base_mae = mean_absolute_error(y_test, base_preds)
    base_rmse = float(np.sqrt(mean_squared_error(y_test, base_preds)))
    base_r2 = r2_score(y_test, base_preds)

    print(f"\n{'Metric':<10}{'Baseline (mean-rate)':>22}{'RandomForest':>16}")
    print(f"{'MAE':<10}{base_mae:>22.1f}{mae:>16.1f}")
    print(f"{'RMSE':<10}{base_rmse:>22.1f}{rmse:>16.1f}")
    print(f"{'R2':<10}{base_r2:>22.3f}{r2:>16.3f}")

    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names
    importances = pipeline.named_steps["regressor"].feature_importances_
    importance_pairs = sorted(zip(feature_names, importances), key=lambda p: -p[1])

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "model": {"mae": round(float(mae), 2), "rmse": round(rmse, 2), "r2": round(float(r2), 4)},
        "baseline": {"mae": round(float(base_mae), 2), "rmse": round(base_rmse, 2), "r2": round(float(base_r2), 4)},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_importance": [
            {"feature": f, "importance": round(float(i), 4)} for f, i in importance_pairs
        ],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model   -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
