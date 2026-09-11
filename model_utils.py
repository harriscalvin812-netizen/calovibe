"""
model_utils.py
--------------
Shared feature-engineering + model I/O helpers used by BOTH train_model.py
(offline training) and app.py (online inference at prediction time), so
training and serving can never silently drift apart.
"""

import json
import os

import joblib
import pandas as pd

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "calorie_burn_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

# These are the ONLY features the model is allowed to use. They are exactly
# the fields the Streamlit sidebar + Workouts tab collect from a real user.
# The training CSV also has Avg_BPM, Fat_Percentage, Water_Intake, etc., but
# those are never available at prediction time for a brand-new user, so
# using them during training would be a data-leakage bug (the model would
# look accurate offline but could never actually be used live). Restricting
# the feature set keeps train/serve consistent.
NUMERIC_FEATURES = ["Age", "Weight (kg)", "Height (m)", "BMI", "Session_Duration (hours)"]
CATEGORICAL_FEATURES = ["Gender", "Workout_Type"]
TARGET = "Calories_Burned"


def build_inference_row(age, sex, height_cm, weight_kg, workout_type, duration_min):
    """Build a single-row DataFrame matching the training schema, from raw
    Streamlit widget values (height in cm, duration in minutes)."""
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    return pd.DataFrame([{
        "Age": age,
        "Weight (kg)": weight_kg,
        "Height (m)": height_m,
        "BMI": bmi,
        "Session_Duration (hours)": duration_min / 60.0,
        "Gender": sex,
        "Workout_Type": workout_type,
    }])


def load_model():
    """Returns the trained sklearn Pipeline, or None if train_model.py hasn't
    been run yet (app should fall back to the baseline coefficient table)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)
