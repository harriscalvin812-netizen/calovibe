"""
prepare_data.py
----------------
Consolidates all raw datasets (USDA foods, Indian foods, branded/allergen foods,
gym exercise library, and gym calorie-burn tracking logs) into a single SQLite
database (fitness_app.db) that the Streamlit app reads from.

Run this once after uploading the raw files to the Colab session:
    python prepare_data.py
"""

import glob
import os
import shutil
import sqlite3
import zipfile

import numpy as np
import pandas as pd

RAW_DIR = "data_raw"
DB_PATH = "fitness_app.db"


# ---------------------------------------------------------------------------
# 0. Unpack everything into a flat raw-data folder
# ---------------------------------------------------------------------------
def stage_raw_files():
    os.makedirs(RAW_DIR, exist_ok=True)

    for z in glob.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
        print(f"Unzipped {z}")

    loose_files = [
        "comprehensive_foods_usda.csv",
        "foods_allergens.csv",
        "foods_dietary_restrictions.csv",
        "foods_health_scores_allergens.csv",
        "healthy_foods_database.csv",
    ]
    for f in loose_files:
        if os.path.exists(f):
            dest = os.path.join(RAW_DIR, f)
            if not os.path.exists(dest):
                shutil.copy(f, dest)


def find_file(substr):
    """Find the first file under RAW_DIR whose name contains substr (case-insensitive)."""
    substr = substr.lower()
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if substr in f.lower():
                return os.path.join(root, f)
    return None


# ---------------------------------------------------------------------------
# 1. Helper: simple 0-100 health score heuristic for datasets that lack one
# ---------------------------------------------------------------------------
def heuristic_health_score(protein, fiber, sugar, sodium_mg, sat_fat=0):
    protein = pd.to_numeric(protein, errors="coerce").fillna(0)
    fiber = pd.to_numeric(fiber, errors="coerce").fillna(0)
    sugar = pd.to_numeric(sugar, errors="coerce").fillna(0)
    sodium_mg = pd.to_numeric(sodium_mg, errors="coerce").fillna(0)
    sat_fat = pd.to_numeric(sat_fat, errors="coerce").fillna(0) if not isinstance(sat_fat, int) else 0
    score = 50 + protein * 1.5 + fiber * 2 - sugar * 0.7 - (sodium_mg / 100.0) - sat_fat * 1.0
    return score.clip(0, 100).round(1)


NUTRISCORE_MAP = {"a": 90, "b": 75, "c": 60, "d": 40, "e": 20}

FOOD_COLUMNS = [
    "name", "source", "food_type", "calories", "protein_g", "carbs_g", "fat_g",
    "fiber_g", "sugar_g", "sodium_mg", "health_score",
    "contains_gluten", "contains_dairy", "contains_nuts", "contains_soy",
    "contains_eggs", "contains_fish", "serving_note",
]


def empty_food_frame():
    return pd.DataFrame(columns=FOOD_COLUMNS)


# ---------------------------------------------------------------------------
# 2. Load + standardize each food source
# ---------------------------------------------------------------------------
def load_usda():
    path = find_file("comprehensive_foods_usda")
    if not path:
        return empty_food_frame()
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "name": df["food_name"],
        "source": "USDA",
        "food_type": df.get("food_category", df.get("food_type", "Unknown")).fillna("Unknown"),
        "calories": pd.to_numeric(df["calories"], errors="coerce"),
        "protein_g": pd.to_numeric(df["protein_g"], errors="coerce"),
        "carbs_g": pd.to_numeric(df["carbs_g"], errors="coerce"),
        "fat_g": pd.to_numeric(df["fat_g"], errors="coerce"),
        "fiber_g": pd.to_numeric(df["fiber_g"], errors="coerce"),
        "sugar_g": pd.to_numeric(df["sugar_g"], errors="coerce"),
        "sodium_mg": pd.to_numeric(df["sodium_mg"], errors="coerce"),
        "health_score": pd.to_numeric(df.get("health_score"), errors="coerce"),
        "contains_gluten": np.nan,
        "contains_dairy": np.nan,
        "contains_nuts": np.nan,
        "contains_soy": np.nan,
        "contains_eggs": np.nan,
        "contains_fish": np.nan,
        "serving_note": df.get("household_serving", "").fillna(""),
    })
    out["health_score"] = out["health_score"].fillna(
        heuristic_health_score(out["protein_g"], out["fiber_g"], out["sugar_g"], out["sodium_mg"])
    )
    return out


def load_indian():
    path = find_file("Indian_Food_Nutrition_Processed")
    if not path:
        return empty_food_frame()
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "name": df["Dish Name"],
        "source": "Indian",
        "food_type": "Indian Dish",
        "calories": pd.to_numeric(df["Calories (kcal)"], errors="coerce"),
        "protein_g": pd.to_numeric(df["Protein (g)"], errors="coerce"),
        "carbs_g": pd.to_numeric(df["Carbohydrates (g)"], errors="coerce"),
        "fat_g": pd.to_numeric(df["Fats (g)"], errors="coerce"),
        "fiber_g": pd.to_numeric(df["Fibre (g)"], errors="coerce"),
        "sugar_g": pd.to_numeric(df["Free Sugar (g)"], errors="coerce"),
        "sodium_mg": pd.to_numeric(df["Sodium (mg)"], errors="coerce"),
        "health_score": np.nan,
        "contains_gluten": np.nan,
        "contains_dairy": np.nan,
        "contains_nuts": np.nan,
        "contains_soy": np.nan,
        "contains_eggs": np.nan,
        "contains_fish": np.nan,
        "serving_note": "1 serving",
    })
    out["health_score"] = heuristic_health_score(
        out["protein_g"], out["fiber_g"], out["sugar_g"], out["sodium_mg"]
    )
    return out


def load_branded():
    path = find_file("foods_health_scores_allergens")
    if not path:
        return empty_food_frame()
    df = pd.read_csv(path, low_memory=False)

    def score_row(grade, protein, fiber, sugar, sodium, satfat):
        g = str(grade).strip().lower()
        if g in NUTRISCORE_MAP:
            return NUTRISCORE_MAP[g]
        return None

    ns_scores = df["nutriscore_grade"].astype(str).str.strip().str.lower().map(NUTRISCORE_MAP)

    out = pd.DataFrame({
        "name": df["product_name"],
        "source": "Branded",
        "food_type": df.get("food_type", "Branded/Packaged").fillna("Branded/Packaged"),
        "calories": pd.to_numeric(df["energy_kcal"], errors="coerce"),
        "protein_g": pd.to_numeric(df["proteins_100g"], errors="coerce"),
        "carbs_g": pd.to_numeric(df["carbs_100g"], errors="coerce"),
        "fat_g": pd.to_numeric(df["fat_100g"], errors="coerce"),
        "fiber_g": pd.to_numeric(df["fiber_100g"], errors="coerce"),
        "sugar_g": pd.to_numeric(df["sugars_100g"], errors="coerce"),
        "sodium_mg": pd.to_numeric(df["sodium_100g"], errors="coerce") * 1000,  # g -> mg
        "health_score": ns_scores,
        "contains_gluten": df.get("contains_gluten"),
        "contains_dairy": df.get("contains_dairy"),
        "contains_nuts": df.get("contains_nuts"),
        "contains_soy": df.get("contains_soy"),
        "contains_eggs": df.get("contains_eggs"),
        "contains_fish": df.get("contains_fish"),
        "serving_note": "per 100g",
    })
    out["health_score"] = out["health_score"].fillna(
        heuristic_health_score(out["protein_g"], out["fiber_g"], out["sugar_g"], out["sodium_mg"])
    )
    return out


def build_foods_table():
    frames = [load_usda(), load_indian(), load_branded()]
    foods = pd.concat(frames, ignore_index=True, sort=False)
    foods = foods.dropna(subset=["name", "calories"])
    foods = foods[foods["calories"] >= 0]
    for col in ["contains_gluten", "contains_dairy", "contains_nuts",
                "contains_soy", "contains_eggs", "contains_fish"]:
        foods[col] = foods[col].astype("object")  # keep NaN = "unknown" distinct from False
    foods = foods.reset_index(drop=True)
    foods.insert(0, "food_id", foods.index + 1)
    return foods


# ---------------------------------------------------------------------------
# 3. Exercise calorie-burn coefficients (calories per minute per kg body weight)
# ---------------------------------------------------------------------------
def build_workout_coefficients():
    path = find_file("gym_members_exercise_tracking")
    if not path:
        return pd.DataFrame(columns=["workout_type", "cal_per_min_per_kg", "n_samples"])
    df = pd.read_csv(path)
    minutes = df["Session_Duration (hours)"] * 60
    df["cal_per_min_per_kg"] = df["Calories_Burned"] / minutes / df["Weight (kg)"]
    df = df[np.isfinite(df["cal_per_min_per_kg"])]
    grouped = (
        df.groupby("Workout_Type")["cal_per_min_per_kg"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"Workout_Type": "workout_type", "mean": "cal_per_min_per_kg", "count": "n_samples"})
    )
    grouped["cal_per_min_per_kg"] = grouped["cal_per_min_per_kg"].round(4)
    return grouped


# ---------------------------------------------------------------------------
# 4. Exercise library (browsable gym exercises)
# ---------------------------------------------------------------------------
def build_exercise_library():
    path = find_file("Gym Exercises Dataset")
    if not path:
        path = find_file("Gym_Exercises_Dataset")
    if not path or not path.lower().endswith((".xlsx", ".xls")):
        # search generically for an xlsx under raw dir
        for root, _, files in os.walk(RAW_DIR):
            for f in files:
                if f.lower().endswith(".xlsx"):
                    path = os.path.join(root, f)
    if not path:
        return pd.DataFrame(columns=["exercise_name", "muscle_group", "equipment", "rating", "description", "url"])

    df = pd.read_excel(path)
    out = pd.DataFrame({
        "exercise_name": df["Exercise_Name"],
        "muscle_group": df["muscle_gp"],
        "equipment": df["Equipment"],
        "rating": df["Rating"],
        "description": df["Description"],
        "url": df.get("Description_URL", ""),
    })
    out = out.dropna(subset=["exercise_name"]).reset_index(drop=True)
    out.insert(0, "exercise_id", out.index + 1)
    return out


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    print("Staging raw files...")
    stage_raw_files()

    print("Building foods table (USDA + Indian + Branded)...")
    foods = build_foods_table()
    print(f"  -> {len(foods):,} food items")

    print("Building workout calorie-burn coefficients...")
    coeffs = build_workout_coefficients()
    print(f"  -> {len(coeffs)} workout types")

    print("Building exercise library...")
    exercises = build_exercise_library()
    print(f"  -> {len(exercises):,} exercises")

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        foods.to_sql("foods", conn, index=False)
        coeffs.to_sql("workout_coefficients", conn, index=False)
        exercises.to_sql("exercise_library", conn, index=False)
        conn.execute("CREATE INDEX idx_foods_name ON foods(name)")
        conn.execute("CREATE INDEX idx_ex_muscle ON exercise_library(muscle_group)")

    print(f"\nDone. Wrote {DB_PATH}")


if __name__ == "__main__":
    main()
