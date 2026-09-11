"""
app.py - All-in-One Fitness & Nutrition Tracker
------------------------------------------------
Run with:  streamlit run app.py
Requires fitness_app.db to already exist (created by prepare_data.py).
"""

import sqlite3
import time
from datetime import datetime

import pandas as pd
import streamlit as st

DB_PATH = "fitness_app.db"

st.set_page_config(page_title="Fitness & Nutrition Tracker", page_icon="🏋️", layout="wide")


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data
def load_foods():
    return pd.read_sql("SELECT * FROM foods", get_conn())


@st.cache_data
def load_coefficients():
    return pd.read_sql("SELECT * FROM workout_coefficients", get_conn())


@st.cache_data
def load_exercises():
    return pd.read_sql("SELECT * FROM exercise_library", get_conn())


foods_df = load_foods()
coeff_df = load_coefficients()
exercise_df = load_exercises()


# ---------------------------------------------------------------------------
# Session state (per-user logs, in-memory for the session)
# ---------------------------------------------------------------------------
if "food_log" not in st.session_state:
    st.session_state.food_log = []
if "workout_log" not in st.session_state:
    st.session_state.workout_log = []


# ---------------------------------------------------------------------------
# Sidebar: profile
# ---------------------------------------------------------------------------
st.sidebar.header("👤 Your Profile")
age = st.sidebar.number_input("Age", 10, 100, 28)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
height_cm = st.sidebar.number_input("Height (cm)", 100.0, 250.0, 170.0)
weight_kg = st.sidebar.number_input("Weight (kg)", 30.0, 250.0, 70.0)
activity = st.sidebar.selectbox(
    "Activity level",
    ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extremely active"],
)
goal = st.sidebar.selectbox("Goal", ["Lose weight", "Maintain weight", "Gain weight"])

ACTIVITY_FACTORS = {
    "Sedentary": 1.2,
    "Lightly active": 1.375,
    "Moderately active": 1.55,
    "Very active": 1.725,
    "Extremely active": 1.9,
}
GOAL_ADJUST = {"Lose weight": -500, "Maintain weight": 0, "Gain weight": 300}

# Mifflin-St Jeor BMR
if sex == "Male":
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
else:
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

tdee = bmr * ACTIVITY_FACTORS[activity]
target_calories = max(1200, tdee + GOAL_ADJUST[goal])
bmi = weight_kg / ((height_cm / 100) ** 2)

if bmi < 18.5:
    bmi_cat = "Underweight"
elif bmi < 25:
    bmi_cat = "Normal"
elif bmi < 30:
    bmi_cat = "Overweight"
else:
    bmi_cat = "Obese"

st.sidebar.markdown("---")
st.sidebar.metric("BMI", f"{bmi:.1f}", bmi_cat)
st.sidebar.metric("TDEE (maintenance)", f"{tdee:.0f} kcal/day")
st.sidebar.metric("Daily calorie target", f"{target_calories:.0f} kcal")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def totals_today():
    eaten = sum(item["calories"] for item in st.session_state.food_log)
    burned = sum(item["calories_burned"] for item in st.session_state.workout_log)
    return eaten, burned


def allergen_filter(df, exclude_gluten, exclude_dairy, exclude_nuts, exclude_soy, exclude_eggs, exclude_fish):
    out = df.copy()
    checks = {
        "contains_gluten": exclude_gluten,
        "contains_dairy": exclude_dairy,
        "contains_nuts": exclude_nuts,
        "contains_soy": exclude_soy,
        "contains_eggs": exclude_eggs,
        "contains_fish": exclude_fish,
    }
    for col, exclude in checks.items():
        if exclude:
            # keep rows where flag is explicitly False; drop True. Unknown (NaN) is kept.
            out = out[out[col] != True]  # noqa: E712
    return out


st.title("🏋️‍♀️ All-in-One Fitness & Nutrition Tracker")

tab_dash, tab_food, tab_workout, tab_reco = st.tabs(
    ["📊 Dashboard", "🍽️ Food Search & Log", "💪 Workouts & Calories Burned", "✨ Recommendations"]
)


# ---------------------------------------------------------------------------
# TAB: Dashboard
# ---------------------------------------------------------------------------
with tab_dash:
    eaten, burned = totals_today()
    net = eaten - burned
    remaining = target_calories - net

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calories eaten", f"{eaten:.0f}")
    c2.metric("Calories burned", f"{burned:.0f}")
    c3.metric("Net calories", f"{net:.0f}")
    c4.metric("Remaining budget", f"{remaining:.0f}")

    progress = min(max(net / target_calories, 0), 1) if target_calories else 0
    st.progress(progress, text=f"{net:.0f} / {target_calories:.0f} kcal target")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Today's meals")
        if st.session_state.food_log:
            fl = pd.DataFrame(st.session_state.food_log)
            st.dataframe(fl[["name", "servings", "calories", "protein_g", "carbs_g", "fat_g"]],
                         use_container_width=True, hide_index=True)
            if st.button("Clear food log"):
                st.session_state.food_log = []
                st.rerun()
            st.bar_chart(fl.set_index("name")[["protein_g", "carbs_g", "fat_g"]])
        else:
            st.info("No meals logged yet. Add some from the Food tab.")

    with col_b:
        st.subheader("Today's workouts")
        if st.session_state.workout_log:
            wl = pd.DataFrame(st.session_state.workout_log)
            st.dataframe(wl[["workout_type", "duration_min", "calories_burned"]],
                         use_container_width=True, hide_index=True)
            if st.button("Clear workout log"):
                st.session_state.workout_log = []
                st.rerun()
        else:
            st.info("No workouts logged yet. Add some from the Workouts tab.")


# ---------------------------------------------------------------------------
# TAB: Food Search & Log
# ---------------------------------------------------------------------------
with tab_food:
    st.subheader("Search the food database")
    st.caption(f"{len(foods_df):,} items from USDA, Indian dishes, and branded/packaged foods.")

    colf1, colf2, colf3 = st.columns([2, 1, 1])
    with colf1:
        query = st.text_input("Search by name", "")
    with colf2:
        source_filter = st.multiselect("Source", sorted(foods_df["source"].unique()),
                                        default=list(foods_df["source"].unique()))
    with colf3:
        max_cal = st.slider("Max calories per item", 0, 900, 900)

    with st.expander("Dietary / allergen filters"):
        e1, e2, e3, e4, e5, e6 = st.columns(6)
        exclude_gluten = e1.checkbox("Exclude gluten")
        exclude_dairy = e2.checkbox("Exclude dairy")
        exclude_nuts = e3.checkbox("Exclude nuts")
        exclude_soy = e4.checkbox("Exclude soy")
        exclude_eggs = e5.checkbox("Exclude eggs")
        exclude_fish = e6.checkbox("Exclude fish")

    results = foods_df[foods_df["source"].isin(source_filter)]
    if query:
        results = results[results["name"].str.contains(query, case=False, na=False)]
    results = results[results["calories"] <= max_cal]
    results = allergen_filter(results, exclude_gluten, exclude_dairy, exclude_nuts,
                               exclude_soy, exclude_eggs, exclude_fish)
    results = results.sort_values("health_score", ascending=False).head(200)

    st.dataframe(
        results[["name", "source", "calories", "protein_g", "carbs_g", "fat_g",
                 "fiber_g", "sugar_g", "sodium_mg", "health_score", "serving_note"]],
        use_container_width=True, hide_index=True, height=320,
    )

    st.markdown("##### Log a food")
    if len(results) > 0:
        pick_name = st.selectbox("Pick an item from results above", results["name"].tolist())
        servings = st.number_input("Servings", 0.25, 10.0, 1.0, step=0.25)
        if st.button("➕ Add to today's log"):
            row = results[results["name"] == pick_name].iloc[0]
            st.session_state.food_log.append({
                "name": row["name"],
                "servings": servings,
                "calories": round(row["calories"] * servings, 1),
                "protein_g": round((row["protein_g"] or 0) * servings, 1),
                "carbs_g": round((row["carbs_g"] or 0) * servings, 1),
                "fat_g": round((row["fat_g"] or 0) * servings, 1),
                "time": datetime.now().strftime("%H:%M"),
            })
            st.success(f"Added {pick_name} ({servings} serving(s))")
    else:
        st.warning("No foods match your filters.")


# ---------------------------------------------------------------------------
# TAB: Workouts
# ---------------------------------------------------------------------------
with tab_workout:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Calorie-burn estimator")
        st.caption("Coefficients learned from real gym member session data.")
        workout_type = st.selectbox("Workout type", coeff_df["workout_type"].tolist())
        duration_min = st.slider("Duration (minutes)", 5, 180, 30)
        coef = coeff_df.loc[coeff_df["workout_type"] == workout_type, "cal_per_min_per_kg"].iloc[0]
        est_calories = coef * duration_min * weight_kg
        st.metric("Estimated calories burned", f"{est_calories:.0f} kcal")

        if st.button("➕ Log this workout"):
            st.session_state.workout_log.append({
                "workout_type": workout_type,
                "duration_min": duration_min,
                "calories_burned": round(est_calories, 1),
                "time": datetime.now().strftime("%H:%M"),
            })
            st.success(f"Logged {workout_type} ({duration_min} min)")

    with right:
        st.subheader("Exercise library")
        muscle_groups = ["All"] + sorted(exercise_df["muscle_group"].dropna().unique().tolist())
        mg = st.selectbox("Muscle group", muscle_groups)
        ex_query = st.text_input("Search exercise name", "")

        ex_results = exercise_df.copy()
        if mg != "All":
            ex_results = ex_results[ex_results["muscle_group"] == mg]
        if ex_query:
            ex_results = ex_results[ex_results["exercise_name"].str.contains(ex_query, case=False, na=False)]

        st.dataframe(
            ex_results[["exercise_name", "muscle_group", "equipment", "rating"]].head(100),
            use_container_width=True, hide_index=True, height=320,
        )


# ---------------------------------------------------------------------------
# TAB: Recommendations
# ---------------------------------------------------------------------------
with tab_reco:
    eaten, burned = totals_today()
    remaining = max(target_calories - eaten + burned, 0)
    st.subheader(f"Suggested foods (fit within ~{remaining/3:.0f} kcal per meal)")

    reco_foods = allergen_filter(foods_df, exclude_gluten, exclude_dairy, exclude_nuts,
                                  exclude_soy, exclude_eggs, exclude_fish) \
        if any([exclude_gluten, exclude_dairy, exclude_nuts, exclude_soy, exclude_eggs, exclude_fish]) \
        else foods_df
    meal_budget = max(remaining / 3, 100)
    reco_foods = reco_foods[reco_foods["calories"] <= meal_budget]
    reco_foods = reco_foods.sort_values("health_score", ascending=False).head(10)
    st.dataframe(
        reco_foods[["name", "source", "calories", "protein_g", "health_score"]],
        use_container_width=True, hide_index=True,
    )

    st.subheader("Suggested workout focus")
    if goal == "Lose weight":
        suggested = ["HIIT", "Cardio"]
        muscle_focus = ["Abdominals", "Quadriceps", "Glutes"]
    elif goal == "Gain weight":
        suggested = ["Strength"]
        muscle_focus = ["Chest", "Lats", "Quadriceps", "Biceps", "Triceps"]
    else:
        suggested = ["Yoga", "Strength", "Cardio"]
        muscle_focus = ["Abdominals", "Chest", "Lats"]

    best = coeff_df[coeff_df["workout_type"].isin(suggested)].sort_values(
        "cal_per_min_per_kg", ascending=False
    )
    st.write(f"Recommended workout types for **{goal}**: " + ", ".join(suggested))
    st.dataframe(best, use_container_width=True, hide_index=True)

    st.write("Matching exercises to try:")
    reco_ex = exercise_df[exercise_df["muscle_group"].isin(muscle_focus)].sample(
        min(8, len(exercise_df[exercise_df["muscle_group"].isin(muscle_focus)]))
    )
    st.dataframe(reco_ex[["exercise_name", "muscle_group", "equipment", "rating"]],
                 use_container_width=True, hide_index=True)

st.caption("⚠️ Estimates are for general fitness tracking, not medical advice.")
