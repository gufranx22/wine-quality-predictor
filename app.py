"""
Wine Quality Predictor - web app
Asansol Engineering College, Department of Computer Applications

Run with:   streamlit run app.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "models", "wine_models.joblib")
DATA_FILE = os.path.join(HERE, "data", "winequality-red.csv")

# name shown on screen, unit/help text
LABELS = {
    "fixed acidity": ("Fixed acidity", "g/dm³ (tartaric acid)"),
    "volatile acidity": ("Volatile acidity", "g/dm³ (acetic acid) - too much gives a vinegar taste"),
    "citric acid": ("Citric acid", "g/dm³"),
    "residual sugar": ("Residual sugar", "g/dm³"),
    "chlorides": ("Chlorides", "g/dm³ (salt)"),
    "free sulfur dioxide": ("Free sulfur dioxide", "mg/dm³"),
    "total sulfur dioxide": ("Total sulfur dioxide", "mg/dm³"),
    "density": ("Density", "g/cm³"),
    "pH": ("pH", "acidity level, lower = more acidic"),
    "sulphates": ("Sulphates", "g/dm³ (potassium sulphate)"),
    "alcohol": ("Alcohol", "% by volume"),
}


@st.cache_resource
def load_models():
    """Load the saved models. If the file cannot be loaded (for example because of a
    different scikit-learn version), train the same models again from the csv file."""
    try:
        return joblib.load(MODEL_FILE)
    except Exception:
        return train_again()


def train_again():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, LinearRegression

    df = pd.read_csv(DATA_FILE, sep=";").drop_duplicates().reset_index(drop=True)
    cols = [c for c in df.columns if c != "quality"]
    q1, q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    X = df[cols].clip(lower=lower, upper=upper, axis=1)
    clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    reg = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    clf.fit(X, (df["quality"] >= 6).astype(int))
    reg.fit(X, df["quality"])
    return {
        "classifier": clf, "regressor": reg,
        "classifier_name": "Logistic Regression", "regressor_name": "Linear Regression",
        "feature_names": cols, "lower_limits": lower.to_dict(), "upper_limits": upper.to_dict(),
        "feature_min": df[cols].min().to_dict(), "feature_max": df[cols].max().to_dict(),
        "feature_median": df[cols].median().to_dict(),
        "classifier_accuracy": float("nan"), "regressor_rmse": float("nan"),
    }


saved = load_models()
features = saved["feature_names"]

# two example wines taken from the dataset, so the user can try the app quickly
EXAMPLES = {
    "good": {"fixed acidity": 7.9, "volatile acidity": 0.35, "citric acid": 0.46, "residual sugar": 3.6,
             "chlorides": 0.078, "free sulfur dioxide": 15.0, "total sulfur dioxide": 37.0, "density": 0.9973,
             "pH": 3.35, "sulphates": 0.86, "alcohol": 12.8},
    "poor": {"fixed acidity": 11.6, "volatile acidity": 0.58, "citric acid": 0.66, "residual sugar": 2.2,
             "chlorides": 0.074, "free sulfur dioxide": 10.0, "total sulfur dioxide": 47.0, "density": 1.0008,
             "pH": 3.25, "sulphates": 0.57, "alcohol": 9.0},
}

# starting values = median of every column
for f in features:
    if f not in st.session_state:
        st.session_state[f] = float(saved["feature_median"][f])


def fill_example(kind):
    for f in features:
        st.session_state[f] = float(EXAMPLES[kind][f])


def reset_values():
    for f in features:
        st.session_state[f] = float(saved["feature_median"][f])


st.title("🍷 Wine Quality Predictor")
st.write("Type the lab test values of a red wine and the app tells you if the wine is likely to be "
         "**good** (quality 6 or more) and what its quality score may be.")

c1, c2, c3 = st.columns(3)
c1.button("Try a good wine", on_click=fill_example, args=("good",), use_container_width=True)
c2.button("Try a poor wine", on_click=fill_example, args=("poor",), use_container_width=True)
c3.button("Reset values", on_click=reset_values, use_container_width=True)

st.subheader("Wine measurements")
left, right = st.columns(2)
values = {}
for i, f in enumerate(features):
    name, help_text = LABELS.get(f, (f, ""))
    lo = float(saved["feature_min"][f])
    hi = float(saved["feature_max"][f])
    step = 0.001 if hi < 2 else (0.01 if hi < 20 else 1.0)
    box = left if i % 2 == 0 else right
    values[f] = box.number_input(name, min_value=0.0, max_value=hi * 2, step=step,
                                 format="%.4f" if step == 0.001 else ("%.2f" if step == 0.01 else "%.0f"),
                                 help=f"{help_text}. Values in the training data go from {lo} to {hi}.",
                                 key=f)

if st.button("Predict quality", type="primary", use_container_width=True):
    row = pd.DataFrame([values])[features]
    # same outlier handling as in training
    lower = pd.Series(saved["lower_limits"])
    upper = pd.Series(saved["upper_limits"])
    row = row.clip(lower=lower, upper=upper, axis=1)

    prob_good = float(saved["classifier"].predict_proba(row)[0, 1])
    score = float(saved["regressor"].predict(row)[0])
    score = min(max(score, 3.0), 8.0)

    st.subheader("Result")
    if prob_good >= 0.5:
        st.success(f"This wine is predicted to be **GOOD**  (chance of being good: {prob_good * 100:.1f}%)")
    else:
        st.error(f"This wine is predicted to be **NOT GOOD**  (chance of being good: {prob_good * 100:.1f}%)")
    st.progress(prob_good)
    st.metric("Predicted quality score (scale 3 to 8)", f"{score:.2f}")

    out_of_range = [LABELS[f][0] for f in features
                    if values[f] < saved["feature_min"][f] or values[f] > saved["feature_max"][f]]
    if out_of_range:
        st.warning("These values are outside the range of the training data, so the prediction may be less "
                   "reliable: " + ", ".join(out_of_range))

with st.expander("About this app"):
    st.write(
        f"- Good / not good is predicted by **{saved['classifier_name']}**.\n"
        f"- The quality score is predicted by **{saved['regressor_name']}**.\n"
        "- Both models were trained on the red wine *Vinho Verde* dataset (UCI Machine Learning Repository) "
        "after removing duplicate rows and capping outliers.\n"
        "- A lab test cannot measure taste, so treat the result as an estimate, not a final verdict."
    )
    st.caption("Project by Roshan Shaw, Rablin Kaur, Shahzad Gufran and Sneha Kumari Sharma - "
               "Asansol Engineering College. Guide: Mr. Animesh Ojha.")
