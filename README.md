# Wine Quality Predictor

A machine learning project that predicts the quality of red wine from its lab test values.
It answers two questions:

1. **Is the wine good?** (quality 6 or more = good) - classification
2. **What is its quality score?** (3 to 8) - regression

Made by Roshan Shaw, Rablin Kaur, Shahzad Gufran and Sneha Kumari Sharma
Department of Computer Applications, Asansol Engineering College
Guide: Mr. Animesh Ojha

## What is inside

| File / folder | What it is |
|---|---|
| `Wine_Quality_Predictor.ipynb` | The full project in a Jupyter notebook (data cleaning, graphs, models, results) |
| `app.py` | The web app (Streamlit) - type the wine values, get a prediction |
| `models/wine_models.joblib` | The trained models used by the app |
| `data/winequality-red.csv` | The dataset (UCI Machine Learning Repository, Cortez et al. 2009) |
| `images/` | All graphs made by the notebook |
| `results/` | Result tables (accuracy, RMSE etc.) saved as csv |

## Data

1,599 red wines (1,359 after removing duplicate rows), 11 lab measurements for each wine
(fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide,
total sulfur dioxide, density, pH, sulphates, alcohol) and a quality score given by wine tasters.

## Models and results

Classification (25% of the data kept for testing):

| Model | Accuracy | AUC |
|---|---|---|
| KNN | 69.1% | 0.773 |
| Naive Bayes | 73.8% | 0.823 |
| Decision Tree | 72.1% | 0.787 |
| **Logistic Regression** | **74.1%** | **0.833** |

Regression (quality score):

| Model | RMSE | R-squared |
|---|---|---|
| KNN Regressor | 0.710 | 0.348 |
| Decision Tree Regressor | 0.719 | 0.330 |
| **Linear Regression** | **0.680** | **0.402** |

## How to run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

To open the notebook:

```bash
pip install -r requirements-notebook.txt
jupyter notebook Wine_Quality_Predictor.ipynb
```

## Limits

The data has only red wine and about 1,300 rows, and taste cannot be fully measured by a lab test,
so the predictions are estimates and not exact.
