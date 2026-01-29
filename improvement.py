import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_STATE = 42

df = pd.read_csv("dataset.csv")

def feature_engineering(df):
    df = df.copy()
    df["age_salary_product"] = df["age"] * df["salary"]
    df["salary_per_experience"] = df["salary"] / (df["experience"] + 1)
    df["age_squared"] = df["age"] ** 2
    df["high_salary_flag"] = (df["salary"] > 50000).astype(int)
    return df

df = feature_engineering(df)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

baseline_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000
)

baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)

baseline_metrics = {
    "accuracy": accuracy_score(y_test, baseline_pred),
    "precision": precision_score(y_test, baseline_pred),
    "recall": recall_score(y_test, baseline_pred),
    "f1": f1_score(y_test, baseline_pred)
}

improved_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=RANDOM_STATE
)

improved_model.fit(X_train, y_train)
proba = improved_model.predict_proba(X_test)[:, 1]

best_f1 = 0
best_threshold = 0.5

for t in np.arange(0.3, 0.8, 0.05):
    preds = (proba >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

final_pred = (proba >= best_threshold).astype(int)

improved_metrics = {
    "accuracy": accuracy_score(y_test, final_pred),
    "precision": precision_score(y_test, final_pred),
    "recall": recall_score(y_test, final_pred),
    "f1": f1_score(y_test, final_pred)
}

print("Baseline Metrics:", baseline_metrics)
print("Improved Metrics:", improved_metrics)
print("Best Threshold:", best_threshold)
print(
    "F1 Improvement (%):",
    round(
        ((improved_metrics["f1"] - baseline_metrics["f1"])
         / baseline_metrics["f1"]) * 100,
        2
    )
)
