import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

before_metrics = []

for _ in range(3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    before_metrics.append(accuracy_score(y_test, y_pred))

after_metrics = []

for _ in range(3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=0.5
    )

    cv_score = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    after_metrics.append(accuracy_score(y_test, y_pred))

final_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

print("Before:", before_metrics)
print("After:", after_metrics)
print("CV Accuracy:", cv_score)
print("Final Metrics:", final_metrics)
