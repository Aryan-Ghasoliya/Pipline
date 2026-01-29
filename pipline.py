import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MODEL_PATH = "trained_model.pkl"

def validate_data(df):
    if df.empty:
        raise ValueError("Dataset is empty")
    if "target" not in df.columns:
        raise ValueError("Target column missing")
    if df.isnull().mean().max() > 0.5:
        raise ValueError("Too many missing values")
    if df.shape[0] < 100:
        raise ValueError("Not enough samples")

def feature_engineering(df):
    df = df.copy()
    df["age_salary_product"] = df["age"] * df["salary"]
    df["salary_per_experience"] = df["salary"] / (df["experience"] + 1)
    df["age_squared"] = df["age"] ** 2
    df["high_salary_flag"] = (df["salary"] > 50000).astype(int)
    return df

def preprocess_data(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def get_model():
    return LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000
    )

def train_model(model, X_train, y_train):
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="accuracy"
    )
    model.fit(X_train, y_train)
    return model, cv_scores.mean()

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def main():
    df = pd.read_csv("dataset.csv")
    validate_data(df)
    df = feature_engineering(df)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    X_train, X_test = preprocess_data(X_train, X_test)
    model = get_model()
    model, cv_score = train_model(model, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)
    print("CV Accuracy:", cv_score)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
