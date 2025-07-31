import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump

def load_data(path='data/processed/data.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df

def preprocess_data(df, target_column='target'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
    print("Accuracy:", acc)
    return acc

def main():
    mlflow.set_experiment("Churn Prediction")  # Ganti sesuai eksperimenmu
    with mlflow.start_run() as run:
        print("Run ID:", run.info.run_id)

        df = load_data()
        X, y, scaler = preprocess_data(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        acc = evaluate_model(model, X_test, y_test)

        # Logging metrics and artifacts
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        # Save scaler
        os.makedirs("outputs", exist_ok=True)
        dump(scaler, "outputs/scaler.joblib")
        mlflow.log_artifact("outputs/scaler.joblib")

        print("MLflow Run Completed.")

if __name__ == "__main__":
    main()
