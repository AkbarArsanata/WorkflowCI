import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import os
import shutil

# --- Load Data ---
try:
    df_train = pd.read_csv('churn_train_preprocessed.csv')
    df_val = pd.read_csv('churn_test_preprocessed.csv')
except FileNotFoundError as e:
    print("ERROR: CSV not found. Current directory:", os.getcwd())
    print("Files:", os.listdir(os.getcwd()))
    exit(1)

X_train = df_train.drop('Exited', axis=1)
y_train = df_train['Exited']
X_val = df_val.drop('Exited', axis=1)
y_val = df_val['Exited']

if not X_train.columns.equals(X_val.columns):
    common_cols = list(set(X_train.columns) & set(X_val.columns))
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]

models = {
    "Random Forest Classifier": RandomForestClassifier(random_state=42)
}

# Menonaktifkan MLflow autologging untuk menghindari masalah serialisasi YAML
# mlflow.sklearn.autolog(log_input_examples=True, log_models=True)

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name}_Training_Val") as run:
        # --- Log Parameter ---
        mlflow.log_param("training_set_size", len(X_train))
        mlflow.log_param("validation_set_size", len(X_val))

        # --- Pelatihan Model ---
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_val_proba = None

        # --- Hitung dan Log Metrik ---
        mlflow.log_metric("val_accuracy", accuracy_score(y_val, y_val_pred))
        mlflow.log_metric("val_precision", precision_score(y_val, y_val_pred, average='weighted', zero_division=0))
        mlflow.log_metric("val_recall", recall_score(y_val, y_val_pred, average='weighted', zero_division=0))
        mlflow.log_metric("val_f1_score", f1_score(y_val, y_val_pred, average='weighted', zero_division=0))

        if y_val_proba is not None:
            try:
                roc_auc = roc_auc_score(y_val, y_val_proba)
                mlflow.log_metric("val_roc_auc_score", roc_auc)
            except ValueError:
                pass

        # --- Buat dan Log Confusion Matrix ---
        cm = confusion_matrix(y_val, y_val_pred)
        cm_file = f"confusion_matrix_{model_name.replace(' ', '_')}.csv"
        np.savetxt(cm_file, cm, delimiter=",", fmt="%d")
        mlflow.log_artifact(cm_file)
        os.remove(cm_file) # Menghapus file lokal setelah dilog

        # --- Set Tag ---
        mlflow.set_tag("Model Type", model_name)
        mlflow.set_tag("Dataset Split", "Train-Validation")

        # --- Logging Model Manual dan Pendaftaran ke Model Registry ---
        model_name_for_registry = "ChurnPredictionModel"
        
        # Inferensi tanda tangan model
        signature = infer_signature(X_val, model.predict(X_val))

        # Log model secara manual
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name_for_registry, # 'artifact_path' is deprecated, but for clarity, using it here.
            registered_model_name=model_name_for_registry,
            signature=signature,
            input_example=X_val.head(5)
        )

        # Buat folder artifacts jika belum ada
        os.makedirs("artifacts", exist_ok=True)

        # Ambil run_id dari run yang aktif
        run_id = run.info.run_id

        # Path ke model dalam mlruns
        source_model_path = f"mlruns/0/{run_id}/artifacts/{model_name_for_registry}"

        # Pastikan directory model ada sebelum copy
        if not os.path.exists(source_model_path):
            raise FileNotFoundError(f"Model directory not found: {source_model_path}")

        # Salin model ke folder artifacts
        destination_path = f"artifacts/{model_name_for_registry}"
        shutil.copytree(source_model_path, destination_path, dirs_exist_ok=True)

        # Menggunakan MlflowClient untuk mengatur alias ke 'Production'
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Ambil versi model yang baru saja terdaftar (versi terbaru)
        latest_version_obj = client.get_latest_versions(model_name_for_registry, stages=["None"])[0]
        latest_version = latest_version_obj.version

        # Set alias 'Production' ke versi terbaru
        client.set_registered_model_alias(
            name=model_name_for_registry,
            alias="Production",
            version=latest_version
        )

        print(f"Model '{model_name_for_registry}' v{latest_version} has been registered and tagged with alias 'Production'.")
