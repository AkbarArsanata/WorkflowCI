import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
import numpy as np
import os

# --- ATUR TRACKING URI DI SINI ---
# PENTING: Baris ini sudah dikomentari/dihapus karena kita menggunakan backend
# file system MLflow lokal untuk GitHub Actions.
# Jika Anda mengaktifkan ini kembali dan ingin terhubung ke server MLflow eksternal,
# pastikan URI sudah benar (misalnya, "https://your-mlflow-server.com").
# mlflow.set_tracking_uri("http://localhost:5000")
# ---------------------------------

# --- 1. Memuat Data ---
# Pastikan 'churn_train_preprocessed.csv' dan 'churn_test_preprocessed.csv'
# berada di lokasi yang dapat diakses oleh skrip ini.
# Jika dataset berada di folder MLProject/ (bersama modelling.py), path ini benar.
# Jika dataset berada di root repositori, pastikan Anda menyesuaikan path ini
# (misalnya, menjadi '../churn_train_preprocessed.csv').
try:
    df_train = pd.read_csv('churn_train_preprocessed.csv')
    df_val = pd.read_csv('churn_test_preprocessed.csv')
except FileNotFoundError as e:
    print(f"ERROR: File not found. Check if 'churn_train_preprocessed.csv' and 'churn_test_preprocessed.csv' are in the correct directory.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Error details: {e}")
    # Berikan detail lebih lanjut tentang isi direktori untuk debugging
    print("Files in current directory:", os.listdir(os.getcwd()))
    exit(1) # Keluar jika file tidak ditemukan

print("Data train head:\n", df_train.head())
print("Data validation head:\n", df_val.head())

# --- 2. Split Data (X dan y) ---
# 'Exited' adalah kolom target (y)

X_train = df_train.drop('Exited', axis=1)
y_train = df_train['Exited']

X_val = df_val.drop('Exited', axis=1)
y_val = df_val['Exited']

# Pastikan kolom X_train dan X_val sama
if not X_train.columns.equals(X_val.columns):
    common_cols = list(set(X_train.columns) & set(X_val.columns))
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    print("Kolom disesuaikan agar cocok antara X_train dan X_val.")


# --- 3. Training Menggunakan Beberapa Model Klasifikasi dengan MLflow Tracking ---

# Daftar model klasifikasi yang akan dibandingkan
models = {
    "Random Forest Classifier": RandomForestClassifier(random_state=42)
}

# Mengaktifkan MLflow autologging untuk scikit-learn
# Ini akan secara otomatis mencatat parameter, metrik, dan model
mlflow.sklearn.autolog(log_input_examples=True, log_models=True)

# Loop melalui setiap model
for model_name, model in models.items():
    # Memulai MLflow run. MLflow akan mencatat ke folder 'mlruns/' secara default
    # jika tidak ada tracking URI eksternal yang disetel.
    with mlflow.start_run(run_name=f"{model_name}_Training_Val"):
        print(f"\n--- Melatih {model_name} ---")

        # Log ukuran dataset sebagai parameter
        mlflow.log_param("training_set_size", len(X_train))
        mlflow.log_param("validation_set_size", len(X_val))

        # Latih model
        model.fit(X_train, y_train)

        # Prediksi pada X_val (data validasi)
        y_val_pred = model.predict(X_val)

        # Prediksi probabilitas untuk ROC AUC
        y_val_proba = None
        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val)[:, 1]

        # --- 4. Model Evaluation pada Data Validasi ---
        # Hitung metrik evaluasi
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

        # Log metrik ke MLflow
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("val_recall", val_recall)
        mlflow.log_metric("val_f1_score", val_f1)

        print(f"Akurasi Validasi: {val_accuracy:.4f}")
        print(f"Presisi Validasi: {val_precision:.4f}")
        print(f"Recall Validasi: {val_recall:.4f}")
        print(f"F1-Score Validasi: {val_f1:.4f}")

        if y_val_proba is not None:
            try:
                val_roc_auc = roc_auc_score(y_val, y_val_proba)
                print(f"ROC AUC Validasi: {val_roc_auc:.4f}")
                mlflow.log_metric("val_roc_auc_score", val_roc_auc)
            except ValueError as e:
                print(f"Tidak dapat menghitung ROC AUC untuk set validasi: {e}")

        # Catat confusion matrix sebagai artefak MLflow DAN simpan lokal untuk GitHub Actions
        cm = confusion_matrix(y_val, y_val_pred)
        cm_filename = f"confusion_matrix_Random_Forest_Classifier.csv" # Nama file yang konsisten
        np.savetxt(cm_filename, cm, delimiter=",", fmt="%d")

        # Log file confusion matrix ke MLflow
        mlflow.log_artifact(cm_filename)
        print(f"Confusion Matrix untuk {model_name} (Set Validasi) disimpan sebagai artefak MLflow.")

        # PENTING: Baris di bawah ini yang sebelumnya menghapus file lokal,
        # sekarang dihapus atau dikomentari agar GitHub Actions bisa mengunggahnya.
        # if os.path.exists(cm_filename):
        #    os.remove(cm_filename)
        #    print(f"File lokal '{cm_filename}' dihapus.")

        # Tambahkan tag kustom ke MLflow run
        mlflow.set_tag("Model Type", model_name)
        mlflow.set_tag("Dataset Split", "Train-Validation")

        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

print("\nPelatihan model selesai. Untuk melihat hasil, unduh artefak 'mlflow-tracking-data' dari GitHub Actions.")
print("Kemudian, jalankan 'mlflow ui --backend-store-uri file:///path/to/downloaded/mlruns' secara lokal.")
