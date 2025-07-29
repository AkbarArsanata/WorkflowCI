# MLProject/Dockerfile

# Base image Micromamba versi 1.5.1 yang stabil.
# Versi Python akan diinstal via conda.yaml.
FROM mambaorg/micromamba:1.5.1

# Tetapkan direktori kerja di dalam container.
WORKDIR /app

# Salin file lingkungan Conda dari lokasinya di MLProject/ ke dalam container.
COPY MLProject/conda.yaml .

# Buat lingkungan Conda dan bersihkan cache.
RUN micromamba create -f conda.yaml -y && micromamba clean --all

# Setel shell default untuk menjalankan perintah di lingkungan Conda yang baru dibuat.
SHELL ["/bin/bash", "-c"]

# Instal Gunicorn dan MLflow di lingkungan Conda.
RUN micromamba run -n mlflow_churn_env pip install gunicorn mlflow

# Paparkan port MLflow scoring server.
EXPOSE 8080

# Salin folder 'mlruns' dari root repositori ke dalam container.
COPY mlruns /app/mlruns

# Salin file-file proyek lainnya dari MLProject/ ke dalam container.
COPY MLProject/modelling.py .
COPY MLProject/MLProject .

# --- PERBAIKAN DI SINI: Salin file data dari MLProject/ ---
# Karena file data berada di dalam folder MLProject/, kita perlu
# menunjuk path relatif yang benar dari konteks build (root repositori).
COPY MLProject/churn_train_preprocessed.csv .
COPY MLProject/churn_test_preprocessed.csv .

# Deklarasikan build argument untuk MLflow Run ID.
ARG MLFLOW_RUN_ID

# Perintah default untuk menjalankan MLflow scoring server.
CMD ["micromamba", "run", "-n", "mlflow_churn_env", "mlflow", "models", "serve", "-m", "runs:/${MLFLOW_RUN_ID}/model", "--host", "0.0.0.0", "--port", "8080"]
