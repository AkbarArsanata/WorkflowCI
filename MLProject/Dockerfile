# MLProject/Dockerfile

# Base image Micromamba versi 1.5.1 yang stabil.
# Image ini tidak memiliki Python bawaan; versi Python akan diinstal via conda.yaml.
FROM mambaorg/micromamba:1.5.1

# Tetapkan direktori kerja di dalam container.
WORKDIR /app

# Salin file lingkungan Conda dari lokasinya di MLProject/ ke dalam container.
COPY MLProject/conda.yaml .

# Buat lingkungan Conda dari conda.yaml dan bersihkan cache.
# Ini akan menginstal Python 3.9 (sesuai definisi di conda.yaml) beserta dependensi lainnya.
RUN micromamba create -f conda.yaml -y && micromamba clean --all

# Setel shell default untuk menjalankan perintah di lingkungan Conda yang baru dibuat.
SHELL ["/bin/bash", "-c"]

# Instal Gunicorn (untuk web server) dan MLflow di lingkungan Conda.
# `micromamba run -n mlflow_churn_env` memastikan instalasi dilakukan di lingkungan yang benar.
RUN micromamba run -n mlflow_churn_env pip install gunicorn mlflow

# Paparkan port yang akan digunakan oleh MLflow scoring server.
# Default MLflow models serve berjalan di port 8080.
EXPOSE 8080

# Salin folder 'mlruns' dari root repositori ke dalam container.
# Karena konteks build adalah root repositori, kita bisa langsung merujuk 'mlruns'.
COPY mlruns /app/mlruns

# Salin file-file proyek lainnya dari folder MLProject/ ke dalam container.
COPY MLProject/modelling.py .
COPY MLProject/MLProject .

# Salin file data dari root repositori (asumsi data ada di root).
COPY churn_train_preprocessed.csv .
COPY churn_test_preprocessed.csv .

# Deklarasikan build argument untuk MLflow Run ID.
# Ini akan diteruskan saat proses `docker build` dari GitHub Actions.
ARG MLFLOW_RUN_ID

# Perintah default yang akan dijalankan saat container Docker dimulai.
# Meluncurkan MLflow scoring server menggunakan model dari Run ID spesifik.
CMD ["micromamba", "run", "-n", "mlflow_churn_env", "mlflow", "models", "serve", "-m", "runs:/${MLFLOW_RUN_ID}/model", "--host", "0.0.0.0", "--port", "8080"]
