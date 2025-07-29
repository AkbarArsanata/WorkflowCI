# MLProject/Dockerfile

# Menggunakan micromamba sebagai base image. Ini sangat ringan dan efisien.
# Pilih tag yang sesuai dengan versi Python Anda, misal: "1.5.1-python=3.9"
# Anda bisa cek tag terbaru di: https://hub.docker.com/r/mambaorg/micromamba/tags
FROM mambaorg/micromamba:1.5.1-python=3.9

# Tetapkan direktori kerja di dalam container.
WORKDIR /app

# Salin file lingkungan Conda ke dalam container.
COPY conda.yaml .

# Buat lingkungan Conda dari conda.yaml menggunakan micromamba.
# Micromamba menggunakan `micromamba install` bukan `conda env create`.
RUN micromamba create -f conda.yaml -y && micromamba clean --all

# Setel shell default untuk menjalankan perintah di lingkungan Conda yang baru dibuat.
# `micromamba shell init -p /opt/micromamba` akan menginisialisasi shell.
# `micromamba activate mlflow_churn_env` akan mengaktifkan lingkungan.
SHELL ["/bin/bash", "-c"]

# Aktifkan lingkungan dan instal gunicorn dan mlflow.
# Gunakan `micromamba run` untuk menjalankan perintah di lingkungan yang diaktifkan.
RUN micromamba run -n mlflow_churn_env pip install gunicorn mlflow

# Paparkan port yang akan digunakan oleh MLflow scoring server.
EXPOSE 8080

# --- PENTING: Salin folder 'mlruns' ke dalam image Docker ---
# Pastikan 'mlruns/' ada di root repositori Anda saat build Docker dijalankan.
COPY mlruns /app/mlruns

# Salin file-file proyek lainnya.
COPY modelling.py .
COPY MLProject .
COPY churn_train_preprocessed.csv .
COPY churn_test_preprocessed.csv .

# Deklarasikan build argument untuk MLflow Run ID (untuk tagging).
ARG MLFLOW_RUN_ID

# Perintah default yang akan dijalankan saat container Docker dimulai.
# Menggunakan `micromamba run` untuk memastikan MLflow dijalankan di lingkungan yang benar.
# `MLFLOW_RUN_ID` akan disetel sebagai environment variable saat runtime.
CMD ["micromamba", "run", "-n", "mlflow_churn_env", "mlflow", "models", "serve", "-m", "runs:/${MLFLOW_RUN_ID}/model", "--host", "0.0.0.0", "--port", "8080"]
