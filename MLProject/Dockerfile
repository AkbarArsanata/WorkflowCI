# MLProject/Dockerfile

# Menggunakan base image Micromamba versi 1.5.1 yang stabil.
# Image ini tidak memiliki Python bawaan; versi Python akan diinstal via conda.yaml.
FROM mambaorg/micromamba:1.5.1

# Tetapkan direktori kerja di dalam container.
# Semua operasi file (COPY, RUN) akan relatif terhadap direktori ini.
WORKDIR /app

# Salin file lingkungan Conda Anda (conda.yaml) ke dalam container.
COPY conda.yaml .

# Buat lingkungan Conda dari conda.yaml menggunakan micromamba.
# Micromamba akan membaca dependensi dari conda.yaml, termasuk versi Python.
# `micromamba create -f conda.yaml -y` akan membuat lingkungan.
# `micromamba clean --all` akan menghapus file-file sementara untuk mengurangi ukuran image.
RUN micromamba create -f conda.yaml -y && micromamba clean --all

# Setel shell default untuk menjalankan perintah di lingkungan Conda yang baru dibuat.
# `bash -c` adalah cara yang aman dan umum untuk menjalankan perintah dalam shell.
SHELL ["/bin/bash", "-c"]

# Aktifkan lingkungan Conda dan instal Gunicorn (untuk web server) dan MLflow.
# `micromamba run -n mlflow_churn_env` memastikan perintah ini dieksekusi
# dalam konteks lingkungan Conda bernama 'mlflow_churn_env' (dari conda.yaml).
RUN micromamba run -n mlflow_churn_env pip install gunicorn mlflow

# Paparkan port yang akan digunakan oleh MLflow scoring server.
# Secara default, MLflow models serve berjalan di port 8080.
EXPOSE 8080

# --- PENTING: Salin folder 'mlruns' ke dalam image Docker ---
# MLflow models serve perlu mengakses artefak model yang dicatat.
# Folder 'mlruns' ini berisi semua data tracking MLflow, termasuk model.
# Karena Dockerfile ini berada di MLProject/, dan folder 'mlruns' Anda
# biasanya dibuat di ROOT repositori, kita perlu naik satu direktori (../mlruns)
# untuk menemukannya dan menyalinnya ke /app/mlruns di dalam container.
COPY ../mlruns /app/mlruns

# Salin file-file proyek lainnya yang mungkin diperlukan oleh model server
# atau untuk referensi di dalam container.
# Pastikan jalur relatif ini benar dari konteks build (MLProject/).
COPY modelling.py .
COPY MLProject .
COPY churn_train_preprocessed.csv .
COPY churn_test_preprocessed.csv .

# Deklarasikan build argument untuk MLflow Run ID.
# Variabel ini akan diteruskan saat proses `docker build` dari GitHub Actions.
# Ini akan digunakan untuk menandai Docker image dan juga dalam perintah CMD.
ARG MLFLOW_RUN_ID

# Perintah default yang akan dijalankan saat container Docker dimulai.
# Ini meluncurkan MLflow scoring server.
# `micromamba run -n mlflow_churn_env` memastikan server berjalan di lingkungan yang benar.
# `-m runs:/${MLFLOW_RUN_ID}/model` menunjuk ke model yang dicatat MLflow menggunakan Run ID.
# `--host 0.0.0.0 --port 8080` membuat server dapat diakses dari luar container.
CMD ["micromamba", "run", "-n", "mlflow_churn_env", "mlflow", "models", "serve", "-m", "runs:/${MLFLOW_RUN_ID}/model", "--host", "0.0.0.0", "--port", "8080"]
