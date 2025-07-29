# MLProject/Dockerfile

# Menggunakan base image Micromamba dengan Python 3.9 terbaru.
# Tag 'latest-py39' adalah format yang valid dan sering digunakan untuk Micromamba
# dengan versi Python 3.9. Ini lebih andal daripada tag yang spesifik atau tidak valid.
FROM mambaorg/micromamba:latest-py39

# Tetapkan direktori kerja di dalam container.
WORKDIR /app

# Salin file lingkungan Conda (conda.yaml) ke dalam container.
COPY conda.yaml .

# Buat lingkungan Conda dari conda.yaml menggunakan micromamba.
# `micromamba create` adalah perintah yang tepat untuk ini, dan `-y` untuk konfirmasi otomatis.
# `micromamba clean --all` akan membersihkan cache setelah instalasi untuk mengurangi ukuran image.
RUN micromamba create -f conda.yaml -y && micromamba clean --all

# Setel shell default untuk menjalankan perintah di lingkungan Conda yang baru dibuat.
# `bash -c` adalah cara umum dan aman untuk menjalankan perintah di shell di Docker.
SHELL ["/bin/bash", "-c"]

# Aktifkan lingkungan dan instal gunicorn serta mlflow.
# Gunakan `micromamba run -n mlflow_churn_env` untuk memastikan perintah dieksekusi
# dalam konteks lingkungan Conda yang telah dibuat.
RUN micromamba run -n mlflow_churn_env pip install gunicorn mlflow

# Paparkan port yang akan digunakan oleh MLflow scoring server.
EXPOSE 8080

# --- PENTING: Salin folder 'mlruns' ke dalam image Docker ---
# Ini sangat penting agar MLflow models serve dapat menemukan model Anda.
# Pastikan folder 'mlruns/' berada di root repositori Anda saat proses build Docker.
# Docker akan menyalinnya dari konteks build (yaitu folder MLProject/ saat ini)
# ke dalam image di /app/mlruns.
COPY mlruns /app/mlruns

# Salin file-file proyek lainnya yang mungkin diperlukan oleh model server
# atau untuk referensi di dalam container.
# Pastikan file-file ini berada di lokasi yang benar relatif terhadap konteks build (MLProject/).
COPY modelling.py .
COPY MLProject .
COPY churn_train_preprocessed.csv .
COPY churn_test_preprocessed.csv .

# Deklarasikan build argument untuk MLflow Run ID.
# Ini akan digunakan untuk menandai (tagging) Docker image dengan ID run yang spesifik,
# dan juga akan digunakan di perintah CMD saat container dijalankan.
ARG MLFLOW_RUN_ID

# Perintah default yang akan dijalankan saat container Docker dimulai.
# Ini akan meluncurkan MLflow scoring server untuk menyajikan model yang telah dilatih.
# Model diidentifikasi menggunakan URI "runs:/${MLFLOW_RUN_ID}/model".
# `micromamba run -n mlflow_churn_env` memastikan server dijalankan dalam lingkungan yang benar.
CMD ["micromamba", "run", "-n", "mlflow_churn_env", "mlflow", "models", "serve", "-m", "runs:/${MLFLOW_RUN_ID}/model", "--host", "0.0.0.0", "--port", "8080"]
