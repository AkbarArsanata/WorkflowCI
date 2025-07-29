# MLProject/Dockerfile

# Menggunakan base image Miniconda3 dengan Python 3.9 terbaru.
# Ini lebih andal daripada tag versi Conda yang sangat spesifik.
FROM continuumio/miniconda3:latest-py39

# Tetapkan direktori kerja di dalam container.
WORKDIR /app

# Salin file lingkungan Conda ke dalam container.
COPY conda.yaml .

# Buat lingkungan Conda dari conda.yaml dan bersihkan cache Conda.
# Ini memastikan semua dependensi proyek diinstal dalam lingkungan yang terisolasi.
RUN conda env create -f conda.yaml && conda clean --all

# Setel shell default untuk menjalankan perintah di lingkungan Conda yang baru dibuat.
# `bash --login -c` memastikan aktivasi Conda bekerja dengan benar.
SHELL ["/bin/bash", "--login", "-c"]

# Tambahkan jalur biner lingkungan Conda ke PATH agar perintah dapat ditemukan.
ENV PATH /opt/conda/envs/mlflow_churn_env/bin:$PATH

# Instal Gunicorn (digunakan oleh MLflow scoring server) dan MLflow di lingkungan Conda.
# Pastikan ini dilakukan SETELAH lingkungan diaktifkan.
RUN conda activate mlflow_churn_env && \
    pip install gunicorn mlflow

# Paparkan port yang akan digunakan oleh MLflow scoring server.
EXPOSE 8080

# --- PENTING: Salin folder 'mlruns' ke dalam image Docker ---
# MLflow models serve memerlukan data run (termasuk model) yang dicatat
# secara lokal. Folder 'mlruns' ini berisi semua artefak dan metadata
# dari run MLflow Anda. Pastikan 'mlruns/' ada di root repositori Anda
# saat build Docker dijalankan.
COPY mlruns /app/mlruns

# Salin file-file proyek lainnya yang mungkin diperlukan untuk menjalankan model
# atau sekadar untuk referensi di dalam container.
# Pastikan file-file ini berada di lokasi yang benar relatif terhadap konteks build (MLProject/).
COPY modelling.py .
COPY MLProject .
COPY churn_train_preprocessed.csv .
COPY churn_test_preprocessed.csv .


# Deklarasikan build argument untuk MLflow Run ID.
# Ini akan digunakan untuk menandai Docker image dengan ID run yang spesifik.
# (Catatan: build args tidak langsung tersedia di CMD, tapi akan digunakan untuk tagging).
ARG MLFLOW_RUN_ID

# Perintah default yang akan dijalankan saat container Docker dimulai.
# Ini akan meluncurkan MLflow scoring server untuk menyajikan model.
# Model diidentifikasi menggunakan URI "runs:/<MLFLOW_RUN_ID>/model".
# <MLFLOW_RUN_ID> di sini diharapkan akan disetel sebagai environment variable
# saat container dijalankan, atau sebagai bagian dari tag image saat build.
CMD ["mlflow", "models", "serve", "-m", "runs:/${MLFLOW_RUN_ID}/model", "--host", "0.0.0.0", "--port", "8080"]
