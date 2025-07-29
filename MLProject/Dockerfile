# MLProject/Dockerfile

# Base image dengan Python dan Conda. Sesuaikan versi Python jika perlu.
# Gunakan Miniconda sebagai base image untuk kemudahan instalasi Conda.
FROM continuumio/miniconda3:py39_4.12.0

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin conda.yaml ke dalam container
COPY conda.yaml .

# Buat lingkungan Conda dari conda.yaml
RUN conda env create -f conda.yaml && conda clean --all

# Aktifkan lingkungan Conda
# SHELL ["conda", "run", "-n", "mlflow_churn_env", "/bin/bash", "-c"]
# Gunakan conda run untuk menjalankan perintah di lingkungan yang benar
SHELL ["/bin/bash", "--login", "-c"]

# Masuk ke lingkungan Conda dan instal gunicorn dan mlflow
ENV PATH /opt/conda/envs/mlflow_churn_env/bin:$PATH
RUN conda activate mlflow_churn_env && \
    pip install gunicorn mlflow

# Paparkan port yang akan digunakan oleh MLflow scoring server
EXPOSE 8080

# Salin modelling.py, data, dan file MLProject lainnya yang mungkin diperlukan
# untuk inferensi, meskipun model utama akan diambil dari MLflow Tracking Server.
COPY modelling.py .
COPY MLProject .
COPY churn_train_preprocessed.csv .
COPY churn_test_preprocessed.csv .

# Perintah default untuk menjalankan MLflow scoring server
# Anda perlu mengganti <MLFLOW_RUN_ID_ANDA> dengan Run ID yang sebenarnya.
# MLflow akan mengunduh model dari tracking server (file system di kasus ini)
# dan menyajikan inferensi.
CMD ["mlflow", "models", "serve", "-m", "runs:/<MLFLOW_RUN_ID_ANDA>/model", "--host", "0.0.0.0", "--port", "8080"]