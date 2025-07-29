# MLProject/Dockerfile

FROM mambaorg/micromamba:1.5.1

WORKDIR /app

# Salin file lingkungan Conda ke dalam container dari lokasinya di root repo
COPY MLProject/conda.yaml . # <--- PERHATIKAN INI

RUN micromamba create -f conda.yaml -y && micromamba clean --all

SHELL ["/bin/bash", "-c"]

RUN micromamba run -n mlflow_churn_env pip install gunicorn mlflow

EXPOSE 8080

# Salin folder 'mlruns' dari root repositori
COPY mlruns /app/mlruns # <--- PERHATIKAN INI (tanpa '../')

# Salin file-file proyek lainnya dari lokasinya di root repo
COPY MLProject/modelling.py .      # <--- PERHATIKAN INI
COPY MLProject/MLProject .         # <--- PERHATIKAN INI
COPY churn_train_preprocessed.csv . # Jika data di root
COPY churn_test_preprocessed.csv .  # Jika data di root

ARG MLFLOW_RUN_ID

CMD ["micromamba", "run", "-n", "mlflow_churn_env", "mlflow", "models", "serve", "-m", "runs:/${MLFLOW_RUN_ID}/model", "--host", "0.0.0.0", "--port", "8080"]
