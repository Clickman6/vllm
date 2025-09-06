FROM ghcr.io/mlflow/mlflow:latest

COPY --from=curlimages/curl:latest /usr/bin/curl /usr/bin/curl
COPY --from=curlimages/curl:latest /lib/ /lib/
COPY --from=curlimages/curl:latest /usr/lib/ /usr/lib/

ARG MLFLOW_PORT=5000
ENV MLFLOW_PORT=${MLFLOW_PORT}

RUN pip install --no-cache-dir mlflow[extras] boto3 psycopg2-binary

HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD curl -f http://localhost:${MLFLOW_PORT}/health || exit 1