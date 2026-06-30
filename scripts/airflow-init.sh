#!/usr/bin/env bash
set -euo pipefail

airflow db migrate
airflow users create \
  --username "${AIRFLOW_ADMIN_USERNAME:-admin}" \
  --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" \
  --firstname Airflow \
  --lastname Admin \
  --role Admin \
  --email admin@example.com || true
