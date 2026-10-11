#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${project_root}/../.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="${PYTHON_BIN:-python}"
fi
export PYTHONPATH="${project_root}/src"

"${python_bin}" -m compileall -q src tests dags scripts
"${python_bin}" -m pytest -q
ruff check src tests dags scripts
"${python_bin}" -m finguard.benchmark \
  --input data/raw/creditcard.csv \
  --output reports/benchmark_ulb.json \
  --tune-xgboost \
  --cv-splits 3 \
  --random-search-iterations 4 \
  --alert-rate 0.01

if [[ -z "${JAVA_HOME:-}" ]]; then
  echo "JAVA_HOME must point to Java 17 for Spark; CI configures this automatically." >&2
  exit 1
fi
JAVA_HOME="${JAVA_HOME}" PATH="${JAVA_HOME}/bin:${PATH}" \
  "${python_bin}" -m finguard.phase1.eda \
  --input data/raw/creditcard.csv --output reports/phase1 --master 'local[2]'
