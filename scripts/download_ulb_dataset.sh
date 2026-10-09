#!/usr/bin/env bash
set -euo pipefail

# Requires a configured Kaggle API token. The raw file is intentionally ignored
# by Git because it is large and subject to the dataset's published terms.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_dir="${project_root}/data/raw"
mkdir -p "${raw_dir}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Install/configure the Kaggle CLI first: https://github.com/Kaggle/kaggle-api" >&2
  exit 1
fi

kaggle datasets download -d mlg-ulb/creditcardfraud -p "${raw_dir}"
unzip -o "${raw_dir}/creditcardfraud.zip" -d "${raw_dir}"
rm -f "${raw_dir}/creditcardfraud.zip"
test -s "${raw_dir}/creditcard.csv"
echo "Downloaded ${raw_dir}/creditcard.csv"

