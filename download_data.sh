#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Replace this value with the published Hugging Face dataset repository ID.
REPO_ID="ORGANIZATION/DATASET"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: bash download_data.sh"
  exit 0
fi
if [ "$#" -ne 0 ]; then
  echo "Usage: bash download_data.sh" >&2
  exit 1
fi
if [ "${REPO_ID}" = "ORGANIZATION/DATASET" ]; then
  echo "The Hugging Face dataset repository has not been configured yet." >&2
  exit 1
fi

mkdir -p "${ROOT}/data"
domains=(navigation driving gaming robot)
splits=(train val_seen val_unseen)

for domain in "${domains[@]}"; do
  for split in "${splits[@]}"; do
    target="${ROOT}/data/${domain}/${split}"
    if [ -e "${target}" ]; then
      echo "Data already exists: ${target}" >&2
      exit 1
    fi
  done
done

for domain in "${domains[@]}"; do
  directory="${ROOT}/data/${domain}"
  mkdir -p "${directory}"

  for split in "${splits[@]}"; do
    relative="${domain}/${split}.tar"
    archive="${directory}/${split}.tar"

    echo "[DOWNLOAD] ${relative}"
    wget -c \
      "https://huggingface.co/datasets/${REPO_ID}/resolve/main/${relative}" \
      -O "${archive}"

    echo "[EXTRACT] ${relative}"
    tar -xf "${archive}" -C "${directory}"

    echo "[CLEAN] removing ${relative}"
    rm -f "${archive}"
  done
done

echo "[DONE] Tiresias data is ready under ${ROOT}/data/"
