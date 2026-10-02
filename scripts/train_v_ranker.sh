#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: bash scripts/train_v_ranker.sh --domain <all|navigation|driving|gaming|robot> [options]"
  exit 0
fi
if [ "${1:-}" != "--domain" ] || [ -z "${2:-}" ]; then
  echo "Usage: bash scripts/train_v_ranker.sh --domain <all|navigation|driving|gaming|robot> [options]" >&2
  exit 1
fi

domain=$2
shift 2
options=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --epochs|--learning-rate|--data_ratio|--seed)
      options+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done
case "${domain}" in
  all) domains=(navigation driving gaming robot) ;;
  navigation|driving|gaming|robot) domains=("${domain}") ;;
  *) echo "Unknown domain: ${domain}" >&2; exit 1 ;;
esac

cd "${ROOT}"
for domain in "${domains[@]}"; do
  ranker_dir="outputs/training/rankers/${domain}"
  python -m rankers.v_ranker.train \
    --train-evidence "${ranker_dir}/train_evidence.pt" \
    --train-labels "${ranker_dir}/train_v_labels.json" \
    --validation-evidence "${ranker_dir}/val_seen_evidence.pt" \
    --validation-labels "${ranker_dir}/val_seen_v_labels.json" \
    --output-dir "weights/rankers/${domain}/8_to_1" \
    ${options[@]+"${options[@]}"}
done
