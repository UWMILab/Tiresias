#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: bash scripts/train_wm.sh --domain <all|navigation|driving|gaming|robot> [options]"
  exit 0
fi
if [ "${1:-}" != "--domain" ] || [ -z "${2:-}" ]; then
  echo "Usage: bash scripts/train_wm.sh --domain <all|navigation|driving|gaming|robot> [options]" >&2
  exit 1
fi

domain=$2
shift 2
num_processes=4
options=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --num-processes)
      num_processes=$2
      shift 2
      ;;
    --epochs|--learning-rate|--data_ratio|--num-workers|--tokenizer-path)
      options+=("$1" "$2")
      shift 2
      ;;
    --dry-run)
      options+=("$1")
      shift
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
  python -m lcwm.main \
    --domain "${domain}" \
    --dataset-root "data/processed/${domain}/wm" \
    --train-manifest "data/processed/${domain}/wm/train.jsonl" \
    --semantic-latent-root "data/processed/${domain}/features" \
    --wan-model-root pretrained_models/Wan2.2-TI2V-5B \
    --output-dir "weights/world_model/${domain}" \
    --num-processes "${num_processes}" \
    ${options[@]+"${options[@]}"}
done
