#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: bash scripts/train_ldp.sh --domain <all|navigation|driving|gaming|robot> [options]"
  exit 0
fi
if [ "${1:-}" != "--domain" ] || [ -z "${2:-}" ]; then
  echo "Usage: bash scripts/train_ldp.sh --domain <all|navigation|driving|gaming|robot> [options]" >&2
  exit 1
fi

domain=$2
shift 2
options=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --updates|--batch-size|--learning-rate|--data_ratio|--seed)
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
  python -m ldp.train \
    --domain "${domain}" \
    --train-manifest "data/processed/${domain}/planning/train.jsonl" \
    --language-cache "data/processed/${domain}/language_train.pt" \
    --output-dir "weights/diffusion_policy/${domain}/proposal" \
    ${options[@]+"${options[@]}"}
done
