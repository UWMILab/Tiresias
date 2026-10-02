#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: bash scripts/train_tr_ranker.sh --domain <all|navigation|driving|gaming|robot> [options]"
  exit 0
fi
if [ "${1:-}" != "--domain" ] || [ -z "${2:-}" ]; then
  echo "Usage: bash scripts/train_tr_ranker.sh --domain <all|navigation|driving|gaming|robot> [options]" >&2
  exit 1
fi

domain=$2
shift 2
options=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --epochs|--batch-size|--learning-rate|--data_ratio|--seed)
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
  processed="data/processed/${domain}"
  ranker_dir="outputs/training/rankers/${domain}"
  command=(python -m rankers.tr_ranker.train \
    --domain "${domain}" \
    --train-candidates "${ranker_dir}/train_candidates.pt" \
    --train-labels "${ranker_dir}/train_labels.pt" \
    --train-cache "${processed}/language_train.pt" \
    --validation-candidates "${ranker_dir}/val_seen_candidates.pt" \
    --validation-labels "${ranker_dir}/val_seen_labels.pt" \
    --validation-cache "${processed}/language_val_seen.pt" \
    --output-dir "weights/rankers/${domain}/128_to_8")
  if [ "${domain}" = gaming ] || [ "${domain}" = robot ]; then
    command+=(--action-config "weights/diffusion_policy/${domain}/proposal/action_config.json")
  fi
  command+=(${options[@]+"${options[@]}"})
  "${command[@]}"
done
