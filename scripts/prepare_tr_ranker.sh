#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: bash scripts/prepare_tr_ranker.sh --domain <all|navigation|driving|gaming|robot>"
  exit 0
fi
if [ "${1:-}" != "--domain" ] || [ -z "${2:-}" ]; then
  echo "Usage: bash scripts/prepare_tr_ranker.sh --domain <all|navigation|driving|gaming|robot>" >&2
  exit 1
fi

case "$2" in
  all) domains=(navigation driving gaming robot) ;;
  navigation|driving|gaming|robot) domains=("$2") ;;
  *) echo "Unknown domain: $2" >&2; exit 1 ;;
esac
shift 2
if [ "$#" -ne 0 ]; then
  echo "Usage: bash scripts/prepare_tr_ranker.sh --domain <all|navigation|driving|gaming|robot>" >&2
  exit 1
fi

cd "${ROOT}"
for domain in "${domains[@]}"; do
  processed="data/processed/${domain}"
  ranker_dir="outputs/training/rankers/${domain}"
  mkdir -p "${ranker_dir}"

  for split in train val_seen; do
    command=(python -m ldp.generate \
      --domain "${domain}" \
      --split "${split}" \
      --checkpoint "weights/diffusion_policy/${domain}/proposal/LDP.pt" \
      --inputs "${processed}/planning/${split}.json" \
      --language-cache "${processed}/language_${split}.pt" \
      --output "${ranker_dir}/${split}_candidates.pt")
    if [ "${domain}" = gaming ] || [ "${domain}" = robot ]; then
      command+=(--action-config "weights/diffusion_policy/${domain}/proposal/action_config.json")
    fi
    "${command[@]}"

    python -m rankers.dataset.build_candidate_metric_labels \
      --domain "${domain}" \
      --split "${split}" \
      --candidate-shards "${ranker_dir}/${split}_candidates.pt" \
      --target-manifest "${processed}/planning/${split}.jsonl" \
      --output "${ranker_dir}/${split}_labels.pt"
  done
done
