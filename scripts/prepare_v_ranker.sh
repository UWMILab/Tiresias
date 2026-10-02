#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: bash scripts/prepare_v_ranker.sh --domain <all|navigation|driving|gaming|robot>"
  exit 0
fi
if [ "${1:-}" != "--domain" ] || [ -z "${2:-}" ]; then
  echo "Usage: bash scripts/prepare_v_ranker.sh --domain <all|navigation|driving|gaming|robot>" >&2
  exit 1
fi

case "$2" in
  all) domains=(navigation driving gaming robot) ;;
  navigation|driving|gaming|robot) domains=("$2") ;;
  *) echo "Unknown domain: $2" >&2; exit 1 ;;
esac
shift 2
if [ "$#" -ne 0 ]; then
  echo "Usage: bash scripts/prepare_v_ranker.sh --domain <all|navigation|driving|gaming|robot>" >&2
  exit 1
fi

cd "${ROOT}"
for domain in "${domains[@]}"; do
  processed="data/processed/${domain}"
  ranker_dir="outputs/training/rankers/${domain}"
  mkdir -p "${ranker_dir}"

  python -m lcwm.datasets.build_config \
    --domain "${domain}" \
    --wan-model-root pretrained_models/Wan2.2-TI2V-5B \
    --dinov3-repo pretrained_models/dinov3 \
    --dinov3-checkpoint pretrained_models/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m.pth \
    --wm-checkpoint "weights/world_model/${domain}/model.safetensors" \
    --output "${ranker_dir}/wm_config.json"

  for split in train val_seen; do
    tr_command=(python -m rankers.tr_ranker.inference \
      --domain "${domain}" \
      --split "${split}" \
      --candidate-shards "${ranker_dir}/${split}_candidates.pt" \
      --language-cache "${processed}/language_${split}.pt" \
      --tr-checkpoint "weights/rankers/${domain}/128_to_8/model.pt" \
      --output "${ranker_dir}/${split}_top8_selection.pt")
    if [ "${domain}" = gaming ] || [ "${domain}" = robot ]; then
      tr_command+=(--action-config "weights/diffusion_policy/${domain}/proposal/action_config.json")
    fi
    "${tr_command[@]}"

    python -m rankers.fusion.shortlist \
      --domain "${domain}" \
      --split "${split}" \
      --candidate-shards "${ranker_dir}/${split}_candidates.pt" \
      --selection "${ranker_dir}/${split}_top8_selection.pt" \
      --output "${ranker_dir}/${split}_top8.json"

    python -m rankers.dataset.build_stage2_initial_observations \
      --domain "${domain}" \
      --split "${split}" \
      --source-manifest "${processed}/planning/${split}.jsonl" \
      --output "${ranker_dir}/${split}_observations.json"

    evidence_command=(python -m rankers.v_ranker.build_evidence \
      --domain "${domain}" \
      --split "${split}" \
      --top8 "${ranker_dir}/${split}_top8.json" \
      --initial-observations "${ranker_dir}/${split}_observations.json" \
      --wm-config "${ranker_dir}/wm_config.json" \
      --output "${ranker_dir}/${split}_evidence.pt")
    if [ "${domain}" = gaming ] || [ "${domain}" = robot ]; then
      evidence_command+=(--action-config "weights/world_model/${domain}/action_config.json")
    fi
    "${evidence_command[@]}"

    python -m rankers.v_ranker.build_labels \
      --domain "${domain}" \
      --split "${split}" \
      --top8 "${ranker_dir}/${split}_top8.json" \
      --candidate-labels "${ranker_dir}/${split}_labels.pt" \
      --target-manifest "${processed}/planning/${split}.jsonl" \
      --output "${ranker_dir}/${split}_v_labels.json"
  done
done
