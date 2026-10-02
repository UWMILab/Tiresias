import argparse
import json
from pathlib import Path

import torch

from lcwm.datasets.protocol import INFERENCE_SPLITS
from ldp.action_utils import fuse_language_wm_top8
from rankers.v_ranker.video_ranker import (
    CHECKPOINT_FORMAT,
    VRanker,
    Config,
    batch,
    load_evidence,
)

def load_model(path: Path, domain: str, device: str) -> VRanker:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("domain") != domain:
        raise ValueError(f"{path} is not a {domain} world-model-ranker checkpoint")
    cfg = Config(**payload["model_config"])
    model = VRanker(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def select(model: VRanker, domain: str, evidence: dict, device: str) -> list[dict]:
    records = []
    for key in sorted(evidence):
        group = evidence[key]
        wan, action, dino, text, language = batch(group, device)
        wm = model(domain, wan, action, dino, text)["score"].float().cpu().tolist()
        candidate_ids = [int(row["candidate_id"]) for row in group["candidates"]]
        if len(candidate_ids) != 8 or len(set(candidate_ids)) != 8:
            raise ValueError(f"{key}: malformed V-Ranker candidate IDs")
        fused = fuse_language_wm_top8(language, wm_score=wm)
        slot = int(fused["selected_top8_slot"])
        records.append(
            {
                "sample_key": key,
                "common_noise_seed": int(group["common_noise_seed"]),
                "candidate_ids": candidate_ids,
                "language_score": [float(value) for value in language],
                "wm_score": [float(value) for value in wm],
                "normalized_language_score": [
                    float(value) for value in fused["normalized_language_score"]
                ],
                "normalized_wm_score": [
                    float(value) for value in fused["normalized_wm_score"]
                ],
                "final_score": [float(value) for value in fused["final_score"]],
                "selected_top8_slot": slot,
                "selected_candidate_id": candidate_ids[slot],
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", choices=("navigation", "driving", "robot", "gaming"), required=True
    )
    parser.add_argument("--split", choices=INFERENCE_SPLITS, default="val_seen")
    parser.add_argument("--evidence", type=Path, nargs="+", required=True)
    parser.add_argument("--v-ranker-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    domain, evidence = load_evidence(args.evidence, args.split)
    if domain != args.domain or not evidence:
        raise ValueError("empty or cross-domain V-Ranker evidence")
    device = "cuda:0"
    model = load_model(args.v_ranker_checkpoint, args.domain, device)
    records = select(model, domain, evidence, device)
    payload = {
        "domain": domain,
        "split": args.split,
        "record_count": len(records),
        "candidate_count_in": 8,
        "candidate_count_out": 1,
        "fusion_weights_language_wm": [2.0, 1.0],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(args.output)
    summary = {key: value for key, value in payload.items() if key != "records"}
    summary["output"] = str(args.output.resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
