import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lcwm.datasets.protocol import INFERENCE_SPLITS
from rankers.fusion.stage2 import (
    actions_for_wm,
    common_chunk_seed,
    load_initial_observations,
    load_top8,
)
from rankers.v_ranker.runtime import WanFeatureRuntime


def validate_config(path: Path, domain: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("domain") != domain:
        raise ValueError(
            f"world-model config is for {value.get('domain')}, not {domain}"
        )
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", choices=("navigation", "driving", "robot", "gaming"), required=True
    )
    parser.add_argument("--split", choices=INFERENCE_SPLITS, default="val_seen")
    parser.add_argument("--top8", type=Path, required=True)
    parser.add_argument("--initial-observations", type=Path, required=True)
    parser.add_argument("--wm-config", type=Path, required=True)
    parser.add_argument("--action-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    action_config = None
    if args.action_config is not None:
        action_config = json.loads(args.action_config.read_text(encoding="utf-8"))
        if action_config.get("domain") != args.domain:
            raise ValueError("world-model action config does not match --domain")
    if args.domain in {"gaming", "robot"} and action_config is None:
        parser.error(f"--action-config is required for {args.domain}")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    top8 = load_top8(args.top8, domain=args.domain, split=args.split)
    observations = load_initial_observations(
        args.initial_observations, domain=args.domain, split=args.split
    )
    config = validate_config(args.wm_config, args.domain)
    rows = sorted(
        top8["records"], key=lambda row: (int(row["dataset_index"]), row["sample_key"])
    )
    if {row["sample_key"] for row in rows} != set(observations):
        raise ValueError("Top-8/initial-observation identities differ")
    rows = [
        row
        for row in rows
        if int(row["dataset_index"]) % args.shard_count == args.shard_index
    ]
    if args.max_samples:
        rows = rows[: args.max_samples]
    runtime = WanFeatureRuntime(config)
    records = []
    for query_index, row in enumerate(rows, start=1):
        key = str(row["sample_key"])
        obs = observations[key]
        if str(row.get("source_name", "")) != str(obs.get("source_name", "")):
            raise ValueError(f"{key}: Top-8/manifest source mismatch")
        with Image.open(obs["initial_observation_path"]) as source:
            image = source.convert("RGB").copy()
        dino = runtime.dino_pooled_tokens(image)
        if dino.shape != (8, 1024) or not torch.isfinite(dino).all():
            raise ValueError(f"{key}: unexpected DINOv3 feature shape")
        candidate_rows = []
        seeds = []
        for slot, candidate_id in enumerate(row["candidate_indices"]):
            length = int(row["candidate_lengths"][slot])
            mask = np.asarray(row["candidate_masks"][slot], dtype=bool)
            action = np.asarray(row["candidate_actions"][slot], dtype=np.float32)
            if length < 1 or not np.array_equal(mask, np.arange(len(mask)) < length):
                raise ValueError(f"{key}: invalid Top-8 length/mask")
            action = actions_for_wm(
                args.domain,
                action[:length],
                source_name=str(obs["source_name"]),
                source_dt=(obs.get("source_dt") if args.domain == "driving" else None),
                action_config=action_config,
            )
            seed = common_chunk_seed(args.base_seed, key, 0)
            features = runtime.rollout_features(
                observation=image,
                instruction=obs["instruction"],
                candidate_actions=action,
                seed=seed,
            )
            expected_shapes = {
                "wan_moments": (144,),
                "action_mean": (3072,),
                "text_mean": (4096,),
            }
            if any(
                tuple(features[name].shape) != shape
                or not torch.isfinite(features[name]).all()
                for name, shape in expected_shapes.items()
            ):
                raise ValueError(f"{key}: malformed compact WM evidence")
            seeds.append(seed)
            candidate_rows.append(
                {
                    "candidate_id": int(candidate_id),
                    "top8_slot": slot,
                    "predicted_action_count": length,
                    "language_score": float(row["selected_language_scores"][slot]),
                    **features,
                }
            )
        if len(set(seeds)) != 1:
            raise ValueError(f"{key}: Top-8 did not receive common diffusion noise")
        records.append(
            {
                "sample_key": key,
                "dataset_index": int(row["dataset_index"]),
                "common_noise_seed": seeds[0],
                "dino_pooled_tokens": dino,
                "candidates": candidate_rows,
            }
        )
        if query_index == 1 or query_index % 10 == 0 or query_index == len(rows):
            print(
                json.dumps({"completed": query_index, "total": len(rows)}), flush=True
            )
    payload = {
        "domain": args.domain,
        "split": args.split,
        "candidate_count_per_sample": 8,
        "base_seed": args.base_seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "record_count": len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    summary = {key: value for key, value in payload.items() if key != "records"}
    summary["output"] = str(args.output.resolve())
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
