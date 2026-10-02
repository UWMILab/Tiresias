import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from lcwm.datasets.protocol import INFERENCE_SPLITS
from ldp.policy import CANDIDATE_FIELDS, HORIZON

from rankers.tr_ranker.trajectory_ranker import (
    CHECKPOINT_FORMAT,
    DROPOUT,
    MODEL_SIZE,
    TrajectoryRanker,
    load_action_adapter,
    make_action_batch,
    make_text_batch,
    trajectory_score,
)

def load_candidates(paths: list[Path], domain: str, split: str):
    records: list[dict[str, Any]] = []
    for path in paths:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (
            value.get("domain") != domain
            or value.get("split") != split
            or int(value.get("horizon", -1)) != HORIZON[domain]
            or tuple(value.get("action_fields", ())) != CANDIDATE_FIELDS[domain]
        ):
            raise ValueError(f"candidate file does not match domain {domain}: {path}")
        if int(value.get("candidate_count", -1)) != 128:
            raise ValueError(f"candidate file does not contain 128 candidates: {path}")
        for row in value.get("records", []):
            actions = torch.as_tensor(row["candidate_actions"])
            masks = torch.as_tensor(row["candidate_masks"]).bool()
            lengths = torch.as_tensor(row["candidate_lengths"]).long()
            if actions.shape[:2] != (128, HORIZON[domain]):
                raise ValueError(
                    {"sample": row.get("sample_id"), "shape": list(actions.shape)}
                )
            if masks.shape != actions.shape[:2] or lengths.shape != (128,):
                raise ValueError(f"malformed mask/length: {row.get('sample_id')}")
            if not bool(masks.all()) or not bool((lengths == HORIZON[domain]).all()):
                raise ValueError(f"non-fixed native H: {row.get('sample_id')}")
            if not torch.isfinite(actions.float()).all():
                raise ValueError(f"non-finite candidate: {row.get('sample_id')}")
            records.append(row)
    records.sort(key=lambda row: (int(row["dataset_index"]), str(row["sample_id"])))
    identities = [str(row["sample_id"]) for row in records]
    if not records or len(set(identities)) != len(identities):
        raise ValueError("empty or duplicate candidate identities")
    return records


def load_cache(path: Path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value.get("token_embeddings"), dict):
        raise ValueError(f"invalid language cache: {path}")
    return value


def load_model(path: Path, domain: str, device: torch.device):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format") != CHECKPOINT_FORMAT or value.get("domain") != domain:
        raise ValueError(f"{path} is not a {domain} trajectory-ranker checkpoint")
    hidden, layers = MODEL_SIZE[domain]
    model = TrajectoryRanker(
        int(value["action_dim"]),
        torch.as_tensor(value["action_mean"]),
        torch.as_tensor(value["action_std"]),
        horizon=HORIZON[domain],
        hidden=int(value.get("hidden", hidden)),
        layers=int(value.get("layers", layers)),
        dropout=float(value.get("dropout", DROPOUT)),
        domain=domain,
    )
    model.load_state_dict(value["model_state"], strict=True)
    return model.to(device).eval(), value


@torch.inference_mode()
def infer(model, rows, cache, domain, adapter, device, batch_size, statistics):
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        text, text_mask = make_text_batch(batch, cache, device)
        actions, action_mask = make_action_batch(batch, device, domain, adapter)
        heads = model(text, text_mask, actions, action_mask)
        scores = trajectory_score(heads, statistics).cpu()
        for row, score in zip(batch, scores):
            ranked = sorted(range(128), key=lambda index: (-float(score[index]), index))
            top8 = ranked[:8]
            output.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "dataset_index": int(row["dataset_index"]),
                    "top8_raw_indices": top8,
                    "top8_scores": [float(score[index]) for index in top8],
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=tuple(HORIZON), required=True)
    parser.add_argument("--split", choices=INFERENCE_SPLITS, default="val_seen")
    parser.add_argument("--candidate-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--language-cache", type=Path, required=True)
    parser.add_argument("--tr-checkpoint", type=Path, required=True)
    parser.add_argument("--action-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = load_candidates(args.candidate_shards, args.domain, args.split)
    cache = load_cache(args.language_cache)
    adapter = load_action_adapter(args.action_config, args.domain)
    device = torch.device("cuda:0")
    model, checkpoint = load_model(args.tr_checkpoint, args.domain, device)
    records = infer(
        model,
        rows,
        cache,
        args.domain,
        adapter,
        device,
        args.batch_size,
        checkpoint["target_statistics"],
    )

    payload = {
        "domain": args.domain,
        "split": args.split,
        "candidate_count_in": 128,
        "candidate_count_out": 8,
        "record_count": len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    summary = {key: value for key, value in payload.items() if key != "records"}
    summary["output"] = str(args.output.resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
