import argparse
import json
from pathlib import Path
from typing import Any

import torch

from lcwm.datasets.protocol import INFERENCE_SPLITS
from ldp.policy import CANDIDATE_FIELDS, HORIZON

DOMAINS = ("navigation", "driving", "robot", "gaming")


def load_candidates(
    paths: list[Path], domain: str, split: str
) -> tuple[dict[str, dict], dict[str, Any]]:
    records: dict[str, dict] = {}
    common: dict[str, Any] | None = None
    required_common = ("domain", "candidate_count", "horizon", "action_fields")
    for path in paths:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (
            value.get("domain") != domain
            or value.get("split") != split
            or int(value.get("horizon", -1)) != HORIZON[domain]
            or tuple(value.get("action_fields", ())) != CANDIDATE_FIELDS[domain]
        ):
            raise ValueError(f"candidate scope mismatch: {path}")
        if int(value.get("candidate_count", -1)) != 128:
            raise ValueError(f"candidate shard is not N=128: {path}")
        current = {key: value.get(key) for key in required_common}
        if common is None:
            common = current
        elif current != common:
            raise ValueError("candidate files use different action schemas")
        for row in value["records"]:
            identity = str(row["sample_id"])
            if identity in records:
                raise ValueError(f"duplicate candidate identity: {identity}")
            actions = torch.as_tensor(row["candidate_actions"])
            masks = torch.as_tensor(row["candidate_masks"]).bool()
            lengths = torch.as_tensor(row["candidate_lengths"]).long()
            if (
                actions.ndim != 3
                or actions.shape[0] != 128
                or masks.shape != actions.shape[:2]
            ):
                raise ValueError(f"invalid candidate tensors: {identity}")
            if lengths.shape != (128,):
                raise ValueError(f"invalid candidate lengths: {identity}")
            expected = torch.arange(actions.shape[1])[None] < lengths[:, None]
            if (
                bool((lengths < 1).any())
                or bool((lengths > actions.shape[1]).any())
                or not torch.equal(masks, expected)
            ):
                raise ValueError(f"candidate mask/length mismatch: {identity}")
            if not torch.isfinite(actions.float()).all():
                raise ValueError(f"non-finite candidate action: {identity}")
            sample_key = str(row.get("sample_key") or "")
            if not sample_key:
                raise ValueError(f"missing candidate sample_key: {identity}")
            records[identity] = row
    if common is None or not records:
        raise ValueError("empty candidate pool")
    return records, common


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--split", choices=INFERENCE_SPLITS, default="val_seen")
    parser.add_argument("--candidate-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    candidates, candidate_contract = load_candidates(
        args.candidate_shards, args.domain, args.split
    )
    selection = torch.load(args.selection, map_location="cpu", weights_only=True)
    expected = {
        "domain": args.domain,
        "split": args.split,
        "candidate_count_in": 128,
        "candidate_count_out": 8,
    }
    for key, wanted in expected.items():
        if selection.get(key) != wanted:
            raise ValueError(
                {
                    "selection_field": key,
                    "expected": wanted,
                    "actual": selection.get(key),
                }
            )
    if int(selection.get("record_count", -1)) != len(candidates):
        raise ValueError("selection/candidate count mismatch")

    output_rows = []
    seen: set[str] = set()
    for chosen in sorted(
        selection["records"],
        key=lambda row: (int(row["dataset_index"]), str(row["sample_id"])),
    ):
        identity = str(chosen["sample_id"])
        if identity in seen or identity not in candidates:
            raise ValueError(f"selection identity mismatch: {identity}")
        seen.add(identity)
        source = candidates[identity]
        if int(chosen["dataset_index"]) != int(source["dataset_index"]):
            raise ValueError(f"dataset index mismatch: {identity}")
        indices = [int(value) for value in chosen["top8_raw_indices"]]
        scores = [float(value) for value in chosen["top8_scores"]]
        if (
            len(indices) != 8
            or len(set(indices)) != 8
            or min(indices) < 0
            or max(indices) >= 128
        ):
            raise ValueError(f"invalid global Top-8: {identity}")
        if len(scores) != 8 or not torch.isfinite(torch.tensor(scores)).all():
            raise ValueError(f"invalid Language scores: {identity}")
        actions = torch.as_tensor(source["candidate_actions"])[indices].float()
        masks = torch.as_tensor(source["candidate_masks"])[indices].bool()
        lengths = torch.as_tensor(source["candidate_lengths"])[indices].long()
        output_rows.append(
            {
                "sample_key": str(source["sample_key"]),
                "dataset_index": int(source["dataset_index"]),
                "source_name": str(source["source_name"]),
                "instruction": str(source["instruction"]),
                "candidate_indices": indices,
                "selected_language_scores": scores,
                "candidate_actions": actions.tolist(),
                "candidate_masks": masks.tolist(),
                "candidate_lengths": lengths.tolist(),
            }
        )
    if seen != set(candidates):
        raise ValueError("selection omitted candidate queries")

    payload = {
        "domain": args.domain,
        "split": args.split,
        "candidate_count": 128,
        "selected_count": 8,
        "record_count": len(output_rows),
        "hmax": int(candidate_contract["horizon"]),
        "action_dim": len(candidate_contract["action_fields"]),
        "action_fields": candidate_contract["action_fields"],
        "records": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(args.output)
    summary = {
        "domain": args.domain,
        "split": args.split,
        "records": len(output_rows),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
