import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lcwm.config import GAMING_METERS_PER_UNIT
from lcwm.datasets.protocol import INFERENCE_SPLITS, read_records
from ldp.action_utils import driving_source_dt
from ldp.policy import CANDIDATE_FIELDS, HORIZON

DOMAINS = tuple(HORIZON)
ACTION_DIM = {domain: len(fields) for domain, fields in CANDIDATE_FIELDS.items()}
INPUT_ACTION_FIELDS = CANDIDATE_FIELDS
OUTPUT_ACTION_FIELDS = {
    "navigation": ("delta_forward", "delta_lateral", "delta_yaw", "dt_seconds"),
    "driving": (
        "delta_forward_at_0p1s",
        "delta_lateral_at_0p1s",
        "delta_yaw_at_0p1s",
        "target_velocity_mps",
        "dt_seconds",
    ),
    "gaming": (
        "delta_forward_m",
        "delta_lateral_m",
        "delta_vertical_m",
        "delta_yaw_radians",
    ),
    "robot": INPUT_ACTION_FIELDS["robot"],
}
FIXED_DT = {"navigation": 0.1, "driving": 0.1}


def identity_candidates(row: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for key in ("sample_key", "trajectory_id", "sample_id")
        if (value := row.get(key)) not in (None, "")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--split", choices=INFERENCE_SPLITS, default="val_seen")
    parser.add_argument("--top8", type=Path, required=True)
    parser.add_argument("--top1", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    top8 = json.loads(args.top8.read_text(encoding="utf-8"))
    top1 = json.loads(args.top1.read_text(encoding="utf-8"))
    if top8.get("domain") != args.domain or top1.get("domain") != args.domain:
        raise ValueError("cross-domain Top-8/Top-1 input")
    if top8.get("split") != args.split or top1.get("split") != args.split:
        raise ValueError("Top-8/Top-1 split does not match --split")
    if (
        top8.get("selected_count") != 8
        or top1.get("candidate_count_in") != 8
        or top1.get("candidate_count_out") != 1
    ):
        raise ValueError("invalid 128-to-8-to-1 selection contract")
    expected_fields = INPUT_ACTION_FIELDS[args.domain]
    if (
        int(top8.get("hmax", -1)) != HORIZON[args.domain]
        or int(top8.get("action_dim", -1)) != ACTION_DIM[args.domain]
        or tuple(top8.get("action_fields", ())) != expected_fields
    ):
        raise ValueError("Top-8 action schema does not match the domain contract")
    top8_records = top8["records"]
    top1_records = top1["records"]
    top8_rows = {str(row["sample_key"]): row for row in top8_records}
    top1_rows = {str(row["sample_key"]): row for row in top1_records}
    if len(top8_rows) != len(top8_records) or len(top1_rows) != len(top1_records):
        raise ValueError("duplicate Top-8 or Top-1 sample identity")
    if not top8_rows or set(top8_rows) != set(top1_rows):
        raise ValueError("Top-8 and Top-1 identities differ")

    source_rows = read_records(args.source_manifest)
    if any(str(row.get("split")) != args.split for row in source_rows):
        raise ValueError("source manifest split does not match --split")
    source_by_key: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        for key in identity_candidates(row):
            if key in source_by_key and source_by_key[key] is not row:
                raise ValueError(f"ambiguous source identity: {key}")
            source_by_key[key] = row

    materialized: list[dict[str, Any]] = []
    for output_index, key in enumerate(sorted(top8_rows)):
        top8_row = top8_rows[key]
        top1_row = top1_rows[key]
        source = source_by_key.get(key)
        if source is None:
            raise KeyError(f"source manifest has no row for {key}")
        if str(top8_row.get("instruction", "")) != str(
            source.get("instruction", "")
        ):
            raise ValueError(f"Top-8/source instruction mismatch: {key}")
        source_name = str(source.get("source_name", ""))
        if not source_name or str(top8_row.get("source_name", "")) != source_name:
            raise ValueError(f"Top-8/source source_name mismatch: {key}")

        indices = [int(value) for value in top8_row["candidate_indices"]]
        top1_indices = [int(value) for value in top1_row["candidate_ids"]]
        if top1_indices != indices:
            raise ValueError(f"Top-1 candidate order differs from Top-8: {key}")
        selected_id = int(top1_row["selected_candidate_id"])
        selected_slot = int(top1_row["selected_top8_slot"])
        if not 0 <= selected_slot < 8 or indices[selected_slot] != selected_id:
            raise ValueError(f"Top-1 slot/candidate mismatch: {key}")

        action = np.asarray(
            top8_row["candidate_actions"][selected_slot], dtype=np.float32
        )
        mask = np.asarray(top8_row["candidate_masks"][selected_slot], dtype=bool)
        length = int(top8_row["candidate_lengths"][selected_slot])
        expected_shape = (HORIZON[args.domain], ACTION_DIM[args.domain])
        if action.shape != expected_shape:
            raise ValueError(f"unexpected action shape for {key}: {action.shape}")
        if length != HORIZON[args.domain] or not bool(mask.all()):
            raise ValueError(f"non-native fixed horizon for {key}")
        if not np.isfinite(action).all():
            raise ValueError(f"non-finite action for {key}")

        selected_action = action
        selected_fields = OUTPUT_ACTION_FIELDS[args.domain]
        if args.domain in FIXED_DT:
            dt = np.full((len(action), 1), FIXED_DT[args.domain], dtype=np.float32)
            selected_action = np.concatenate((action, dt), axis=1)

        trajectory_id = str(
            source.get("trajectory_id") or source.get("sample_id") or key
        )
        instruction = str(source.get("instruction") or "").strip()
        if not trajectory_id or not instruction:
            raise ValueError(f"source row lacks trajectory identity/instruction: {key}")
        source_dt = None
        if args.domain == "driving":
            source_dt = driving_source_dt(
                source, context=f"{key} Driving metadata"
            )
        row: dict[str, Any] = {
            "sample_key": key,
            "dataset_index": int(
                source.get("dataset_index", source.get("sample_idx", output_index))
            ),
            "trajectory_id": trajectory_id,
            "instruction": instruction,
            "source_name": source_name,
            "selected_candidate_id": selected_id,
            "selected_top8_slot": selected_slot,
            "selected_action": selected_action.tolist(),
            "selected_action_fields": list(selected_fields),
            "future_mask": mask.tolist(),
        }
        if args.domain == "driving":
            row["selected_action_4d"] = action.tolist()
            row["selected_action_4d_fields"] = list(selected_fields[:4])
            row["source_dt"] = float(source_dt)
        elif args.domain == "gaming":
            scale = float(source.get("translation_scale_to_meter", 0))
            expected_scale = GAMING_METERS_PER_UNIT.get(source_name)
            if expected_scale is None or not np.isclose(
                scale, expected_scale, atol=1e-8, rtol=0
            ):
                raise ValueError(f"invalid Gaming source scale: {key}")
            row["translation_scale_to_meter"] = scale
        materialized.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in materialized:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    summary = {
        "domain": args.domain,
        "record_count": len(materialized),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
