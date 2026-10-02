import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lcwm.config import DOMAINS
from lcwm.datasets.protocol import (
    PLANNING_ANCHOR_FRAME_IDX,
    INFERENCE_SPLITS,
    legal_frame_count,
    read_records,
    write_records,
)
from ldp.action_utils import driving_source_dt
from ldp.policy import CANDIDATE_FIELDS, HORIZON
from rankers.fusion.export_actions import FIXED_DT, OUTPUT_ACTION_FIELDS
from rankers.fusion.stage2 import actions_for_wm


def resolve_video(value: str, root: Path | None) -> str:
    path = Path(value)
    if not path.is_absolute():
        if root is None:
            raise ValueError("relative video path requires --wm-root")
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def selected_action(domain: str, row: dict[str, Any]) -> np.ndarray:
    horizon = HORIZON[domain]
    fields = OUTPUT_ACTION_FIELDS[domain]
    action = np.asarray(row["selected_action"], dtype=np.float32)
    mask = np.asarray(row["future_mask"], dtype=bool)
    if (
        tuple(row.get("selected_action_fields", ())) != fields
        or action.shape != (horizon, len(fields))
        or mask.shape != (horizon,)
        or not mask.all()
        or not np.isfinite(action).all()
    ):
        raise ValueError(f"invalid fixed-horizon {domain} action")
    if domain in FIXED_DT:
        if not np.allclose(action[:, -1], FIXED_DT[domain], atol=5e-3, rtol=0):
            raise ValueError(f"invalid {domain} action interval")
        action = action[:, :-1]
    return action


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--wm-manifest", type=Path, required=True)
    parser.add_argument("--wm-root", type=Path)
    parser.add_argument("--action-config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=INFERENCE_SPLITS, required=True)
    args = parser.parse_args()

    config = None
    if args.action_config is not None:
        config = json.loads(args.action_config.read_text(encoding="utf-8"))
    if args.domain in {"gaming", "robot"} and (
        config is None or config.get("domain") != args.domain
    ):
        parser.error(f"--action-config is required for {args.domain}")

    selections = read_records(args.selection)
    wm_rows = read_records(args.wm_manifest)
    wm_by_key = {str(row["sample_key"]): row for row in wm_rows}
    if len(wm_by_key) != len(wm_rows):
        raise ValueError("duplicate sample_key in world-model manifest")
    if {str(row["sample_key"]) for row in selections} != set(wm_by_key):
        raise ValueError("selection and world-model manifest contain different samples")

    output_root = args.output_root.resolve()
    action_root = output_root / "actions" / args.split
    action_root.mkdir(parents=True, exist_ok=False)
    fields = CANDIDATE_FIELDS[args.domain]
    output_rows = []

    for selection in selections:
        key = str(selection["sample_key"])
        source = wm_by_key[key]
        trajectory_id = str(selection["trajectory_id"])
        source_name = str(selection["source_name"])
        if (
            source.get("domain") != args.domain
            or source.get("split") != args.split
            or str(source.get("trajectory_id")) != trajectory_id
            or str(source.get("instruction")) != str(selection.get("instruction"))
            or str(source.get("source_name")) != source_name
            or tuple(str(source.get("action_fields", "")).split(";")) != fields
        ):
            raise ValueError(f"{key}: selection and world-model row do not match")

        source_dt = None
        if args.domain == "driving":
            source_dt = driving_source_dt(
                source, context=f"{key} Driving world-model metadata"
            )
            if not np.isclose(
                float(selection.get("source_dt")), source_dt, atol=1e-6, rtol=0
            ):
                raise ValueError(f"{key}: source_dt mismatch")
        action = actions_for_wm(
            args.domain,
            selected_action(args.domain, selection),
            source_name=source_name,
            source_dt=source_dt,
            action_config=config,
        )
        horizon = HORIZON[args.domain]
        if action.shape != (horizon, len(fields)):
            raise ValueError(f"{key}: invalid world-model action shape")

        action_path = action_root / f"{trajectory_id}.npz"
        np.savez_compressed(
            action_path,
            action=action,
            wm_fields=np.asarray(fields),
        )
        output_rows.append(
            {
                "domain": args.domain,
                "split": args.split,
                "sample_idx": int(selection["dataset_index"]),
                "sample_key": key,
                "trajectory_id": trajectory_id,
                "instruction": str(selection["instruction"]),
                "source_name": source_name,
                "video": resolve_video(str(source["video"]), args.wm_root),
                "array": str(action_path.relative_to(output_root)),
                "action_fields": ";".join(fields),
                "action_dim": len(fields),
                "orig_num_frames": horizon + 1,
                "future_horizon": horizon,
                "context_frame_idx": PLANNING_ANCHOR_FRAME_IDX,
                "wan_num_frames": legal_frame_count(horizon + 1),
                "action_source": "diffusion_policy_top1",
            }
        )

    manifest = output_root / f"{args.split}.jsonl"
    write_records(manifest, output_rows)
    print(
        json.dumps(
            {
                "domain": args.domain,
                "records": len(output_rows),
                "output": str(manifest),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
