import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from lcwm.datasets.protocol import PUBLIC_SPLITS, read_records
from ldp.policy import HORIZON
from rankers.dataset.build_candidate_metric_labels import (
    METRIC_DEFINITION,
    domain_average_step,
    integrate_candidates,
    integrate_target,
    target_action,
)
from rankers.fusion.stage2 import load_top8


def turn_error(domain: str, action: np.ndarray, truth: np.ndarray) -> float:
    if domain == "robot":
        return float((action[-1, 6] >= 0.5) != (truth[-1, 6] >= 0.5))
    yaw_index = 3 if domain == "gaming" else 2
    predicted_turn = float(
        np.arctan2(
            np.sin(action[:, yaw_index].sum()), np.cos(action[:, yaw_index].sum())
        )
    )
    target_turn = float(
        np.arctan2(np.sin(truth[:, yaw_index].sum()), np.cos(truth[:, yaw_index].sum()))
    )
    straight = np.deg2rad(5.0)
    minimum = np.deg2rad(2.5)
    if abs(target_turn) >= straight:
        correct = abs(predicted_turn) >= minimum and np.sign(predicted_turn) == np.sign(
            target_turn
        )
    else:
        correct = abs(predicted_turn) < straight
    return float(not correct)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", choices=("navigation", "driving", "robot", "gaming"), required=True
    )
    parser.add_argument(
        "--split",
        choices=PUBLIC_SPLITS,
        required=True,
    )
    parser.add_argument("--top8", type=Path, required=True)
    parser.add_argument("--candidate-labels", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    top8 = load_top8(args.top8, domain=args.domain, split=args.split)
    label_source = torch.load(
        args.candidate_labels, map_location="cpu", weights_only=True
    )
    if (
        not isinstance(label_source, Mapping)
        or label_source.get("domain") != args.domain
        or label_source.get("split") != args.split
        or int(label_source.get("candidate_count", -1)) != 128
        or label_source.get("metric_definition") != METRIC_DEFINITION
    ):
        raise ValueError(
            "candidate labels must match the requested domain/split and RMSE "
            "metric_definition; rebuild candidate labels before preparing "
            "V-Ranker labels"
        )
    label_lookup = {str(row["sample_key"]): row for row in label_source["records"]}
    targets = {
        str(row["sample_key"]): row for row in read_records(args.target_manifest)
    }
    keys = {str(row["sample_key"]) for row in top8["records"]}
    if keys != set(label_lookup) or keys != set(targets):
        raise ValueError("Top-8/label/target identities differ")
    threshold = domain_average_step(args.domain, args.target_manifest)
    output = []
    for row in top8["records"]:
        key = str(row["sample_key"])
        source = label_lookup[key]
        truth_action = target_action(
            args.domain, targets[key], args.target_manifest.resolve().parent
        )
        for slot, candidate_id in enumerate(row["candidate_indices"]):
            candidate_id = int(candidate_id)
            length = int(row["candidate_lengths"][slot])
            if length != HORIZON[args.domain]:
                raise ValueError(f"{key}: invalid candidate horizon")
            action = np.asarray(row["candidate_actions"][slot], dtype=np.float64)[
                :length
            ]
            predicted, _ = integrate_candidates(args.domain, action[None])
            truth_path, _ = integrate_target(args.domain, truth_action)
            terminal = (
                float(np.linalg.norm(predicted[0, -1] - truth_path[-1])) / threshold
            )
            success = float(torch.as_tensor(source["SR"])[candidate_id])
            ndtw = float(torch.as_tensor(source["nDTW"])[candidate_id])
            ate = float(torch.as_tensor(source["ATE"])[candidate_id])
            rpe = float(torch.as_tensor(source["RPE"])[candidate_id])
            common = min(len(action), len(truth_action))
            if common < 1:
                raise ValueError(f"{key}: empty valid GT/action prefix")
            turn = turn_error(args.domain, action[:common], truth_action[:common])
            normalized_ate, normalized_rpe = ate / threshold, rpe / threshold
            values = {
                "quality_label": 4.0 * success
                - terminal
                - 0.5 * normalized_ate
                - 0.5 * normalized_rpe
                - turn,
                "language_label": 1.0 - turn,
                "semantic_label": ndtw,
                "motion_label": -(0.5 * normalized_ate + 0.5 * normalized_rpe),
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{key}: non-finite V-Ranker target")
            output.append(
                {
                    "sample_key": key,
                    "candidate_id": candidate_id,
                    "top8_slot": slot,
                    "SR": success,
                    "nDTW": ndtw,
                    "ATE": ate,
                    "RPE": rpe,
                    "normalized_terminal_error": terminal,
                    "turn_error": turn,
                    **values,
                }
            )
    payload = {
        "domain": args.domain,
        "split": args.split,
        "metric_definition": dict(METRIC_DEFINITION),
        "candidate_count_per_sample": 8,
        "record_count": len(output),
        "records": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(args.output)
    print(json.dumps({"records": len(output), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
