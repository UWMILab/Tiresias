import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lcwm.datasets.protocol import PUBLIC_SPLITS, read_records
from ldp.action_utils import (
    DRIVING_ACTION_FIELDS,
    driving_to_canonical,
    euler_zyx_matrix,
    integrate_planar_ego,
    integrate_robot_local,
    load_driving_reference,
    validate_source_dt,
)
from ldp.policy import ACTION_FIELDS, CANDIDATE_FIELDS, HORIZON

DIM = {domain: len(fields) for domain, fields in ACTION_FIELDS.items()}
METRIC_DEFINITION = {
    "ATE": "rmse_position_l2",
    "RPE": "rmse_step_displacement_l2",
}


def identifier(domain: str, row: dict[str, Any]) -> str:
    return str(row["trajectory_id"] if domain == "navigation" else row["sample_key"])


def driving_cadence_adapter(action: np.ndarray, source_dt: float) -> np.ndarray:
    return driving_to_canonical(action, validate_source_dt(source_dt)).astype(
        np.float64, copy=False
    )


def target_action(
    domain: str,
    row: dict[str, Any],
    manifest_root: Path | None = None,
) -> np.ndarray:
    path = Path(str(row.get("action_path") or row.get("array") or ""))
    if not path.is_absolute() and manifest_root is not None:
        path = manifest_root / path
    if domain == "driving":
        fields = [
            item.strip()
            for item in str(row.get("action_fields", "")).replace(",", ";").split(";")
            if item.strip()
        ]
        if tuple(fields) != DRIVING_ACTION_FIELDS:
            raise ValueError(
                f"{row.get('trajectory_id')}: unsupported Driving action fields {fields!r}"
            )
        if manifest_root is None:
            raise ValueError("Driving target_action requires manifest_root metadata")
        payload, path, source_dt = load_driving_reference(
            row, manifest_root, context=f"{row.get('trajectory_id', '<unknown>')} Driving target"
        )
        if "physical_action" not in payload:
            raise ValueError(f"{path}: Driving reference is missing physical_action")
        action = np.asarray(payload["physical_action"], dtype=np.float64)
    else:
        with np.load(path, allow_pickle=False) as payload:
            key = "physical_action" if "physical_action" in payload else "action"
            action = np.asarray(payload[key], dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != DIM[domain] or len(action) < 1:
        raise ValueError({"target": str(path), "shape": list(action.shape)})
    declared = row.get("future_horizon")
    true_length = (
        int(declared)
        if declared not in (None, "")
        else min(len(action), HORIZON[domain])
    )
    if not 1 <= true_length <= HORIZON[domain] or true_length > len(action):
        raise ValueError(
            {
                "target": str(path),
                "future_horizon": true_length,
                "stored_actions": len(action),
            }
        )
    # Keep the real T_gt; never manufacture a zero-action reference tail.
    action = action[:true_length]
    if domain == "driving":
        action = driving_cadence_adapter(action, source_dt)
    return action


def domain_average_step(domain: str, manifest: Path) -> float:
    config_path = manifest.resolve().parent / "action_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("domain") != domain:
        raise ValueError(f"action config does not match {domain}: {config_path}")
    step = float(config["average_step"])
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"invalid domain average step: {step}")
    return step


def integrate_candidates(
    domain: str, action: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    if domain in {"navigation", "driving", "gaming"}:
        vertical = domain == "gaming"
        yaw_index = 3 if vertical else 2
        yaw_before = np.cumsum(action[..., yaw_index], axis=1) - action[..., yaw_index]
        forward, lateral = action[..., 0], action[..., 1]
        dx = np.cos(yaw_before) * forward - np.sin(yaw_before) * lateral
        dy = np.sin(yaw_before) * forward + np.cos(yaw_before) * lateral
        xy = np.stack((np.cumsum(dx, axis=1), np.cumsum(dy, axis=1)), axis=-1)
        if vertical:
            return np.concatenate(
                (xy, np.cumsum(action[..., 2], axis=1)[..., None]), axis=-1
            ), None
        return xy, None
    count, horizon = action.shape[:2]
    position = np.zeros((count, 3), dtype=np.float64)
    rotation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    positions = []
    for step in range(horizon):
        position = position + np.einsum("nij,nj->ni", rotation, action[:, step, :3])
        rotation = np.einsum(
            "nij,njk->nik", rotation, euler_zyx_matrix(action[:, step, 3:6])
        )
        positions.append(position.copy())
    return np.stack(positions, axis=1), np.clip(action[..., 6], 0.0, 1.0)


def integrate_target(
    domain: str, action: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    if domain in {"navigation", "driving"}:
        return integrate_planar_ego(action[:, :3]), None
    if domain == "gaming":
        return integrate_planar_ego(action, vertical=True), None
    result = integrate_robot_local(action)
    return np.asarray(result["position"]), np.asarray(result["gripper"])


def batched_dtw(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    count, p_len = predicted.shape[:2]
    t_len = len(truth)
    cost = np.full((count, p_len + 1, t_len + 1), np.inf, dtype=np.float64)
    cost[:, 0, 0] = 0.0
    for i in range(1, p_len + 1):
        for j in range(1, t_len + 1):
            local = np.linalg.norm(predicted[:, i - 1] - truth[None, j - 1], axis=-1)
            cost[:, i, j] = local + np.minimum.reduce(
                (cost[:, i - 1, j - 1], cost[:, i - 1, j], cost[:, i, j - 1])
            )
    return cost[:, p_len, t_len]


def metric_vectors(
    domain: str, action: np.ndarray, truth_action: np.ndarray, threshold: float
) -> dict[str, torch.Tensor]:
    predicted, predicted_gripper = integrate_candidates(domain, action)
    truth, truth_gripper = integrate_target(domain, truth_action)
    common = min(predicted.shape[1], len(truth))
    if common < 1:
        raise ValueError("predicted and reference trajectories must be non-empty")
    # ATE/RPE use the corresponding prefix; SR/nDTW keep the full prediction.
    predicted_valid = predicted[:, :common]
    truth_valid = truth[:common]
    terminal = np.linalg.norm(predicted[:, -1] - truth[-1], axis=-1)
    success = terminal <= threshold
    if domain == "robot":
        success &= (predicted_gripper[:, -1] >= 0.5) == (truth_gripper[-1] >= 0.5)
    if predicted.shape[1] != HORIZON[domain]:
        raise ValueError({"predicted_h": predicted.shape[1], "truth_h": len(truth)})
    error = predicted_valid - truth_valid[None]
    ate = np.sqrt(np.square(error).sum(axis=-1).mean(axis=-1))
    if common == 1:
        rpe = np.zeros(len(predicted), dtype=np.float64)
    else:
        relative = (
            np.diff(predicted_valid, axis=1)
            - np.diff(truth_valid, axis=0)[None]
        )
        rpe = np.sqrt(np.square(relative).sum(axis=-1).mean(axis=-1))
    dtw = batched_dtw(predicted, truth)
    ndtw = np.exp(-dtw / (threshold * len(truth)))
    return {
        "SR": torch.from_numpy(success.astype(np.float32)),
        "nDTW": torch.from_numpy(ndtw.astype(np.float32)),
        "ATE": torch.from_numpy(ate.astype(np.float32)),
        "RPE": torch.from_numpy(rpe.astype(np.float32)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=tuple(HORIZON), required=True)
    parser.add_argument(
        "--split",
        choices=PUBLIC_SPLITS,
        required=True,
    )
    parser.add_argument("--candidate-shards", nargs="+", required=True, type=Path)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    threshold = domain_average_step(args.domain, args.target_manifest)
    payloads = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in args.candidate_shards
    ]
    for payload in payloads:
        if (
            payload.get("domain") != args.domain
            or payload.get("split") != args.split
            or int(payload.get("candidate_count", -1)) != 128
            or int(payload.get("horizon", -1)) != HORIZON[args.domain]
            or tuple(payload.get("action_fields", ()))
            != CANDIDATE_FIELDS[args.domain]
        ):
            raise ValueError("invalid candidate shard")
    candidates = [row for payload in payloads for row in payload["records"]]
    candidates.sort(key=lambda row: (int(row["dataset_index"]), str(row["sample_id"])))
    if len({str(row["sample_id"]) for row in candidates}) != len(candidates):
        raise ValueError("duplicate candidate identity")
    target_rows = read_records(args.target_manifest)
    if any(
        str(row.get("domain")) != args.domain
        or str(row.get("split")) != args.split
        for row in target_rows
    ):
        raise ValueError("target manifest does not match --domain/--split")
    targets = {identifier(args.domain, row): row for row in target_rows}
    if len(targets) != len(target_rows):
        raise ValueError("duplicate target identity")
    if set(targets) != {str(row["sample_id"]) for row in candidates}:
        raise ValueError(
            {"candidate_count": len(candidates), "target_count": len(targets)}
        )
    result = []
    for index, row in enumerate(candidates, start=1):
        action = torch.as_tensor(row["candidate_actions"]).double().numpy()
        lengths = torch.as_tensor(row["candidate_lengths"]).long().numpy()
        if len({int(value) for value in lengths}) != 1:
            raise ValueError("all candidates in a query must use the same horizon")
        length = int(lengths[0])
        if length != HORIZON[args.domain]:
            raise ValueError(
                f"candidate horizon {length} does not match {HORIZON[args.domain]}"
            )
        values = metric_vectors(
            args.domain,
            action,
            target_action(
                args.domain,
                targets[str(row["sample_id"])],
                args.target_manifest.resolve().parent,
            ),
            threshold,
        )
        result.append(
            {
                "sample_id": str(row["sample_id"]),
                "sample_key": str(row["sample_key"]),
                "dataset_index": int(row["dataset_index"]),
                **values,
            }
        )
        if index == 1 or index % 100 == 0 or index == len(candidates):
            print(
                json.dumps({"completed": index, "total": len(candidates)}), flush=True
            )
    payload = {
        "domain": args.domain,
        "split": args.split,
        "candidate_count": 128,
        "record_count": len(result),
        "metric_names": ["SR", "nDTW", "ATE", "RPE"],
        "metric_definition": METRIC_DEFINITION,
        "records": result,
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
