import argparse
import json
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lcwm.config import DOMAINS, GAMING_METERS_PER_UNIT
from lcwm.datasets.protocol import (
    PLANNING_ANCHOR_FRAME_IDX,
    PUBLIC_SPLITS,
    episode_id,
    legal_frame_count,
    write_records,
)
from ldp.action_utils import driving_to_canonical
from ldp.policy import CANDIDATE_FIELDS as ACTION_FIELDS
from ldp.policy import HORIZON

SPLITS = PUBLIC_SPLITS
PLAN_START = PLANNING_ANCHOR_FRAME_IDX
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ROOT = Path(__file__).resolve().parents[1]


class InvalidTrajectory(ValueError):
    pass


class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core."):
            module = "numpy.core." + module.removeprefix("numpy._core.")
        return super().find_class(module, name)


@dataclass
class Sample:
    split: str
    trajectory_id: str
    instruction: str
    source: str
    directory: Path
    frames: list[Path]
    frame_indices: np.ndarray
    physical_action: np.ndarray
    source_action: np.ndarray | None = None
    source_dt: float | None = None


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def euler_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    roll, pitch, yaw = np.moveaxis(value, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    row0 = np.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), -1)
    row1 = np.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), -1)
    row2 = np.stack((-sp, cp * sr, cp * cr), -1)
    return np.stack((row0, row1, row2), -2)


def matrix_euler(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    pitch = np.arcsin(np.clip(-matrix[..., 2, 0], -1.0, 1.0))
    regular = np.abs(np.cos(pitch)) > 1e-8
    roll = np.where(
        regular,
        np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2]),
        np.zeros_like(pitch),
    )
    yaw = np.where(
        regular,
        np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]),
        np.arctan2(-matrix[..., 0, 1], matrix[..., 1, 1]),
    )
    return wrap_angle(np.stack((roll, pitch, yaw), axis=-1))


def numbered_frames(directory: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, int | str]:
        try:
            return 0, int(path.stem)
        except ValueError:
            return 1, path.name

    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=key,
    )


def frame_indices(
    count: int, maximum: int | None = None, minimum: int = 2
) -> np.ndarray:
    if count < minimum:
        raise InvalidTrajectory(
            f"trajectory has {count} frames; at least {minimum} are required"
        )
    if maximum is not None and count > maximum:
        return np.rint(np.linspace(0, count - 1, maximum)).astype(np.int64)
    return np.arange(count, dtype=np.int64)


def instruction(payload: dict[str, Any]) -> str:
    for key in ("instruction", "language_instruction", "prompt", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError("traj_data.pkl has no instruction")


def gaming_source(directory: Path, payload: dict[str, Any]) -> str:
    sources = set()
    aliases = {"cs": "CS", "csgo": "CS", "mc": "MC", "minecraft": "MC"}
    for key in ("source_name", "game_domain"):
        if key in payload:
            value = aliases.get(str(payload[key]).strip().lower())
            if value is None:
                raise ValueError(f"unknown Gaming source in {key}: {payload[key]!r}")
            sources.add(value)
    for part in directory.parts:
        if part.lower() == "cs_segment":
            sources.add("CS")
        elif part.lower() == "mc_segment":
            sources.add("MC")
    lower = directory.name.lower()
    if lower.startswith(("aba_", "abca_")):
        sources.add("MC")
    elif lower.startswith(("dust2_", "inferno_", "mirage_")):
        sources.add("CS")
    if len(sources) != 1:
        raise ValueError(f"cannot identify Gaming source from {directory}")
    return sources.pop()


def source_name(domain: str, directory: Path, payload: dict[str, Any]) -> str:
    sample_name = directory.name
    lower = sample_name.lower()
    if domain == "navigation":
        return "TartanDrive" if re.match(r"^\d{8}_", sample_name) else "LCVN"
    if domain == "driving":
        if lower.startswith("scene"):
            return "nuScenes"
        if lower.startswith("2013_05_28_drive_"):
            return "KITTI360"
        first = sample_name.split("_", 1)[0]
        return "Waymo" if first.isdigit() and len(first) > 10 else "DrivingDojo"
    if domain == "gaming":
        return gaming_source(directory, payload)
    if lower.startswith("traj_droid_"):
        return "DROID"
    if lower.startswith("bridgev2_"):
        return "BridgeDataV2"
    if lower.startswith("rt1_"):
        return "RT-1"
    if lower.startswith("libero_") and "_wrist_" in lower:
        return "LIBERO_wrist"
    if lower.startswith("libero_") and "_agentview_" in lower:
        return "LIBERO_agent"
    raise ValueError(f"cannot identify Robot source from {sample_name}")


def driving_action(
    world_delta: np.ndarray,
    heading: np.ndarray,
    delta_yaw: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(world_delta, dtype=np.float64)[:, :2]
    heading = np.asarray(heading, dtype=np.float64).reshape(-1)
    forward = np.cos(heading) * delta[:, 0] + np.sin(heading) * delta[:, 1]
    lateral = -np.sin(heading) * delta[:, 0] + np.cos(heading) * delta[:, 1]
    return np.stack((forward, lateral, wrap_angle(delta_yaw)), axis=-1)


def gaming_action(
    source: str,
    world_delta: np.ndarray,
    yaw: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(world_delta, dtype=np.float64)
    heading = np.asarray(yaw, dtype=np.float64).reshape(-1)
    if source == "CS":
        world_x, world_side, vertical = delta[:, 0], delta[:, 1], delta[:, 2]
        forward = np.cos(heading) * world_x + np.sin(heading) * world_side
        lateral = -np.sin(heading) * world_x + np.cos(heading) * world_side
    elif source == "MC":
        world_x, world_side, vertical = delta[:, 0], delta[:, 1], delta[:, 2]
        forward = -np.sin(heading) * world_x + np.cos(heading) * world_side
        lateral = -np.cos(heading) * world_x - np.sin(heading) * world_side
    else:
        raise ValueError(f"unsupported Gaming source: {source}")
    scale = GAMING_METERS_PER_UNIT[source]
    return np.stack(
        (forward * scale, lateral * scale, vertical * scale, wrap_angle(delta[:, 3])),
        axis=-1,
    )


def navigation_action(position: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float32)
    yaw = np.asarray(yaw, dtype=np.float32).reshape(-1)
    delta = np.diff(position[:, :2], axis=0)
    cos_yaw = np.cos(yaw[:-1])
    sin_yaw = np.sin(yaw[:-1])
    delta_yaw = (np.diff(yaw) + np.float32(np.pi)) % np.float32(
        2.0 * np.pi
    ) - np.float32(np.pi)
    return np.stack(
        (
            cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1],
            -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1],
            delta_yaw,
        ),
        axis=-1,
    )


def closedness(source: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if source in {"RT-1", "LIBERO_wrist", "LIBERO_agent"}:
        return np.clip((value + 1.0) * 0.5, 0.0, 1.0)
    if source == "BridgeDataV2":
        return np.clip(1.0 - value, 0.0, 1.0)
    return np.clip(value, 0.0, 1.0)


def robot_absolute_action(
    pose: np.ndarray, grip: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    rotation = euler_matrix(pose[:, 3:6])
    values = []
    for start, end in zip(indices[:-1], indices[1:]):
        relative_rotation = rotation[start].T @ rotation[end]
        translation = rotation[start].T @ (pose[end, :3] - pose[start, :3])
        values.append(
            np.concatenate((translation, matrix_euler(relative_rotation), [grip[end]]))
        )
    return np.asarray(values, dtype=np.float64)


def robot_relative_action(raw: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = []
    for start, end in zip(indices[:-1], indices[1:]):
        position = np.zeros(3, dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        for step in raw[int(start) : int(end)]:
            position += rotation @ step[:3]
            rotation = rotation @ euler_matrix(step[3:6])
        values.append(
            np.concatenate((position, matrix_euler(rotation), [raw[int(end) - 1, 6]]))
        )
    return np.asarray(values, dtype=np.float64)


def robot_source_action(
    source: str,
    pose: np.ndarray,
    grip: np.ndarray,
    indices: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    grip = np.asarray(grip, dtype=np.float32)
    if source == "DROID":
        transition = pose[1:frame_count] - pose[: frame_count - 1]
        transition[:, 3:6] = (transition[:, 3:6] + np.pi) % (2.0 * np.pi) - np.pi
        transition_grip = grip[1:frame_count]
    else:
        transition = pose[: frame_count - 1].copy()
        transition_grip = grip[: frame_count - 1]
    step_action = np.concatenate((transition, transition_grip[:, None]), axis=-1)
    values = []
    for start, end in zip(indices[:-1], indices[1:]):
        segment = step_action[int(start) : int(end)]
        values.append(np.concatenate((segment[:, :6].sum(axis=0), segment[-1, 6:7])))
    return np.asarray(values, dtype=np.float32)


def load_sample(
    domain: str,
    split: str,
    split_root: Path,
    pkl_path: Path,
) -> Sample:
    directory = pkl_path.parent
    relative = directory.relative_to(split_root)
    with pkl_path.open("rb") as stream:
        payload = NumpyUnpickler(stream).load()
    frames = numbered_frames(directory)
    source = source_name(domain, directory, payload)
    if domain in {"navigation", "driving", "gaming"}:
        position = np.asarray(payload["position"], dtype=np.float64)
        yaw = np.asarray(payload["yaw"], dtype=np.float64).reshape(-1)
        available = min(len(frames), len(position), len(yaw))
        velocity = None
        if domain == "driving":
            raw_delta = np.asarray(payload["delta"], dtype=np.float64)
            velocity = np.asarray(
                payload.get("target_velocity", payload.get("velocity")),
                dtype=np.float64,
            ).reshape(-1)
            available = min(available, len(velocity), len(raw_delta) + 1)
            indices = frame_indices(available, minimum=PLAN_START + 2)
            delta_yaw = raw_delta[: available - 1, 2]
            fallback = wrap_angle(np.diff(yaw[:available]))
            delta_yaw = np.where(np.isfinite(delta_yaw), delta_yaw, fallback)
            action = driving_action(
                raw_delta[: available - 1, :2],
                yaw[: available - 1],
                delta_yaw,
            )
            action = np.concatenate((action, velocity[1:available, None]), axis=-1)
        elif domain == "gaming":
            raw_delta = np.asarray(payload["delta"], dtype=np.float64)
            available = min(available, len(raw_delta) + 1)
            indices = frame_indices(available, minimum=PLAN_START + 2)
            action = gaming_action(
                source,
                raw_delta[: available - 1],
                yaw[: available - 1],
            )
        else:
            indices = frame_indices(available, minimum=PLAN_START + 2)
            action = navigation_action(position[:available], yaw[:available])
    else:
        pose = np.stack(
            [
                np.asarray(payload[name], dtype=np.float64).reshape(-1)
                for name in (
                    "delta_x",
                    "delta_y",
                    "delta_z",
                    "delta_roll",
                    "delta_pitch",
                    "delta_yaw",
                )
            ],
            axis=-1,
        )
        grip = closedness(source, payload["gripper"])
        if source == "DROID":
            available = min(len(frames), len(pose), len(grip))
            indices = frame_indices(available, maximum=33, minimum=5)
            action = robot_absolute_action(pose, grip, indices)
        else:
            available = min(len(frames), len(pose) + 1, len(grip) + 1)
            indices = frame_indices(available, maximum=33, minimum=5)
            usable = available - 1
            raw = np.concatenate((pose[:usable], grip[:usable, None]), axis=-1)
            action = robot_relative_action(raw, indices)
        source_action = robot_source_action(source, pose, grip, indices, available)
    if not np.isfinite(action).all():
        raise ValueError(f"{pkl_path}: non-finite action")
    trajectory_id = "__".join(relative.parts)
    sample_dt = driving_frame_interval(action) if domain == "driving" else None
    return Sample(
        split=split,
        trajectory_id=trajectory_id,
        instruction=instruction(payload),
        source=source,
        directory=directory,
        frames=frames,
        frame_indices=indices,
        physical_action=action.astype(np.float32),
        source_action=(source_action.astype(np.float32) if domain == "robot" else None),
        source_dt=sample_dt,
    )


def driving_frame_interval(action: np.ndarray) -> float:
    speed = np.abs(action[:, 3])
    displacement = np.linalg.norm(action[:, :2], axis=-1)
    usable = (
        (speed > 0.5)
        & (displacement > 1e-8)
        & np.isfinite(speed)
        & np.isfinite(displacement)
    )
    ratios = displacement[usable] / speed[usable]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) < 2:
        raise InvalidTrajectory(
            "Driving sample has no reliable per-sample source_dt: "
            "at least two moving reference steps are required"
        )
    value = float(np.median(ratios))
    if not 0.05 <= value <= 1.0:
        raise InvalidTrajectory(
            f"Driving source_dt is outside [0.05, 1.0]: {value}"
        )
    return value


def driving_target(action: np.ndarray, source_dt: float) -> np.ndarray:
    return driving_to_canonical(action, source_dt)


def mean_std(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    merged = np.concatenate(values, axis=0).astype(np.float64)
    return merged.mean(axis=0), np.maximum(merged.std(axis=0), 1e-6)


def robust_statistics(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    merged = np.concatenate(values, axis=0).astype(np.float64)
    center = np.median(merged, axis=0)
    mad = np.median(np.abs(merged - center), axis=0)
    scale = np.maximum.reduce(
        (1.4826 * mad, 0.25 * merged.std(axis=0), np.full(merged.shape[1], 1e-6))
    )
    return center, scale


def average_step(domain: str, train: list[Sample]) -> float:
    dimensions = 3 if domain in {"gaming", "robot"} else 2
    steps = np.concatenate(
        [
            np.linalg.norm(
                (
                    driving_target(sample.physical_action, sample.source_dt)
                    if domain == "driving"
                    else sample.physical_action
                )[:, :dimensions],
                axis=-1,
            )
            for sample in train
        ]
    )
    value = float(steps.mean())
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {domain} average step")
    return value


def target_segment(action: np.ndarray, domain: str) -> np.ndarray:
    start = PLAN_START
    return action[start : start + HORIZON[domain]]


def padded_target(action: np.ndarray, domain: str) -> np.ndarray:
    segment = target_segment(action, domain)
    target = np.zeros((HORIZON[domain], action.shape[1]), dtype=np.float32)
    target[: len(segment)] = segment
    return target


def statistics(domain: str, train: list[Sample]) -> dict[str, Any]:
    domain_step = average_step(domain, train)
    if domain == "gaming":
        source_stats = {}
        for source in sorted({sample.source for sample in train}):
            center, scale = robust_statistics(
                [
                    target_segment(sample.physical_action, domain)
                    for sample in train
                    if sample.source == source
                ]
            )
            source_stats[source] = {"mean": center.tolist(), "std": scale.tolist()}
        zeros = [0.0] * 4
        ones = [1.0] * 4
        return {
            "domain": domain,
            "average_step": domain_step,
            "mean": zeros,
            "std": ones,
            "ldp_native_statistics": {"mean": zeros, "std": ones},
            "statistics": source_stats,
        }
    if domain == "robot":
        translation_scale_by_source = {}
        for source in sorted({sample.source for sample in train}):
            steps = np.concatenate(
                [
                    np.linalg.norm(
                        target_segment(sample.physical_action, domain)[:, :3], axis=-1
                    )
                    for sample in train
                    if sample.source == source
                ]
            )
            translation_scale_by_source[source] = max(float(steps.mean()), 1e-6)
        targets = []
        for sample in train:
            value = padded_target(sample.physical_action, domain)[:, :6]
            value[:, :3] /= translation_scale_by_source[sample.source]
            targets.append(value)
        center, scale = mean_std(targets)
        return {
            "domain": domain,
            "average_step": domain_step,
            "mean": center.tolist(),
            "std": scale.tolist(),
            "default_translation_scale": domain_step,
            "translation_scale_by_source": translation_scale_by_source,
        }
    targets = [padded_target(sample.physical_action, domain) for sample in train]
    if domain == "driving":
        targets = [
            driving_target(value, sample.source_dt)
            for value, sample in zip(targets, train)
        ]
    center, scale = mean_std(targets)
    return {
        "domain": domain,
        "average_step": domain_step,
        "mean": center.tolist(),
        "std": scale.tolist(),
    }


def wm_statistics(domain: str, train: list[Sample]) -> dict[str, Any]:
    if domain == "gaming":
        sources = {}
        for source in sorted({sample.source for sample in train}):
            values = np.concatenate(
                [sample.physical_action for sample in train if sample.source == source],
                axis=0,
            ).astype(np.float64)
            source_scale = GAMING_METERS_PER_UNIT[source]
            source_values = values.copy()
            source_values[:, :3] /= source_scale
            sources[source] = {
                "translation_xy_scale": max(
                    float(
                        np.percentile(np.linalg.norm(source_values[:, :2], axis=-1), 95)
                    ),
                    1e-6,
                ),
                "vertical_scale": max(
                    float(np.percentile(np.abs(source_values[:, 2]), 95)), 1e-6
                ),
                "yaw_scale": 1.0,
            }
        return {"domain": domain, "sources": sources}
    if domain == "robot":
        sources = {}
        all_values = np.concatenate(
            [sample.source_action[:, :6] for sample in train], axis=0
        )
        for source in sorted({sample.source for sample in train}):
            values = np.concatenate(
                [
                    sample.source_action[:, :6]
                    for sample in train
                    if sample.source == source
                ],
                axis=0,
            )
            sources[source] = {
                "pose_scale": np.maximum(
                    np.quantile(np.abs(values), 0.99, axis=0), 1e-4
                ).tolist()
            }
        return {
            "domain": domain,
            "clip": 5.0,
            "default_pose_scale": np.maximum(
                np.quantile(np.abs(all_values), 0.99, axis=0), 1e-4
            ).tolist(),
            "sources": sources,
        }
    return {}


def planning_action(sample: Sample, config: dict[str, Any]) -> np.ndarray:
    action = padded_target(sample.physical_action, config["domain"])
    if config["domain"] == "gaming":
        stat = config["statistics"][sample.source]
        source_z = (action - np.asarray(stat["mean"], dtype=np.float32)) / np.asarray(
            stat["std"], dtype=np.float32
        )
        native = config.get("ldp_native_statistics")
        if native is None:
            return source_z
        return (
            np.asarray(native["mean"], dtype=np.float32)
            + np.asarray(native["std"], dtype=np.float32) * source_z
        )
    if config["domain"] == "robot":
        scale = config["translation_scale_by_source"].get(
            sample.source, config["default_translation_scale"]
        )
        action[:, :3] /= float(scale)
    return action


def world_model_action(sample: Sample, config: dict[str, Any]) -> np.ndarray:
    action = sample.physical_action.copy()
    if not config:
        return action
    sources = config["sources"]
    if config["domain"] == "gaming":
        source_scale = GAMING_METERS_PER_UNIT[sample.source]
        raw = action.copy()
        raw[:, :3] /= source_scale
        value = sources[sample.source]
        raw[:, :2] /= float(value["translation_xy_scale"])
        raw[:, 2] /= float(value["vertical_scale"])
        raw[:, 3] /= float(value["yaw_scale"])
        return raw
    if config["domain"] != "robot":
        raise ValueError(f"unsupported world-model action config: {config['domain']}")
    if sample.source_action is None:
        raise ValueError("Robot sample is missing source actions")
    values = sources.get(sample.source)
    scale = values["pose_scale"] if values else config["default_pose_scale"]
    scales = np.asarray(scale, dtype=np.float32)
    action = sample.source_action.copy()
    clip = float(config["clip"])
    action[:, :6] = np.clip(action[:, :6] / scales[None], -clip, clip)
    return action


def materialize_frames(sample: Sample, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, source_index in enumerate(sample.frame_indices):
        source = sample.frames[int(source_index)]
        shutil.copy2(source, destination / f"{index:06d}{source.suffix.lower()}")


def prepare(
    domain: str,
    input_root: Path,
    output_root: Path,
) -> None:
    if output_root.exists():
        raise FileExistsError(output_root)
    samples: dict[str, list[Sample]] = {split: [] for split in SPLITS}
    skipped = 0
    required = ("train", "val_seen", "val_unseen")
    missing = [split for split in required if not (input_root / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"{input_root} is missing split directories: {', '.join(missing)}"
        )
    split_roots = {
        split: input_root / split
        for split in SPLITS
        if (input_root / split).is_dir()
    }
    discovered = [
        (split, root, path)
        for split, root in split_roots.items()
        for path in sorted(root.rglob("traj_data.pkl"))
    ]
    for split, root, path in discovered:
        try:
            samples[split].append(load_sample(domain, split, root, path))
        except InvalidTrajectory:
            skipped += 1
    samples = {split: values for split, values in samples.items() if values}
    if "train" not in samples:
        raise ValueError("input contains no training trajectories")
    for split, values in samples.items():
        identities = [sample.trajectory_id for sample in values]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate trajectory in {split}")
    split_episodes = {
        split: {episode_id(sample.trajectory_id) for sample in values}
        for split, values in samples.items()
    }
    for index, left in enumerate(split_episodes):
        for right in list(split_episodes)[index + 1 :]:
            overlap = split_episodes[left] & split_episodes[right]
            if overlap:
                raise ValueError(
                    f"trajectory styles cross {left}/{right}: {sorted(overlap)[:3]}"
                )
    full_train = samples["train"]
    plan_config = statistics(domain, full_train)
    wm_config = wm_statistics(domain, full_train)
    output_root.mkdir(parents=True)
    (output_root / "planning" / "action_config.json").parent.mkdir(parents=True)
    (output_root / "planning" / "action_config.json").write_text(
        json.dumps(plan_config, indent=2) + "\n", encoding="utf-8"
    )
    if wm_config:
        (output_root / "wm" / "action_config.json").parent.mkdir(parents=True)
        (output_root / "wm" / "action_config.json").write_text(
            json.dumps(wm_config, indent=2) + "\n", encoding="utf-8"
        )

    for split, values in samples.items():
        plan_rows = []
        wm_rows = []
        queries = []
        for index, sample in enumerate(values):
            plan_action = planning_action(sample, plan_config)
            wm_action = world_model_action(sample, wm_config)
            plan_path = (
                output_root
                / "planning"
                / "actions"
                / split
                / f"{sample.trajectory_id}.npz"
            )
            wm_path = (
                output_root / "wm" / "actions" / split / f"{sample.trajectory_id}.npz"
            )
            video_path = output_root / "wm" / "videos" / split / sample.trajectory_id
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            wm_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                plan_path,
                action=plan_action.astype(np.float32),
                physical_action=padded_target(sample.physical_action, domain),
                action_fields=np.asarray(ACTION_FIELDS[domain]),
                **(
                    {"source_dt": np.asarray(sample.source_dt, dtype=np.float32)}
                    if domain == "driving"
                    else {}
                ),
            )
            np.savez_compressed(
                wm_path,
                action=wm_action.astype(np.float32),
                physical_action=sample.physical_action.astype(np.float32),
                action_fields=np.asarray(ACTION_FIELDS[domain]),
                **(
                    {"source_dt": np.asarray(sample.source_dt, dtype=np.float32)}
                    if domain == "driving"
                    else {}
                ),
            )
            materialize_frames(sample, video_path)
            sample_key = f"{split}:{sample.trajectory_id}"
            sample_id = sample.trajectory_id if domain == "navigation" else sample_key
            common = {
                "domain": domain,
                "split": split,
                "dataset_index": index,
                "sample_idx": index,
                "sample_id": sample_id,
                "sample_key": sample_key,
                "trajectory_id": sample.trajectory_id,
                "instruction": sample.instruction,
                "source_name": sample.source,
            }
            if domain == "driving":
                if sample.source_dt is None:
                    raise ValueError(
                        f"{sample.trajectory_id}: missing per-sample Driving source_dt"
                    )
                common["source_dt"] = float(sample.source_dt)
            plan_rows.append(
                {
                    **common,
                    "video": str(
                        Path("..") / "wm" / "videos" / split / sample.trajectory_id
                    ),
                    "first_rgb_path": str(
                        Path("..")
                        / "wm"
                        / "videos"
                        / split
                        / sample.trajectory_id
                        / (
                            f"{PLAN_START:06d}"
                            f"{sample.frames[int(sample.frame_indices[PLAN_START])].suffix.lower()}"
                        )
                    ),
                    "action_path": str(Path("actions") / split / plan_path.name),
                    "action_fields": ";".join(ACTION_FIELDS[domain]),
                    "future_horizon": len(
                        target_segment(sample.physical_action, domain)
                    ),
                }
            )
            if domain == "gaming":
                plan_rows[-1]["translation_scale_to_meter"] = GAMING_METERS_PER_UNIT[
                    sample.source
                ]
            wm_rows.append(
                {
                    **common,
                    "video": str(Path("videos") / split / sample.trajectory_id),
                    "array": str(Path("actions") / split / wm_path.name),
                    "action_fields": ";".join(ACTION_FIELDS[domain]),
                    "action_dim": wm_action.shape[1],
                    "orig_num_frames": len(sample.frame_indices),
                    "future_horizon": len(wm_action),
                    "context_frame_idx": 0,
                    "wan_num_frames": legal_frame_count(len(sample.frame_indices)),
                }
            )
            queries.append(
                {
                    "domain": domain,
                    "split": split,
                    "dataset_index": index,
                    "sample_id": sample_id,
                    "sample_key": sample_key,
                    "trajectory_id": sample.trajectory_id,
                    "instruction": sample.instruction,
                    "source_name": sample.source,
                }
            )
        write_records(output_root / "planning" / f"{split}.jsonl", plan_rows)
        write_records(output_root / "wm" / f"{split}.jsonl", wm_rows)
        (output_root / "planning" / f"{split}.json").write_text(
            json.dumps({"records": queries}, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"Prepared {sum(map(len, samples.values()))} {domain} trajectories "
        f"in {output_root} ({skipped} skipped)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=(*DOMAINS, "all"), required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    domains = DOMAINS if args.domain == "all" else (args.domain,)
    for domain in domains:
        target = output_root / domain
        if target.exists():
            raise FileExistsError(target)
    for domain in domains:
        prepare(domain, input_root / domain, output_root / domain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
