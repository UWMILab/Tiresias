from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


DRIVING_DT = 0.1
DRIVING_ACTION_FIELDS = (
    "delta_forward",
    "delta_lateral",
    "delta_yaw",
    "target_velocity",
)


def validate_source_dt(value: Any, *, context: str = "source_dt") -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{context} must be a scalar")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result) or not 0.05 <= result <= 1.0:
        raise ValueError(f"{context} must be in [0.05, 1.0]")
    return result


def driving_action_fields(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(
            item.strip()
            for item in value.replace(",", ";").split(";")
            if item.strip()
        )
    return tuple(str(item) for item in np.asarray(value).reshape(-1).tolist())


def driving_source_dt(
    row: Mapping[str, Any], *, context: str = "Driving metadata"
) -> float:
    if driving_action_fields(row.get("action_fields")) != DRIVING_ACTION_FIELDS:
        raise ValueError(f"{context} action fields do not match Driving")
    if "source_dt" not in row:
        raise ValueError(f"{context} is missing source_dt")
    return validate_source_dt(row["source_dt"], context=f"{context}.source_dt")


def validate_driving_reference(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    context: str = "driving reference",
) -> float:
    manifest_dt = driving_source_dt(row, context=f"{context}.manifest")
    if driving_action_fields(payload.get("action_fields")) != DRIVING_ACTION_FIELDS:
        raise ValueError(f"{context} NPZ action fields do not match Driving")
    action = np.asarray(payload.get("action"), dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 4 or not np.isfinite(action).all():
        raise ValueError(f"{context} has an invalid action array")
    if "physical_action" in payload:
        physical = np.asarray(payload["physical_action"], dtype=np.float32)
        if physical.shape != action.shape or not np.array_equal(physical, action):
            raise ValueError(f"{context} action and physical_action differ")
    if "source_dt" not in payload:
        raise ValueError(f"{context} is missing source_dt")
    payload_dt = validate_source_dt(
        payload["source_dt"], context=f"{context}.npz.source_dt"
    )
    if not np.isclose(manifest_dt, payload_dt, atol=1e-6, rtol=0.0):
        raise ValueError(f"{context} source_dt mismatch")
    return manifest_dt


def _driving_action(action: np.ndarray, *, context: str) -> np.ndarray:
    output = np.asarray(action, dtype=np.float32)
    if output.ndim != 2 or output.shape[1] != len(DRIVING_ACTION_FIELDS):
        raise ValueError(f"{context} must have shape [T,4], got {output.shape}")
    if len(output) < 1 or not np.isfinite(output).all():
        raise ValueError(f"{context} must be non-empty and finite")
    return output.copy()


def driving_to_canonical(action: np.ndarray, source_dt: Any) -> np.ndarray:
    output = _driving_action(action, context="source-cadence action")
    output[:, :3] *= np.float32(DRIVING_DT / validate_source_dt(source_dt))
    return output


def driving_to_source(action: np.ndarray, source_dt: Any) -> np.ndarray:
    output = _driving_action(action, context="canonical action")
    output[:, :3] *= np.float32(validate_source_dt(source_dt) / DRIVING_DT)
    return output


def load_driving_reference(
    row: Mapping[str, Any],
    root: Path,
    *,
    context: str = "driving reference",
) -> tuple[dict[str, Any], Path, float]:
    value = row.get("action_path") or row.get("array")
    if value in (None, ""):
        raise ValueError(f"{context} manifest row is missing action_path/array")
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as raw:
        payload = {key: raw[key] for key in raw.files}
    source_dt = validate_driving_reference(
        row, payload, context=f"{context} {path}"
    )
    return payload, path, source_dt


def robust_z8(
    values: Sequence[float],
    *,
    epsilon: float = 1e-6,
    mad_scale_multiplier: float = 1.4826,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (8,) or not np.isfinite(array).all():
        raise ValueError(f"robust_z8 expects eight finite scores, got {array.shape}")
    ordered = np.sort(array)
    center = 0.5 * (ordered[3] + ordered[4])
    deviations = np.sort(np.abs(array - center))
    mad = 0.5 * (deviations[3] + deviations[4])
    if mad <= epsilon:
        return np.zeros(8, dtype=np.float64)
    scale = max(mad_scale_multiplier * mad, epsilon)
    return (array - center) / scale


def fuse_language_wm_top8(
    language_score: Sequence[float],
    *,
    wm_score: Sequence[float] | None = None,
    wm_cost: Sequence[float] | None = None,
) -> dict[str, np.ndarray | int | str]:
    if (wm_score is None) == (wm_cost is None):
        raise ValueError("provide exactly one of wm_score or wm_cost")
    language = np.asarray(language_score, dtype=np.float64)
    wm_higher = (
        np.asarray(wm_score, dtype=np.float64)
        if wm_score is not None
        else -np.asarray(wm_cost, dtype=np.float64)
    )
    language_z = robust_z8(language)
    wm_z = robust_z8(wm_higher)
    final = 2.0 * language_z + wm_z
    candidate_ids = np.arange(8, dtype=np.int64)
    slot = max(
        range(8),
        key=lambda index: (
            float(final[index]),
            float(language_z[index]),
            float(wm_z[index]),
            -int(candidate_ids[index]),
        ),
    )
    return {
        "normalized_language_score": language_z,
        "normalized_wm_score": wm_z,
        "final_score": final,
        "selected_top8_slot": int(slot),
    }


def euler_zyx_matrix(euler: np.ndarray | Iterable[float]) -> np.ndarray:
    value = np.asarray(euler, dtype=np.float64)
    roll, pitch, yaw = np.moveaxis(value, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    row0 = np.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), -1)
    row1 = np.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), -1)
    row2 = np.stack((-sp, cp * sr, cp * cr), -1)
    return np.stack((row0, row1, row2), -2)


def integrate_planar_ego(actions: np.ndarray, *, vertical: bool = False) -> np.ndarray:
    action = np.asarray(actions, dtype=np.float64)
    expected = 4 if vertical else 3
    if action.ndim != 2 or action.shape[1] < expected or len(action) < 1:
        raise ValueError(f"expected [T,{expected}+], got {action.shape}")
    yaw_index = 3 if vertical else 2
    forward, lateral = action[:, 0], action[:, 1]
    delta_yaw = action[:, yaw_index]
    yaw_before = np.cumsum(delta_yaw) - delta_yaw
    dx = np.cos(yaw_before) * forward - np.sin(yaw_before) * lateral
    dy = np.sin(yaw_before) * forward + np.cos(yaw_before) * lateral
    xy = np.stack((np.cumsum(dx), np.cumsum(dy)), axis=-1)
    if not vertical:
        return xy
    z = np.cumsum(action[:, 2])
    return np.concatenate((xy, z[:, None]), axis=-1)


def integrate_robot_local(actions: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(actions, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 7 or len(action) < 1:
        raise ValueError(f"expected [T,7], got {action.shape}")
    current_position = np.zeros(3, dtype=np.float64)
    current_rotation = np.eye(3, dtype=np.float64)
    positions, rotations = [], []
    for step in action:
        current_position = current_position + current_rotation @ step[:3]
        current_rotation = current_rotation @ euler_zyx_matrix(step[3:6])
        positions.append(current_position.copy())
        rotations.append(current_rotation.copy())
    return {
        "position": np.stack(positions),
        "rotation_matrix": np.stack(rotations),
        "gripper": np.clip(action[:, 6], 0.0, 1.0),
    }
