import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lcwm.config import GAMING_METERS_PER_UNIT
from ldp.action_utils import driving_to_source, validate_source_dt
from ldp.policy import CANDIDATE_FIELDS, HORIZON


def common_chunk_seed(base_seed: int, sample_key: str, chunk_id: int) -> int:
    entropy = [int(base_seed), int(chunk_id), *sample_key.encode("utf-8")]
    words = np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32)
    return ((int(words[0]) << 32) | int(words[1])) % (2**63 - 1)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_top8(path: Path, *, domain: str, split: str) -> dict[str, Any]:
    value = load_json(path)
    expected = {
        "domain": domain,
        "split": split,
        "candidate_count": 128,
        "selected_count": 8,
        "hmax": HORIZON[domain],
        "action_dim": len(CANDIDATE_FIELDS[domain]),
        "action_fields": list(CANDIDATE_FIELDS[domain]),
    }
    mismatch = {
        key: (wanted, value.get(key))
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if mismatch:
        raise ValueError(f"Top-8 file does not match the requested run: {mismatch}")
    rows = value.get("records")
    if (
        not isinstance(rows, list)
        or len(rows) != int(value.get("record_count", -1))
        or not rows
    ):
        raise ValueError("Top-8 records are empty or count-mismatched")
    horizon = HORIZON[domain]
    identities: set[str] = set()
    for row in rows:
        key = str(row.get("sample_key") or "")
        if not key or key in identities:
            raise ValueError(f"duplicate/empty Top-8 identity: {key!r}")
        identities.add(key)
        ids = [int(item) for item in row.get("candidate_indices", [])]
        scores = np.asarray(row.get("selected_language_scores", []), dtype=np.float64)
        actions = np.asarray(row.get("candidate_actions", []), dtype=np.float64)
        masks = np.asarray(row.get("candidate_masks", []), dtype=bool)
        lengths = np.asarray(row.get("candidate_lengths", []), dtype=np.int64)
        action_dim = int(value["action_dim"])
        if len(ids) != 8 or len(set(ids)) != 8 or min(ids) < 0 or max(ids) >= 128:
            raise ValueError(f"{key}: invalid candidate IDs")
        if scores.shape != (8,) or not np.isfinite(scores).all():
            raise ValueError(f"{key}: invalid Language scores")
        if (
            actions.shape != (8, horizon, action_dim)
            or masks.shape != (8, horizon)
            or lengths.shape != (8,)
        ):
            raise ValueError(f"{key}: malformed Top-8 action tensors")
        if not np.isfinite(actions).all():
            raise ValueError(f"{key}: non-finite action")
        if not np.all(lengths == horizon) or not masks.all():
            raise ValueError(f"{key}: expected the full {domain} action horizon")
    return value


def load_initial_observations(
    path: Path, *, domain: str, split: str
) -> dict[str, dict[str, Any]]:
    value = load_json(path)
    expected = {"domain": domain, "split": split}
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise ValueError("initial observations do not match the requested run")
    allowed = {
        "sample_key",
        "dataset_index",
        "domain",
        "instruction",
        "initial_observation_path",
        "source_name",
    }
    output: dict[str, dict[str, Any]] = {}
    rows = value.get("records")
    if not isinstance(rows, list) or len(rows) != int(value.get("record_count", -1)):
        raise ValueError("initial-observation record count mismatch")
    for raw in rows:
        if not isinstance(raw, Mapping) or not allowed.issubset(raw):
            raise ValueError("initial-observation row is missing required fields")
        row = dict(raw)
        key = str(row["sample_key"])
        if not key or key in output or row["domain"] != domain:
            raise ValueError("duplicate/empty/domain-mismatched initial observation")
        image = Path(str(row["initial_observation_path"])).resolve()
        if not image.is_file():
            raise ValueError(f"{key}: missing initial observation")
        if not str(row["instruction"]).strip():
            raise ValueError(f"{key}: empty instruction")
        if domain == "driving":
            validate_source_dt(
                row.get("source_dt"), context=f"{key}.source_dt"
            )
        output[key] = row
    return output


def driving_source_interval(source: str, source_dt: Any | None = None) -> float:
    if source_dt is None:
        raise ValueError(
            f"missing per-sample Driving source_dt for source {source!r}"
        )
    return validate_source_dt(source_dt, context=f"{source}.source_dt")


def driving_action_to_wm(
    action: np.ndarray,
    source: str,
    source_dt: Any | None = None,
) -> np.ndarray:
    interval = driving_source_interval(source, source_dt)
    return driving_to_source(action, interval)


def gaming_action_to_wm(
    action: np.ndarray, config: Mapping[str, Any], source: str
) -> np.ndarray:
    if source not in GAMING_METERS_PER_UNIT or source not in config["sources"]:
        raise ValueError(f"missing Gaming action adapter for {source}")
    values = config["sources"][source]
    scales = np.asarray(
        [
            values["translation_xy_scale"],
            values["translation_xy_scale"],
            values["vertical_scale"],
            values["yaw_scale"],
        ],
        dtype=np.float32,
    )
    if scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError(f"invalid Gaming action scales for {source}")
    output = action.astype(np.float32, copy=True)
    output[:, :3] /= GAMING_METERS_PER_UNIT[source]
    output /= scales
    return output


def robot_action_to_wm(
    action: np.ndarray, config: Mapping[str, Any], source: str
) -> np.ndarray:
    gripper = action[:, 6]
    if np.any(gripper < 0) or np.any(gripper > 1):
        raise ValueError("gripper_closedness lies outside [0,1]")
    values = config["sources"].get(source)
    scale = np.asarray(
        values["pose_scale"] if values else config["default_pose_scale"],
        dtype=np.float32,
    )
    if scale.shape != (6,) or np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError(f"invalid Robot pose scale for {source}")
    output = action.astype(np.float32, copy=True)
    output[:, :6] = np.clip(
        output[:, :6] / scale[None], -float(config["clip"]), float(config["clip"])
    )
    return output


def actions_for_wm(
    domain: str,
    candidate_actions: Sequence[Sequence[float]],
    *,
    source_name: str,
    source_dt: Any | None = None,
    action_config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    action = np.asarray(candidate_actions, dtype=np.float32)
    if action.ndim != 2 or len(action) < 1 or not np.isfinite(action).all():
        raise ValueError("candidate actions must be a finite non-empty [T,C] array")
    expected_dim = len(CANDIDATE_FIELDS[domain])
    if action.shape[1] != expected_dim:
        raise ValueError(
            f"{domain}: expected action dim {expected_dim}, got {action.shape[1]}"
        )
    if domain in {"navigation", "driving"}:
        output = action
    else:
        if action_config is None or action_config.get("domain") != domain:
            raise ValueError(f"missing {domain} world-model action config")
    if domain == "driving":
        output = driving_action_to_wm(
            action, source_name, source_dt=source_dt
        )
    elif domain == "gaming":
        output = gaming_action_to_wm(action, action_config, source_name)
    elif domain == "robot":
        output = robot_action_to_wm(action, action_config, source_name)
    if not np.isfinite(output).all():
        raise ValueError(f"{domain}: non-finite world-model action")
    return np.ascontiguousarray(output)
