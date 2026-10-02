import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np


IMAGE_SIZE = 256
PUBLIC_SPLITS = ("train", "val_seen", "val_unseen")
INFERENCE_SPLITS = (*PUBLIC_SPLITS, "input")

T = TypeVar("T")

# Planning starts at source frame 3; WM training keeps the full F/F-1 protocol.
PLANNING_ANCHOR_FRAME_IDX = 3

LORA_RANK = 32
WEIGHT_DECAY = 1e-2

ACTION_EMBED_DIM = 256
ACTION_CONTEXT_TOKENS = 8
ACTION_TIME_HIDDEN_DIM = 512
ACTION_TIME_SCALE = 1.0

SEMANTIC_LATENT_DIM = 1024
SEMANTIC_CONTEXT_TOKENS = 8
SEMANTIC_CONTEXT_SCALE = 1.0
SEMANTIC_FRAME_INDEX = 0

DINO_MODEL_NAME = "dinov3_vitl16"
DINO_IMAGE_SIZE = 256
LORA_ALPHA = 1.0
MIN_WAN_FRAMES = 1


def legal_frame_count(count: int) -> int:
    return ((int(count) - 1 + 3) // 4) * 4 + 1


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix != ".jsonl":
        raise ValueError(f"expected a JSONL file: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"empty or invalid records: {path}")
    return rows


def write_records(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def episode_id(trajectory_id: str) -> str:
    """Return the shared episode id for the _0/_1/_2 trajectory styles."""

    return re.sub(r"_[012]$", "", str(trajectory_id))


def select_training_items(
    values: Sequence[T], ratio: float, *, key: Callable[[T], str]
) -> list[T]:
    """Select complete episode groups for a training-only data ratio."""

    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("data ratio must be in (0, 1]")
    values = list(values)
    if ratio >= 1.0:
        return values
    groups: dict[str, list[T]] = {}
    for value in values:
        groups.setdefault(episode_id(key(value)), []).append(value)
    count = max(1, round(len(groups) * ratio))
    group_ids = sorted(groups)
    np.random.default_rng(0).shuffle(group_ids)
    selected = set(group_ids[:count])
    return [value for value in values if episode_id(key(value)) in selected]
