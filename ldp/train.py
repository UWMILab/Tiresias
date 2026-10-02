import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lcwm.datasets.protocol import read_records, select_training_items
from ldp.action_utils import (
    DRIVING_ACTION_FIELDS,
    driving_to_canonical,
    load_driving_reference,
    validate_source_dt,
)
from ldp.policy import ACTION_FIELDS, HORIZON, build_policy, continuous_dim, save_policy

ACTION_DIM = {domain: len(fields) for domain, fields in ACTION_FIELDS.items()}


def parse_fields(value: str) -> list[str]:
    value = str(value).strip()
    if value.startswith("["):
        return list(json.loads(value))
    return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]


def cadence_adapter(action: np.ndarray, source_dt: float) -> np.ndarray:
    return driving_to_canonical(action, validate_source_dt(source_dt))


def load_target(domain: str, row: dict[str, str], root: Path) -> tuple[np.ndarray, int]:
    path = Path(row.get("action_path") or row.get("array") or "")
    if not path.is_absolute():
        path = root / path
    if domain == "driving":
        fields = parse_fields(row.get("action_fields", ""))
        if tuple(fields) != DRIVING_ACTION_FIELDS:
            raise ValueError(
                f"{row.get('trajectory_id')}: unsupported Driving action fields {fields!r}"
            )
        payload, path, source_dt = load_driving_reference(
            row, root, context=f"{row.get('trajectory_id', '<unknown>')} Driving target"
        )
        action = np.asarray(payload["action"], dtype=np.float32)
    else:
        with np.load(path, allow_pickle=False) as payload:
            action = np.asarray(payload["action"], dtype=np.float32)
    if domain == "driving":
        action = cadence_adapter(action, source_dt)
    elif domain == "navigation":
        action = action[:, :3]
    elif domain == "robot":
        fields = parse_fields(row.get("action_fields", ""))
        if fields and fields[-1] == "gripper_closedness":
            action = action.copy()
            action[:, -1] = 1.0 - np.clip(action[:, -1], 0.0, 1.0)
    elif domain == "gaming":
        fields = parse_fields(row.get("action_fields", ""))
        expected = ["delta_forward", "delta_lateral", "delta_vertical", "delta_yaw"]
        if fields and fields != expected:
            raise ValueError(f"unsupported Gaming action fields: {fields}")
    if (
        action.ndim != 2
        or action.shape[1] != ACTION_DIM[domain]
        or not np.isfinite(action).all()
    ):
        raise ValueError(
            {"sample": row.get("trajectory_id"), "shape": list(action.shape)}
        )
    if len(action) == 0:
        raise ValueError(f"{row.get('trajectory_id')}: empty action sequence")
    target = np.zeros((HORIZON[domain], ACTION_DIM[domain]), dtype=np.float32)
    declared_value = row.get("future_horizon")
    declared = int(declared_value) if declared_value not in (None, "") else len(action)
    if declared < 1 or declared > len(action):
        raise ValueError(
            f"{row.get('trajectory_id')}: invalid reference horizon {declared} for {len(action)} actions"
        )
    # Short expert sequences supervise only their real prefix.
    length = min(declared, HORIZON[domain])
    if length < 1:
        raise ValueError(f"{row.get('trajectory_id')}: empty action horizon")
    target[:length] = action[:length]
    return target, length


class TrajectoryDataset(Dataset):
    def __init__(
        self, domain: str, manifest: Path, cache_path: Path, data_ratio: float
    ):
        self.domain = domain
        rows = read_records(manifest)
        self.rows = select_training_items(
            rows, data_ratio, key=lambda row: str(row["trajectory_id"])
        )
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.tokens = cache["token_embeddings"]
        self.token_masks = cache["token_masks"]
        loaded = [
            load_target(domain, row, manifest.resolve().parent) for row in self.rows
        ]
        self.targets = [target for target, _ in loaded]
        self.lengths = [length for _, length in loaded]
        missing = {row["instruction"] for row in self.rows} - set(self.tokens)
        if missing:
            raise KeyError({"missing_instruction_embeddings": len(missing)})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.rows[index]
        raw = torch.from_numpy(self.targets[index]).float()
        target = raw
        mask = torch.arange(HORIZON[self.domain]) < self.lengths[index]
        text = row["instruction"]
        return {
            "tokens": torch.as_tensor(self.tokens[text]),
            "language_mask": torch.as_tensor(self.token_masks[text]).bool(),
            "target": target,
            "future_mask": mask,
        }


def collate(rows: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        "tokens": torch.nn.utils.rnn.pad_sequence(
            [row["tokens"] for row in rows], batch_first=True
        ),
        "language_mask": torch.nn.utils.rnn.pad_sequence(
            [row["language_mask"] for row in rows], batch_first=True
        ),
        "target": torch.stack([row["target"] for row in rows]),
        "future_mask": torch.stack([row["future_mask"] for row in rows]),
    }


def build_model(domain: str, action_config: dict[str, Any]):
    return build_policy(
        domain,
        tuple(float(value) for value in action_config["mean"]),
        tuple(float(value) for value in action_config["std"]),
    )


def masked_mse(value: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid = mask.unsqueeze(-1).to(value.dtype)
    return ((value - target).square() * valid).sum() / (
        valid.sum().clamp_min(1) * value.shape[-1]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=tuple(HORIZON), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--language-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not np.isfinite(args.data_ratio) or not 0 < args.data_ratio <= 1:
        parser.error("--data_ratio must be in (0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_config_path = args.train_manifest.parent / "action_config.json"
    if not action_config_path.is_file():
        parser.error(f"Action config not found: {action_config_path}")
    norm = json.loads(action_config_path.read_text(encoding="utf-8"))
    if norm.get("domain") not in (None, args.domain):
        raise ValueError("action config does not match --domain")
    dataset = TrajectoryDataset(
        args.domain, args.train_manifest, args.language_cache, args.data_ratio
    )
    if len(dataset) < args.batch_size:
        parser.error(
            f"training subset has {len(dataset)} trajectories, fewer than "
            f"--batch-size {args.batch_size}"
        )
    updates = max(1, round(args.updates * args.data_ratio))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0")
    model = build_model(args.domain, norm)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    iterator = iter(loader)
    started = time.time()
    learned_dim = continuous_dim(args.domain)
    for update in range(1, updates + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        tokens = batch["tokens"].to(device)
        language_mask = batch["language_mask"].to(device)
        target = batch["target"].to(device)
        future_mask = batch["future_mask"].to(device)
        continuous = target[..., :learned_dim]
        normalized = model.normalize_actions(continuous)
        timestep = torch.randint(
            0, model.config.diffusion_steps, (len(target),), device=device
        )
        noise = torch.randn_like(normalized) * future_mask.unsqueeze(-1)
        alpha = model.alpha_bars[timestep].view(-1, 1, 1)
        noisy = (
            alpha.sqrt() * normalized + (1 - alpha).sqrt() * noise
        ) * future_mask.unsqueeze(-1)
        drop = (
            torch.rand(len(target), device=device) < model.config.cfg_dropout_prob
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(
                noisy, timestep, tokens, language_mask, future_mask, drop
            )
            epsilon_loss = masked_mse(prediction, noise, future_mask)
            gripper_loss = epsilon_loss.new_zeros(())
            if args.domain == "robot":
                logits = model.gripper_logits(tokens, language_mask, drop)
                gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[future_mask],
                    target[..., 6][future_mask],
                )
            loss = epsilon_loss + 0.5 * gripper_loss
        if not torch.isfinite(loss):
            raise FloatingPointError({"update": update, "loss": float(loss)})
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(gradient):
            raise FloatingPointError({"update": update, "gradient": gradient})
        optimizer.step()
        if update == 1 or update % 25 == 0 or update == updates:
            row = {
                "update": update,
                "loss": float(loss.detach()),
                "gradient_norm": gradient,
                "elapsed_seconds": time.time() - started,
            }
            print(json.dumps(row), flush=True)
    checkpoint = args.output_dir / "LDP.pt"
    save_policy(checkpoint, args.domain, model.eval())
    if args.domain in {"gaming", "robot"}:
        shutil.copyfile(action_config_path, args.output_dir / "action_config.json")
    summary = {
        "domain": args.domain,
        "checkpoint": str(checkpoint),
        "training_records": len(dataset),
        "updates": updates,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
