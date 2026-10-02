import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lcwm.datasets.protocol import INFERENCE_SPLITS
from ldp.policy import (
    CANDIDATE_FIELDS,
    HORIZON,
    continuous_dim,
    load_policy,
)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def batches(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def load_inputs(args: argparse.Namespace):
    payload = json.loads(args.inputs.read_text())
    if isinstance(payload, list):
        rows = list(payload)
        payload = {"records": rows}
    elif isinstance(payload, dict):
        rows = list(payload.get("records") or payload.get("predictions") or [])
    else:
        raise TypeError("--inputs must contain a JSON list or object")
    if args.max_records:
        rows = rows[: args.max_records]
    rows = [
        row
        for row in rows
        if int(row["dataset_index"]) % args.shard_count == args.shard_index
    ]
    if not rows:
        raise ValueError("input contains no records")
    if any(
        str(row.get("domain")) != args.domain
        or str(row.get("split")) != args.split
        for row in rows
    ):
        raise ValueError("input records do not match --domain/--split")
    identities = [str(row.get("sample_key") or "") for row in rows]
    if not all(identities) or len(identities) != len(set(identities)):
        raise ValueError("input records have empty or duplicate sample_key")
    cache = torch.load(args.language_cache, map_location="cpu", weights_only=True)
    embeddings = cache["token_embeddings"]
    masks = cache.get("token_masks")
    missing = {str(row["instruction"]) for row in rows} - set(embeddings)
    if missing:
        raise KeyError({"missing_instructions": len(missing)})
    return payload, rows, embeddings, masks


def token_batch(rows, embeddings, masks):
    values = []
    for row in rows:
        text = str(row["instruction"])
        value = torch.as_tensor(embeddings[text])
        if masks is not None:
            mask = torch.as_tensor(masks[text]).bool()
            value = value[mask]
        values.append(value)
    length = max(len(value) for value in values)
    tokens = torch.zeros(len(rows), length, values[0].shape[-1], dtype=torch.float16)
    token_masks = torch.zeros(len(rows), length, dtype=torch.bool)
    for index, value in enumerate(values):
        tokens[index, : len(value)] = value.to(torch.float16)
        token_masks[index, : len(value)] = True
    return tokens, token_masks


def load_model(domain: str, checkpoint: Path, device: torch.device):
    model, payload = load_policy(checkpoint, domain, device)
    learned_dim = continuous_dim(domain)
    if (
        int(model.config.horizon) != HORIZON[domain]
        or int(model.config.action_dim) != learned_dim
    ):
        raise ValueError("checkpoint shape contract mismatch")
    return model.eval(), payload


def independent_noise(rows: list[dict[str, Any]], domain: str, base_seed: int):
    learned_dim = continuous_dim(domain)
    horizon = HORIZON[domain]
    noise, masks = [], []
    for row in rows:
        row_noise = []
        for candidate in range(128):
            seed = base_seed + int(row["dataset_index"]) * 1000003 + candidate * 1009
            generator = torch.Generator(device="cpu").manual_seed(seed)
            row_noise.append(torch.randn(horizon, learned_dim, generator=generator))
        noise.append(torch.stack(row_noise))
        masks.append(torch.ones(128, horizon, dtype=torch.bool))
    return torch.stack(noise), torch.stack(masks)


@torch.inference_mode()
def generate(
    model,
    domain,
    tokens,
    language_mask,
    noise,
    masks,
    ddim_steps,
    guidance,
    *,
    robot_gripper_semantics: str | None = None,
):
    batch, candidates, horizon, learned_dim = noise.shape
    expected_horizon = HORIZON[domain]
    expected_dim = continuous_dim(domain)
    if (horizon, learned_dim) != (expected_horizon, expected_dim):
        raise ValueError(
            {
                "noise_shape": list(noise.shape),
                "expected": ["B", "N", expected_horizon, expected_dim],
            }
        )
    if masks.shape != (batch, candidates, expected_horizon):
        raise ValueError(
            {"mask_shape": list(masks.shape), "expected": [batch, candidates, expected_horizon]}
        )
    # Reference/GT metadata is not an inference-time truncation signal.
    if not bool(masks.all()):
        raise ValueError("LDP inference requires a complete native action horizon")
    device = tokens.device
    valid = masks.to(device).reshape(batch * candidates, horizon)
    state = noise.to(device=device, dtype=model.noisy_action_proj.weight.dtype).reshape(
        batch * candidates, horizon, learned_dim
    ) * valid.unsqueeze(-1)
    memory, padding = model.language_memory(tokens, language_mask)
    empty_memory, empty_padding = model.language_memory(
        tokens, language_mask, torch.ones(batch, device=device, dtype=torch.bool)
    )
    memory, padding = (
        memory.repeat_interleave(candidates, 0),
        padding.repeat_interleave(candidates, 0),
    )
    empty_memory = empty_memory.repeat_interleave(candidates, 0)
    empty_padding = empty_padding.repeat_interleave(candidates, 0)
    indices = (
        torch.linspace(model.config.diffusion_steps - 1, 0, ddim_steps, device=device)
        .round()
        .long()
        .unique_consecutive()
    )
    if int(indices[-1]) != 0:
        indices = torch.cat((indices, indices.new_zeros(1)))
    for position, value in enumerate(indices):
        timestep = torch.full(
            (len(state),), int(value), device=device, dtype=torch.long
        )
        conditional = model.denoise_with_memory(state, timestep, memory, padding, valid)
        unconditional = model.denoise_with_memory(
            state, timestep, empty_memory, empty_padding, valid
        )
        epsilon = unconditional + float(guidance) * (conditional - unconditional)
        clean = model.predict_x0_from_epsilon(state, timestep, epsilon).clamp(-8.0, 8.0)
        if position + 1 == len(indices):
            state = clean
            break
        previous = model.alpha_bars[indices[position + 1]].view(1, 1, 1)
        state = (
            previous.sqrt() * clean + (1.0 - previous).sqrt() * epsilon
        ) * valid.unsqueeze(-1)
    action = model.denormalize_actions(state).reshape(
        batch, candidates, horizon, learned_dim
    )
    if domain == "robot":
        probability = torch.sigmoid(model.gripper_logits(tokens, language_mask))
        gripper = (
            (probability >= 0.5).to(action.dtype)[:, None].expand(-1, candidates, -1)
        )
        if robot_gripper_semantics == "gripper_open=1":
            gripper = 1.0 - gripper
        elif robot_gripper_semantics != "gripper_closedness=1":
            raise ValueError(
                f"unknown Robot gripper checkpoint semantics: {robot_gripper_semantics!r}"
            )
        action = torch.cat((action, gripper.unsqueeze(-1)), dim=-1)
    action = action * masks.to(device).unsqueeze(-1)
    return action.float().cpu()


def decode_adapter(
    domain: str,
    rows,
    action: Tensor,
    adapter: dict[str, Any] | None,
    checkpoint: dict[str, Any],
) -> Tensor:
    if domain not in {"robot", "gaming"}:
        return action
    if adapter is None:
        raise ValueError(f"{domain} requires --action-config")
    result = action.clone()
    for index, row in enumerate(rows):
        source = str(row["source_name"])
        if domain == "robot":
            scales = adapter["translation_scale_by_source"]
            scale = float(scales.get(source, adapter.get("default_translation_scale", 0)))
            if scale <= 0:
                raise ValueError(f"Robot action config has no scale for {source!r}")
            result[index, :, :, :3] *= scale
        else:
            config = checkpoint.get("config", {})
            native_mean = torch.tensor(
                config.get("action_mean", ()), dtype=result.dtype
            )
            native_std = torch.tensor(config.get("action_std", ()), dtype=result.dtype)
            if (
                native_mean.shape != (4,)
                or native_std.shape != (4,)
                or bool((native_std <= 0).any())
            ):
                raise ValueError("Gaming checkpoint is missing action statistics")
            declared = adapter.get("ldp_native_statistics")
            if declared is not None:
                expected_mean = torch.tensor(declared["mean"], dtype=result.dtype)
                expected_std = torch.tensor(declared["std"], dtype=result.dtype)
                if not torch.allclose(
                    native_mean, expected_mean, atol=1e-7, rtol=0
                ) or not torch.allclose(native_std, expected_std, atol=1e-7, rtol=0):
                    raise ValueError("Gaming LDP and adapter statistics do not match")
            stats = adapter["statistics"][source]
            mean = torch.tensor(stats["mean"], dtype=result.dtype).view(1, 1, 4)
            std = torch.tensor(stats["std"], dtype=result.dtype).view(1, 1, 4)
            native_z = (result[index] - native_mean.view(1, 1, 4)) / native_std.view(
                1, 1, 4
            )
            result[index] = native_z * std + mean
            result[index, :, :, 3] = torch.atan2(
                torch.sin(result[index, :, :, 3]), torch.cos(result[index, :, :, 3])
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=tuple(HORIZON), required=True)
    parser.add_argument("--split", choices=INFERENCE_SPLITS, default="val_seen")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--language-cache", required=True, type=Path)
    parser.add_argument("--action-config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ddim-steps", type=int, default=12)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    _, rows, embeddings, token_masks = load_inputs(args)
    adapter = json.loads(args.action_config.read_text()) if args.action_config else None
    device = torch.device("cuda:0")
    model, checkpoint = load_model(args.domain, args.checkpoint, device)
    robot_gripper_semantics = None
    if args.domain == "robot":
        robot_gripper_semantics = str(
            checkpoint.get("contract", {}).get("gripper_semantics", "")
        )
        if robot_gripper_semantics not in {"gripper_open=1", "gripper_closedness=1"}:
            raise ValueError(
                f"Robot LDP checkpoint lacks an unambiguous gripper contract: "
                f"{robot_gripper_semantics!r}"
            )
    started, records = time.time(), []
    for batch_index, batch_rows in enumerate(batches(rows, args.batch_size), start=1):
        tokens, language_mask = token_batch(batch_rows, embeddings, token_masks)
        tokens, language_mask = tokens.to(device), language_mask.to(device)
        noise, candidate_masks = independent_noise(batch_rows, args.domain, args.seed)
        actions = generate(
            model,
            args.domain,
            tokens,
            language_mask,
            noise,
            candidate_masks,
            args.ddim_steps,
            args.guidance,
            robot_gripper_semantics=robot_gripper_semantics,
        )
        actions = decode_adapter(args.domain, batch_rows, actions, adapter, checkpoint)
        lengths = candidate_masks.sum(-1).to(torch.int16)
        for index, row in enumerate(batch_rows):
            records.append(
                {
                    "dataset_index": int(row["dataset_index"]),
                    "sample_id": str(row["sample_id"]),
                    "sample_key": str(row["sample_key"]),
                    "instruction": str(row["instruction"]),
                    "source_name": str(row["source_name"]),
                    "candidate_lengths": lengths[index],
                    "candidate_masks": candidate_masks[index],
                    "candidate_actions": actions[index],
                }
            )
        if batch_index == 1 or batch_index % 10 == 0 or len(records) == len(rows):
            elapsed = time.time() - started
            rate = len(records) / max(elapsed, 1e-6)
            print(
                json.dumps(
                    {
                        "completed": len(records),
                        "total": len(rows),
                        "eta_seconds": (len(rows) - len(records)) / max(rate, 1e-6),
                    }
                ),
                flush=True,
            )
    payload = {
        "domain": args.domain,
        "split": args.split,
        "candidate_count": 128,
        "record_count": len(records),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "horizon": HORIZON[args.domain],
        "action_fields": list(CANDIDATE_FIELDS[args.domain]),
        "elapsed_seconds": time.time() - started,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    summary = {
        "domain": args.domain,
        "output": str(args.output.resolve()),
        "record_count": len(records),
        "candidate_count": 128,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
