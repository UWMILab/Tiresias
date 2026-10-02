import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lcwm.action_conditioning import (
    ActionContextEncoder,
    ActionTimeConditioner,
    Dinov3SemanticEncoder,
)
from lcwm.config import DOMAINS, NEGATIVE_PROMPTS, joint_text
from lcwm.wan22.diffsynth.core.loader.config import ModelConfig
from lcwm.wan22.diffsynth.core.loader.file import load_state_dict
from lcwm.wan22.diffsynth.pipelines.wan_video import WanVideoPipeline
from lcwm.datasets.protocol import (
    DINO_IMAGE_SIZE,
    DINO_MODEL_NAME,
    IMAGE_SIZE,
    LORA_ALPHA,
    MIN_WAN_FRAMES,
    PLANNING_ANCHOR_FRAME_IDX,
    read_records,
)
from lcwm.datasets.video import (
    contain_and_pad,
    content_box_from_image,
    mask_padding_tensor,
    preprocessing_contract,
)
from lcwm.semantic_backbone import encode_tokens, load_dinov3
from ldp.policy import CANDIDATE_FIELDS, HORIZON


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def numbered_frames(frame_dir: Path) -> list[Path]:
    frames = sorted(
        (
            path
            for path in frame_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ),
        key=_frame_sort_key,
    )
    if not frames:
        raise FileNotFoundError(f"no RGB frames found in {frame_dir}")
    return frames


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.name


def load_raw_frame(path: Path, frame_index: int) -> Image.Image:
    frames = numbered_frames(path)
    if not 0 <= frame_index < len(frames):
        raise IndexError(f"frame {frame_index} is unavailable in {path}")
    with Image.open(frames[frame_index]) as image:
        return image.convert("RGB").copy()


def load_video_frame(
    video_path: Path, frame_index: int, width: int, height: int
) -> Image.Image:
    if video_path.is_dir():
        image = load_raw_frame(video_path, frame_index)
    else:
        import imageio.v2 as imageio

        reader = imageio.get_reader(video_path)
        try:
            image = Image.fromarray(reader.get_data(frame_index)).convert("RGB")
        finally:
            reader.close()
    return contain_and_pad(image, (width, height))


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_tensor_image(frame: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.detach().cpu().float().clamp(0, 1)
    array = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def as_frame_list(video: Any) -> list[torch.Tensor]:
    if isinstance(video, list):
        frames = []
        for item in video:
            if isinstance(item, Image.Image):
                frames.append(image_to_tensor(item.convert("RGB")))
                continue
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"unsupported frame type: {type(item)}")
            tensor = item.detach().cpu().float()
            if tensor.ndim == 3 and tensor.shape[0] == 3:
                frames.append(tensor.clamp(0, 1))
            elif tensor.ndim == 3 and tensor.shape[-1] == 3:
                frames.append(tensor.permute(2, 0, 1).clamp(0, 1))
            else:
                raise ValueError(
                    f"unsupported tensor frame shape: {tuple(tensor.shape)}"
                )
        return frames

    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise TypeError("generated video must be a frame list or rank-4 tensor")
    tensor = video.detach().cpu().float()
    if tensor.shape[1] == 3:
        return [frame.clamp(0, 1) for frame in tensor]
    if tensor.shape[-1] == 3:
        return [frame.permute(2, 0, 1).clamp(0, 1) for frame in tensor]
    raise ValueError(
        f"cannot infer channel axis from video shape {tuple(tensor.shape)}"
    )


def save_mp4(frames: list[torch.Tensor], path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [
        (frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        for frame in frames
    ]
    imageio.mimsave(path, arrays, fps=fps)


def build_pipeline(args: argparse.Namespace) -> WanVideoPipeline:
    wan_root = Path(args.wan_root)
    shards = sorted(wan_root.glob("diffusion_pytorch_model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"Wan2.2 diffusion weights not found in {wan_root}")
    model_configs = [
        ModelConfig([str(path) for path in shards]),
        ModelConfig(str(wan_root / "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(str(wan_root / "Wan2.2_VAE.pth")),
    ]
    tokenizer_dir = wan_root / "google" / "umt5-xxl"
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype(args.torch_dtype),
        device="cuda:0",
        model_configs=model_configs,
        tokenizer_config=ModelConfig(str(tokenizer_dir)),
    )
    pipe.load_lora(pipe.dit, args.lora_path, alpha=LORA_ALPHA)
    return pipe


def encode_conditioning_semantic_tokens(
    image: Image.Image,
    model: torch.nn.Module,
    image_size: int,
    device: str,
) -> torch.Tensor:
    return encode_tokens(image, model, image_size, device)


def _checkpoint_group(
    state: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    values = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not values:
        raise RuntimeError(f"checkpoint does not contain {prefix.removesuffix('.')}")
    return values


def _load_checkpoint(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    return load_state_dict(
        args.lora_path,
        torch_dtype=torch_dtype(args.torch_dtype),
        device="cuda:0",
    )


def build_action_context_encoder(
    args: argparse.Namespace,
    pipe: WanVideoPipeline,
    state: dict[str, torch.Tensor] | None = None,
) -> ActionContextEncoder:
    state = _load_checkpoint(args) if state is None else state
    values = _checkpoint_group(state, "action_context_encoder.")
    encoder = ActionContextEncoder(
        args.domain,
        hidden_dim=int(values["net.0.weight"].shape[0]),
        out_dim=pipe.dit.dim,
        num_tokens=int(values["pos"].shape[1]),
    ).to(device="cuda:0", dtype=torch_dtype(args.torch_dtype))
    encoder.load_state_dict(values, strict=True)
    return encoder.eval()


def build_semantic_context_encoder(
    args: argparse.Namespace,
    pipe: WanVideoPipeline,
    state: dict[str, torch.Tensor] | None = None,
) -> Dinov3SemanticEncoder:
    state = _load_checkpoint(args) if state is None else state
    values = _checkpoint_group(state, "semantic_context_encoder.")
    encoder = Dinov3SemanticEncoder(
        latent_dim=int(values["net.0.weight"].shape[1]),
        hidden_dim=int(values["net.0.weight"].shape[0]),
        out_dim=pipe.dit.dim,
        num_tokens=int(values["pos"].shape[1]),
    ).to(device="cuda:0", dtype=torch_dtype(args.torch_dtype))
    encoder.load_state_dict(values, strict=True)
    return encoder.eval()


def install_action_time_conditioner(
    args: argparse.Namespace,
    pipe: WanVideoPipeline,
    state: dict[str, torch.Tensor] | None = None,
) -> None:
    state = _load_checkpoint(args) if state is None else state
    values = _checkpoint_group(state, "action_time_conditioner.")
    conditioner = ActionTimeConditioner(
        dim=pipe.dit.dim,
        hidden_dim=int(values["input_proj.weight"].shape[0]),
    ).to(device="cuda:0", dtype=torch_dtype(args.torch_dtype))
    conditioner.load_state_dict(values, strict=True)
    pipe.dit.action_time_conditioner = conditioner.eval()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Tiresias world-model inference.")
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--wan-root", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--dinov3-repo", required=True)
    parser.add_argument("--semantic-weights", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    domain = args.domain
    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.lora_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    records = read_records(manifest_path)
    records = records[args.start_index :]
    if args.limit is not None:
        records = records[: args.limit]
    records = [
        dict(row, __global_idx__=args.start_index + index)
        for index, row in enumerate(records)
    ]
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    records = [
        row
        for row in records
        if int(row["__global_idx__"]) % args.num_shards == args.shard_index
    ]
    sources = {str(row.get("action_source", "")) for row in records}
    if sources != {"diffusion_policy_top1"}:
        raise ValueError(
            "manifest action_source must be 'diffusion_policy_top1', "
            f"got {sorted(sources)}"
        )
    domains = {str(row.get("domain", "")) for row in records}
    if domains != {domain}:
        raise ValueError(f"manifest domain must be {domain!r}, got {sorted(domains)}")
    expected_fields = tuple(CANDIDATE_FIELDS[domain])
    fields = {
        tuple(str(row.get("action_fields", "")).split(";")) for row in records
    }
    if fields != {expected_fields}:
        raise ValueError(
            f"manifest action fields must be {expected_fields}, got {sorted(fields)}"
        )

    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(args)
    checkpoint = _load_checkpoint(args)
    install_action_time_conditioner(args, pipe, checkpoint)
    action_encoder = build_action_context_encoder(args, pipe, checkpoint)
    semantic_encoder = build_semantic_context_encoder(args, pipe, checkpoint)
    dino = load_dinov3(
        Path(args.dinov3_repo),
        Path(args.semantic_weights),
        DINO_MODEL_NAME,
        "cuda:0",
    )
    dataset_root = (
        manifest_path.parent.parent
        if manifest_path.parent.name == "metadata"
        else manifest_path.parent
    )

    for record in records:
        sample_index = int(record.get("sample_idx", record["__global_idx__"]))
        trajectory_id = str(record.get("trajectory_id", f"sample_{sample_index:03d}"))
        sample_dir = output_root / f"sample_{sample_index:03d}_{trajectory_id}"
        original_frames = int(record["orig_num_frames"])
        requested_frames = int(record["wan_num_frames"])
        wan_frames = max(requested_frames, MIN_WAN_FRAMES)
        if (wan_frames - 1) % 4:
            wan_frames = ((wan_frames - 1 + 3) // 4) * 4 + 1
        horizon = int(record.get("future_horizon", original_frames - 1))
        context_index = int(record.get("context_frame_idx", 0) or 0)
        if horizon <= 0:
            continue
        expected_horizon = HORIZON[domain]
        if horizon != expected_horizon or original_frames != expected_horizon + 1:
            raise ValueError(
                f"{trajectory_id}: final rollout must be anchor+H={expected_horizon} "
                f"(frames={expected_horizon + 1}), got horizon={horizon}, "
                f"frames={original_frames}"
            )
        if context_index != PLANNING_ANCHOR_FRAME_IDX:
            raise ValueError(
                f"{trajectory_id}: final rollout anchor must be source frame "
                f"{PLANNING_ANCHOR_FRAME_IDX}, got {context_index}"
            )
        if sample_dir.exists():
            raise FileExistsError(sample_dir)

        video_path = dataset_root / record["video"]
        with np.load(dataset_root / record["array"], allow_pickle=False) as arrays:
            action = arrays["action"].astype(np.float32)
            stored_fields = tuple(
                str(value) for value in arrays["wm_fields"].tolist()
            )
        if stored_fields != expected_fields:
            raise ValueError(
                f"{trajectory_id}: stored action fields {stored_fields} do not match "
                f"{expected_fields}"
            )
        if action.shape != (horizon, len(expected_fields)) or not np.isfinite(action).all():
            raise ValueError(
                f"{trajectory_id}: action must be finite with shape "
                f"({horizon}, {len(expected_fields)}), got {action.shape}"
            )
        prompt = joint_text(domain, record.get("instruction", ""))
        image = load_video_frame(video_path, context_index, IMAGE_SIZE, IMAGE_SIZE)
        content_box = content_box_from_image(image)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device="cuda:0")
        with torch.no_grad():
            action_context = action_encoder(action_tensor)
            semantic_tokens = encode_conditioning_semantic_tokens(
                image, dino, image_size=DINO_IMAGE_SIZE, device="cuda:0"
            )
            representation_context = semantic_encoder(semantic_tokens)

        generated = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPTS[domain],
            input_image=image,
            action_context=action_context,
            representation_context=representation_context,
            num_frames=wan_frames,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            seed=args.seed,
            tiled=True,
            cfg_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
        )
        frames = [
            mask_padding_tensor(frame, content_box)
            for frame in as_frame_list(generated)
        ]
        if len(frames) < wan_frames:
            raise RuntimeError(f"generated {len(frames)} frames, expected {wan_frames}")
        aligned_frames = frames[:original_frames]
        predicted = aligned_frames[1:]
        if len(predicted) != horizon:
            raise RuntimeError(f"future length {len(predicted)} != {horizon}")

        pred_dir = sample_dir / "pred_rgb"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for step, frame in enumerate(predicted, start=1):
            save_tensor_image(frame, pred_dir / f"step_{step:02d}.png")
        if args.save_mp4:
            save_mp4(aligned_frames, sample_dir / "pred_video.mp4", args.fps)
        metadata = {
            "trajectory_id": trajectory_id,
            "action_source": "diffusion_policy_top1",
            "future_horizon": horizon,
            "context_frame_idx": context_index,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "video_preprocessing": preprocessing_contract(),
        }
        (sample_dir / "meta.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
