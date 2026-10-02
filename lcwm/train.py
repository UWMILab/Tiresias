import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import accelerate
import imageio.v2 as imageio
import numpy as np
import torch
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
from tqdm import tqdm

from lcwm.action_conditioning import (
    ActionContextEncoder,
    ActionTimeConditioner,
    Dinov3SemanticEncoder,
)
from lcwm.config import ACTION_DIMS, DOMAINS, joint_text
from lcwm.datasets.operators import LetterboxResize
from lcwm.datasets.protocol import (
    ACTION_CONTEXT_TOKENS,
    ACTION_EMBED_DIM,
    ACTION_TIME_HIDDEN_DIM,
    ACTION_TIME_SCALE,
    IMAGE_SIZE,
    LORA_RANK,
    SEMANTIC_CONTEXT_SCALE,
    SEMANTIC_CONTEXT_TOKENS,
    SEMANTIC_FRAME_INDEX,
    SEMANTIC_LATENT_DIM,
    WEIGHT_DECAY,
    legal_frame_count,
    read_records,
    select_training_items,
)
from lcwm.datasets.video import frame_indices_with_wan_padding
from lcwm.wan22.diffsynth.core.loader.config import ModelConfig
from lcwm.wan22.diffsynth.pipelines.wan_video import WanVideoPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
LORA_TARGETS = ("q", "k", "v", "o", "ffn.0", "ffn.2")


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.name


class VariableLengthVideoLoader:
    def __init__(self, base_path: str, height: int, width: int):
        self.base_path = Path(base_path)
        self.resize = LetterboxResize(height, width, 16, 16)

    def __call__(self, row: dict[str, Any]) -> list[Image.Image]:
        path = self.base_path / str(row["video"])
        num_frames = int(row["wan_num_frames"])
        if path.is_dir():
            frame_paths = sorted(
                (
                    item
                    for item in path.iterdir()
                    if item.suffix.lower() in IMAGE_SUFFIXES
                ),
                key=_frame_sort_key,
            )
            if not frame_paths:
                raise FileNotFoundError(f"no image frames found in {path}")
            frames = []
            for index in frame_indices_with_wan_padding(len(frame_paths), num_frames):
                frame_path = frame_paths[index]
                with Image.open(frame_path) as image:
                    frames.append(self.resize(image.convert("RGB")))
            return frames
        if path.suffix.lower() in IMAGE_SUFFIXES:
            if num_frames != 1:
                raise ValueError(
                    f"single image cannot provide {num_frames} video frames: {path}"
                )
            with Image.open(path) as image:
                return [self.resize(image.convert("RGB"))]
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"unsupported video file: {path}")

        reader = imageio.get_reader(path)
        try:
            available = int(reader.count_frames())
            if available < 1:
                raise ValueError(f"video contains no frames: {path}")
            return [
                self.resize(Image.fromarray(reader.get_data(index)).convert("RGB"))
                for index in frame_indices_with_wan_padding(available, num_frames)
            ]
        finally:
            reader.close()


class ActionArrayLoader:
    def __init__(self, base_path: str, domain: str):
        self.base_path = Path(base_path)
        self.domain = domain

    def __call__(self, row: dict[str, Any]) -> np.ndarray:
        with np.load(self.base_path / str(row["array"]), allow_pickle=False) as arrays:
            action = arrays["action"].astype(np.float32)
            if self.domain != "gaming":
                return action
            fields = (
                [str(value) for value in arrays["action_fields"].tolist()]
                if "action_fields" in arrays
                else []
            )
            expected_fields = [
                "delta_forward",
                "delta_lateral",
                "delta_vertical",
                "delta_yaw",
            ]
            if fields and fields != expected_fields:
                raise ValueError(f"unsupported Gaming action fields: {fields}")
            return action


class ActionConditionedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        domain: str,
        semantic_latent_root: str,
        data_ratio: float,
    ):
        self.base_path = Path(base_path)
        self.domain = domain
        rows = read_records(metadata_path)
        self.rows = select_training_items(
            rows, data_ratio, key=lambda row: str(row["trajectory_id"])
        )
        self.video_loader = VariableLengthVideoLoader(base_path, IMAGE_SIZE, IMAGE_SIZE)
        self.action_loader = ActionArrayLoader(base_path, domain)
        self.semantic_root = Path(semantic_latent_root) / self.base_path.name

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        row["video"] = self.video_loader(row)
        row["action"] = self.action_loader(row)
        frame_count = int(row["orig_num_frames"])
        wan_count = int(row["wan_num_frames"])
        expected_action_shape = (frame_count - 1, ACTION_DIMS[self.domain])
        if (
            row["action"].shape != expected_action_shape
            or len(row["video"]) != wan_count
            or wan_count != legal_frame_count(frame_count)
            or not np.isfinite(row["action"]).all()
        ):
            raise ValueError(f"misaligned trajectory: {row.get('trajectory_id', index)}")
        trajectory_id = str(row["trajectory_id"])
        payload = torch.load(
            self.semantic_root / f"{trajectory_id}_semantic.pt",
            map_location="cpu",
            weights_only=True,
        )
        frame = SEMANTIC_FRAME_INDEX
        row["semantic_tokens0"] = torch.cat(
            [payload["cls_tokens"][frame : frame + 1], payload["patch_tokens"][frame]],
            dim=0,
        ).to(torch.float32)
        return row


def _to_device(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.to(device)
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_to_device(item, device, dtype) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device, dtype) for key, item in value.items()}
    return value


class WorldModelTrainingModule(torch.nn.Module):
    def __init__(self, domain: str, model_paths: str, tokenizer_path: str, device: str):
        super().__init__()
        self.domain = domain
        configs = [ModelConfig(path) for path in json.loads(model_paths)]
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=configs,
            tokenizer_config=ModelConfig(tokenizer_path),
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([])
        lora = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK,
            target_modules=list(LORA_TARGETS),
        )
        self.pipe.dit = inject_adapter_in_model(lora, self.pipe.dit)
        for parameter in self.pipe.dit.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(self.pipe.torch_dtype)

        self.action_context_encoder = ActionContextEncoder(
            domain,
            hidden_dim=ACTION_EMBED_DIM,
            out_dim=self.pipe.dit.dim,
            num_tokens=ACTION_CONTEXT_TOKENS,
        )
        self.pipe.dit.action_time_conditioner = ActionTimeConditioner(
            dim=self.pipe.dit.dim,
            hidden_dim=ACTION_TIME_HIDDEN_DIM,
            scale=ACTION_TIME_SCALE,
        ).to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
        self.semantic_context_encoder = Dinov3SemanticEncoder(
            latent_dim=SEMANTIC_LATENT_DIM,
            hidden_dim=ACTION_EMBED_DIM,
            out_dim=self.pipe.dit.dim,
            num_tokens=SEMANTIC_CONTEXT_TOKENS,
            scale=SEMANTIC_CONTEXT_SCALE,
        )

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def export_state_dict(self, state_dict: dict[str, torch.Tensor]):
        trainable = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        output = {}
        prefix = "pipe.dit."
        for name, value in state_dict.items():
            if name not in trainable:
                continue
            output[name[len(prefix) :] if name.startswith(prefix) else name] = value
        return output

    def _pipeline_inputs(self, data: dict[str, Any]):
        video = data["video"]
        shared = {
            "input_video": video,
            "input_image": video[0],
            "height": video[0].height,
            "width": video[0].width,
            "num_frames": len(video),
            "cfg_scale": 1.0,
            "tiled": False,
            "rand_device": self.pipe.device,
            "action_raw": data["action"],
            "semantic_tokens0": data["semantic_tokens0"],
        }
        return shared, {"prompt": joint_text(self.domain, data["instruction"])}, {}

    def forward(self, data: dict[str, Any]) -> torch.Tensor:
        inputs = _to_device(
            self._pipeline_inputs(data), self.pipe.device, self.pipe.torch_dtype
        )
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        shared, positive, _ = inputs
        action_context = self.action_context_encoder(shared["action_raw"])
        representation_context = self.semantic_context_encoder(
            shared["semantic_tokens0"]
        )
        return self._flow_matching_loss(
            latents=shared["input_latents"],
            first_frame=shared.get("first_frame_latents"),
            context=positive["context"],
            action_context=action_context,
            representation_context=representation_context,
            fuse_first_frame=shared.get("fuse_vae_embedding_in_latents", False),
        )

    def _flow_matching_loss(
        self,
        *,
        latents: torch.Tensor,
        first_frame: torch.Tensor | None,
        context: torch.Tensor,
        action_context: torch.Tensor,
        representation_context: torch.Tensor,
        fuse_first_frame: bool,
    ) -> torch.Tensor:
        step = torch.randint(0, len(self.pipe.scheduler.timesteps), (1,))
        timestep = self.pipe.scheduler.timesteps[step].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        noise = torch.randn_like(latents)
        noisy = self.pipe.scheduler.add_noise(latents, noise, timestep)
        target = self.pipe.scheduler.training_target(latents, noise, timestep)
        if first_frame is not None:
            noisy[:, :, :1] = first_frame
        prediction = self.pipe.model_fn(
            dit=self.pipe.dit,
            latents=noisy,
            timestep=timestep,
            context=context,
            action_context=action_context,
            representation_context=representation_context,
            fuse_vae_embedding_in_latents=fuse_first_frame,
            use_gradient_checkpointing=True,
        )
        if first_frame is not None:
            prediction = prediction[:, :, 1:]
            target = target[:, :, 1:]
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        return loss * self.pipe.scheduler.training_weight(timestep)


def _save_epoch(
    accelerator: accelerate.Accelerator,
    model: WorldModelTrainingModule,
    output_path: Path,
    epoch: int,
) -> None:
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        state_dict = unwrapped.export_state_dict(state_dict)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint = output_path / f"epoch-{epoch}.safetensors"
        accelerator.save(state_dict, checkpoint, safe_serialization=True)
        shutil.copyfile(checkpoint, output_path / "model.safetensors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Tiresias world model.")
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--dataset-base-path", required=True)
    parser.add_argument("--dataset-metadata-path", required=True)
    parser.add_argument("--semantic-latent-root", required=True)
    parser.add_argument("--model-paths", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dataset-num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--data_ratio", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be at least 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if not 0 < args.data_ratio <= 1:
        raise ValueError("--data_ratio must be in (0, 1]")
    if args.dataset_num_workers < 0:
        raise ValueError("--dataset-num-workers must be non-negative")
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)
        ]
    )
    dataset = ActionConditionedDataset(
        args.dataset_base_path,
        args.dataset_metadata_path,
        args.domain,
        args.semantic_latent_root,
        args.data_ratio,
    )
    model = WorldModelTrainingModule(
        args.domain,
        args.model_paths,
        args.tokenizer_path,
        str(accelerator.device),
    )
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        collate_fn=lambda batch: batch[0],
        num_workers=args.dataset_num_workers,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    for epoch in range(args.num_epochs):
        progress = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for data in progress:
            optimizer.zero_grad()
            loss = model(data)
            accelerator.backward(loss)
            optimizer.step()
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")
        _save_epoch(accelerator, model, Path(args.output_path), epoch)


if __name__ == "__main__":
    main()
