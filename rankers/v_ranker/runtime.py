from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lcwm.config import NEGATIVE_PROMPTS, joint_text
from lcwm.datasets.protocol import DINO_IMAGE_SIZE, DINO_MODEL_NAME, IMAGE_SIZE
from lcwm.datasets.video import contain_and_pad
from lcwm.inference import (
    _load_checkpoint,
    build_action_context_encoder,
    build_pipeline,
    build_semantic_context_encoder,
    encode_conditioning_semantic_tokens,
    install_action_time_conditioner,
)
from lcwm.semantic_backbone import load_dinov3


@contextmanager
def capture_predecode_latent(vae: Any):
    original = vae.decode
    captured: list[torch.Tensor] = []

    def wrapped(latent, *_args, **_kwargs):
        if captured or not isinstance(latent, torch.Tensor):
            raise RuntimeError("VAE.decode must be called exactly once")
        captured.append(latent.detach().float().cpu().contiguous().clone())
        frames = (int(latent.shape[2]) - 1) * 4 + 1
        return torch.zeros((1, 3, frames, 1, 1), dtype=torch.float32)

    vae.decode = wrapped
    try:
        yield captured
    finally:
        vae.decode = original


def wan_sufficient_statistics(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 48:
        raise RuntimeError(
            f"expected Wan latent [1,48,T,H,W], got {tuple(latent.shape)}"
        )
    if not torch.isfinite(latent).all():
        raise RuntimeError("non-finite Wan latent")
    future = latent[:, :, 1:] if latent.shape[2] > 1 else latent
    future = future.float()
    mean = future.mean(dim=(2, 3, 4))
    std = future.std(dim=(2, 3, 4), unbiased=False)
    if future.shape[2] > 1:
        delta = (future[:, :, 1:] - future[:, :, :-1]).abs().mean(dim=(2, 3, 4))
    else:
        delta = torch.zeros_like(mean)
    return torch.cat((mean, std, delta), dim=-1).squeeze(0).contiguous()


class WanFeatureRuntime:
    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self._loaded = False

    def _parse_args(self) -> Any:
        from argparse import Namespace

        return Namespace(
            domain=str(self.config["domain"]),
            wan_root=str(self.config["wan_root"]),
            lora_path=str(self.config["wm_checkpoint_path"]),
            device=str(self.config["device"]),
            torch_dtype=str(self.config["torch_dtype"]),
            guidance_scale=float(self.config.get("guidance_scale", 5.0)),
            num_inference_steps=int(self.config.get("num_inference_steps", 30)),
        )

    def _load(self) -> None:
        if self._loaded:
            return
        args = self._parse_args()
        pipe = build_pipeline(args)
        state = _load_checkpoint(args)
        install_action_time_conditioner(args, pipe, state)
        action_encoder = build_action_context_encoder(args, pipe, state)
        semantic_encoder = build_semantic_context_encoder(args, pipe, state)
        for model in (pipe.dit, action_encoder, semantic_encoder):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        dino = load_dinov3(
            Path(str(self.config["dinov3_repo"])),
            Path(str(self.config["dinov3_checkpoint_path"])),
            DINO_MODEL_NAME,
            str(self.config["device"]),
        )
        self.args, self.pipe = args, pipe
        self.action_encoder, self.semantic_encoder, self.dino = (
            action_encoder,
            semantic_encoder,
            dino,
        )
        self._loaded = True

    def _prompt(self, instruction: str) -> str:
        return joint_text(str(self.config["domain"]), instruction)

    def dino_pooled_tokens(self, observation: Image.Image) -> torch.Tensor:
        self._load()
        image = contain_and_pad(observation.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE))
        with torch.inference_mode():
            tokens = encode_conditioning_semantic_tokens(
                image,
                self.dino,
                image_size=DINO_IMAGE_SIZE,
                device=self.args.device,
            )
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 3 or tokens.shape[-1] != 1024:
            raise RuntimeError(f"unexpected DINOv3 tokens: {tuple(tokens.shape)}")
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            tokens.float().transpose(1, 2), 8
        ).transpose(1, 2)
        return pooled.squeeze(0).cpu().contiguous()

    def _masked_text_mean(self, prompt: str) -> torch.Tensor:
        pipe = self.pipe
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(pipe.device), mask.to(pipe.device)
        with torch.inference_mode():
            token = pipe.text_encoder(ids, mask).float()
        weight = mask.bool().float().unsqueeze(-1)
        if (weight.sum(dim=1) < 1).any():
            raise RuntimeError("empty UMT5 token mask")
        return (
            ((token * weight).sum(dim=1) / weight.sum(dim=1))
            .squeeze(0)
            .cpu()
            .contiguous()
        )

    def rollout_features(
        self,
        *,
        observation: Image.Image,
        instruction: str,
        candidate_actions: Sequence[Sequence[float]],
        seed: int,
    ) -> dict[str, torch.Tensor]:
        self._load()
        actions = np.asarray(candidate_actions, dtype=np.float32)
        if actions.ndim != 2 or len(actions) < 1:
            raise RuntimeError("candidate action must be non-empty [T,C]")
        valid_frames = len(actions) + 1
        wan_frames = ((valid_frames - 1 + 3) // 4) * 4 + 1
        image = contain_and_pad(observation.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE))
        prompt = self._prompt(instruction)
        semantic_tokens0 = encode_conditioning_semantic_tokens(
            image,
            self.dino,
            image_size=DINO_IMAGE_SIZE,
            device=self.args.device,
        )
        with torch.inference_mode():
            semantic_context = self.semantic_encoder(semantic_tokens0)
        action_tensor = torch.as_tensor(
            actions, dtype=torch.float32, device=self.args.device
        )
        with torch.inference_mode():
            action_context = self.action_encoder(action_tensor)
        text_mean = self._masked_text_mean(prompt)
        negative = NEGATIVE_PROMPTS[str(self.config["domain"])]
        call = {
            "prompt": prompt,
            "negative_prompt": negative,
            "input_image": image,
            "action_context": action_context,
            "representation_context": semantic_context,
            "num_frames": wan_frames,
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
            "seed": int(seed),
            "tiled": True,
            "cfg_scale": float(self.args.guidance_scale),
            "num_inference_steps": int(self.args.num_inference_steps),
        }
        with capture_predecode_latent(self.pipe.vae) as captured:
            self.pipe(**call)
        if len(captured) != 1:
            raise RuntimeError("pre-VAE Wan latent was not captured exactly once")
        action_mean = (
            action_context.detach().float().mean(dim=1).squeeze(0).cpu().contiguous()
        )
        if action_mean.shape != (3072,) or text_mean.shape != (4096,):
            raise RuntimeError(
                f"unexpected context dimensions: action={tuple(action_mean.shape)}, text={tuple(text_mean.shape)}"
            )
        return {
            "wan_moments": wan_sufficient_statistics(captured[0]),
            "action_mean": action_mean,
            "text_mean": text_mean,
        }
