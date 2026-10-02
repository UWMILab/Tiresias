import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lcwm.datasets.video import contain_and_pad

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def image_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = contain_and_pad(image.convert("RGB"), (size, size))
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(
        2, 0, 1
    )
    return (tensor - _MEAN) / _STD


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        value: Any = load_file(str(path), device="cpu")
    else:
        value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, Mapping):
        for key in ("state_dict", "model", "teacher"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                value = nested
                break
    if not isinstance(value, Mapping):
        raise TypeError(f"unsupported DINOv3 checkpoint: {path}")
    return {
        str(key).removeprefix("module."): tensor
        for key, tensor in value.items()
        if isinstance(tensor, torch.Tensor)
    }


def load_dinov3(
    repo: Path,
    checkpoint: Path,
    model_name: str,
    device: str,
) -> torch.nn.Module:
    repo = repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    builder = getattr(importlib.import_module("dinov3.hub.backbones"), model_name)
    model = builder(pretrained=False)
    model.load_state_dict(_state_dict(checkpoint.resolve()), strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_tokens(
    image: Image.Image,
    model: torch.nn.Module,
    size: int,
    device: str,
) -> torch.Tensor:
    tensor = image_tensor(image, size).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = model.forward_features(tensor)
    cls = features["x_norm_clstoken"].detach().float()
    patch = features["x_norm_patchtokens"].detach().float()
    return torch.cat([cls.unsqueeze(1), patch], dim=1).squeeze(0)
