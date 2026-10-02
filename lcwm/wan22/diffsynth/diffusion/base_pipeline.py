from collections.abc import Mapping, Sequence

import numpy as np
import torch
from einops import reduce, repeat
from PIL import Image

from ..core.loader.file import load_state_dict
from ..utils.lora.general import GeneralLoRALoader


class PipelineUnit:
    def __init__(
        self,
        shared_inputs: Sequence[str] = (),
        positive_inputs: Mapping[str, str] | None = None,
        negative_inputs: Mapping[str, str] | None = None,
    ):
        self.shared_inputs = tuple(shared_inputs)
        self.positive_inputs = positive_inputs
        self.negative_inputs = negative_inputs

    def process(self, pipe, **kwargs) -> dict:
        raise NotImplementedError


class PipelineUnitRunner:
    def __call__(
        self,
        unit: PipelineUnit,
        pipe: "BasePipeline",
        shared: dict,
        positive: dict,
        negative: dict,
    ) -> tuple[dict, dict, dict]:
        if unit.positive_inputs is None:
            values = {name: shared.get(name) for name in unit.shared_inputs}
            shared.update(unit.process(pipe, **values))
            return shared, positive, negative

        common = {name: shared.get(name) for name in unit.shared_inputs}
        values = {
            name: positive.get(source)
            for name, source in unit.positive_inputs.items()
        }
        positive_output = unit.process(pipe, **common, **values)
        positive.update(positive_output)
        if shared["cfg_scale"] == 1:
            negative.update(positive_output)
        else:
            values = {
                name: negative.get(source)
                for name, source in (unit.negative_inputs or {}).items()
            }
            negative.update(unit.process(pipe, **common, **values))
        return shared, positive, negative


class BasePipeline(torch.nn.Module):
    def __init__(
        self,
        *,
        device: str | torch.device,
        torch_dtype: torch.dtype,
        height_division_factor: int,
        width_division_factor: int,
        time_division_factor: int,
        time_division_remainder: int,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.unit_runner = PipelineUnitRunner()

    def to(self, *args, **kwargs):
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        return super().to(*args, **kwargs)

    def check_resize_height_width(
        self, height: int, width: int, num_frames: int
    ) -> tuple[int, int, int]:
        height = (
            (height + self.height_division_factor - 1)
            // self.height_division_factor
            * self.height_division_factor
        )
        width = (
            (width + self.width_division_factor - 1)
            // self.width_division_factor
            * self.width_division_factor
        )
        if num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames = (
                (num_frames + self.time_division_factor - 1)
                // self.time_division_factor
                * self.time_division_factor
                + self.time_division_remainder
            )
        return height, width, num_frames

    def preprocess_image(
        self,
        image: Image.Image,
        *,
        pattern: str = "B C H W",
        minimum: float = -1,
        maximum: float = 1,
    ) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
        tensor = tensor.to(dtype=self.torch_dtype, device=self.device)
        tensor = tensor * ((maximum - minimum) / 255) + minimum
        return repeat(tensor, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))

    def preprocess_video(self, video: Sequence[Image.Image]) -> torch.Tensor:
        frames = [self.preprocess_image(image) for image in video]
        return torch.stack(frames, dim=2)

    def vae_output_to_video(self, output: torch.Tensor) -> list[Image.Image]:
        frames = reduce(output, "B C T H W -> T H W C", reduction="mean")
        frames = ((frames + 1) * 127.5).clamp(0, 255).to("cpu", torch.uint8)
        return [Image.fromarray(frame.numpy()) for frame in frames]

    def generate_noise(
        self,
        shape: tuple[int, ...],
        *,
        seed: int | None,
        rand_device: str,
    ) -> torch.Tensor:
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device)
        return noise.to(dtype=self.torch_dtype, device=self.device)

    def freeze_except(self, names: Sequence[str]) -> None:
        self.eval()
        self.requires_grad_(False)
        for name in names:
            module = self
            for part in name.split("."):
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            module.train().requires_grad_(True)

    def load_lora(self, module: torch.nn.Module, path: str, alpha: float = 1.0) -> None:
        state = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        loader.fuse_lora_to_base_model(module, state, alpha=alpha)
