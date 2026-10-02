from collections.abc import Callable, Mapping

import torch

from ..vram.disk_map import DiskMap
from ..vram.initialization import skip_model_initialization


def load_model(
    model_class: type[torch.nn.Module],
    path: str | list[str],
    *,
    config: dict | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    state_dict_converter: Callable[[Mapping], Mapping] | None = None,
) -> torch.nn.Module:
    with skip_model_initialization():
        model = model_class(**(config or {}))
    state: Mapping = DiskMap(path, device, torch_dtype=torch_dtype)
    if state_dict_converter is not None:
        state = state_dict_converter(state)
    else:
        state = {name: state[name] for name in state}
    model.load_state_dict(state, assign=True)
    return model.to(dtype=torch_dtype, device=device).eval()
