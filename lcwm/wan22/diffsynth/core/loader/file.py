import torch
from safetensors import safe_open


def load_state_dict(
    path: str | list[str],
    torch_dtype: torch.dtype | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    if isinstance(path, list):
        state = {}
        for item in path:
            state.update(load_state_dict(item, torch_dtype, device))
        return state
    if path.endswith(".safetensors"):
        state = {}
        with safe_open(path, framework="pt", device=str(device)) as source:
            for name in source.keys():
                tensor = source.get_tensor(name)
                state[name] = tensor.to(torch_dtype) if torch_dtype is not None else tensor
        return state
    payload = torch.load(path, map_location=device, weights_only=True)
    for key in ("state_dict", "module", "model_state"):
        if isinstance(payload, dict) and len(payload) == 1 and key in payload:
            payload = payload[key]
            break
    if torch_dtype is not None:
        payload = {
            name: value.to(torch_dtype) if isinstance(value, torch.Tensor) else value
            for name, value in payload.items()
        }
    return payload
