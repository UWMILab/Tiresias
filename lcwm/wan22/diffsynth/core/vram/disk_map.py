from collections.abc import Iterator, Mapping

import torch
from safetensors import safe_open


class _TorchFile:
    def __init__(self, path: str, device: str | torch.device):
        self.state = torch.load(path, weights_only=True, map_location=device)

    def keys(self):
        return self.state.keys()

    def tensor(self, name: str) -> torch.Tensor:
        return self.state[name]


class _SafeFile:
    def __init__(self, path: str, device: str | torch.device):
        self.state = safe_open(path, framework="pt", device=str(device))

    def keys(self):
        return self.state.keys()

    def tensor(self, name: str) -> torch.Tensor:
        return self.state.get_tensor(name)


class DiskMap(Mapping[str, torch.Tensor]):
    def __init__(
        self,
        path: str | list[str],
        device: str | torch.device,
        torch_dtype: torch.dtype | None = None,
    ):
        paths = path if isinstance(path, list) else [path]
        self.files = [
            _SafeFile(item, device) if item.endswith(".safetensors") else _TorchFile(item, device)
            for item in paths
        ]
        self.locations = {
            name: file_index
            for file_index, source in enumerate(self.files)
            for name in source.keys()
        }
        self.torch_dtype = torch_dtype

    def __getitem__(self, name: str) -> torch.Tensor:
        tensor = self.files[self.locations[name]].tensor(name)
        return tensor.to(self.torch_dtype) if self.torch_dtype is not None else tensor

    def __iter__(self) -> Iterator[str]:
        return iter(self.locations)

    def __len__(self) -> int:
        return len(self.locations)
