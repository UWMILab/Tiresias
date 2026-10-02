from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    path: str | list[str]
