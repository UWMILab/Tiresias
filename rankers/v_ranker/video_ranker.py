from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

CHECKPOINT_FORMAT = "tiresias_v_ranker"
DOMAINS = ("navigation", "driving", "robot", "gaming")


@dataclass(frozen=True)
class Config:
    hidden_dim: int = 128
    dropout: float = 0.10
    dinov3_dim: int = 1024
    semantic_hidden: int = 256
    semantic_tokens: int = 8


class VRanker(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        hidden = cfg.hidden_dim
        self.semantic_position = nn.Parameter(
            torch.zeros(1, cfg.semantic_tokens, cfg.semantic_hidden)
        )
        self.semantic_input = nn.Linear(cfg.dinov3_dim, cfg.semantic_hidden)
        self.semantic_rest = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(cfg.semantic_hidden),
            nn.Linear(cfg.semantic_hidden, 3072),
        )
        self.wan = nn.Sequential(nn.LayerNorm(144), nn.Linear(144, hidden), nn.SiLU())
        self.action = nn.Sequential(
            nn.LayerNorm(3072), nn.Linear(3072, hidden), nn.SiLU()
        )
        self.semantic = nn.Sequential(
            nn.LayerNorm(3072), nn.Linear(3072, hidden), nn.SiLU()
        )
        self.language = nn.Sequential(
            nn.LayerNorm(4096), nn.Linear(4096, hidden), nn.SiLU()
        )
        self.domain_embedding = nn.Embedding(len(DOMAINS), hidden)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 8),
            nn.Linear(hidden * 8, hidden * 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
        )
        self.total_head = nn.Linear(hidden, 1)
        self.language_head = nn.Linear(hidden, 1)
        self.semantic_head = nn.Linear(hidden, 1)
        self.motion_head = nn.Linear(hidden, 1)

    def forward(
        self, domain: str, wan: Tensor, action: Tensor, dino: Tensor, text: Tensor
    ) -> dict[str, Tensor]:
        batch_size = wan.shape[0]
        expected = (
            (wan, (batch_size, 144)),
            (action, (batch_size, 3072)),
            (dino, (batch_size, 8, 1024)),
            (text, (batch_size, 4096)),
        )
        if any(
            tuple(value.shape) != shape or not torch.isfinite(value).all()
            for value, shape in expected
        ):
            raise ValueError("invalid world-model evidence")
        semantic_tokens = self.semantic_rest(
            self.semantic_input(dino) + self.semantic_position
        )
        wan_features = self.wan(wan)
        action_features = self.action(action)
        semantic_features = self.semantic(semantic_tokens.mean(dim=1))
        language_features = self.language(text)
        domain_id = torch.full(
            (batch_size,), DOMAINS.index(domain), dtype=torch.long, device=wan.device
        )
        domain_features = self.domain_embedding(domain_id)
        features = self.fusion(
            torch.cat(
                (
                    wan_features,
                    action_features,
                    semantic_features,
                    language_features,
                    wan_features * action_features,
                    wan_features * semantic_features,
                    wan_features * language_features,
                    domain_features,
                ),
                dim=-1,
            )
        )
        total = self.total_head(features).squeeze(-1)
        language = self.language_head(features).squeeze(-1)
        semantic = self.semantic_head(features).squeeze(-1)
        motion = self.motion_head(features).squeeze(-1)
        return {
            "score": total + 0.5 * language + 0.5 * semantic + 0.25 * motion,
            "language": language,
            "semantic": semantic,
            "motion": motion,
        }


def load_evidence(
    paths: list[Path], split: str
) -> tuple[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, Any]] = {}
    domain = ""
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            int(payload.get("candidate_count_per_sample", -1)) != 8
            or payload.get("split") != split
        ):
            raise ValueError(f"invalid evidence file: {path}")
        domain = domain or str(payload["domain"])
        if str(payload["domain"]) != domain:
            raise ValueError("evidence files contain multiple domains")
        for row in payload["records"]:
            key = str(row["sample_key"])
            if key in groups or len(row["candidates"]) != 8:
                raise ValueError("duplicate or incomplete evidence group")
            groups[key] = row
    if not groups:
        raise ValueError("empty evidence")
    return domain, groups


def batch(
    group: dict[str, Any], device: str
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[float]]:
    candidates = group["candidates"]
    wan = torch.stack([item["wan_moments"] for item in candidates]).float().to(device)
    action = (
        torch.stack([item["action_mean"] for item in candidates]).float().to(device)
    )
    text = torch.stack([item["text_mean"] for item in candidates]).float().to(device)
    dino = group["dino_pooled_tokens"].float().unsqueeze(0).expand(8, -1, -1).to(device)
    language = [float(item["language_score"]) for item in candidates]
    return wan, action, dino, text, language
