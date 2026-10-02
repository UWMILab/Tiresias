import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

CHECKPOINT_FORMAT = "tiresias_tr_ranker"
METRICS = ("SR", "nDTW", "ATE", "RPE")
MODEL_SIZE = {
    "navigation": (512, 4),
    "driving": (256, 2),
    "gaming": (256, 2),
    "robot": (256, 2),
}
DROPOUT = 0.10


def load_action_adapter(path: Path | None, domain: str) -> dict[str, Any] | None:
    if path is None:
        if domain in {"gaming", "robot"}:
            raise ValueError(f"{domain} requires --action-config")
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if domain == "robot" and not value.get("translation_scale_by_source"):
        raise ValueError("Robot action config is missing its translation scales")
    if domain == "gaming" and "statistics" not in value:
        raise ValueError("Gaming action config is missing source statistics")
    return value


def apply_action_adapter(
    actions: Tensor,
    rows: list[dict[str, Any]],
    domain: str,
    adapter: dict[str, Any] | None,
) -> Tensor:
    result = actions.clone()
    if domain == "robot":
        if adapter is None:
            raise ValueError("robot requires an action adapter")
        scales = adapter["translation_scale_by_source"]
        for index, row in enumerate(rows):
            source = str(row["source_name"])
            scale = float(scales.get(source, adapter.get("default_translation_scale", 0)))
            if scale <= 0:
                raise ValueError(f"Robot action config has no scale for {source!r}")
            result[index, ..., :3] /= scale
    elif domain == "gaming":
        if adapter is None:
            raise ValueError("gaming requires an action adapter")
        statistics = adapter["statistics"]
        for index, row in enumerate(rows):
            source = str(row["source_name"])
            spec = statistics[source]
            mean = torch.as_tensor(
                spec["mean"], dtype=result.dtype, device=result.device
            )
            std = torch.as_tensor(spec["std"], dtype=result.dtype, device=result.device)
            result[index] = (result[index] - mean) / std.clamp_min(1e-6)
    return result


class TrajectoryRanker(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_mean: Tensor,
        action_std: Tensor,
        horizon: int = 16,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = DROPOUT,
        domain: str = "navigation",
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.domain = domain
        self.feature_dim = int(action_mean.numel())
        self.register_buffer("action_mean", action_mean.reshape(1, 1, 1, -1).float())
        self.register_buffer(
            "action_std", action_std.reshape(1, 1, 1, -1).float().clamp_min(1e-6)
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(4096), nn.Linear(4096, hidden)
        )
        self.action_projection = nn.Linear(self.feature_dim, hidden)
        self.temporal_position = nn.Parameter(torch.randn(horizon, hidden) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden, 8, dropout=dropout, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            hidden,
            8,
            hidden * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(layer, layers)
        self.output_norm = nn.LayerNorm(hidden)
        self.heads = nn.Linear(hidden, 4)

    def forward(
        self,
        text_tokens: Tensor,
        text_mask: Tensor,
        actions: Tensor,
        action_mask: Tensor,
    ) -> dict[str, Tensor]:
        batch, candidates, horizon, _ = actions.shape
        if candidates != 128 or horizon > self.horizon:
            raise ValueError({"candidates": candidates, "horizon": horizon})
        query = self.action_projection((actions - self.action_mean) / self.action_std)
        query = (query + self.temporal_position[:horizon]).reshape(
            batch * candidates, horizon, self.hidden
        )
        memory = self.text_projection(text_tokens)
        memory = memory[:, None].expand(-1, candidates, -1, -1)
        memory = memory.reshape(batch * candidates, memory.shape[2], self.hidden)
        padding = (~text_mask.bool())[:, None].expand(-1, candidates, -1)
        padding = padding.reshape(batch * candidates, text_mask.shape[1])
        attended, _ = self.cross_attention(
            query, memory, memory, key_padding_mask=padding, need_weights=False
        )
        valid = action_mask.reshape(batch * candidates, horizon).bool()
        encoded = self.temporal_transformer(
            query + attended, src_key_padding_mask=~valid
        )
        pooled = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(
            1, keepdim=True
        ).clamp_min(1)
        raw = self.heads(self.output_norm(pooled)).reshape(batch, candidates, 4)
        return {
            "sr_logit": raw[..., 0],
            "ndtw_z": raw[..., 1],
            "ate_z": raw[..., 2],
            "rpe_z": raw[..., 3],
        }


def trajectory_score(
    outputs: dict[str, Tensor], statistics: dict[str, dict[str, float]]
) -> Tensor:
    sr = torch.sigmoid(outputs["sr_logit"])
    sr_z = (sr - statistics["SR"]["mean"]) / max(statistics["SR"]["std"], 1e-6)
    return sr_z + outputs["ndtw_z"] - outputs["ate_z"] - outputs["rpe_z"]


def make_text_batch(
    rows: list[dict[str, Any]],
    cache: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    tokens: list[Tensor] = []
    for row in rows:
        text = str(row["instruction"])
        if text not in cache["token_embeddings"]:
            raise KeyError(f"instruction is missing from the language cache: {text}")
        value = torch.as_tensor(cache["token_embeddings"][text]).float()
        mask = (
            torch.as_tensor(cache["token_masks"][text]).bool()
            if "token_masks" in cache
            else torch.ones(len(value), dtype=torch.bool)
        )
        tokens.append(value[mask])
    length = max(map(len, tokens))
    token_batch = torch.zeros(len(tokens), length, tokens[0].shape[-1])
    mask_batch = torch.zeros(len(tokens), length, dtype=torch.bool)
    for index, value in enumerate(tokens):
        token_batch[index, : len(value)] = value
        mask_batch[index, : len(value)] = True
    return token_batch.to(device), mask_batch.to(device)


def make_action_batch(
    rows: list[dict[str, Any]],
    device: torch.device,
    domain: str,
    adapter: dict[str, Any] | None,
) -> tuple[Tensor, Tensor]:
    actions = torch.stack(
        [torch.as_tensor(row["candidate_actions"]).float() for row in rows]
    )
    masks = torch.stack(
        [torch.as_tensor(row["candidate_masks"]).bool() for row in rows]
    )
    return apply_action_adapter(actions, rows, domain, adapter).to(device), masks.to(
        device
    )
