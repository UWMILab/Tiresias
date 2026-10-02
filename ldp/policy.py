import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

CHECKPOINT_FORMAT = "tiresias_ldp"

HORIZON = {
    "navigation": 16,
    "driving": 16,
    "gaming": 16,
    "robot": 8,
}

ACTION_FIELDS = {
    "navigation": ("delta_forward", "delta_lateral", "delta_yaw"),
    "driving": (
        "delta_forward",
        "delta_lateral",
        "delta_yaw",
        "target_velocity",
    ),
    "gaming": (
        "delta_forward",
        "delta_lateral",
        "delta_vertical",
        "delta_yaw",
    ),
    "robot": (
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_roll",
        "delta_pitch",
        "delta_yaw",
        "gripper_open",
    ),
}

CANDIDATE_FIELDS = {
    **ACTION_FIELDS,
    "robot": (*ACTION_FIELDS["robot"][:-1], "gripper_closedness"),
}

TEXT_LENGTH = {
    "navigation": 72,
    "driving": 96,
    "gaming": 96,
    "robot": 96,
}


@dataclass(frozen=True)
class PolicyConfig:
    horizon: int = 16
    action_dim: int = 4
    umt5_token_dim: int = 4096
    umt5_max_length: int = 96
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    context_encoder_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.1
    diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    cfg_dropout_prob: float = 0.2
    cfg_guidance_scale: float = 1.5
    action_mean: tuple[float, ...] = ()
    action_std: tuple[float, ...] = ()


def continuous_dim(domain: str) -> int:
    return 6 if domain == "robot" else len(ACTION_FIELDS[domain])


def validate_config(domain: str, config: PolicyConfig) -> None:
    expected_dim = continuous_dim(domain)
    if (config.horizon, config.action_dim) != (HORIZON[domain], expected_dim):
        raise ValueError(f"invalid {domain} horizon or action dimension")
    if (
        len(config.action_mean) != expected_dim
        or len(config.action_std) != expected_dim
    ):
        raise ValueError(f"invalid {domain} action normalization")
    if min(config.action_std) <= 0 or config.hidden_dim % config.num_heads:
        raise ValueError("invalid normalization or attention width")


def policy_config(
    domain: str,
    action_mean: tuple[float, ...],
    action_std: tuple[float, ...],
) -> PolicyConfig:
    config = PolicyConfig(
        horizon=HORIZON[domain],
        action_dim=continuous_dim(domain),
        umt5_max_length=TEXT_LENGTH[domain],
        action_mean=action_mean,
        action_std=action_std,
    )
    validate_config(domain, config)
    return config


def checkpoint_contract(domain: str) -> dict[str, Any]:
    value = {
        "condition_type": "instruction_only",
        "horizon": HORIZON[domain],
        "action_fields": list(ACTION_FIELDS[domain]),
    }
    if domain in {"navigation", "driving"}:
        value["fixed_dt"] = 0.1
    if domain in {"navigation", "gaming"}:
        value["action_dim"] = len(ACTION_FIELDS[domain])
    if domain == "robot":
        value["gripper_semantics"] = "gripper_open=1"
    return value


def sinusoidal_timestep_embedding(timesteps: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(
        -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = torch.nn.functional.pad(
            embedding, (0, dimension - embedding.shape[-1])
        )
    return embedding


class DiffusionPolicy(nn.Module):
    def __init__(self, domain: str, config: PolicyConfig) -> None:
        super().__init__()
        validate_config(domain, config)
        self.domain = domain
        self.config = config
        hidden = config.hidden_dim
        self.instruction_token_proj = nn.Sequential(
            nn.LayerNorm(config.umt5_token_dim),
            nn.Linear(config.umt5_token_dim, hidden),
        )
        self.language_modality_embed = nn.Parameter(torch.zeros(hidden))
        context_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=hidden * config.ff_mult,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            context_layer, num_layers=config.context_encoder_layers
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=hidden * config.ff_mult,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_layers
        )
        self.noisy_action_proj = nn.Linear(config.action_dim, hidden)
        self.step_embed = nn.Embedding(config.horizon, hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.eps_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden // 2, config.action_dim),
        )
        mean = torch.tensor(config.action_mean, dtype=torch.float32)
        std = torch.tensor(config.action_std, dtype=torch.float32).clamp_min(1e-6)
        self.register_buffer("action_mean", mean.view(1, 1, -1), persistent=True)
        self.register_buffer("action_std", std.view(1, 1, -1), persistent=True)
        betas = torch.linspace(
            config.beta_start,
            config.beta_end,
            config.diffusion_steps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer(
            "alpha_bars", torch.cumprod(alphas, dim=0), persistent=True
        )
        if domain == "robot":
            self.gripper_queries = nn.Parameter(
                torch.randn(config.horizon, hidden) * 0.02
            )
            self.gripper_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

    @property
    def contract(self) -> dict[str, Any]:
        return checkpoint_contract(self.domain)

    def _validate_language(
        self, language_tokens: Tensor, language_mask: Tensor
    ) -> tuple[int, int]:
        if (
            language_tokens.ndim != 3
            or language_tokens.shape[-1] != self.config.umt5_token_dim
        ):
            raise ValueError({"language_tokens": list(language_tokens.shape)})
        if language_mask.shape != language_tokens.shape[:2]:
            raise ValueError({"language_mask": list(language_mask.shape)})
        if not torch.isfinite(language_tokens).all():
            raise ValueError("language tokens must be finite")
        valid = language_mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("each row requires one valid language token")
        return language_tokens.shape[0], min(
            language_tokens.shape[1], self.config.umt5_max_length
        )

    def language_memory(
        self,
        language_tokens: Tensor,
        language_mask: Tensor,
        drop_language_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, length = self._validate_language(language_tokens, language_mask)
        dtype = self.instruction_token_proj[1].weight.dtype
        valid = language_mask[:, :length].bool()
        memory = self.instruction_token_proj(language_tokens[:, :length].to(dtype))
        if drop_language_mask is not None:
            drop = drop_language_mask.to(memory.device).bool().reshape(-1)
            if drop.shape[0] != batch:
                raise ValueError({"drop_language_mask": list(drop.shape)})
            memory = torch.where(drop[:, None, None], torch.zeros_like(memory), memory)
            valid = valid & ~drop[:, None]
        memory = memory + self.language_modality_embed.view(1, 1, -1)
        padding = ~valid
        all_masked = padding.all(dim=1)
        if bool(all_masked.any()):
            memory = memory.clone()
            padding = padding.clone()
            memory[all_masked, 0] = 0
            padding[all_masked, 0] = False
        return self.context_encoder(memory, src_key_padding_mask=padding), padding

    def action_queries(self, noisy_actions: Tensor, timesteps: Tensor) -> Tensor:
        self._validate_actions(noisy_actions)
        batch = noisy_actions.shape[0]
        if timesteps.shape != (batch,):
            raise ValueError({"timesteps": list(timesteps.shape), "batch": batch})
        steps = torch.arange(self.config.horizon, device=noisy_actions.device)
        diffusion_time = self.time_mlp(
            sinusoidal_timestep_embedding(timesteps, self.config.hidden_dim)
        ).unsqueeze(1)
        return (
            self.noisy_action_proj(noisy_actions)
            + self.step_embed(steps)[None]
            + diffusion_time
        )

    def denoise_with_memory(
        self,
        noisy_actions: Tensor,
        timesteps: Tensor,
        memory: Tensor,
        memory_padding: Tensor,
        future_mask: Tensor,
    ) -> Tensor:
        query = self.action_queries(noisy_actions, timesteps)
        hidden = self.action_decoder(
            query,
            memory,
            tgt_key_padding_mask=~future_mask.bool(),
            memory_key_padding_mask=memory_padding,
        )
        return self.eps_head(hidden) * future_mask.unsqueeze(-1).to(hidden.dtype)

    def forward(
        self,
        noisy_normalized_actions: Tensor,
        timesteps: Tensor,
        language_tokens: Tensor,
        language_mask: Tensor,
        future_mask: Tensor,
        drop_language_mask: Tensor | None = None,
    ) -> Tensor:
        memory, padding = self.language_memory(
            language_tokens,
            language_mask,
            drop_language_mask,
        )
        return self.denoise_with_memory(
            noisy_normalized_actions,
            timesteps,
            memory,
            padding,
            future_mask,
        )

    def gripper_logits(
        self,
        tokens: Tensor,
        language_mask: Tensor,
        drop_language_mask: Tensor | None = None,
    ) -> Tensor:
        if self.domain != "robot":
            raise RuntimeError("gripper prediction is only available for Robot")
        memory, padding = self.language_memory(
            tokens, language_mask, drop_language_mask
        )
        query = self.gripper_queries.unsqueeze(0).expand(len(tokens), -1, -1)
        hidden = self.action_decoder(query, memory, memory_key_padding_mask=padding)
        return self.gripper_head(hidden).squeeze(-1)

    def _validate_actions(self, actions: Tensor) -> None:
        expected = (self.config.horizon, self.config.action_dim)
        if actions.ndim != 3 or tuple(actions.shape[1:]) != expected:
            raise ValueError(
                {"action_shape": list(actions.shape), "expected": ["B", *expected]}
            )
        if not torch.isfinite(actions).all():
            raise ValueError("actions must be finite")

    def normalize_actions(self, actions: Tensor) -> Tensor:
        self._validate_actions(actions)
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions: Tensor) -> Tensor:
        self._validate_actions(actions)
        return actions * self.action_std + self.action_mean

    def predict_x0_from_epsilon(
        self, noisy_actions: Tensor, timesteps: Tensor, predicted_epsilon: Tensor
    ) -> Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        return (
            noisy_actions
            - torch.sqrt(1.0 - alpha_bar).clamp_min(1e-8) * predicted_epsilon
        ) / torch.sqrt(alpha_bar).clamp_min(1e-8)


def build_policy(
    domain: str,
    action_mean: tuple[float, ...],
    action_std: tuple[float, ...],
) -> DiffusionPolicy:
    return DiffusionPolicy(domain, policy_config(domain, action_mean, action_std))


def save_policy(path: str | Path, domain: str, model: DiffusionPolicy) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "domain": domain,
            "config": asdict(model.config),
            "model": model.state_dict(),
            "contract": model.contract,
        },
        path,
    )


def load_policy(
    path: str | Path,
    domain: str,
    device: str | torch.device = "cpu",
) -> tuple[DiffusionPolicy, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("domain") != domain:
        raise ValueError(f"{path} is not a {domain} Tiresias policy checkpoint")
    if "config" not in payload or "model" not in payload:
        raise ValueError(f"{path} is missing policy state")
    config = PolicyConfig(**payload["config"])
    model = DiffusionPolicy(domain, config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload
