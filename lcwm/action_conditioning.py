import torch

from lcwm.config import ACTION_DIMS

ACTION_FEATURE_DIMS = {
    "navigation": 5,
    "driving": 6,
    "gaming": 6,
    "robot": 22,
}


def _as_action_batch(action: torch.Tensor, domain: str) -> torch.Tensor:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    expected = ACTION_DIMS[domain]
    if action.ndim != 3 or action.shape[-1] != expected:
        raise ValueError(
            f"{domain} action must have shape [T,{expected}] or [B,T,{expected}]"
        )
    return action


def _phase_slices(length: int, count: int, device: torch.device):
    boundaries = torch.linspace(0, length, count + 1, device=device).round().long()
    for index in range(count):
        start = int(boundaries[index].item())
        end = int(boundaries[index + 1].item())
        if end <= start:
            end = min(start + 1, length)
            start = max(0, end - 1)
        yield index, start, end


def _integrate_planar(
    action: torch.Tensor, *, yaw_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dx_local = action[..., 0]
    dy_local = action[..., 1]
    delta_yaw = action[..., yaw_index]
    yaw_before = torch.cumsum(delta_yaw, dim=1) - delta_yaw
    cos_yaw = torch.cos(yaw_before)
    sin_yaw = torch.sin(yaw_before)
    dx_world = cos_yaw * dx_local - sin_yaw * dy_local
    dy_world = sin_yaw * dx_local + cos_yaw * dy_local
    x = torch.cumsum(dx_world, dim=1)
    y = torch.cumsum(dy_world, dim=1)
    yaw = torch.cumsum(delta_yaw, dim=1)
    speed = torch.sqrt(torch.clamp(dx_local * dx_local + dy_local * dy_local, min=0.0))
    return x, y, yaw, speed


def _planar_features(
    action: torch.Tensor,
    num_tokens: int,
    *,
    yaw_index: int,
    extra: torch.Tensor | None = None,
    sum_extra: bool = False,
) -> torch.Tensor:
    x, y, yaw, speed = _integrate_planar(action, yaw_index=yaw_index)
    delta_yaw = action[..., yaw_index]
    features = []
    for index, start, end in _phase_slices(action.shape[1], num_tokens, action.device):
        segment_yaw = delta_yaw[:, start:end]
        signed_turn = torch.tanh(segment_yaw.sum(dim=1) * 6.0)
        speed_mean = speed[:, start:end].mean(dim=1)
        phase = torch.full_like(
            speed_mean, float(index + 1) / float(max(num_tokens, 1))
        )
        phase_signed = signed_turn * phase
        values = [
            x[:, start:end][:, -1] - x[:, start:end][:, 0],
            y[:, start:end][:, -1] - y[:, start:end][:, 0],
            segment_yaw.mean(dim=1),
            yaw[:, start:end][:, -1],
            signed_turn + 0.25 * phase_signed,
        ]
        if extra is not None:
            segment_extra = extra[:, start:end]
            values.append(
                segment_extra.sum(dim=1) if sum_extra else segment_extra.mean(dim=1)
            )
        features.append(torch.stack(values, dim=-1))
    return torch.stack(features, dim=1)


def summarize_action(
    domain: str, action: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    action = _as_action_batch(action, domain)
    if domain == "navigation":
        return _planar_features(action, num_tokens, yaw_index=2)
    if domain == "driving":
        return _planar_features(
            action[..., :3], num_tokens, yaw_index=2, extra=action[..., 3]
        )
    if domain == "gaming":
        return _planar_features(
            action,
            num_tokens,
            yaw_index=3,
            extra=action[..., 2],
            sum_extra=True,
        )

    pose = action[..., :6]
    gripper = action[..., 6]
    features = []
    for index, start, end in _phase_slices(action.shape[1], num_tokens, action.device):
        segment_pose = pose[:, start:end]
        segment_gripper = gripper[:, start:end]
        gripper_mean = segment_gripper.mean(dim=1, keepdim=True)
        phase = torch.full_like(
            gripper_mean, float(index + 1) / float(max(num_tokens, 1))
        )
        features.append(
            torch.cat(
                [
                    segment_pose.mean(dim=1),
                    segment_pose.sum(dim=1),
                    segment_pose.abs().mean(dim=1),
                    gripper_mean,
                    segment_gripper[:, -1:].contiguous(),
                    (segment_gripper[:, -1] - segment_gripper[:, 0]).unsqueeze(-1),
                    phase,
                ],
                dim=-1,
            )
        )
    return torch.stack(features, dim=1)


class ActionContextEncoder(torch.nn.Module):
    def __init__(
        self,
        domain: str,
        hidden_dim: int = 256,
        out_dim: int = 3072,
        num_tokens: int = 8,
        scale: float = 1.0,
    ):
        super().__init__()
        if domain not in ACTION_DIMS:
            raise ValueError(f"unsupported domain: {domain}")
        self.domain = domain
        self.num_tokens = int(num_tokens)
        self.scale = float(scale)
        self.pos = torch.nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(ACTION_FEATURE_DIMS[domain], hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(action, device=self.pos.device)
        features = summarize_action(self.domain, action.float(), self.num_tokens)
        dtype = self.net[0].weight.dtype
        features = features.to(device=self.pos.device, dtype=dtype)
        hidden = self.net[0](features) + self.pos.to(dtype=dtype)
        for layer in self.net[1:]:
            hidden = layer(hidden)
        return hidden * self.scale


class ActionTimeConditioner(torch.nn.Module):
    def __init__(
        self,
        dim: int = 3072,
        hidden_dim: int = 512,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scale = float(scale)
        self.input_proj = torch.nn.Linear(dim, hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, dim)
        torch.nn.init.zeros_(self.output_proj.weight)
        torch.nn.init.zeros_(self.output_proj.bias)

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        if action_tokens.ndim != 3:
            raise ValueError("action_tokens must have shape [batch,tokens,dim]")
        hidden = torch.nn.functional.silu(self.input_proj(action_tokens.mean(dim=1)))
        return self.output_proj(hidden) * self.scale


class Dinov3SemanticEncoder(torch.nn.Module):
    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 256,
        out_dim: int = 3072,
        num_tokens: int = 8,
        scale: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_tokens = int(num_tokens)
        self.scale = float(scale)
        self.pos = torch.nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
        self.separator = torch.nn.Parameter(torch.zeros(1, 1, out_dim))
        torch.nn.init.normal_(self.separator, mean=0.0, std=0.02)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def summarize(self, semantic_tokens: torch.Tensor) -> torch.Tensor:
        if semantic_tokens.ndim == 2:
            semantic_tokens = semantic_tokens.unsqueeze(0)
        if semantic_tokens.ndim != 3:
            raise ValueError("semantic tokens must have shape [N,D] or [B,N,D]")
        if semantic_tokens.shape[-1] != self.latent_dim:
            raise ValueError(
                f"DINOv3 feature dimension must be {self.latent_dim}, "
                f"got {semantic_tokens.shape[-1]}"
            )
        pooled = []
        for _, start, end in _phase_slices(
            semantic_tokens.shape[1], self.num_tokens, semantic_tokens.device
        ):
            pooled.append(semantic_tokens[:, start:end].mean(dim=1))
        return torch.stack(pooled, dim=1)

    def forward(self, semantic_tokens: torch.Tensor) -> torch.Tensor:
        semantic_tokens = torch.as_tensor(
            semantic_tokens, dtype=torch.float32, device=self.pos.device
        )
        features = self.summarize(semantic_tokens)
        dtype = self.net[0].weight.dtype
        features = features.to(dtype=dtype)
        hidden = self.net[0](features) + self.pos.to(dtype=dtype)
        for layer in self.net[1:]:
            hidden = layer(hidden)
        hidden = hidden * self.scale
        separator = self.separator.to(dtype=hidden.dtype).expand(
            hidden.shape[0], -1, -1
        )
        return torch.cat([hidden, separator], dim=1)
