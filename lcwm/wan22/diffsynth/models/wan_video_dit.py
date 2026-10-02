import math
from typing import Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from ..core.gradient.gradient_checkpoint import gradient_checkpoint_forward


try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        output = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(output, tuple):
            output = output[0]
        return rearrange(output, "b s n d -> b s (n d)", n=num_heads)
    if FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        output = flash_attn.flash_attn_func(q, k, v)
        return rearrange(output, "b s n d -> b s (n d)", n=num_heads)
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    output = sageattn(q, k, v) if SAGE_ATTN_AVAILABLE else F.scaled_dot_product_attention(q, k, v)
    return rearrange(output, "b n s d -> b s (n d)", n=num_heads)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(
        position.to(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device)
            / (dim // 2),
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(position.dtype)


def precompute_freqs_cis(
    dim: int, end: int = 1024, theta: float = 10000.0
) -> torch.Tensor:
    frequencies = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim)
    )
    frequencies = torch.outer(torch.arange(end, device=frequencies.device), frequencies)
    return torch.polar(torch.ones_like(frequencies), frequencies)


def precompute_freqs_cis_3d(
    dim: int, end: int = 1024, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        precompute_freqs_cis(dim - 2 * (dim // 3), end, theta),
        precompute_freqs_cis(dim // 3, end, theta),
        precompute_freqs_cis(dim // 3, end, theta),
    )


def rope_apply(x: torch.Tensor, frequencies: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    complex_x = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    output = torch.view_as_real(complex_x * frequencies).flatten(2)
    return output.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return flash_attention(q, k, v, self.num_heads)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(num_heads)

    def forward(self, x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        q = rope_apply(self.norm_q(self.q(x)), frequencies, self.num_heads)
        k = rope_apply(self.norm_k(self.k(x)), frequencies, self.num_heads)
        return self.o(self.attn(q, k, self.v(x)))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(num_heads)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        return self.o(self.attn(q, k, self.v(context)))


class GateModule(nn.Module):
    def forward(
        self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6
    ):
        super().__init__()
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        has_sequence_timestep = timestep.ndim == 4
        chunk_dim = 2 if has_sequence_timestep else 1
        values = (self.modulation.to(timestep) + timestep).chunk(6, dim=chunk_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = values
        if has_sequence_timestep:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                value.squeeze(2) for value in values
            )
        residual = self.self_attn(
            modulate(self.norm1(x), shift_msa, scale_msa), frequencies
        )
        x = self.gate(x, gate_msa, residual)
        x = x + self.cross_attn(self.norm3(x), context)
        residual = self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return self.gate(x, gate_mlp, residual)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(timestep) + timestep.unsqueeze(2)
            ).chunk(2, dim=2)
            return self.head(
                self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)
            )
        shift, scale = (self.modulation.to(timestep) + timestep).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class WanModel(nn.Module):
    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        separated_timestep: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.separated_timestep = separated_timestep
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(dim // num_heads)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)

    def unpatchify(
        self, x: torch.Tensor, grid_size: tuple[int, int, int]
    ) -> torch.Tensor:
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        embedded_time = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype)
        )
        time_modulation = self.time_projection(embedded_time).unflatten(
            1, (6, self.dim)
        )
        context = self.text_embedding(context)
        x = self.patchify(x)
        frames, height, width = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        frequencies = torch.cat(
            [
                self.freqs[0][:frames]
                .view(frames, 1, 1, -1)
                .expand(frames, height, width, -1),
                self.freqs[1][:height]
                .view(1, height, 1, -1)
                .expand(frames, height, width, -1),
                self.freqs[2][:width]
                .view(1, 1, width, -1)
                .expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(frames * height * width, 1, -1).to(x.device)
        for block in self.blocks:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                x,
                context,
                time_modulation,
                frequencies,
            )
        x = self.head(x, embedded_time)
        return self.unpatchify(x, (frames, height, width))
