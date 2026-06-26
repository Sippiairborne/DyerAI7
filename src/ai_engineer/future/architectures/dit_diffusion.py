# Copyright 2026 Matt Dyer / Dyer-Tech
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusion Transformer (DiT) — replace UNet with ViT in diffusion models."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiTConfig:
    img_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    d_model: int = 1152
    depth: int = 28
    n_heads: int = 16
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    learn_sigma: bool = True


class TimestepEmbedder(nn.Module):
    def __init__(self, d: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d), nn.SiLU(), nn.Linear(d, d))

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10_000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t))


class LabelEmbedder(nn.Module):
    def __init__(self, n_classes: int, d: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_classes + 1, d)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.embed(y)


class DiTBlock(nn.Module):
    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, int(cfg.d_model * cfg.mlp_ratio)), nn.GELU(), nn.Linear(int(cfg.d_model * cfg.mlp_ratio), cfg.d_model))
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(cfg.d_model, 6 * cfg.d_model))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.ada_ln(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa[:, None, :] * attn_out
        h = self.norm2(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.mlp(h)
        return x


class DiT(nn.Module):
    """Diffusion Transformer — class-conditional latent diffusion."""

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(cfg.in_channels, cfg.d_model, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, cfg.d_model))
        self.t_embed = TimestepEmbedder(cfg.d_model)
        self.y_embed = LabelEmbedder(cfg.num_classes, cfg.d_model)
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        out_dim = cfg.in_channels * 2 if cfg.learn_sigma else cfg.in_channels
        self.head = nn.Linear(cfg.d_model, cfg.patch_size * cfg.patch_size * out_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        c = self.t_embed(t) + self.y_embed(y)
        for block in self.blocks:
            x = block(x, c)
        x = self.head(self.norm(x))
        return x


class FlowMatching:
    """Rectified flow matching: simpler, more stable training than diffusion."""

    @staticmethod
    def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Linear interpolation and target velocity."""
        t = t[:, None, None, None] if x0.ndim == 4 else t[:, None]
        x_t = (1 - t) * x0 + t * x1
        v = x1 - x0
        return x_t, v

    @staticmethod
    def step(model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        """Euler step using predicted velocity."""
        return x_t + (t_next - t) * model_output
