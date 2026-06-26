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

"""Mamba/SSM hybrid (state-space models) — linear-time sequence modeling."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    d_model: int = 768
    n_layers: int = 24
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    vocab_size: int = 50_000
    max_seq_len: int = 2048
    use_mamba2: bool = True


class MambaBlock(nn.Module):
    """Single Mamba block: SSM + gating + MLP."""

    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.in_proj = nn.Linear(d, cfg.expand * d * 2)
        self.conv1d = nn.Conv1d(cfg.expand * d, cfg.expand * d, kernel_size=cfg.d_conv, groups=cfg.expand * d, padding=cfg.d_conv - 1)
        self.x_proj = nn.Linear(cfg.expand * d, cfg.d_state * 2 + cfg.d_state)  # dt, B, C
        self.dt_proj = nn.Linear(cfg.d_state, cfg.expand * d)
        A = torch.arange(1, cfg.d_state + 1, dtype=torch.float32).repeat(cfg.expand * d, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(cfg.expand * d))
        self.out_proj = nn.Linear(cfg.expand * d, d)
        self.d_inner = cfg.expand * d
        self.d_state = cfg.d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = F.silu(x.transpose(1, 2))
        y = self._ssm(x)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        # Selective scan (simplified — uses python loop; production uses CUDA kernel)
        b, l, d = x.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_proj = self.x_proj(x)
        dt, B, C = torch.split(x_proj, [self.d_state, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        h = torch.zeros(b, d, self.d_state, device=x.device)
        ys = []
        for i in range(l):
            dA = torch.exp(dt[:, i, :, None] * A[None, :, :])
            dB = dt[:, i, :, None] * B[:, i, None, :]
            h = dA * h + dB * x[:, i, :, None]
            ys.append((h @ C[:, i, :, None]).squeeze(-1) + D * x[:, i, :])
        return torch.stack(ys, dim=1)


class MambaModel(nn.Module):
    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([MambaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie weights
        self.head.weight = self.embed.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm(x)
        return self.head(x)


class MambaTransformerHybrid(nn.Module):
    """Jamba-style: alternate Mamba + Attention blocks for best of both."""

    def __init__(self, cfg: MambaConfig, mamba_to_attn_ratio: int = 7) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        layer_list = []
        for i in range(cfg.n_layers):
            if i % (mamba_to_attn_ratio + 1) == mamba_to_attn_ratio:
                layer_list.append(nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=12, dim_feedforward=cfg.d_model * 4, batch_first=True, activation=F.silu))
            else:
                layer_list.append(MambaBlock(cfg))
        self.layers = nn.ModuleList(layer_list)
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for layer in self.layers:
            x = x + layer(x) if isinstance(layer, MambaBlock) else layer(x)
        return self.head(self.norm(x))
