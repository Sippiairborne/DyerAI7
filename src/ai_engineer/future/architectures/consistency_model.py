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

"""Consistency Models (Song et al. 2023) — single-step generation via distillation from diffusion."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConsistencyConfig:
    img_size: int = 32
    in_channels: int = 4
    d_model: int = 512
    depth: int = 12
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    n_steps: int = 18


class ConsistencyModel(nn.Module):
    """Predicts the denoised x_0 from any noisy x_t along the PF ODE."""

    def __init__(self, cfg: ConsistencyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Simplified U-Net-like body (real impl uses full UNet)
        self.body = nn.Sequential(
            nn.Conv2d(cfg.in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, cfg.d_model),
        )
        self.time_embed = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, cfg.d_model))
        self.head = nn.Linear(cfg.d_model, cfg.in_channels * cfg.img_size * cfg.img_size)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        s = sigma.view(b, 1)
        f = self.body(x)
        t = self.time_embed(s)
        out = self.head(f + t)
        return out.view(b, self.cfg.in_channels, self.cfg.img_size, self.cfg.img_size)

    def loss(self, x: torch.Tensor, teacher_fn) -> torch.Tensor:
        b = x.size(0)
        # Sample sigma
        sigmas = torch.rand(b, device=x.device) * (self.cfg.sigma_max - self.cfg.sigma_min) + self.cfg.sigma_min
        noise = torch.randn_like(x)
        x_noisy = x + sigmas.view(b, 1, 1, 1) * noise
        x_pred = self.forward(x_noisy, sigmas)
        # Teacher at next sigma
        sigmas_next = (sigmas / 1.1).clamp(min=self.cfg.sigma_min)
        x_next = x + sigmas_next.view(b, 1, 1, 1) * noise
        with torch.no_grad():
            x_target = teacher_fn(x_next, sigmas_next)
        return F.mse_loss(x_pred, x_target)

    @torch.no_grad()
    def sample(self, batch_size: int = 1, device: str = "cuda") -> torch.Tensor:
        x = torch.randn(batch_size, self.cfg.in_channels, self.cfg.img_size, self.cfg.img_size, device=device)
        for _ in range(self.cfg.n_steps):
            sigma = torch.full((batch_size,), self.cfg.sigma_min, device=device)
            x = x + self.cfg.sigma_min * self.forward(x, sigma)
        return x
