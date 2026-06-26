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

"""Flow Matching / Continuous Normalizing Flows (Lipman et al. 2023)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ai_engineer.future.architectures.dit_diffusion import DiT, DiTConfig


class FlowMatchingModel:
    """Wrapper around DiT for flow-matching training."""

    def __init__(self, dit: DiT | None = None, cfg: DiTConfig | None = None) -> None:
        self.cfg = cfg or DiTConfig()
        self.model = dit or DiT(self.cfg)

    def loss(self, x1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.size(0), device=x1.device)
        t_b = t[:, None, None, None]
        x_t = (1 - t_b) * x0 + t_b * x1
        v_pred = self.model(x_t, t, y)
        v_target = x1 - x0
        return ((v_pred - v_target) ** 2).mean()

    @torch.no_grad()
    def sample(self, y: torch.Tensor, n_steps: int = 50, device: str = "cuda") -> torch.Tensor:
        x = torch.randn(y.size(0), self.cfg.in_channels, self.cfg.img_size, self.cfg.img_size, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((y.size(0),), i * dt, device=device)
            v = self.model(x, t, y)
            x = x + dt * v
        return x
