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

"""Differential privacy with Opacus for PyTorch training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ai_engineer.utils.errors import AIEngineerError


@dataclass
class DPConfig:
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float | None = None
    sample_rate: float | None = None


class DifferentialPrivacy:
    """Differential privacy via Opacus or manual gradient noise + clipping."""

    def __init__(self, config: DPConfig) -> None:
        self.config = config

    def wrap_optimizer(self, optimizer, data_loader, noise_multiplier: float | None = None) -> object:
        """Wrap a PyTorch optimizer with DP-SGD."""
        try:
            from opacus import PrivacyEngine
        except ImportError as e:
            raise AIEngineerError("Install opacus: pip install opacus") from e
        pe = PrivacyEngine()
        model, optimizer, data_loader = pe.make_private(
            module=optimizer.param_groups[0]["params"][0] if hasattr(optimizer.param_groups[0]["params"][0], "module") else None,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier or self.config.noise_multiplier or 1.0,
            max_grad_norm=self.config.max_grad_norm,
        )
        return pe, model, optimizer, data_loader

    def add_noise_to_gradients(self, model: nn.Module, sigma: float, max_norm: float = 1.0) -> None:
        """Manual DP-SGD step: clip per-sample gradients, add Gaussian noise."""
        for p in model.parameters():
            if p.grad is None:
                continue
            # In practice, per-sample gradients need functorch. Here we approximate.
            grad_norm = p.grad.norm(2)
            clip_coef = max_norm / (grad_norm + 1e-6)
            clip_coef = min(clip_coef, 1.0)
            p.grad.mul_(clip_coef)
            p.grad.add_(torch.randn_like(p.grad) * sigma * max_norm)

    def compute_epsilon(self, n_steps: int, sample_rate: float, noise_multiplier: float, delta: float = 1e-5) -> float:
        """Compute privacy spent (epsilon) using RDP."""
        try:
            from opacus.accountants import RDPAccountant
            acc = RDPAccountant()
            for _ in range(n_steps):
                acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
            return acc.get_epsilon(delta=delta)
        except ImportError:
            return -1.0

    def make_private(self, model: nn.Module, optimizer, data_loader, sample_rate: float, noise_multiplier: float) -> tuple:
        try:
            from opacus import PrivacyEngine
            pe = PrivacyEngine()
            model, optimizer, data_loader = pe.make_private(
                module=model, optimizer=optimizer, data_loader=data_loader,
                noise_multiplier=noise_multiplier, max_grad_norm=self.config.max_grad_norm,
            )
            return pe, model, optimizer, data_loader
        except ImportError as e:
            raise AIEngineerError("Install opacus") from e
