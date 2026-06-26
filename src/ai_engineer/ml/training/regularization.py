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

"""Regularization techniques: mixup, cutmix, dropout, label smoothing (loss side), etc."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    h, w = x.shape[-2], x.shape[-1]
    cut_rat = (1.0 - lam) ** 0.5
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cy - cut_h // 2, 0, h)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, w), np.clip(cy + cut_h // 2, 0, h)
    x_new = x.clone()
    x_new[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    return x_new, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep
