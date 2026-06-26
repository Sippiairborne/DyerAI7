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

"""Multi-task training with shared backbone and task-specific heads."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class MultiTaskModel(nn.Module):
    def __init__(self, backbone: nn.Module, task_configs: dict[str, int]) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict({task: nn.Linear(self._feat_dim(), n) for task, n in task_configs.items()})
        self.task_configs = task_configs

    def _feat_dim(self) -> int:
        for p in self.backbone.parameters():
            return p.shape[-1] if p.ndim >= 2 else 1
        return 1

    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        feats = self.backbone(x)
        return self.heads[task](feats)


def uncertainty_weighting(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    """Homogeneous uncertainty weighting (Kendall et al. 2018)."""
    total = 0.0
    for name, loss in losses.items():
        s = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        total = total + 0.5 * torch.exp(-s) * loss + s
    return total
