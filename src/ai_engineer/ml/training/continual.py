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

"""Continual learning: EWC, replay, LwF."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC:
    """Elastic Weight Consolidation."""

    def __init__(self, model: nn.Module, dataset: torch.utils.data.DataLoader, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataset)
        self._opt_params = {}

    def _compute_fisher(self, dl: torch.utils.data.DataLoader) -> dict[str, torch.Tensor]:
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for batch in dl:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()} if isinstance(batch, dict) else [v.to(self.device) if torch.is_tensor(v) else v for v in batch]
            self.model.zero_grad()
            x = batch[0] if isinstance(batch, (tuple, list)) else batch["x"]
            y = batch[1] if isinstance(batch, (tuple, list)) else batch["y"]
            out = self.model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        return {n: v / max(len(dl), 1) for n, v in fisher.items()}

    def penalty(self, model: nn.Module, lam: float = 1e3) -> torch.Tensor:
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return lam * loss


class ReplayBuffer:
    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self.x: list = []
        self.y: list = []

    def add(self, x, y) -> None:
        if len(self.x) >= self.capacity:
            self.x.pop(0)
            self.y.pop(0)
        self.x.append(x.detach().cpu())
        self.y.append(y.detach().cpu())

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        import random
        idx = random.sample(range(len(self.x)), min(n, len(self.x)))
        return torch.cat([self.x[i] for i in idx]), torch.cat([self.y[i] for i in idx])
