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

"""Mixture of Experts (MoE) — sparse activation with top-k routing."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    d_model: int = 1024
    n_experts: int = 8
    top_k: int = 2
    expert_dim: int = 4096
    capacity_factor: float = 1.25
    load_balance_coef: float = 0.01
    use_shared_experts: bool = False
    n_shared: int = 1


class Expert(nn.Module):
    def __init__(self, d: int, ed: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, ed, bias=False)
        self.fc2 = nn.Linear(ed, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MoE(nn.Module):
    """Switch Transformer / Mixtral style MoE."""

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg.d_model, cfg.expert_dim) for _ in range(cfg.n_experts)])
        if cfg.use_shared_experts:
            self.shared = nn.ModuleList([Expert(cfg.d_model, cfg.expert_dim) for _ in range(cfg.n_shared)])
        self.aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        scores = self.gate(flat)
        top = F.softmax(scores, dim=-1).topk(self.cfg.top_k, dim=-1)
        weights, idx = top.values, top.indices
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        out = torch.zeros_like(flat)
        # Load balancing
        importance = F.softmax(scores, dim=-1).sum(0)
        load = torch.zeros(self.cfg.n_experts, device=flat.device)
        for k in range(self.cfg.top_k):
            for e in range(self.cfg.n_experts):
                mask = idx[:, k] == e
                load[e] += mask.sum().item()
                if mask.any():
                    expert_in = flat[mask]
                    expert_out = self.experts[e](expert_in)
                    out[mask] += weights[mask, k].unsqueeze(-1) * expert_out
        # Load balancing loss
        total = load.sum() + 1e-9
        load_dist = load / total
        uniform = 1.0 / self.cfg.n_experts
        self.aux_loss = self.cfg.load_balance_coef * self.cfg.n_experts * (load_dist * uniform).sum()
        out = out.reshape(b, t, d)
        if self.cfg.use_shared_experts:
            for s in self.shared:
                out = out + s(x)
        return out
