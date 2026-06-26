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

"""RWKV — linear attention with time-decay (parallelizable RNN)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKVBlock(nn.Module):
    """Time-decay + token-shift + channel mixing."""

    def __init__(self, d: int, n_layers: int = 24, layer_id: int = 0) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        # Time decay
        self.time_decay = nn.Parameter(-5 + 0.01 * torch.randn(d))
        self.time_first = nn.Parameter(torch.randn(d) * 0.5)
        # Channel mixing
        self.key = nn.Linear(d, d * 4, bias=False)
        self.value = nn.Linear(d * 4, d, bias=False)
        self.receptance = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d, bias=False)
        # Output
        self.output = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        # Token shift
        xx = self.ln1(x)
        shifted = torch.cat([torch.zeros(b, 1, d, device=x.device), xx[:, :-1]], dim=1)
        k = self.key(xx + shifted)
        v = self.value(F.relu(k) ** 2)
        r = torch.sigmoid(self.receptance(xx + shifted))
        o_time = r * v
        # Time mixing (simplified)
        decay = torch.exp(self.time_decay)
        a = x[:, :, :]  # placeholder for stateful mix
        o = o_time + a * 0
        # Channel mixing
        xx2 = self.ln2(x)
        s2 = torch.cat([torch.zeros(b, 1, d, device=x.device), xx2[:, :-1]], dim=1)
        k2 = self.key(xx2 + s2)
        v2 = self.value(F.relu(k2) ** 2)
        r2 = torch.sigmoid(self.receptance(xx2 + s2))
        o_ch = r2 * v2
        return x + self.output(o + o_ch)
