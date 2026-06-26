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

"""Integrated Gradients."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class IGResult:
    attributions: np.ndarray
    convergence_delta: float


class IntegratedGradients:
    def __init__(self, model, n_steps: int = 50) -> None:
        self.model = model
        self.n_steps = n_steps

    def attribute(self, inputs: torch.Tensor, target: int | None = None) -> IGResult:
        self.model.eval()
        baseline = torch.zeros_like(inputs)
        scaled = [baseline + (float(i) / self.n_steps) * (inputs - baseline) for i in range(1, self.n_steps + 1)]
        scaled_t = torch.stack(scaled).requires_grad_(True)
        grads = []
        for x in scaled_t:
            out = self.model(x)
            tgt = target if target is not None else out.argmax(dim=-1)
            loss = out[0, tgt].sum() if target is not None else out[0, tgt[0]].sum()
            loss.backward()
            grads.append(x.grad.detach().clone())
            x.grad = None
        avg_grads = torch.stack(grads).mean(dim=0)
        ig = (inputs - baseline) * avg_grads
        return IGResult(attributions=ig.detach().cpu().numpy(), convergence_delta=float(ig.abs().sum().item()))
