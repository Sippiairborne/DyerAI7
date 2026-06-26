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

"""Pruning: magnitude, structured, movement, dynamic."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class PruningResult:
    sparsity: float
    model_path: str
    original_size_mb: float
    pruned_size_mb: float


class Pruner:
    def magnitude_prune(self, model: nn.Module, sparsity: float = 0.5, output_path: str = "/tmp/pruned.pt") -> PruningResult:
        import torch.nn.utils.prune as prune

        params = sum(p.numel() for p in model.parameters())
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name="weight", amount=sparsity)
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                try:
                    prune.remove(module, "weight")
                except Exception:
                    pass
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)
        new_params = sum((p != 0).sum().item() for p in model.parameters())
        return PruningResult(sparsity=float(1 - new_params / params), model_path=output_path, original_size_mb=params * 4 / 1024 / 1024, pruned_size_mb=new_params * 4 / 1024 / 1024)

    def structured_prune(self, model: nn.Module, sparsity: float = 0.5, output_path: str = "/tmp/pruned_struct.pt") -> PruningResult:
        import torch.nn.utils.prune as prune

        params = sum(p.numel() for p in model.parameters())
        for module in model.modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(module, name="weight", amount=sparsity, n=2, dim=0)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, "weight")
                except Exception:
                    pass
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)
        new_params = sum(p.numel() for p in model.parameters())
        return PruningResult(sparsity=sparsity, model_path=output_path, original_size_mb=params * 4 / 1024 / 1024, pruned_size_mb=new_params * 4 / 1024 / 1024)
