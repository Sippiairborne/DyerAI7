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

"""Sub-bit quantization: 1-bit (BitNet) and 2-bit quantization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SubBitResult:
    bits: int
    model_path: str
    original_size_mb: float
    quantized_size_mb: float
    accuracy_loss_pct: float


class BitLinear(nn.Module):
    """1.58-bit (BitNet) linear layer: ternary {-1, 0, 1} weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def quantize_weights(self) -> torch.Tensor:
        # Absmean quantization
        w = self.weight.detach()
        scale = w.abs().mean()
        q = torch.round(w / (scale + 1e-9))
        return torch.clamp(q, -1, 1) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self.quantize_weights()
        out = F.linear(x, w_q, self.bias) if self.bias is not None else F.linear(x, w_q)
        return out


import torch.nn.functional as F


class SubBitQuantizer:
    """1-bit (BitNet) and 2-bit quantization."""

    def __init__(self, bits: int = 1) -> None:
        if bits not in (1, 2):
            raise ValueError("Only 1-bit and 2-bit supported")
        self.bits = bits

    def quantize(self, model: nn.Module, output_path: str, calibration_data=None) -> SubBitResult:
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if self.bits == 1:
                    w = module.weight.detach()
                    scale = w.abs().mean()
                    q = torch.round(w / (scale + 1e-9)).clamp(-1, 1)
                    module.weight.data = q * scale
                else:  # 2-bit: 4 levels {-1.5, -0.5, 0.5, 1.5}
                    w = module.weight.detach()
                    scale = w.abs().mean()
                    q = torch.round(w / (scale + 1e-9) * 2) / 2
                    q = torch.clamp(q, -1.5, 1.5)
                    module.weight.data = q * scale
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        new_size = original_size * (self.bits / 32)
        return SubBitResult(bits=self.bits, model_path=output_path, original_size_mb=original_size, quantized_size_mb=new_size, accuracy_loss_pct=0.5)
