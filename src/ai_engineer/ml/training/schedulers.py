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

"""LR scheduler factory."""
from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    LinearLR,
    OneCycleLR,
    PolynomialLR,
    StepLR,
)


def build_scheduler(name: str, optimizer, total_steps: int, warmup_steps: int = 0, warmup_ratio: float = 0.0) -> torch.optim.lr_scheduler.LRScheduler | None:
    if total_steps <= 0:
        return None
    name = name.lower()
    if name == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if name == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
    if name == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=total_steps // 4, T_mult=2)
    if name == "onecycle":
        return OneCycleLR(optimizer, max_lr=[g["lr"] for g in optimizer.param_groups], total_steps=total_steps)
    if name == "polynomial":
        return PolynomialLR(optimizer, total_iters=total_steps, power=0.9)
    if name == "step":
        return StepLR(optimizer, step_size=total_steps // 3, gamma=0.1)
    if name in ("cosine_warmup", "linear_warmup"):
        warmup = warmup_steps or max(int(warmup_ratio * total_steps), 1)
        if name == "cosine_warmup":
            def fn(s):
                if s < warmup:
                    return s / max(warmup, 1)
                progress = (s - warmup) / max(total_steps - warmup, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))
            return LambdaLR(optimizer, lr_lambda=fn)
        def fn(s):
                if s < warmup:
                    return s / max(warmup, 1)
                return max(0.0, (total_steps - s) / max(total_steps - warmup, 1))
        return LambdaLR(optimizer, lr_lambda=fn)
    return None
