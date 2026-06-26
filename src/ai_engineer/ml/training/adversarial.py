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

"""Adversarial training: FGSM, PGD, TRADES."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float = 8 / 255) -> torch.Tensor:
    x_adv = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return (x_adv + eps * grad.sign()).detach()


def pgd(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float = 8 / 255, alpha: float = 2 / 255, iters: int = 7) -> torch.Tensor:
    x_adv = x.detach().clone() + (torch.rand_like(x) * 2 - 1) * eps
    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    return x_adv.detach()


def trades_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float = 8 / 255, beta: float = 6.0) -> torch.Tensor:
    x_adv = pgd(model, x, y, eps=eps)
    logits_clean = F.log_softmax(model(x), dim=1)
    logits_adv = F.log_softmax(model(x_adv), dim=1)
    ce = F.cross_entropy(model(x), y)
    kl = F.kl_div(logits_adv, logits_clean, reduction="batchmean")
    return ce + beta * kl
