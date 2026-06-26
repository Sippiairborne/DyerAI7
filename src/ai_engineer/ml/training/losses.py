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

"""Loss function library."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target, num_classes=logits.size(1)).permute(0, -1, *range(2, logits.dim()))
        intersection = (probs * target_oh).flatten(1).sum(1)
        denom = (probs + target_oh).flatten(1).sum(1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        d = F.pairwise_distance(x1, x2)
        return ((label) * d**2 + (1 - label) * F.relu(self.margin - d) ** 2).mean()


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al. 2020)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        sim = features @ features.T / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim)
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        exp_sim = exp_sim * (1 - torch.eye(len(labels), device=features.device))
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return -mean_log_prob_pos.mean()


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return F.triplet_margin_loss(anchor, positive, negative, margin=self.margin)


class KLDivLoss(nn.Module):
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
        return F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta)


def build_loss(name: str, **kwargs: object) -> nn.Module:
    return {
        "cross_entropy": lambda: nn.CrossEntropyLoss(**kwargs),  # type: ignore
        "mse": lambda: nn.MSELoss(**kwargs),  # type: ignore
        "l1": lambda: nn.L1Loss(**kwargs),  # type: ignore
        "bce": lambda: nn.BCEWithLogitsLoss(**kwargs),  # type: ignore
        "focal": lambda: FocalLoss(**kwargs),  # type: ignore
        "label_smoothing": lambda: LabelSmoothingLoss(**kwargs),  # type: ignore
        "dice": lambda: DiceLoss(**kwargs),  # type: ignore
        "contrastive": lambda: ContrastiveLoss(**kwargs),  # type: ignore
        "supcon": lambda: SupConLoss(**kwargs),  # type: ignore
        "triplet": lambda: TripletLoss(**kwargs),  # type: ignore
        "kl_div": lambda: KLDivLoss(**kwargs),  # type: ignore
        "huber": lambda: HuberLoss(**kwargs),  # type: ignore
    }[name]()
