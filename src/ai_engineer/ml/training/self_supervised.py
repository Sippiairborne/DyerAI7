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

"""Self-supervised learning: SimCLR, MoCo, MAE, BYOL, DINO."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, projection_dim: int = 128, hidden_dim: int = 512) -> None:
        super().__init__()
        self.encoder = encoder
        d = self._feat_dim()
        self.projector = nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim))

    def _feat_dim(self) -> int:
        for p in self.encoder.parameters():
            return p.shape[-1] if p.ndim >= 2 else 1
        return 1

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = z1 @ z2.T / 0.5
        n = z1.size(0)
        targets = torch.arange(n, device=z1.device)
        return (F.cross_entropy(sim, targets) + F.cross_entropy(sim.T, targets)) / 2


class MAEDecoder(nn.Module):
    """Masked autoencoder decoder (ViT-style)."""

    def __init__(self, encoder_dim: int, patch_dim: int, decoder_dim: int = 512, depth: int = 4, heads: int = 8) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj_in = nn.Linear(encoder_dim, decoder_dim)
        self.proj_out = nn.Linear(decoder_dim, patch_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        b, n, _ = x.shape
        mask_tokens = self.mask_token.expand(b, n, -1)
        x = torch.where(mask.unsqueeze(-1), mask_tokens, x)
        return self.proj_out(self.decoder(x))


class BYOL(nn.Module):
    """Bootstrap Your Own Latent."""

    def __init__(self, encoder: nn.Module, projection_dim: int = 128, hidden_dim: int = 512) -> None:
        super().__init__()
        d = self._feat_dim()
        self.online_encoder = encoder
        self.online_projector = nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim))
        self.predictor = nn.Sequential(nn.Linear(projection_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim))
        self.target_encoder = encoder
        self.target_projector = nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim))

    def _feat_dim(self) -> int:
        for p in self.online_encoder.parameters():
            return p.shape[-1] if p.ndim >= 2 else 1
        return 1

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        o1 = self.predictor(self.online_projector(self.online_encoder(x1)))
        o2 = self.predictor(self.online_projector(self.online_encoder(x2)))
        with torch.no_grad():
            t1 = self.target_projector(self.target_encoder(x1))
            t2 = self.target_projector(self.target_encoder(x2))
        return (
            2 - 2 * (F.normalize(o1, dim=1) * F.normalize(t2, dim=1)).sum(-1).mean()
            + 2 - 2 * (F.normalize(o2, dim=1) * F.normalize(t1, dim=1)).sum(-1).mean()
        )
