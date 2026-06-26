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

"""SimPO — Simple Preference Optimization without reference model (Meng et al. 2024)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class SimPO:
    """SimPO: length-normalized preference optimization without a reference model."""

    def __init__(self, model, tokenizer, beta: float = 2.0, gamma_beta_ratio: float = 0.5, learning_rate: float = 5e-7) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta
        self.gamma = gamma_beta_ratio * beta
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def step(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor) -> float:
        chosen_logp = self._norm_logp(self.model, chosen_ids)
        rejected_logp = self._norm_logp(self.model, rejected_ids)
        logits = self.beta * (chosen_logp - rejected_logp) - self.gamma
        loss = -F.logsigmoid(logits).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _norm_logp(self, model, ids: torch.Tensor) -> torch.Tensor:
        out = model(ids)
        logits = out.logits[:, :-1, :]
        targets = ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_logp.sum(-1) / targets.size(1)
