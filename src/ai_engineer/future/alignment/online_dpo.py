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

"""Online DPO — iteratively sample, judge, and update."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OnlineDPOConfig:
    n_iterations: int = 5
    n_samples_per_prompt: int = 4
    beta: float = 0.1
    learning_rate: float = 5e-7
    max_length: int = 1024
    reference_free: bool = False


class OnlineDPO:
    """Online DPO (Guo et al. 2024) — continuously updates from freshly sampled data."""

    def __init__(self, llm: LLMClient, model, tokenizer, ref_model=None, config: OnlineDPOConfig | None = None) -> None:
        self.llm = llm
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.config = config or OnlineDPOConfig()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

    async def train(self, prompts: list[str], judge_fn, output_path: str | Path) -> dict:
        history: list[dict] = []
        for it in range(self.config.n_iterations):
            new_pairs: list[dict] = []
            for prompt in prompts:
                samples = await self._sample(prompt, self.config.n_samples_per_prompt)
                for i in range(len(samples)):
                    for j in range(i + 1, len(samples)):
                        winner, loser = await judge_fn(prompt, samples[i], samples[j])
                        if winner is None:
                            continue
                        wi = samples.index(winner)
                        li = samples.index(loser)
                        new_pairs.append({"prompt": prompt, "chosen": samples[wi], "rejected": samples[li]})
            if not new_pairs:
                continue
            # Inner DPO step
            loss_val = self._dpo_step(new_pairs)
            history.append({"iter": it, "n_pairs": len(new_pairs), "loss": loss_val})
            logger.info("online_dpo.iter", iter=it, n_pairs=len(new_pairs), loss=loss_val)
        with Path(output_path).open("w") as f:
            for h in history:
                f.write(json.dumps(h) + "\n")
        return {"iterations": self.config.n_iterations, "history": history}

    async def _sample(self, prompt: str, n: int) -> list[str]:
        out: list[str] = []
        for _ in range(n):
            r = await self.llm.complete(messages=[Message(role="user", content=prompt)], temperature=0.9, max_tokens=512)
            out.append(r.content)
        return out

    def _dpo_step(self, pairs: list[dict]) -> float:
        self.model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
        total_loss = 0.0
        for p in pairs:
            chosen_ids = self.tokenizer(p["prompt"] + p["chosen"], return_tensors="pt", truncation=True, max_length=self.config.max_length).input_ids
            rejected_ids = self.tokenizer(p["prompt"] + p["rejected"], return_tensors="pt", truncation=True, max_length=self.config.max_length).input_ids
            chosen_logp = self._sequence_logp(self.model, chosen_ids)
            rejected_logp = self._sequence_logp(self.model, rejected_ids)
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_chosen_logp = self._sequence_logp(self.ref_model, chosen_ids)
                    ref_rejected_logp = self._sequence_logp(self.ref_model, rejected_ids)
            else:
                ref_chosen_logp = ref_rejected_logp = 0.0
            logits = self.config.beta * ((chosen_logp - rejected_logp) - (ref_chosen_logp - ref_rejected_logp))
            loss = -F.logsigmoid(logits).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(pairs), 1)

    def _sequence_logp(self, model, ids: torch.Tensor) -> torch.Tensor:
        out = model(ids)
        logits = out.logits[:, :-1, :]
        targets = ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_logp.sum(-1).mean()
