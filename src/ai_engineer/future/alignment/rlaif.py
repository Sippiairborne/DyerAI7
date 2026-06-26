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

"""RLAIF (Constitutional AI at scale) — Reinforcement Learning from AI Feedback."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.future.reasoning.constitutional_ai import ConstitutionalAI
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RLAIFTrajectory:
    prompt: str
    candidates: list[str] = field(default_factory=list)
    preferences: list[tuple[int, int]] = field(default_factory=list)  # (winner_idx, loser_idx)
    metadata: dict = field(default_factory=dict)


class RLAIFTrainer:
    """Generate preference data from AI judges, then train a reward model and policy."""

    def __init__(self, llm: LLMClient, cai: ConstitutionalAI | None = None) -> None:
        self.llm = llm
        self.cai = cai or ConstitutionalAI(llm)
        self.trajectories: list[RLAIFTrajectory] = []

    async def collect_preferences(self, prompt: str, candidates: list[str]) -> RLAIFTrajectory:
        """Use CAI to generate preference pairs between candidates."""
        traj = RLAIFTrajectory(prompt=prompt, candidates=candidates)
        # For each pair, get CAI winner/loser
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                winner, loser = await self.cai.rlaif_preferences(prompt, candidates[i], candidates[j])
                wi = candidates.index(winner)
                li = candidates.index(loser)
                traj.preferences.append((wi, li))
        self.trajectories.append(traj)
        return traj

    async def collect_batch(self, prompts: list[str], n_candidates: int = 4) -> list[RLAIFTrajectory]:
        out: list[RLAIFTrajectory] = []
        for p in prompts:
            cands = await self._sample_candidates(p, n_candidates)
            traj = await self.collect_preferences(p, cands)
            out.append(traj)
        return out

    async def _sample_candidates(self, prompt: str, n: int) -> list[str]:
        out: list[str] = []
        for _ in range(n):
            r = await self.llm.complete(messages=[Message(role="user", content=prompt)], temperature=0.9, max_tokens=512)
            out.append(r.content)
        return out

    def export_dpo_format(self, output_path: str | Path) -> int:
        """Export collected preferences as DPO training data."""
        n = 0
        with Path(output_path).open("w") as f:
            for t in self.trajectories:
                for wi, li in t.preferences:
                    f.write(json.dumps({"prompt": t.prompt, "chosen": t.candidates[wi], "rejected": t.candidates[li]}) + "\n")
                    n += 1
        logger.info("rlaif.export", path=str(output_path), examples=n)
        return n
