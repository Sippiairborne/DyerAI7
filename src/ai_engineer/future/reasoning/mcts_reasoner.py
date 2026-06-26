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

"""MCTS-based reasoning — Monte Carlo Tree Search with self-evaluating rollouts."""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from uuid import uuid4

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MCTSNode:
    id: str
    state: str
    parent_id: str | None
    visits: int = 0
    value: float = 0.0
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    terminal: bool = False


class MCTSReasoner:
    """MCTS for reasoning. Uses UCT for selection."""

    def __init__(self, llm: LLMClient, n_rollouts: int = 8, max_depth: int = 8, c_explore: float = 1.414, n_children: int = 3) -> None:
        self.llm = llm
        self.n_rollouts = n_rollouts
        self.max_depth = max_depth
        self.c = c_explore
        self.n_children = n_children
        self.tree: dict[str, MCTSNode] = {}

    async def solve(self, problem: str) -> tuple[str, MCTSNode]:
        root_id = str(uuid4())
        self.tree[root_id] = MCTSNode(id=root_id, state=problem, parent_id=None, depth=0)
        for _ in range(self.n_rollouts):
            await self._rollout(root_id)
        best_child = max(self.tree[root_id].children_ids, key=lambda cid: self.tree[cid].visits)
        return self.tree[best_child].state, self.tree[best_child]

    async def _rollout(self, root_id: str) -> None:
        path: list[str] = [root_id]
        node_id = root_id
        # Selection
        while self.tree[node_id].children_ids and self.tree[node_id].depth < self.max_depth and not self.tree[node_id].terminal:
            node_id = self._select_child(node_id)
            path.append(node_id)
            if self.tree[node_id].terminal:
                break
        # Expansion
        if not self.tree[node_id].terminal and self.tree[node_id].depth < self.max_depth:
            await self._expand(node_id)
            if self.tree[node_id].children_ids:
                node_id = self.tree[node_id].children_ids[0]
                path.append(node_id)
        # Simulation (rollout)
        value = await self._simulate(self.tree[node_id].state)
        # Backprop
        for nid in path:
            n = self.tree[nid]
            n.visits += 1
            n.value += value

    def _select_child(self, node_id: str) -> str:
        node = self.tree[node_id]
        log_n = math.log(max(node.visits, 1))
        def uct(cid: str) -> float:
            c = self.tree[cid]
            if c.visits == 0:
                return float("inf")
            return c.value / c.visits + self.c * math.sqrt(log_n / c.visits)
        return max(node.children_ids, key=uct)

    async def _expand(self, node_id: str) -> None:
        node = self.tree[node_id]
        if node.terminal or node.depth >= self.max_depth:
            return
        children_states = await self._propose(node.state, self.n_children)
        for s in children_states:
            cid = str(uuid4())
            self.tree[cid] = MCTSNode(id=cid, state=s, parent_id=node_id, depth=node.depth + 1)
            node.children_ids.append(cid)
            if self._is_terminal(s):
                self.tree[cid].terminal = True

    async def _propose(self, state: str, n: int) -> list[str]:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Generate distinct next reasoning steps."),
                Message(role="user", content=f"State:\n{state}\n\nNext {n} steps, each prefixed 'STEP:'."),
            ],
            temperature=0.9,
            max_tokens=600,
        )
        return [line.split("STEP:", 1)[1].strip() for line in resp.content.splitlines() if "STEP:" in line][:n]

    async def _simulate(self, state: str) -> float:
        # Greedy rollout: keep extending until terminal or depth limit
        s = state
        for _ in range(self.max_depth):
            if self._is_terminal(s):
                break
            s = (await self._propose(s, 1) or [s])[0]
        return await self._score(s)

    async def _score(self, state: str) -> float:
        resp = await self.llm.complete(
            messages=[Message(role="system", content="Rate 0-1:"), Message(role="user", content=state)],
            temperature=0.0, max_tokens=10,
        )
        import re
        m = re.search(r"(\d+\.?\d*)", resp.content)
        return float(m.group(1)) if m else 0.5

    def _is_terminal(self, s: str) -> bool:
        sl = s.strip().lower()
        return any(sl.startswith(p) for p in ("answer:", "the answer is", "conclusion:", "final answer:"))
