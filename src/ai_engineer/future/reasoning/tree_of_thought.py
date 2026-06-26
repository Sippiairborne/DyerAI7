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

"""Tree of Thoughts — explore multiple reasoning paths with BFS/DFS and self-evaluation."""
from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ThoughtNode:
    id: str
    state: str
    parent_id: str | None
    depth: int
    score: float
    children_ids: list[str] = field(default_factory=list)
    is_terminal: bool = False
    answer: str | None = None
    visited: int = 0


@dataclass
class ToTResult:
    best_path: list[ThoughtNode]
    best_answer: str
    total_nodes: int
    pruned_nodes: int
    elapsed_s: float


class TreeOfThoughts:
    """Tree of Thoughts reasoning with self-evaluation and voting."""

    def __init__(self, llm: LLMClient, n_branches: int = 3, max_depth: int = 5, prune_threshold: float = 0.3) -> None:
        self.llm = llm
        self.n_branches = n_branches
        self.max_depth = max_depth
        self.prune_threshold = prune_threshold
        self.nodes: dict[str, ThoughtNode] = {}

    async def solve(
        self,
        problem: str,
        generate_fn: Callable[[str, int], list[str]] | None = None,
        evaluate_fn: Callable[[str], float] | None = None,
        method: str = "bfs",
    ) -> ToTResult:
        from uuid import uuid4

        start = time.time()
        root_id = str(uuid4())
        root = ThoughtNode(id=root_id, state=problem, parent_id=None, depth=0, score=1.0)
        self.nodes[root_id] = root
        frontier = [root_id]
        best_path: list[ThoughtNode] = [root]
        pruned = 0
        gen = generate_fn or self._default_generate
        ev = evaluate_fn or self._default_evaluate

        while frontier and root.depth < self.max_depth:
            if method == "dfs":
                current_id = frontier.pop()
            else:
                current_id = frontier.pop(0)
            current = self.nodes[current_id]
            if current.is_terminal:
                continue
            current.visited += 1
            candidates = await gen(current.state, self.n_branches)
            scored: list[tuple[float, str]] = []
            for c in candidates:
                cid = str(uuid4())
                child = ThoughtNode(id=cid, state=c, parent_id=current_id, depth=current.depth + 1)
                self.nodes[cid] = child
                current.children_ids.append(cid)
                s = await ev(c)
                child.score = s
                if s < self.prune_threshold:
                    self.nodes[cid].is_terminal = True
                    pruned += 1
                    continue
                scored.append((s, cid))
                # Heuristic: terminal if looks like an answer
                if self._looks_like_final_answer(c):
                    child.is_terminal = True
                    child.answer = c
                    if child.score > best_path[-1].score:
                        best_path = self._path_to(cid) + [child]
            scored.sort(key=lambda x: -x[0])
            for _, cid in scored:
                heapq.heappush(frontier, (-self.nodes[cid].score, self.nodes[cid].depth, cid))
        if not best_path[-1].answer:
            best_path[-1].answer = best_path[-1].state
        return ToTResult(
            best_path=best_path,
            best_answer=best_path[-1].answer or best_path[-1].state,
            total_nodes=len(self.nodes),
            pruned_nodes=pruned,
            elapsed_s=time.time() - start,
        )

    def _path_to(self, node_id: str) -> list[ThoughtNode]:
        path: list[ThoughtNode] = []
        cur = self.nodes[node_id]
        while cur:
            path.append(cur)
            cur = self.nodes[cur.parent_id] if cur.parent_id else None  # type: ignore
        return list(reversed(path))

    def _looks_like_final_answer(self, s: str) -> bool:
        s_lower = s.strip().lower()
        return any(s_lower.startswith(p) for p in ("answer:", "the answer is", "conclusion:", "final answer:", "therefore,"))

    async def _default_generate(self, state: str, n: int) -> list[str]:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="You are a thoughtful reasoner. Generate distinct next steps."),
                Message(role="user", content=f"Current reasoning state:\n{state}\n\nGenerate {n} distinct next reasoning steps. Each on a separate line, prefixed with 'STEP:'."),
            ],
            temperature=0.8,
            max_tokens=600,
        )
        steps = []
        for line in resp.content.splitlines():
            if line.strip().startswith("STEP:"):
                steps.append(line.split("STEP:", 1)[1].strip())
        return steps[:n] or [resp.content]

    async def _default_evaluate(self, state: str) -> float:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="You evaluate reasoning steps. Respond with a single number 0-1 indicating how promising this step is towards solving the problem."),
                Message(role="user", content=state),
            ],
            temperature=0.0,
            max_tokens=10,
        )
        try:
            import re
            m = re.search(r"(\d+\.?\d*)", resp.content)
            if m:
                v = float(m.group(1))
                return max(0.0, min(1.0, v))
        except Exception:
            pass
        return 0.5
