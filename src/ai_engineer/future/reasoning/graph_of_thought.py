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

"""Graph of Thoughts — reasoning as a graph with merging/thought networks."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GoTNode:
    id: str
    content: str
    score: float
    in_edges: list[str] = field(default_factory=list)
    out_edges: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)


@dataclass
class GoTEdge:
    src: str
    dst: str
    operation: str  # generate | aggregate | refine | keep_best


@dataclass
class GoTResult:
    best_node: GoTNode
    nodes: dict[str, GoTNode]
    edges: list[GoTEdge]
    rounds: int
    elapsed_s: float


class GraphOfThoughts:
    """Graph of Thoughts with generate, aggregate, refine, score operations."""

    def __init__(self, llm: LLMClient, k: int = 3, max_rounds: int = 4, threshold: float = 0.5) -> None:
        self.llm = llm
        self.k = k
        self.max_rounds = max_rounds
        self.threshold = threshold
        self.nodes: dict[str, GoTNode] = {}
        self.edges: list[GoTEdge] = []

    async def solve(self, problem: str) -> GoTResult:
        start = time.time()
        # Seed
        root_id = str(uuid4())
        self.nodes[root_id] = GoTNode(id=root_id, content=problem, score=1.0)
        frontier = [root_id]
        rounds = 0
        while frontier and rounds < self.max_rounds:
            rounds += 1
            new_frontier = []
            for node_id in frontier:
                # Generate k branches
                branches = await self._generate(self.nodes[node_id].content, self.k)
                for b in branches:
                    cid = str(uuid4())
                    self.nodes[cid] = GoTNode(id=cid, content=b, score=await self._score(b))
                    self.edges.append(GoTEdge(src=node_id, dst=cid, operation="generate"))
                    self.nodes[node_id].out_edges.append(cid)
                    self.nodes[cid].in_edges.append(node_id)
                    if self.nodes[cid].score >= self.threshold:
                        new_frontier.append(cid)
            # Aggregate: merge pairs
            if len(new_frontier) >= 2:
                for i in range(0, len(new_frontier) - 1, 2):
                    a, b = new_frontier[i], new_frontier[i + 1]
                    agg = await self._aggregate(self.nodes[a].content, self.nodes[b].content)
                    aid = str(uuid4())
                    self.nodes[aid] = GoTNode(id=aid, content=agg, score=await self._score(agg))
                    for src in (a, b):
                        self.edges.append(GoTEdge(src=src, dst=aid, operation="aggregate"))
                        self.nodes[src].out_edges.append(aid)
                        self.nodes[aid].in_edges.append(src)
                    new_frontier.append(aid)
            # Refine: top 3 by score
            new_frontier = sorted(new_frontier, key=lambda nid: -self.nodes[nid].score)[: self.k]
            frontier = new_frontier
        # Refine top node
        if frontier:
            top = max(frontier, key=lambda nid: self.nodes[nid].score)
            refined = await self._refine(self.nodes[top].content)
            rid = str(uuid4())
            self.nodes[rid] = GoTNode(id=rid, content=refined, score=await self._score(refined))
            self.edges.append(GoTEdge(src=top, dst=rid, operation="refine"))
            self.nodes[top].out_edges.append(rid)
            self.nodes[rid].in_edges.append(top)
            best = self.nodes[rid]
        else:
            best = self.nodes[root_id]
        return GoTResult(best_node=best, nodes=self.nodes, edges=self.edges, rounds=rounds, elapsed_s=time.time() - start)

    async def _generate(self, state: str, n: int) -> list[str]:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Generate distinct next reasoning steps."),
                Message(role="user", content=f"State:\n{state}\n\nGenerate {n} distinct continuations, each prefixed 'STEP:'."),
            ],
            temperature=0.8,
            max_tokens=600,
        )
        return [line.split("STEP:", 1)[1].strip() for line in resp.content.splitlines() if "STEP:" in line][:n]

    async def _aggregate(self, a: str, b: str) -> str:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Merge two reasoning paths into a stronger combined path."),
                Message(role="user", content=f"Path A:\n{a}\n\nPath B:\n{b}\n\nCombined:"),
            ],
            temperature=0.4,
            max_tokens=600,
        )
        return resp.content

    async def _refine(self, s: str) -> str:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Critique and refine this reasoning for correctness and clarity."),
                Message(role="user", content=s),
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return resp.content

    async def _score(self, s: str) -> float:
        resp = await self.llm.complete(
            messages=[Message(role="system", content="Rate 0-1:"), Message(role="user", content=s)],
            temperature=0.0,
            max_tokens=10,
        )
        import re
        m = re.search(r"(\d+\.?\d*)", resp.content)
        return float(m.group(1)) if m else 0.5
