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

"""GraphRAG (Microsoft 2024) — community detection over knowledge graph for global QA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.ml.features.text import TextVectorizer
from ai_engineer.ml.memory.graph_store import GraphStore
from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GraphRAGResult:
    answer: str
    communities_used: list[int]
    entities: list[str]
    chunks: list[str]


class GraphRAG:
    """Build entity graph over corpus, detect communities, generate community summaries, answer globally."""

    def __init__(self, llm: LLMClient, graph: GraphStore | None = None) -> None:
        self.llm = llm
        self.graph = graph or GraphStore()
        self.vectorizer = TextVectorizer(kind="sentence")
        self.chunks: list[dict] = []  # {"id", "text", "embedding", "entities"}
        self.communities: dict[int, list[str]] = {}
        self.community_summaries: dict[int, str] = {}

    async def index(self, documents: list[str]) -> None:
        # Chunk
        chunks = [d for d in documents]  # simple: one chunk per doc; in production, recursive chunking
        for i, c in enumerate(chunks):
            cid = f"chunk_{i}"
            entities = await self._extract_entities(c)
            self.chunks.append({"id": cid, "text": c, "entities": entities})
            for e in entities:
                await self.graph.add_relationship(e, cid, "mentioned_in")
        # Build community summaries
        self._build_communities()
        for cid, ents in self.communities.items():
            chunks_in = [c["text"] for c in self.chunks if any(e in c["entities"] for e in ents)]
            joined = "\n".join(chunks_in[:30])
            self.community_summaries[cid] = await self._summarize(joined)

    async def query(self, question: str, top_k_communities: int = 3) -> GraphRAGResult:
        # Map question to communities via entities
        ents = await self._extract_entities(question)
        scores: dict[int, int] = {}
        for ent in ents:
            neighbors = await self.graph.neighbors(ent, depth=2)
            for n in neighbors:
                if "Chunk" in n.get("labels", [""])[0] if n.get("labels") else False:
                    # find community
                    for cid, members in self.communities.items():
                        if any(e in ents for e in members):
                            scores[cid] = scores.get(cid, 0) + 1
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_k_communities]
        if not top:
            top = [(cid, 1) for cid in list(self.community_summaries.keys())[:top_k_communities]]
        summaries = "\n\n".join(f"[Community {cid}]\n{self.community_summaries[cid]}" for cid, _ in top)
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Use the community summaries to answer the question with supporting details."),
                Message(role="user", content=f"Question: {question}\n\nCommunity Summaries:\n{summaries}\n\nAnswer:"),
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return GraphRAGResult(
            answer=resp.content,
            communities_used=[c for c, _ in top],
            entities=ents,
            chunks=[c["text"] for c in self.chunks[:5]],
        )

    def _build_communities(self) -> None:
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities
        except ImportError:
            return
        G = nx.Graph()
        for c in self.chunks:
            for e1 in c["entities"]:
                for e2 in c["entities"]:
                    if e1 != e2:
                        if G.has_edge(e1, e2):
                            G[e1][e2]["weight"] += 1
                        else:
                            G.add_edge(e1, e2, weight=1)
        if G.number_of_edges() == 0:
            return
        try:
            comms = louvain_communities(G, weight="weight", resolution=1.0)
            for i, c in enumerate(comms):
                self.communities[i] = list(c)
        except Exception:
            entities = sorted({e for c in self.chunks for e in c["entities"]})
            for i in range(0, len(entities), 10):
                self.communities[len(self.communities)] = entities[i:i + 10]

    async def _extract_entities(self, text: str) -> list[str]:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Extract key named entities (people, places, organizations, concepts, technical terms). Output as a comma-separated list."),
                Message(role="user", content=text),
            ],
            temperature=0.0,
            max_tokens=200,
        )
        return [e.strip() for e in resp.content.split(",") if e.strip()][:20]

    async def _summarize(self, text: str) -> str:
        if not text.strip():
            return ""
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Summarize this content into a tight paragraph capturing the key facts and relationships."),
                Message(role="user", content=text[:8000]),
            ],
            temperature=0.1,
            max_tokens=400,
        )
        return resp.content
