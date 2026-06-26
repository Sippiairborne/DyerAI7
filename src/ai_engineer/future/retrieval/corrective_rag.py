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

"""Corrective RAG (Yan et al. 2024) — grade documents and rewrite query if needed."""
from __future__ import annotations

from dataclasses import dataclass

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CRAGResult:
    answer: str
    graded: list[dict]
    action: str  # correct | incorrect | ambiguous
    refined_query: str | None = None


class CorrectiveRAG:
    """CRAG: retrieve → grade → conditionally refine → web-search fallback → generate."""

    def __init__(self, llm: LLMClient, retriever, web_search=None) -> None:
        self.llm = llm
        self.retriever = retriever
        self.web_search = web_search

    async def answer(self, question: str, top_k: int = 5) -> CRAGResult:
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        graded: list[dict] = []
        for r in retrieved:
            grade = await self.llm.complete(
                messages=[
                    Message(role="system", content="Grade this document: CORRECT, INCORRECT, or AMBIGUOUS based on whether it contains accurate, relevant information."),
                    Message(role="user", content=f"Q: {question}\nDoc: {r['text']}\n\nGrade:"),
                ],
                temperature=0.0,
                max_tokens=10,
            )
            label = "ambiguous"
            g = grade.content.upper()
            if "CORRECT" in g and "INCORRECT" not in g:
                label = "correct"
            elif "INCORRECT" in g:
                label = "incorrect"
            graded.append({"text": r["text"], "grade": label})
        counts = {"correct": 0, "incorrect": 0, "ambiguous": 0}
        for g in graded:
            counts[g["grade"]] += 1
        ctx: list[str] = []
        refined: str | None = None
        if counts["correct"] >= max(1, top_k // 2):
            ctx = [g["text"] for g in graded if g["grade"] == "correct"]
            action = "correct"
        elif counts["ambiguous"] > 0:
            refined = await self._refine_query(question)
            new_docs = self.retriever.retrieve(refined, top_k=top_k)
            ctx = [g["text"] for g in new_docs]
            ctx += [g["text"] for g in graded if g["grade"] == "ambiguous"]
            action = "ambiguous"
        else:
            if self.web_search:
                try:
                    ws = self.web_search.search(question)
                    ctx = [r["body"] for r in ws.results[:5]]
                    action = "web_search"
                except Exception:
                    action = "skip"
            else:
                action = "skip"
        joined = "\n\n".join(ctx)
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Answer using the provided context. Be precise and cite which context supports each claim."),
                Message(role="user", content=f"Q: {question}\n\nContext:\n{joined}\n\nAnswer:"),
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return CRAGResult(answer=resp.content, graded=graded, action=action, refined_query=refined)

    async def _refine_query(self, q: str) -> str:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Rewrite the query to be more specific and search-friendly."),
                Message(role="user", content=q),
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return resp.content
