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

"""Self-RAG (Asai et al. 2024) — adaptive retrieval with self-reflection tokens."""
from __future__ import annotations

from dataclasses import dataclass

from ai_engineer.core.llm import LLMClient, Message


@dataclass
class SelfRAGResult:
    answer: str
    retrieved_chunks: list[str]
    retrieval_tokens: list[str]
    support_tokens: list[str]
    usefulness_tokens: list[str]


class SelfRAG:
    """Self-RAG: model emits special tokens [Retrieve], [IsRel], [IsSup], [IsUse]."""

    SPECIAL_TOKENS = ["[Retrieve]", "[No Retrieve]", "[Relevant]", "[Irrelevant]", "[Fully supported]", "[Partially supported]", "[No support]", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]

    def __init__(self, llm: LLMClient, retriever) -> None:
        self.llm = llm
        self.retriever = retriever

    async def answer(self, question: str, top_k: int = 4) -> SelfRAGResult:
        # Step 1: Decide whether to retrieve
        decision = await self.llm.complete(
            messages=[
                Message(role="system", content=f"Decide if external knowledge is needed. Respond with one of: {self.SPECIAL_TOKENS[:2]}"),
                Message(role="user", content=question),
            ],
            temperature=0.0,
            max_tokens=20,
        )
        retrieved = []
        support_tokens: list[str] = []
        use_tokens: list[str] = []
        relevance_tokens: list[str] = []
        if "[Retrieve]" in decision.content:
            retrieved = self.retriever.retrieve(question, top_k=top_k)
            for r in retrieved:
                # Step 2: For each chunk, judge relevance
                rel = await self.llm.complete(
                    messages=[
                        Message(role="system", content=f"Is this chunk relevant? Respond with: {self.SPECIAL_TOKENS[2]} or {self.SPECIAL_TOKENS[3]}"),
                        Message(role="user", content=f"Q: {question}\n\nChunk: {r['text']}"),
                    ],
                    temperature=0.0,
                    max_tokens=10,
                )
                relevance_tokens.append(rel.content)
                if "[Irrelevant]" in rel.content:
                    continue
                # Step 3: Judge support
                sup = await self.llm.complete(
                    messages=[
                        Message(role="system", content=f"Does the chunk support the answer? {self.SPECIAL_TOKENS[4:7]}"),
                        Message(role="user", content=f"Q: {question}\nChunk: {r['text']}"),
                    ],
                    temperature=0.0,
                    max_tokens=10,
                )
                support_tokens.append(sup.content)
        ctx = "\n\n".join(r["text"] for r in retrieved)
        ans = await self.llm.complete(
            messages=[
                Message(role="system", content=f"Answer using the context. Then judge usefulness: {self.SPECIAL_TOKENS[7:]}"),
                Message(role="user", content=f"Q: {question}\n\nContext:\n{ctx}\n\nAnswer:"),
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        for tok in self.SPECIAL_TOKENS[7:]:
            if tok in ans.content:
                use_tokens.append(tok)
        return SelfRAGResult(answer=ans.content, retrieved_chunks=[r["text"] for r in retrieved], retrieval_tokens=[decision.content.strip()], support_tokens=support_tokens, usefulness_tokens=use_tokens)
