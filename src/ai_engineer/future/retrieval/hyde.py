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

"""HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer then embed for retrieval."""
from __future__ import annotations

from dataclasses import dataclass

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.ml.features.text import TextVectorizer


@dataclass
class HydeResult:
    query: str
    hypothetical: str
    retrieved: list[dict]


class HyDE:
    """Hypothetical Document Embeddings — improves retrieval by embedding the answer, not the question."""

    def __init__(self, llm: LLMClient, vectorizer: TextVectorizer | None = None, store=None) -> None:
        self.llm = llm
        self.vectorizer = vectorizer or TextVectorizer(kind="sentence")
        self.store = store  # Anything with a .search(text, top_k) method

    async def index(self, documents: list[str]) -> None:
        self.vectorizer.fit_transform(documents)

    async def retrieve(self, query: str, top_k: int = 5) -> HydeResult:
        # Generate hypothetical
        hypo_resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Write a short passage that would perfectly answer the question. Be specific and informative."),
                Message(role="user", content=query),
            ],
            temperature=0.7,
            max_tokens=400,
        )
        # Embed hypothetical
        embedding = self.vectorizer.transform([hypo_resp.content])[0]
        # Search store
        retrieved: list[dict] = []
        if self.store is not None:
            try:
                retrieved = self.store.search_by_vector(embedding, top_k=top_k)
            except Exception:
                retrieved = []
        return HydeResult(query=query, hypothetical=hypo_resp.content, retrieved=retrieved)
