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

"""Vector store abstraction over Qdrant."""
from __future__ import annotations

import time
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from ai_engineer.config import get_settings
from ai_engineer.core.llm import LLMClient
from ai_engineer.utils.errors import MemoryError_ as MemError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Persistent vector store for episodic and semantic memory."""

    def __init__(self, client: LLMClient, qdrant: AsyncQdrantClient | None = None) -> None:
        self.client = client
        self.settings = get_settings()
        self.qdrant = qdrant or AsyncQdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key or None,
        )
        self.collection = self.settings.qdrant_collection
        self.dim = self.settings.embedding_dim

    async def init(self) -> None:
        try:
            await self.qdrant.get_collection(self.collection)
        except UnexpectedResponse:
            await self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=self.dim, distance=qmodels.Distance.COSINE),
                optimizers_config=qmodels.OptimizersConfigDiff(default_segment_number=2),
            )
            logger.info("vector_store.collection_created", name=self.collection)

    async def add(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if not texts:
            return []
        embeddings = await self.client.embed(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadata = metadata or [{} for _ in texts]
        # Add timestamp to metadata
        ts = time.time()
        for m in metadata:
            m.setdefault("timestamp", ts)
        await self.qdrant.upsert(
            collection_name=self.collection,
            points=qmodels.Batch(ids=ids, vectors=embeddings, payloads=metadata),
        )
        return ids

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        embedding = (await self.client.embed([query]))[0]
        results = await self.qdrant.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=top_k,
            query_filter=qmodels.Filter(must=[qmodels.FieldCondition(**f) for f in filter_]) if filter_ else None,
            score_threshold=score_threshold,
        )
        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload or {}}
            for r in results
        ]

    async def get(self, ids: list[str]) -> list[dict[str, Any]]:
        results = await self.qdrant.retrieve(self.collection, ids=ids)
        return [{"id": str(r.id), "payload": r.payload or {}} for r in results]

    async def delete(self, ids: list[str]) -> None:
        await self.qdrant.delete(self.collection, points_selector=qmodels.PointIdsList(points=ids))

    async def close(self) -> None:
        await self.qdrant.close()
