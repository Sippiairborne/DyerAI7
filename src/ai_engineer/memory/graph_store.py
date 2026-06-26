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

"""Graph store over Neo4j for semantic relationships."""
from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase

from ai_engineer.config import get_settings
from ai_engineer.utils.errors import MemoryError_ as MemError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


class GraphStore:
    """Stores concepts and relationships for semantic reasoning."""

    def __init__(self) -> None:
        s = get_settings()
        self._driver = AsyncGraphDatabase.driver(
            s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)
        )

    async def init(self) -> None:
        async with self._driver.session() as session:
            await session.run(
                "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE"
            )

    async def close(self) -> None:
        await self._driver.close()

    async def add_relationship(
        self,
        source: str,
        target: str,
        relation: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                f"""
                MERGE (s:Concept {{name: $source}})
                MERGE (t:Concept {{name: $target}})
                MERGE (s)-[r:RELATES {{type: $relation}}]->(t)
                SET r += $properties
                """,
                source=source,
                target=target,
                relation=relation,
                properties=properties or {},
            )

    async def query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(cypher, parameters or {})
            return [dict(record) async for record in result]

    async def neighbors(self, concept: str, depth: int = 1) -> list[dict[str, Any]]:
        return await self.query(
            f"""
            MATCH path = (c:Concept {{name: $name}})-[*1..{depth}]-(n)
            RETURN DISTINCT n.name AS name, labels(n) AS labels
            """,
            {"name": concept},
        )
