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

"""Initialize the database."""
from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from ai_engineer.config import get_settings


async def main() -> None:
    s = get_settings()
    # Parse URL
    from urllib.parse import urlparse

    p = urlparse(s.database_url.replace("+asyncpg", ""))
    conn = await asyncpg.connect(
        host=p.hostname, port=p.port or 5432, user=p.username, password=p.password, database=p.path.lstrip("/")
    )
    sql_path = Path(__file__).parent.parent / "src" / "ai_engineer" / "db" / "migrations" / "001_init.sql"
    sql = sql_path.read_text()
    await conn.execute(sql)
    print(f"DB initialized: {p.path.lstrip('/')}")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
