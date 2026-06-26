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

import pytest

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.errors import LLMError


@pytest.mark.asyncio
async def test_complete(llm: LLMClient) -> None:
    resp = await llm.complete(
        messages=[
            Message(role="system", content="You are a calculator. Reply with just the number."),
            Message(role="user", content="2+2="),
        ],
        max_tokens=10,
        temperature=0.0,
    )
    assert resp.content
    assert "4" in resp.content


@pytest.mark.asyncio
async def test_structured(llm: LLMClient) -> None:
    from pydantic import BaseModel

    class Ans(BaseModel):
        answer: int

    out = await llm.structured(
        messages=[Message(role="user", content="What is 2+2?")],
        schema=Ans,
        max_tokens=20,
    )
    assert out.answer == 4
