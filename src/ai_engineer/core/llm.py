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

"""LLM client supporting any OpenAI-compatible endpoint (your own model)."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ai_engineer.config import get_settings
from ai_engineer.utils.errors import LLMError
from ai_engineer.utils.logging import get_logger
from ai_engineer.utils.retry import async_retry

logger = get_logger(__name__)


class Message(BaseModel):
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: dict[str, str]


class CompletionRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    top_p: float = 0.95
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, str] | None = None
    response_format: dict[str, str] | None = None
    stream: bool = False


class CompletionResponse(BaseModel):
    content: str
    finish_reason: str
    usage: dict[str, int] = Field(default_factory=dict)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class LLMClient:
    """Unified async LLM client. Talks to any OpenAI-compatible server, including your own model."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.default_model = settings.llm_model
        self.timeout = settings.llm_timeout
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,
        )
        self._http = httpx.AsyncClient(timeout=self.timeout)

    async def close(self) -> None:
        await self._client.close()
        await self._http.aclose()

    @async_retry(LLMError)
    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        response_format: dict[str, str] | None = None,
        stop: list[str] | None = None,
    ) -> CompletionResponse:
        """Chat completion call."""
        model = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        if stop:
            kwargs["stop"] = stop

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}") from e

        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        if getattr(msg, "tool_calls", None):
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function={"name": tc.function.name, "arguments": tc.function.arguments},
                )
                for tc in msg.tool_calls
            ]
        return CompletionResponse(
            content=msg.content or "",
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            },
            tool_calls=tool_calls,
            raw=resp.model_dump(),
        )

    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        model = model or self.default_model
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=[m.model_dump(exclude_none=True) for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            raise LLMError(f"Stream failed: {e}") from e

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings using the configured embedding model."""
        from sentence_transformers import SentenceTransformer

        if not hasattr(self, "_embedder"):
            settings = get_settings()
            self._embedder = SentenceTransformer(settings.embedding_model)
        vectors = self._embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    async def structured(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_attempts: int = 3,
    ) -> BaseModel:
        """Generate a structured response conforming to a Pydantic schema.

        Tries JSON mode first; falls back to extraction with repair.
        """
        schema_instructions = (
            f"\n\nYou MUST respond with a single JSON object matching this schema:\n"
            f"```json\n{json.dumps(schema.model_json_schema(), indent=2)}\n```\n"
            f"Return ONLY the JSON object, no commentary."
        )
        augmented = list(messages)
        if augmented and augmented[-1].role == "user":
            augmented[-1] = Message(
                role=augmented[-1].role,
                content=augmented[-1].content + schema_instructions,
            )
        else:
            augmented.append(Message(role="user", content=schema_instructions))

        last_err: Exception | None = None
        for attempt in range(max_attempts):
            resp = await self.complete(
                messages=augmented,
                model=model,
                temperature=temperature if attempt == 0 else min(0.5, temperature * (attempt + 1)),
                max_tokens=max_tokens,
            )
            text = resp.content.strip()
            # Strip code fences
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            try:
                data = json.loads(text)
                return schema.model_validate(data)
            except Exception as e:
                last_err = e
                # Add a repair message
                augmented.append(
                    Message(
                        role="user",
                        content=f"Your previous response was invalid: {e}\nFix and return ONLY valid JSON matching the schema.",
                    )
                )
        raise LLMError(f"Failed to get structured output after {max_attempts} attempts: {last_err}")
