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

"""Retry helpers."""
from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ai_engineer.config import get_settings
from ai_engineer.utils.errors import LLMError, SandboxError, ToolError

T = TypeVar("T")


def async_retry(
    *exceptions: type[BaseException],
    max_attempts: int | None = None,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for async retry with exponential backoff."""
    settings = get_settings()
    attempts = max_attempts or settings.llm_max_retries
    retryable = exceptions or (LLMError, SandboxError, ToolError)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=min_wait, max=max_wait),
                retry=retry_if_exception_type(retryable),
                reraise=True,
            ):
                with attempt:
                    return await func(*args, **kwargs)
            raise RuntimeError("Unreachable")

        return wrapper

    return decorator


def run_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine in a sync context, used by tools that may be sync."""
    return asyncio.get_event_loop().run_until_complete(coro) if asyncio.get_event_loop().is_running() else asyncio.run(coro)
