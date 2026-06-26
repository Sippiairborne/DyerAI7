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

"""Tool registry — turns Python functions into LLM-callable tools."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from ai_engineer.utils.errors import ToolError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]


def tool(name: str | None = None, description: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator marking a function as a tool."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            t = hints.get(pname, str)
            tname = getattr(t, "__name__", str(t))
            schema_type = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
            }.get(tname, "string")
            properties[pname] = {"type": schema_type, "description": param.annotation.__doc__ or ""}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        func.__tool_spec__ = ToolSpec(
            name=name or func.__name__,
            description=description or (func.__doc__ or "").strip(),
            parameters={"type": "object", "properties": properties, "required": required},
            func=func,
        )
        return func

    return decorator


class ToolRegistry:
    """Holds all available tools and renders them for the LLM."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def decorate(self, func: Callable[..., Any]) -> Callable[..., Any]:
        spec = getattr(func, "__tool_spec__", None)
        if spec:
            self.register(spec)
        return func

    def add(self, func: Callable[..., Any]) -> None:
        self.decorate(func)

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        return self._tools[name]

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def as_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": s.name, "description": s.description, "parameters": s.parameters}}
            for s in self._tools.values()
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        spec = self.get(name)
        try:
            result = spec.func(**arguments)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as e:
            raise ToolError(f"Tool {name} failed: {e}") from e
