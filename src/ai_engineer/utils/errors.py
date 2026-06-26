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

"""Custom exception hierarchy."""
from __future__ import annotations


class AIEngineerError(Exception):
    """Base exception for the system."""


class LLMError(AIEngineerError):
    """LLM provider failed."""


class ToolError(AIEngineerError):
    """A tool execution failed."""


class SandboxError(AIEngineerError):
    """Sandbox/container failed."""


class PlanningError(AIEngineerError):
    """Planning step failed."""


class MemoryError_(AIEngineerError):
    """Memory subsystem failed."""


class PipelineError(AIEngineerError):
    """Pipeline execution failed."""


class ValidationError(AIEngineerError):
    """Input validation failed."""


class ResourceError(AIEngineerError):
    """Resource exhausted (GPU, memory, etc.)."""
