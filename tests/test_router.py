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

from ai_engineer.core.llm import LLMClient
from ai_engineer.core.router import ModelRouter


def test_route_code() -> None:
    r = ModelRouter(LLMClient())
    d = r.route("code", 0.5)
    assert d.model


def test_route_fast() -> None:
    r = ModelRouter(LLMClient())
    d = r.route("fast", 0.1)
    assert d.model
