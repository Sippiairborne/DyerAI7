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
from fastapi.testclient import TestClient


def test_health() -> None:
    from ai_engineer.api.server import app
    from ai_engineer.api.deps import get_state

    get_state()  # init
    with TestClient(app) as client:
        # We can't fully test without orchestrator init, so just ensure it doesn't crash on import
        assert app.title == "AI Engineer"
