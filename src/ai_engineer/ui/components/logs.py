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

"""Live logs component."""
from __future__ import annotations

import json

import streamlit as st


def render_logs(events: list[dict]) -> None:
    st.subheader("📜 Live Logs")
    placeholder = st.empty()
    text = ""
    for e in events[-100:]:
        text += f"[{e.get('type', '')}] {json.dumps(e)[:300]}\n"
    placeholder.code(text, language="json")
