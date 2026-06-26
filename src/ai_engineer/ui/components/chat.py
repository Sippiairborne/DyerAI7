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

"""Chat interface for submitting tasks."""
from __future__ import annotations

import json
import time

import requests
import streamlit as st


def render_chat() -> None:
    st.subheader("💬 Chat with AI Engineer")

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("events"):
                with st.expander("Events"):
                    for e in msg["events"][-20:]:
                        st.json(e)

    if prompt := st.chat_input("Describe your ML task..."):
        st.session_state.setdefault("messages", []).append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                r = requests.post(
                    f"{st.session_state.api_url}/api/tasks",
                    json={"goal": prompt, "stream": False},
                    timeout=10,
                )
                r.raise_for_status()
                task = r.json()
                st.session_state.current_task = task["id"]
                placeholder.success(f"Task submitted: `{task['id']}`")
            except Exception as e:
                placeholder.error(f"Failed: {e}")
