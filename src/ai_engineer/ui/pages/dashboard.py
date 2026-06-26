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

"""Dashboard page."""
from __future__ import annotations

import time

import requests
import streamlit as st

from ai_engineer.ui.components.chat import render_chat
from ai_engineer.ui.components.logs import render_logs


def render() -> None:
    st.header("🏠 Dashboard")
    render_chat()

    task_id = st.session_state.get("current_task")
    if task_id:
        st.divider()
        st.subheader(f"Task: `{task_id}`")
        col1, col2 = st.columns([3, 1])
        with col1:
            events_placeholder = st.empty()
        with col2:
            if st.button("Refresh"):
                st.rerun()

        try:
            r = requests.get(f"{st.session_state.api_url}/api/tasks/{task_id}/events", timeout=5)
            events = r.json() if r.status_code == 200 else []
            render_logs(events)
            task = requests.get(f"{st.session_state.api_url}/api/tasks/{task_id}", timeout=5).json()
            status = task.get("status", "unknown")
            if status in ("done", "failed"):
                st.success(f"Status: {status}")
                if task.get("result"):
                    st.json(task["result"])
                st.session_state.current_task = None
        except Exception as e:
            st.error(f"Error: {e}")
        time.sleep(1)
        st.rerun()
