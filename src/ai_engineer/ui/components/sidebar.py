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

"""Sidebar navigation."""
from __future__ import annotations

import streamlit as st

PAGES = ["Dashboard", "Projects", "Training", "Evaluation", "Settings"]


def render_sidebar() -> None:
    with st.sidebar:
        st.title("🧠 AI Engineer")
        st.caption("v1.0.0")
        st.divider()
        page = st.radio("Navigation", PAGES, index=PAGES.index(st.session_state.get("page", "Dashboard")))
        st.session_state.page = page
        st.divider()
        # Health check
        import requests

        try:
            r = requests.get(f"{st.session_state.api_url}/api/system/health", timeout=2)
            if r.status_code == 200:
                st.success("● API Connected")
            else:
                st.error("● API Error")
        except Exception:
            st.error("● API Offline")
