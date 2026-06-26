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

"""Projects page."""
from __future__ import annotations

import requests
import streamlit as st


def render() -> None:
    st.header("📁 Projects")
    with st.form("new_project"):
        name = st.text_input("Project name")
        desc = st.text_area("Description")
        if st.form_submit_button("Create"):
            try:
                r = requests.post(
                    f"{st.session_state.api_url}/api/projects",
                    params={"name": name, "description": desc},
                    timeout=10,
                )
                r.raise_for_status()
                st.success("Created")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Existing Projects")
    try:
        r = requests.get(f"{st.session_state.api_url}/api/projects", timeout=5)
        if r.status_code == 200:
            for p in r.json():
                with st.expander(p["name"]):
                    st.write(p["description"])
                    st.caption(f"Created: {p['created_at']}")
    except Exception as e:
        st.error(str(e))
