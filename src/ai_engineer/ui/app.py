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

"""Main Streamlit application."""
from __future__ import annotations

import os

import requests
import streamlit as st

from ai_engineer.ui.components.sidebar import render_sidebar
from ai_engineer.ui.pages import dashboard, evaluation, projects, settings as settings_page, training
from ai_engineer.ui.state import init_state

st.set_page_config(page_title="AI Engineer", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

init_state()
render_sidebar()

API_URL = os.environ.get("UI_API_URL", "http://localhost:8080")
st.session_state.api_url = API_URL

page = st.session_state.get("page", "Dashboard")
if page == "Dashboard":
    dashboard.render()
elif page == "Projects":
    projects.render()
elif page == "Training":
    training.render()
elif page == "Evaluation":
    evaluation.render()
elif page == "Settings":
    settings_page.render()
