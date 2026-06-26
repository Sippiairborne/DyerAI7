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

"""Metrics dashboard."""
from __future__ import annotations

import streamlit as st


def render_metrics(metrics: dict[str, float]) -> None:
    if not metrics:
        st.info("No metrics yet")
        return
    cols = st.columns(min(4, len(metrics)))
    for i, (k, v) in enumerate(metrics.items()):
        with cols[i % len(cols)]:
            st.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))
