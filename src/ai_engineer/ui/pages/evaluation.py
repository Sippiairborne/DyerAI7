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

"""Evaluation page."""
from __future__ import annotations

import requests
import streamlit as st


def render() -> None:
    st.header("📊 Evaluation")
    with st.form("eval"):
        st.text_input("Model path", key="eval_model")
        st.text_input("Dataset path", key="eval_dataset")
        st.text_input("Metrics (comma-separated)", value="accuracy,f1", key="eval_metrics")
        if st.form_submit_button("Run Evaluation"):
            try:
                r = requests.post(
                    f"{st.session_state.api_url}/api/evaluation/run",
                    json={
                        "model_path": st.session_state.eval_model,
                        "dataset_path": st.session_state.eval_dataset,
                        "metrics": [m.strip() for m in st.session_state.eval_metrics.split(",")],
                    },
                    timeout=3600,
                )
                r.raise_for_status()
                st.json(r.json())
            except Exception as e:
                st.error(str(e))
