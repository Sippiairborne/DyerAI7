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

"""Training page."""
from __future__ import annotations

import requests
import streamlit as st


def render() -> None:
    st.header("🚀 Training")
    with st.form("train"):
        st.text_input("Model name", value="meta-llama/Llama-3.2-3B-Instruct", key="model_name")
        st.text_input("Dataset path", key="dataset_path")
        st.text_input("Output dir", value="/data/models/run1", key="output_dir")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Epochs", 1, 100, 3, key="epochs")
        c2.number_input("Batch size", 1, 256, 4, key="bs")
        c3.number_input("Learning rate", value=2e-5, format="%.1e", key="lr")
        st.checkbox("Use LoRA", value=True, key="lora")
        if st.form_submit_button("Start Training"):
            try:
                r = requests.post(
                    f"{st.session_state.api_url}/api/training/start",
                    json={
                        "model_name": st.session_state.model_name,
                        "dataset_path": st.session_state.dataset_path,
                        "output_dir": st.session_state.output_dir,
                        "num_epochs": st.session_state.epochs,
                        "batch_size": st.session_state.bs,
                        "learning_rate": st.session_state.lr,
                        "use_lora": st.session_state.lora,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                st.success(f"Started: {r.json()}")
            except Exception as e:
                st.error(str(e))
