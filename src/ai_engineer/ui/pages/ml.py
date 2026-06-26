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

"""ML pipeline UI — full lifecycle from data to deployment."""
from __future__ import annotations

import json

import requests
import streamlit as st


def render() -> None:
    st.header("🧪 ML Pipeline")
    tabs = st.tabs(["Data", "Features", "Train", "Evaluate", "Optimize", "Deploy", "Monitor"])

    api = st.session_state.api_url

    with tabs[0]:
        st.subheader("Data Profiling")
        path = st.text_input("Data path", key="ml_data_path")
        target = st.text_input("Target column", key="ml_data_target")
        if st.button("Profile", key="profile_btn"):
            r = requests.post(f"{api}/api/ml/data/profile", json={"path": path, "target": target}, timeout=60)
            if r.status_code == 200:
                report = r.json()
                st.json({k: v for k, v in report.items() if k != "sample"})
                st.write(f"Rows: {report['n_rows']:,}, Cols: {report['n_columns']}")
            else:
                st.error(r.text)

    with tabs[1]:
        st.subheader("Feature Engineering")
        in_p = st.text_input("Input path", key="fe_in")
        out_p = st.text_input("Output path", key="fe_out")
        if st.button("Engineer", key="fe_btn"):
            r = requests.post(f"{api}/api/ml/features/engineer", json={"input_path": in_p, "output_path": out_p, "target": target}, timeout=120)
            if r.status_code == 200:
                st.success(r.json())
            else:
                st.error(r.text)

    with tabs[2]:
        st.subheader("Training")
        task = st.selectbox("Task", ["classification", "regression"], key="task_sel")
        library = st.selectbox("Library", ["lightgbm", "xgboost", "catboost"], key="lib_sel")
        in_p = st.text_input("Train data", key="tr_in")
        target = st.text_input("Target", key="tr_target", value=target)
        reg = st.text_input("Register name (optional)", key="tr_reg")
        if st.button("Train", key="train_btn"):
            r = requests.post(f"{api}/api/ml/train/classical", json={"input_path": in_p, "target": target, "library": library, "task": task, "register_name": reg}, timeout=600)
            if r.status_code == 200:
                d = r.json()
                st.success(f"Trained! CV: {d['metrics']}")
                st.json(d.get("top_features"))
            else:
                st.error(r.text)

    with tabs[3]:
        st.subheader("Evaluation")
        mp = st.text_input("Model path", key="ev_mp")
        dp = st.text_input("Dataset path", key="ev_dp")
        tgt = st.text_input("Target", key="ev_tgt", value=target)
        sens = st.text_input("Sensitive column (optional)", key="ev_sens")
        if st.button("Evaluate", key="ev_btn"):
            r = requests.post(f"{api}/api/ml/evaluate", json={"model_path": mp, "dataset_path": dp, "target": tgt, "sensitive_column": sens, "task": task}, timeout=300)
            if r.status_code == 200:
                d = r.json()
                st.json(d["metrics"])
                st.warning(d.get("fairness_warnings", []))
                st.markdown(f"[Open report]({d['report']})")
            else:
                st.error(r.text)

    with tabs[4]:
        st.subheader("Optimization")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Hyperparameter Tuning**")
            n_trials = st.number_input("Trials", 10, 500, 50)
            if st.button("Run HPO"):
                r = requests.post(f"{api}/api/ml/tune", json={"input_path": in_p, "target": target, "n_trials": int(n_trials)}, timeout=1800)
                if r.status_code == 200:
                    st.json(r.json())
        with c2:
            st.markdown("**Quantization**")
            qm = st.text_input("Model to quantize", key="qmodel")
            qm_out = st.text_input("Output path", key="qmodel_out")
            qmethod = st.selectbox("Method", ["bnb_4bit", "bnb_8bit", "dynamic", "ptq"], key="qmethod")
            if st.button("Quantize", key="qbtn"):
                r = requests.post(f"{api}/api/ml/optimize/quantize", json={"model_path": qm, "output_path": qm_out, "method": qmethod}, timeout=600)
                if r.status_code == 200:
                    st.json(r.json())

    with tabs[5]:
        st.subheader("Deployment")
        mp = st.text_input("Model to deploy", key="dep_mp")
        fw = st.selectbox("Framework", ["vllm", "tgi", "triton", "fastapi", "bentoml", "ray"], key="dep_fw")
        port = st.number_input("Port", 8000, 65535, 8000)
        if st.button("Deploy", key="dep_btn"):
            r = requests.post(f"{api}/api/ml/deploy_serving", json={"model_path": mp, "framework": fw, "port": int(port)})
            st.json(r.json())

    with tabs[6]:
        st.subheader("Monitoring")
        ref = st.text_input("Reference data", key="ref_p")
        cur = st.text_input("Current data", key="cur_p")
        if st.button("Detect drift"):
            r = requests.post(f"{api}/api/ml/monitor/drift", json={"reference_path": ref, "current_path": cur}, timeout=60)
            st.json(r.json())
