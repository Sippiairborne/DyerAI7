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

"""Next-gen capabilities page."""
from __future__ import annotations

import json

import requests
import streamlit as st


def render() -> None:
    st.header("🚀 Next-Gen AI")
    tabs = st.tabs(["Reasoning", "Retrieval 2.0", "Alignment", "Privacy", "Safety", "Frontier Archs", "Self-Evolving", "Agent Mesh"])

    api = st.session_state.api_url

    with tabs[0]:
        st.subheader("Advanced Reasoning")
        method = st.selectbox("Method", ["tot", "got", "self_refine", "mcts", "constitutional"], key="reason_m")
        problem = st.text_area("Problem", height=150, key="reason_p")
        if st.button("Solve", key="reason_btn"):
            r = requests.post(f"{api}/api/future/reason", json={"method": method, "problem": problem, "params": {}}, timeout=120)
            if r.status_code == 200:
                d = r.json()
                key = "best_answer" if "best_answer" in d else "answer" if "answer" in d else "final" if "final" in d else "best"
                st.markdown(f"### {key}\n{d.get(key, str(d))}")
                with st.expander("Details"):
                    st.json(d)
            else:
                st.error(r.text)

    with tabs[1]:
        st.subheader("Retrieval 2.0")
        rag_method = st.selectbox("RAG", ["graph", "hyde"], key="rag_m")
        doc_dir = st.text_input("Document directory", key="rag_dir")
        query = st.text_input("Question", key="rag_q")
        if st.button("Query", key="rag_btn"):
            endpoint = "/api/future/rag/graph" if rag_method == "graph" else "/api/future/rag/hyde"
            r = requests.post(f"{api}{endpoint}", json={"doc_dir": doc_dir, "question" if rag_method == "graph" else "query": query}, timeout=120)
            if r.status_code == 200:
                d = r.json()
                st.markdown(f"### Answer\n{d.get('answer', d.get('hypothetical', str(d)))}")
                with st.expander("Details"):
                    st.json(d)
            else:
                st.error(r.text)

    with tabs[2]:
        st.subheader("RLAIF: Generate AI Preference Data")
        prompts = st.text_area("Prompts (one per line)", key="rlaif_p").split("\n")
        out = st.text_input("Output path", value="/tmp/dpo_data.jsonl", key="rlaif_out")
        if st.button("Collect", key="rlaif_btn"):
            r = requests.post(f"{api}/api/future/alignment/rlaif", json={"prompts": [p for p in prompts if p.strip()], "output_path": out, "n_candidates": 4}, timeout=600)
            if r.status_code == 200:
                st.success(r.json())
            else:
                st.error(r.text)

    with tabs[3]:
        st.subheader("Differential Privacy")
        c1, c2, c3 = st.columns(3)
        noise = c1.number_input("Noise multiplier", 0.1, 10.0, 1.0)
        grad_norm = c2.number_input("Max grad norm", 0.1, 10.0, 1.0)
        eps = c3.number_input("Target epsilon", 0.1, 100.0, 8.0)
        if st.button("Apply DP"):
            st.success(f"DP config set: noise={noise}, grad_norm={grad_norm}, ε={eps}")

    with tabs[4]:
        st.subheader("Safety & Detection")
        text = st.text_area("Text to analyze", height=150, key="safe_text")
        col1, col2, col3 = st.columns(3)
        if col1.button("Jailbreak check"):
            r = requests.post(f"{api}/api/future/safety/jailbreak", json={"text": text}, timeout=30)
            if r.status_code == 200:
                d = r.json()
                color = "green" if d["is_safe"] else "red"
                st.markdown(f":{color}[{'SAFE' if d['is_safe'] else 'UNSAFE'}] Risk: {d['risk_score']:.2f} | Rec: {d['recommendation']}")
                st.json(d)
        if col2.button("AI detection"):
            r = requests.post(f"{api}/api/future/safety/ai-detect", json={"text": text}, timeout=30)
            if r.status_code == 200:
                d = r.json()
                st.json(d)
        if col3.button("Redact PII"):
            r = requests.post(f"{api}/api/future/safety/redact", json={"text": text}, timeout=30)
            if r.status_code == 200:
                d = r.json()
                st.text(d["redacted"])

    with tabs[5]:
        st.subheader("Frontier Architectures")
        arch = st.selectbox("Architecture", ["mamba", "moe", "dit", "flow_matching"], key="arch_sel")
        ds = st.text_input("Dataset path", key="arch_ds")
        out = st.text_input("Output dir", key="arch_out")
        if st.button("Build", key="arch_btn"):
            r = requests.post(f"{api}/api/future/train/architecture", json={"arch": arch, "dataset_path": ds, "output_dir": out, "params": {}}, timeout=60)
            st.json(r.json())

    with tabs[6]:
        st.subheader("Self-Evolving Agent")
        base = st.text_area("Base system prompt", height=100, key="evo_base")
        tasks = st.text_area("Evaluation tasks (one per line)", key="evo_tasks").split("\n")
        gens = st.number_input("Generations", 1, 20, 5)
        if st.button("Evolve", key="evo_btn"):
            r = requests.post(f"{api}/api/future/self/evolve", json={"base_prompt": base, "eval_tasks": [t for t in tasks if t.strip()], "n_generations": int(gens)}, timeout=1800)
            if r.status_code == 200:
                d = r.json()
                st.success(f"Best score: {d['best_score']:.2f}")
                st.text_area("Best evolved prompt", d["best_prompt"], height=300)
            else:
                st.error(r.text)

    with tabs[7]:
        st.subheader("Multi-Agent Mesh")
        agents_json = st.text_area("Agents (JSON)", value='[{"name": "researcher", "role": "research", "system_prompt": "You are a research agent.", "tools": []}, {"name": "coder", "role": "implementation", "system_prompt": "You are an expert programmer.", "tools": []}]', key="mesh_agents")
        goal = st.text_area("Goal", key="mesh_goal")
        if st.button("Solve", key="mesh_btn"):
            try:
                agents = json.loads(agents_json)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                return
            r = requests.post(f"{api}/api/future/mesh/solve", json={"goal": goal, "agents": agents}, timeout=600)
            if r.status_code == 200:
                st.json(r.json())
            else:
                st.error(r.text)
