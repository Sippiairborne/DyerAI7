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

"""Future capabilities API routes."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ai_engineer.api.deps import get_state

router = APIRouter(prefix="/api/future", tags=["future"])


# === Reasoning ===
class ReasonRequest(BaseModel):
    method: str  # tot | got | self_refine | mcts | constitutional
    problem: str
    params: dict = Field(default_factory=dict)


@router.post("/reason")
async def reason(req: ReasonRequest) -> dict:
    state = get_state()
    if not state.llm:
        raise HTTPException(503, "LLM not initialized")
    method = req.method
    if method == "tot":
        from ai_engineer.future.reasoning.tree_of_thought import TreeOfThoughts
        solver = TreeOfThoughts(state.llm, **req.params)
        r = await solver.solve(req.problem)
        return {"best_answer": r.best_answer, "nodes": r.total_nodes, "pruned": r.pruned_nodes, "elapsed_s": r.elapsed_s}
    if method == "got":
        from ai_engineer.future.reasoning.graph_of_thought import GraphOfThoughts
        solver = GraphOfThoughts(state.llm, **req.params)
        r = await solver.solve(req.problem)
        return {"best": r.best_node.content, "rounds": r.rounds}
    if method == "self_refine":
        from ai_engineer.future.reasoning.self_refine import SelfRefine
        refiner = SelfRefine(state.llm)
        r = await refiner.refine(req.problem)
        return {"final": r.final_output, "iterations": r.iterations}
    if method == "mcts":
        from ai_engineer.future.reasoning.mcts_reasoner import MCTSReasoner
        solver = MCTSReasoner(state.llm, **req.params)
        ans, _ = await solver.solve(req.problem)
        return {"answer": ans}
    if method == "constitutional":
        from ai_engineer.future.reasoning.constitutional_ai import ConstitutionalAI
        cai = ConstitutionalAI(state.llm)
        traj = await cai.train(req.problem, req.params.get("response", ""))
        return {"final": traj.final, "iterations": traj.iterations}
    raise HTTPException(400, f"Unknown method: {method}")


# === Retrieval ===
class GraphRAGRequest(BaseModel):
    doc_dir: str
    question: str


@router.post("/rag/graph")
async def rag_graph(req: GraphRAGRequest) -> dict:
    state = get_state()
    if not state.llm:
        raise HTTPException(503, "LLM not initialized")
    from ai_engineer.future.retrieval.graph_rag import GraphRAG
    docs = []
    for f in Path(req.doc_dir).iterdir():
        if f.is_file():
            docs.append(f.read_text(errors="replace"))
    rag = GraphRAG(state.llm)
    await rag.index(docs)
    r = await rag.query(req.question)
    return {"answer": r.answer, "entities": r.entities, "communities": r.communities_used}


class HydeRequest(BaseModel):
    query: str
    doc_dir: str


@router.post("/rag/hyde")
async def rag_hyde(req: HydeRequest) -> dict:
    state = get_state()
    from ai_engineer.future.retrieval.hyde import HyDE
    from ai_engineer.ml.features.text import TextVectorizer
    import numpy as np
    docs = []
    for f in Path(req.doc_dir).iterdir():
        if f.is_file():
            docs.append(f.read_text(errors="replace"))
    v = TextVectorizer(kind="sentence")
    v.fit_transform(docs)
    embs = v.transform(docs)
    class S:
        def search_by_vector(self, emb, top_k=5):
            sc = [(float(np.dot(emb, e) / max(np.linalg.norm(emb) * np.linalg.norm(e), 1e-9)), i) for i, e in enumerate(embs)]
            sc.sort(key=lambda x: -x[0])
            return [{"text": docs[i], "score": s} for s, i in sc[:top_k]]
    hyde = HyDE(state.llm, vectorizer=v, store=S())
    r = await hyde.retrieve(req.query)
    return {"hypothetical": r.hypothetical, "retrieved": r.retrieved}


# === Safety ===
class SafetyRequest(BaseModel):
    text: str


@router.post("/safety/jailbreak")
async def jailbreak(req: SafetyRequest) -> dict:
    from ai_engineer.future.safety.jailbreak_detector import JailbreakDetector
    r = JailbreakDetector(state.llm if get_state().llm else None).check(req.text)
    return {"is_safe": r.is_safe, "risk_score": r.risk_score, "patterns": r.detected_patterns, "recommendation": r.recommendation}


@router.post("/safety/ai-detect")
async def ai_detect(req: SafetyRequest) -> dict:
    from ai_engineer.future.safety.ai_detector import AITextDetector
    r = AITextDetector().detect(req.text)
    return {"is_ai": r.is_ai, "confidence": r.confidence, "perplexity": r.perplexity, "burstiness": r.burstiness, "entropy": r.entropy}


@router.post("/safety/redact")
async def redact(req: SafetyRequest) -> dict:
    from ai_engineer.future.safety.pii_redactor import PIIRedactor
    redactor = PIIRedactor()
    matches = redactor.detect(req.text)
    return {"matches": [{"kind": m.kind, "value": m.value} for m in matches], "redacted": redactor.redact(req.text)}


# === Alignment ===
class RLAIFRequest(BaseModel):
    prompts: list[str]
    output_path: str
    n_candidates: int = 4


@router.post("/alignment/rlaif")
async def rlaif(req: RLAIFRequest) -> dict:
    state = get_state()
    from ai_engineer.future.alignment.rlaif import RLAIFTrainer
    trainer = RLAIFTrainer(state.llm)
    await trainer.collect_batch(req.prompts, n_candidates=req.n_candidates)
    n = trainer.export_dpo_format(req.output_path)
    return {"n_examples": n, "path": req.output_path}


# === Privacy ===
class DPRequest(BaseModel):
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    target_epsilon: float = 8.0


@router.post("/privacy/dp-config")
async def dp_config(req: DPRequest) -> dict:
    return {"status": "ok", "config": req.model_dump()}


# === Architectures ===
class ArchRequest(BaseModel):
    arch: str  # mamba | moe | dit | flow_matching | consistency
    dataset_path: str
    output_dir: str
    params: dict = Field(default_factory=dict)


@router.post("/train/architecture")
async def train_arch(req: ArchRequest) -> dict:
    if req.arch == "mamba":
        from ai_engineer.future.architectures.mamba import MambaModel, MambaConfig
        cfg = MambaConfig(**req.params) if req.params else MambaConfig()
        model = MambaModel(cfg)
        n = sum(p.numel() for p in model.parameters())
        Path(req.output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"{req.output_dir}/init.pt")
        return {"arch": "mamba", "params": n}
    if req.arch == "moe":
        from ai_engineer.future.architectures.mixture_of_experts import MoE, MoEConfig
        moe = MoE(MoEConfig(**req.params))
        return {"arch": "moe", "params": sum(p.numel() for p in moe.parameters())}
    if req.arch in ("dit", "flow_matching"):
        from ai_engineer.future.architectures.dit_diffusion import DiT, DiTConfig
        cfg = DiTConfig(**req.params) if req.params else DiTConfig()
        model = DiT(cfg)
        return {"arch": req.arch, "params": sum(p.numel() for p in model.parameters())}
    raise HTTPException(400, f"Unknown arch: {req.arch}")


# === Self-evolving ===
class EvolveRequest(BaseModel):
    base_prompt: str
    eval_tasks: list[str]
    n_generations: int = 5


@router.post("/self/evolve")
async def evolve(req: EvolveRequest) -> dict:
    state = get_state()
    from ai_engineer.future.self_evolving.evolution import SelfEvolvingAgent
    agent = SelfEvolvingAgent(state.llm, req.base_prompt, tools=[], eval_tasks=req.eval_tasks)
    r = await agent.evolve(n_generations=req.n_generations)
    return {"best_score": r.best_variant.score, "best_prompt": r.best_variant.system_prompt, "generations": r.generations}


# === Agent Mesh ===
class MeshRequest(BaseModel):
    goal: str
    agents: list[dict]


@router.post("/mesh/solve")
async def mesh_solve(req: MeshRequest) -> dict:
    state = get_state()
    from ai_engineer.future.agent_mesh.coordinator import AgentMeshCoordinator, AgentSpec
    coords = AgentMeshCoordinator(state.llm, [AgentSpec(**a) for a in req.agents])
    return await coords.solve(req.goal)
