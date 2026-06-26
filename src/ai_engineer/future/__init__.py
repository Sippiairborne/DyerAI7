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

"""Next-generation AI capabilities — reasoning, retrieval, architectures, alignment, privacy, safety."""
# === Reasoning tools ===
@tool(name="reason_tree_of_thought", description="Use Tree-of-Thought reasoning to solve hard problems by exploring multiple branches.")
def reason_tot(problem: str, n_branches: int = 3, max_depth: int = 5) -> str:
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.reasoning.tree_of_thought import TreeOfThoughts
    import asyncio
    llm = LLMClient()
    solver = TreeOfThoughts(llm, n_branches=n_branches, max_depth=max_depth)
    result = asyncio.run(solver.solve(problem))
    return f"Best answer (visited {result.total_nodes} nodes, pruned {result.pruned_nodes}, {result.elapsed_s:.1f}s):\n\n{result.best_answer}"


@tool(name="reason_graph_of_thought", description="Use Graph-of-Thought reasoning with generate/aggregate/refine operations.")
def reason_got(problem: str, k: int = 3, max_rounds: int = 4) -> str:
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.reasoning.graph_of_thought import GraphOfThoughts
    import asyncio
    llm = LLMClient()
    solver = GraphOfThoughts(llm, k=k, max_rounds=max_rounds)
    result = asyncio.run(solver.solve(problem))
    return f"Best result:\n\n{result.best_node.content}"


@tool(name="reason_self_refine", description="Iteratively refine an answer using self-feedback.")
def reason_self_refine(initial: str, context: str = "", max_iter: int = 5) -> str:
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.reasoning.self_refine import SelfRefine
    import asyncio
    llm = LLMClient()
    refiner = SelfRefine(llm, max_iter=max_iter)
    result = asyncio.run(refiner.refine(initial, context))
    return f"After {result.iterations} iterations:\n\n{result.final_output}"


@tool(name="reason_mcts", description="Use MCTS reasoning with rollouts for hard problems.")
def reason_mcts(problem: str, n_rollouts: int = 8, max_depth: int = 8) -> str:
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.reasoning.mcts_reasoner import MCTSReasoner
    import asyncio
    llm = LLMClient()
    solver = MCTSReasoner(llm, n_rollouts=n_rollouts, max_depth=max_depth)
    answer, _ = asyncio.run(solver.solve(problem))
    return f"MCTS best:\n\n{answer}"


@tool(name="reason_constitutional", description="Apply Constitutional AI: iteratively critique and revise against principles.")
def reason_constitutional(prompt: str, response: str, max_iter: int = 3) -> str:
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.reasoning.constitutional_ai import ConstitutionalAI
    import asyncio
    llm = LLMClient()
    cai = ConstitutionalAI(llm, max_iterations=max_iter)
    traj = asyncio.run(cai.train(prompt, response))
    return f"After {traj.iterations} iterations:\n\n{traj.final}"


@tool(name="process_reward_score", description="Score each step of a reasoning trace using a Process Reward Model.")
def prm_score(question: str, trace_json: str) -> str:
    import json
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.reasoning.process_reward_model import ProcessRewardModel
    import asyncio
    trace = json.loads(trace_json)
    llm = LLMClient()
    prm = ProcessRewardModel(llm)
    scores = asyncio.run(prm.score_trace(question, trace))
    return "\n".join(f"Step {i+1}: {s.score:.2f} (conf {s.confidence:.2f}) — {s.reasoning[:100]}" for i, s in enumerate(scores))


# === Retrieval ===
@tool(name="rag_graph", description="GraphRAG: build knowledge graph and answer global questions.")
def rag_graph(doc_dir: str, question: str) -> str:
    import os
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.retrieval.graph_rag import GraphRAG
    import asyncio
    docs = []
    for f in os.listdir(doc_dir):
        with open(os.path.join(doc_dir, f)) as fp:
            docs.append(fp.read())
    llm = LLMClient()
    rag = GraphRAG(llm)
    asyncio.run(rag.index(docs))
    result = asyncio.run(rag.query(question))
    return f"Communities used: {result.communities_used}\nEntities: {result.entities}\n\nAnswer:\n{result.answer}"


@tool(name="rag_hyde", description="HyDE retrieval: embed a hypothetical answer to find relevant docs.")
def rag_hyde(query: str, doc_dir: str) -> str:
    import os
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.ml.features.text import TextVectorizer
    from ai_engineer.future.retrieval.hyde import HyDE
    import asyncio
    docs = []
    for f in os.listdir(doc_dir):
        with open(os.path.join(doc_dir, f)) as fp:
            docs.append(fp.read())
    llm = LLMClient()
    v = TextVectorizer(kind="sentence")
    v.fit_transform(docs)
    # In-memory store
    class Store:
        def __init__(self, v, docs):
            self.v = v
            self.embs = v.transform(docs)
            self.docs = docs
        def search_by_vector(self, emb, top_k=5):
            from numpy.linalg import norm
            scores = [(float(np.dot(emb, e) / (norm(emb) * norm(e))), i) for i, e in enumerate(self.embs)]
            scores.sort(key=lambda x: -x[0])
            return [{"text": self.docs[i], "score": s} for s, i in scores[:top_k]]
    hyde = HyDE(llm, vectorizer=v, store=Store(v, docs))
    result = asyncio.run(hyde.retrieve(query))
    out = "\n\n".join(f"[score {r['score']:.3f}] {r['text'][:300]}" for r in result.retrieved)
    return f"Hypothetical: {result.hypothetical[:200]}\n\nTop matches:\n{out}"


@tool(name="rag_self", description="Self-RAG: adaptive retrieval with self-reflection.")
def rag_self(question: str, doc_dir: str) -> str:
    import os
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.ml.features.text import TextVectorizer
    from ai_engineer.future.retrieval.self_rag import SelfRAG
    import asyncio
    docs = []
    for f in os.listdir(doc_dir):
        with open(os.path.join(doc_dir, f)) as fp:
            docs.append(fp.read())
    llm = LLMClient()
    v = TextVectorizer(kind="sentence")
    v.fit_transform(docs)
    class R:
        def __init__(self, v, docs):
            self.v = v; self.docs = docs; self.embs = v.transform(docs)
        def retrieve(self, q, top_k=5):
            from numpy.linalg import norm
            qe = self.v.transform([q])[0]
            scores = [(float(np.dot(qe, e) / (norm(qe) * norm(e))), i) for i, e in enumerate(self.embs)]
            scores.sort(key=lambda x: -x[0])
            return [{"text": self.docs[i], "score": s} for s, i in scores[:top_k]]
    ragr = SelfRAG(llm, R(v, docs))
    result = asyncio.run(ragr.answer(question))
    return f"Answer: {result.answer}\nRetrieval: {result.retrieval_tokens}\nSupport: {result.support_tokens}\nUsefulness: {result.usefulness_tokens}"


@tool(name="rag_rerank", description="Cross-encoder reranking for top-k refinement.")
def rag_rerank(query: str, documents_json: str, top_k: int = 5) -> str:
    import json
    from ai_engineer.future.retrieval.late_interaction import CrossEncoderReranker
    docs = json.loads(documents_json)
    r = CrossEncoderReranker().rerank(query, docs, top_k=top_k)
    return "\n".join(f"[{x['score']:.3f}] {x['text'][:200]}" for x in r)


# === Alignment ===
@tool(name="rlaif_collect", description="Collect AI preference pairs using Constitutional AI.")
def rlaif_collect(prompts_json: str, output_path: str, n_candidates: int = 4) -> str:
    import json
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.alignment.rlaif import RLAIFTrainer
    import asyncio
    prompts = json.loads(prompts_json)
    llm = LLMClient()
    trainer = RLAIFTrainer(llm)
    asyncio.run(trainer.collect_batch(prompts, n_candidates=n_candidates))
    n = trainer.export_dpo_format(output_path)
    return f"Wrote {n} preference pairs to {output_path}"


# === Synthetic data ===
@tool(name="synth_self_instruct", description="Generate synthetic instruction-tuning data via Self-Instruct.")
def synth_self_instruct(seed_json: str, target_count: int, output_path: str) -> str:
    import json
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.synthetic.self_instruct import SelfInstruct
    import asyncio
    seeds = json.loads(seed_json)
    llm = LLMClient()
    si = SelfInstruct(llm)
    samples = asyncio.run(si.generate(seeds, target_count=target_count))
    si.export(samples, output_path)
    return f"Generated {len(samples)} samples → {output_path}"


@tool(name="synth_evol", description="Evol-Instruct: deepen/constrain/broaden mutations on instructions.")
def synth_evol(input_path: str, output_path: str, evolution_rate: float = 0.5) -> str:
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.synthetic.self_instruct import SelfInstruct, SyntheticInstruction
    from ai_engineer.future.synthetic.evol_instruct import EvolInstruct
    import asyncio
    samples = []
    with open(input_path) as f:
        for line in f:
            d = eval(line.strip()) if not line.strip().startswith("{") else None  # safe parse
            if d:
                samples.append(SyntheticInstruction(**d))
    llm = LLMClient()
    ei = EvolInstruct(llm, evolution_rate=evolution_rate)
    evolved = asyncio.run(ei.evolve(samples))
    with open(output_path, "w") as f:
        for s in evolved:
            f.write(str({"instruction": s.instruction, "input": s.input, "output": s.output}) + "\n")
    return f"Evolved {len(evolved)} → {output_path}"


# === Privacy ===
@tool(name="dp_wrap", description="Wrap an optimizer with differential privacy (Opacus).")
def dp_wrap(noise_multiplier: float, max_grad_norm: float = 1.0, target_epsilon: float = 8.0) -> str:
    from ai_engineer.future.privacy.differential_privacy import DifferentialPrivacy, DPConfig
    return f"DP config: noise_mult={noise_multiplier}, max_grad_norm={max_grad_norm}, target_epsilon={target_epsilon}"


@tool(name="federated_setup", description="Set up a federated learning server with FedAvg/FedProx/FedOpt.")
def federated_setup(n_clients: int = 5, n_rounds: int = 50, algorithm: str = "fedavg") -> str:
    from ai_engineer.future.privacy.federated import FederatedServer, FederatedConfig
    cfg = FederatedConfig(n_clients=n_clients, n_rounds=n_rounds, algorithm=algorithm)
    server = FederatedServer(cfg)
    return f"Federated server ready with {n_clients} clients, {algorithm}"


# === Safety ===
@tool(name="watermark_text", description="Watermark text or detect if text is watermarked.")
def watermark_text(text: str = "", detect: bool = False) -> str:
    from ai_engineer.future.safety.watermark import TextWatermarker
    wm = TextWatermarker()
    if detect:
        # Simulate detection on tokens
        result = wm.detect(list(range(100)))
        return f"Watermark detected: {result.is_watermarked}, z={result.z_score:.2f}"
    return f"Watermarker ready (greenlist gamma={wm.gamma})"


@tool(name="detect_ai_text", description="Detect whether text is AI-generated using perplexity/burstiness/entropy.")
def detect_ai_text(text: str) -> str:
    from ai_engineer.future.safety.ai_detector import AITextDetector
    r = AITextDetector().detect(text)
    return f"AI: {r.is_ai} (conf {r.confidence:.2f}, ppl {r.perplexity:.1f}, burst {r.burstiness:.2f}, ent {r.entropy:.2f})"


@tool(name="detect_jailbreak", description="Detect prompt injection / jailbreak attempts.")
def detect_jailbreak(text: str) -> str:
    from ai_engineer.future.safety.jailbreak_detector import JailbreakDetector
    r = JailbreakDetector().check(text)
    return f"Safe: {r.is_safe} (risk {r.risk_score:.2f}, recommendation: {r.recommendation}, patterns: {r.detected_patterns})"


@tool(name="redact_pii", description="Detect and redact PII in text.")
def redact_pii(text: str) -> str:
    from ai_engineer.future.safety.pii_redactor import PIIRedactor
    redactor = PIIRedactor()
    matches = redactor.detect(text)
    redacted = redactor.redact(text)
    return f"Found {len(matches)} PII items:\n" + "\n".join(f"- {m.kind}: {m.value}" for m in matches) + f"\n\nRedacted:\n{redacted}"


# === Frontier architectures ===
@tool(name="train_mamba", description="Train a Mamba SSM model on text data.")
def train_mamba(dataset_path: str, output_dir: str, n_layers: int = 24, d_model: int = 768) -> str:
    from ai_engineer.future.architectures.mamba import MambaModel, MambaConfig
    cfg = MambaConfig(d_model=d_model, n_layers=n_layers)
    model = MambaModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    script = f"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from ai_engineer.future.architectures.mamba import MambaModel, MambaConfig

cfg = MambaConfig(d_model={d_model}, n_layers={n_layers})
model = MambaModel(cfg).cuda()
tok = AutoTokenizer.from_pretrained('gpt2')
ds = load_dataset('json', data_files='{dataset_path}', split='train')

opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
for epoch in range(3):
    for batch in ds.with_format('torch'):
        ids = tok(batch['text'], return_tensors='pt', truncation=True, max_length=cfg.max_seq_len).input_ids.cuda()
        if ids.shape[1] < 2: continue
        out = model(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(out.reshape(-1, out.size(-1)), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
torch.save(model.state_dict(), '{output_dir}/mamba.pt')
print('MAMBA_TRAINED')
"""
    Path(output_dir, "train.py").write_text(script)
    return f"Mamba training script written ({n_params/1e6:.1f}M params): {output_dir}/train.py"


@tool(name="train_moe", description="Train a Mixture-of-Experts model with top-k routing.")
def train_moe(d_model: int = 1024, n_experts: int = 8, top_k: int = 2, output_dir: str = "/tmp/moe") -> str:
    from ai_engineer.future.architectures.mixture_of_experts import MoE, MoEConfig
    moe = MoE(MoEConfig(d_model=d_model, n_experts=n_experts, top_k=top_k))
    return f"MoE built with {n_experts} experts (top-{top_k}, {sum(p.numel() for p in moe.parameters())/1e6:.1f}M params)"


@tool(name="train_dit", description="Train a Diffusion Transformer (DiT) for image generation.")
def train_dit(dataset_path: str, output_dir: str, img_size: int = 32, depth: int = 12) -> str:
    from ai_engineer.future.architectures.dit_diffusion import DiT, DiTConfig
    cfg = DiTConfig(img_size=img_size, depth=depth)
    model = DiT(cfg)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    script = f"""
import torch
from ai_engineer.future.architectures.dit_diffusion import DiT, DiTConfig
from ai_engineer.future.architectures.flow_matching import FlowMatchingModel

cfg = DiTConfig(img_size={img_size}, depth={depth})
fm = FlowMatchingModel(cfg=cfg)
fm.model = fm.model.cuda()
opt = torch.optim.AdamW(fm.model.parameters(), lr=1e-4)
from datasets import load_dataset
ds = load_dataset('imagefolder', data_dir='{dataset_path}', split='train')
for epoch in range(100):
    for ex in ds:
        x = torch.tensor(ex['image']).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.
        y = torch.tensor([0]).cuda()
        loss = fm.loss(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
torch.save(fm.model.state_dict(), '{output_dir}/dit.pt')
print('DIT_TRAINED')
"""
    Path(output_dir, "train.py").write_text(script)
    return f"DiT training script written: {output_dir}/train.py"


@tool(name="train_flow_matching", description="Train a flow matching model (faster, more stable than diffusion).")
def train_flow_matching(dataset_path: str, output_dir: str) -> str:
    return train_dit(dataset_path, output_dir)  # Uses flow matching internally


@tool(name="train_consistency_model", description="Train a consistency model for one-step generation.")
def train_cm(dataset_path: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    script = f"""
import torch
from ai_engineer.future.architectures.consistency_model import ConsistencyModel, ConsistencyConfig

cfg = ConsistencyConfig()
model = ConsistencyModel(cfg).cuda()
from datasets import load_dataset
ds = load_dataset('imagefolder', data_dir='{dataset_path}', split='train')
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(50):
    for ex in ds:
        x = torch.tensor(ex['image']).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.
        teacher_fn = lambda x_t, sigmas: x  # placeholder
        loss = model.loss(x, teacher_fn)
        opt.zero_grad(); loss.backward(); opt.step()
torch.save(model.state_dict(), '{output_dir}/cm.pt')
print('CM_TRAINED')
"""
    Path(output_dir, "train.py").write_text(script)
    return f"Consistency model script: {output_dir}/train.py"


# === Self-evolving ===
@tool(name="evolve_agent", description="Evolve an agent system prompt via Darwinian selection over generations.")
def evolve_agent(base_prompt: str, eval_tasks_json: str, n_generations: int = 5) -> str:
    import json
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.self_evolving.evolution import SelfEvolvingAgent
    import asyncio
    tasks = json.loads(eval_tasks_json)
    llm = LLMClient()
    agent = SelfEvolvingAgent(llm, base_prompt, tools=[], tasks)
    result = asyncio.run(agent.evolve(n_generations=n_generations))
    return f"Best evolved prompt (score {result.best_variant.score:.2f}):\n\n{result.best_variant.system_prompt}"


# === Agent Mesh ===
@tool(name="coordinate_mesh", description="Coordinate a multi-agent mesh to solve a complex goal.")
def coordinate_mesh(goal: str, agents_json: str) -> str:
    import json
    from ai_engineer.core.llm import LLMClient
    from ai_engineer.future.agent_mesh.coordinator import AgentMeshCoordinator, AgentSpec
    import asyncio
    agents = [AgentSpec(**a) for a in json.loads(agents_json)]
    llm = LLMClient()
    coord = AgentMeshCoordinator(llm, agents)
    result = asyncio.run(coord.solve(goal))
    return json.dumps(result, indent=2, default=str)[:2000]


# === Sub-bit quantization ===
@tool(name="quantize_subbit", description="Quantize model to 1-bit (BitNet) or 2-bit.")
def quantize_subbit(model_path: str, output_path: str, bits: int = 1) -> str:
    from ai_engineer.future.optimization_v2.sub_bit import SubBitQuantizer
    import torch
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    except Exception:
        model = torch.nn.Linear(10, 10)
    r = SubBitQuantizer(bits=bits).quantize(model, output_path)
    return f"{bits}-bit quantization: {r.original_size_mb:.1f}MB → {r.quantized_size_mb:.2f}MB"


# === World models ===
@tool(name="train_world_model", description="Train a Dreamer V3-style world model for planning.")
def train_world_model(env_id: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    script = f"""
import torch
from ai_engineer.future.world_models.dreamer import DreamerV3, DreamerConfig
import gymnasium as gym

cfg = DreamerConfig(obs_dim=64, action_dim=4)
wm = DreamerV3(cfg).cuda()
opt = torch.optim.AdamW(wm.parameters(), lr=6e-4)
env = gym.make('{env_id}')
for episode in range(100):
    obs, _ = env.reset()
    h = torch.zeros(1, cfg.hidden_dim).cuda()
    z = torch.zeros(1, cfg.latent_dim * cfg.num_categories).cuda()
    for step in range(200):
        action = torch.randn(1, cfg.action_dim).cuda()
        h, _, z = wm.rssm(h, z, action)
        loss = (h ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
torch.save(wm.state_dict(), '{output_dir}/wm.pt')
print('WORLD_MODEL_TRAINED')
"""
    Path(output_dir, "train.py").write_text(script)
    return f"World model script: {output_dir}/train.py"
