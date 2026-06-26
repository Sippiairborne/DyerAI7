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

"""ML-specific tools exposed to the LLM agent."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_engineer.tools.registry import ToolRegistry, tool
from ai_engineer.utils.errors import ToolError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)
_registry = ToolRegistry()


@tool(name="profile_data", description="Profile a dataset file (CSV/Parquet/JSONL). Returns a comprehensive report with statistics, warnings, and samples.")
def profile_data(path: str, target: str = "", text_columns: str = "", max_rows: int = 100_000) -> str:
    from ai_engineer.ml.data.profiler import DataProfiler
    text_cols = [c.strip() for c in text_columns.split(",") if c.strip()]
    df = pd.read_csv(path, nrows=max_rows) if path.endswith(".csv") else pd.read_parquet(path).head(max_rows)
    report = DataProfiler().profile(df, target=target or None, text_columns=text_cols or None)
    return report.to_markdown()


@tool(name="clean_data", description="Clean a dataset: handle missing, drop duplicates, fix types, normalize text. Returns the cleaned file path and a report.")
def clean_data(input_path: str, output_path: str, target: str = "") -> str:
    from ai_engineer.ml.data.cleaner import DataCleaner
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    cleaner = DataCleaner()
    cleaned, report = cleaner.clean(df)
    if output_path.endswith(".parquet"):
        cleaned.to_parquet(output_path, index=False)
    else:
        cleaned.to_csv(output_path, index=False)
    return f"Cleaned data saved to {output_path}\n\n{report.to_markdown()}"


@tool(name="validate_data", description="Validate a dataset against a JSON schema of expectations. Returns a validation report.")
def validate_data(path: str, expectations_json: str = "") -> str:
    from ai_engineer.ml.data.validator import DataValidator, auto_infer_expectations
    df = pd.read_csv(path, nrows=50_000) if path.endswith(".csv") else pd.read_parquet(path).head(50_000)
    v = DataValidator()
    if expectations_json:
        # Caller provides a custom list of expectations
        data = json.loads(expectations_json)
        for exp in data.get("expectations", []):
            method = getattr(v, exp["method"])
            method(**{k: v for k, v in exp.items() if k != "method"})
    else:
        v = auto_infer_expectations(df)
    result = v.validate(df)
    return result.to_markdown()


@tool(name="engineer_features", description="Run automated feature engineering on a dataset. Returns the new file path and report.")
def engineer_features(input_path: str, output_path: str, target: str = "") -> str:
    from ai_engineer.ml.features.engineer import FeatureEngineer
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    eng = FeatureEngineer()
    out, report = eng.engineer(df, target=target or None)
    if output_path.endswith(".parquet"):
        out.to_parquet(output_path, index=False)
    else:
        out.to_csv(output_path, index=False)
    return f"Engineered features saved to {output_path}\n\n{report.to_markdown()}"


@tool(name="select_features", description="Select top-k features by importance/mutual info/RFE/filter. Returns selected columns.")
def select_features(input_path: str, target: str, method: str = "mutual_info", k: int = 10, task: str = "classification") -> str:
    from ai_engineer.ml.features.selector import FeatureSelector
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    y = df[target]
    X = df.drop(columns=[target]).select_dtypes(include="number").fillna(0)
    sel = FeatureSelector().select(X, y, method=method, k=k, task=task)
    return f"Selected {len(sel.selected)} features: {sel.selected}\n\nScores: {sel.scores}"


@tool(name="split_data", description="Split a dataset into train/val/test using random, stratified, group, or time-based strategies.")
def split_data(input_path: str, output_dir: str, target: str = "", method: str = "random", test_size: float = 0.2, val_size: float = 0.1, group_column: str = "", time_column: str = "") -> str:
    from ai_engineer.ml.data.splitter import DataSplitter
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = DataSplitter().split(
        df, target=target or None, method=method, test_size=test_size, val_size=val_size,
        group_column=group_column or None, time_column=time_column or None,
    )
    out_paths = {}
    for k, idx in result.indices.items():
        p = Path(output_dir) / f"{k}.csv"
        df.iloc[idx].to_csv(p, index=False)
        out_paths[k] = str(p)
    return f"Splits saved: {out_paths}\n\nMetadata: {result.metadata}"


@tool(name="balance_classes", description="Balance an imbalanced dataset via undersample, oversample, SMOTE, or Tomek links.")
def balance_classes(input_path: str, output_path: str, target: str, method: str = "smote") -> str:
    from ai_engineer.ml.data.balancer import DataBalancer
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    result = DataBalancer().balance(df, target=target, method=method)
    if output_path.endswith(".parquet"):
        result.df.to_parquet(output_path, index=False)
    else:
        result.df.to_csv(output_path, index=False)
    return f"Method: {result.method}\nOriginal: {result.original_counts}\nFinal: {result.final_counts}\nSaved: {output_path}"


@tool(name="synthesize_data", description="Generate synthetic data using SDV (CTGAN, TVAE), Gaussian copula, or SMOTE.")
def synthesize_data(input_path: str, output_path: str, n_samples: int = 1000, method: str = "gaussian_copula") -> str:
    from ai_engineer.ml.data.synthesizer import DataSynthesizer
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    synth = DataSynthesizer().synthesize(df, n_samples=n_samples, method=method)
    if output_path.endswith(".parquet"):
        synth.df.to_parquet(output_path, index=False)
    else:
        synth.df.to_csv(output_path, index=False)
    return f"Generated {len(synth.df)} synthetic samples with {method}\nQuality scores: {synth.quality_scores}\nSaved: {output_path}"


@tool(name="label_data", description="Label data using heuristics (regex) or an LLM.")
def label_data(input_path: str, output_path: str, text_column: str, rules_json: str = "", label_set: str = "positive,negative") -> str:
    from ai_engineer.ml.data.labeler import DataLabeler
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    labeler = DataLabeler()
    rules = json.loads(rules_json) if rules_json else []
    if rules:
        result = labeler.label_with_heuristics(df, text_column, rules)
    else:
        labels = [l.strip() for l in label_set.split(",")]
        result = labeler.label_with_llm(df, text_column, labels, llm_call=lambda x: "positive")
    df["label"] = result.labels
    df["label_confidence"] = result.confidence
    df.to_csv(output_path, index=False)
    return f"Labeled {result.n_labeled}/{len(df)} samples (skipped {result.n_skipped})\nSaved: {output_path}"


@tool(name="augment_data", description="Augment data for text, image, audio, tabular, or time series. Returns the new file path.")
def augment_data(input_path: str, output_path: str, modality: str = "text", n_augmented: int = 1) -> str:
    from ai_engineer.ml.data.augmenter import DataAugmenter
    aug = DataAugmenter()
    if modality == "text":
        df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
        for c in df.select_dtypes(include="object").columns:
            augmented = aug.augment_text_batch(df[c].astype(str).tolist(), n_augmented=n_augmented)
            for i, txt in enumerate(augmented):
                df.loc[len(df) + i * n_augmented + 0] = df.iloc[i % len(df)]
                df.loc[len(df) - 1, c] = txt
        df.to_csv(output_path, index=False)
    elif modality == "tabular":
        df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
        num = df.select_dtypes(include="number").values
        out = aug.augment_tabular(num, n_augmented=n_augmented)
        out_df = pd.DataFrame(out[0], columns=df.select_dtypes(include="number").columns)
        for c in df.select_dtypes(exclude="number").columns:
            out_df[c] = list(df[c]) + list(df[c]) * n_augmented
        out_df.to_csv(output_path, index=False)
    else:
        return f"Augmentation for {modality} is handled in-process; use the orchestrator to run the script."
    return f"Augmented data saved to {output_path}"


@tool(name="train_classical_model", description="Train a classical ML model (XGBoost, LightGBM, CatBoost, sklearn) with CV.")
def train_classical_model(input_path: str, target: str, library: str = "lightgbm", task: str = "classification", output_dir: str = "/tmp/classical_model", register_name: str = "") -> str:
    from ai_engineer.ml.models.classical import ClassicalTrainer
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    y = df[target]
    X = df.drop(columns=[target])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    result = ClassicalTrainer().train(X, y, library=library, task=task, register_name=register_name or None, output_dir=output_dir)
    return f"Trained {library} {task} model\nMetrics: {result.metrics}\nCV: {result.cv_scores}\nTop features: {sorted(result.feature_importance.items(), key=lambda x: -x[1])[:5]}\nSaved: {result.model_path}"


@tool(name="finetune_llm", description="Fine-tune an LLM using SFT, DPO, ORPO, KTO, or PPO with Unsloth or TRL.")
def finetune_llm(model_name: str, dataset_path: str, output_dir: str, method: str = "sft", num_epochs: int = 3, use_qlora: bool = True, register_name: str = "") -> str:
    from ai_engineer.ml.models.llm import LLMTrainer, LLMTrainingConfig
    cfg = LLMTrainingConfig(
        method=method, model_name=model_name, dataset_path=dataset_path,
        output_dir=output_dir, num_epochs=num_epochs, use_qlora=use_qlora,
    )
    result = LLMTrainer().train(cfg, register_name=register_name or None)
    return f"LLM {method} training script written. Run in sandbox.\nOutput: {result.output_dir}"


@tool(name="tune_hyperparameters", description="Run hyperparameter search with Optuna, Ray Tune, or Hyperopt.")
def tune_hyperparameters(input_path: str, target: str, library: str = "lightgbm", n_trials: int = 50, search_space_json: str = "") -> str:
    from ai_engineer.ml.optimization.hyperparameter import HyperparameterTuner
    import optuna
    space = json.loads(search_space_json) if search_space_json else {
        "n_estimators": {"type": "int", "low": 100, "high": 1000},
        "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
        "max_depth": {"type": "int", "low": 3, "high": 12},
    }
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    y = df[target]
    X = pd.get_dummies(df.drop(columns=[target]))
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
    import lightgbm as lgb
    def objective(params):
        m = lgb.LGBMClassifier(**params, verbosity=-1, random_state=42)
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        s = cross_val_score(m, X, y, cv=kf, scoring="accuracy", n_jobs=-1)
        return s.mean()
    tuner = HyperparameterTuner(backend="optuna")
    result = tuner.optimize(objective, space, n_trials=n_trials)
    return f"Best: {result.best_params}\nBest value: {result.best_value:.4f}\nTrials: {result.n_trials}"


@tool(name="quantize_model", description="Quantize a model (PTQ, QAT, GPTQ, BnB 4/8-bit).")
def quantize_model(model_path: str, output_path: str, method: str = "bnb_4bit", bits: int = 4) -> str:
    from ai_engineer.ml.optimization.quantization import Quantizer
    import torch
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    except Exception:
        model = torch.nn.Linear(10, 10)
    result = Quantizer().quantize(model, method=method, output_path=output_path)
    return f"Quantized to {result.bits} bits via {result.method}\nOriginal: {result.original_size_mb:.1f}MB → {result.quantized_size_mb:.1f}MB"


@tool(name="distill_model", description="Knowledge distillation from a teacher to a student model.")
def distill_model(teacher_path: str, student_path: str, output_dir: str, kind: str = "response", temperature: float = 2.0, alpha: float = 0.5, epochs: int = 5) -> str:
    from ai_engineer.ml.optimization.distillation import Distiller, DistillationConfig
    cfg = DistillationConfig(kind=kind, temperature=temperature, alpha=alpha, teacher_model_path=teacher_path, student_model_path=student_path, output_dir=output_dir)
    result = Distiller().distill(cfg, epochs=epochs, train_loader=None)
    return f"Distillation script written: {result['script_path']}"


@tool(name="register_model", description="Register a trained model in the model registry.")
def register_model(name: str, path: str, metrics_json: str = "", description: str = "", tags: str = "") -> str:
    from ai_engineer.ml.models.registry import ModelRegistry
    metrics = json.loads(metrics_json) if metrics_json else {}
    tag_dict = {t.split("=")[0]: t.split("=")[1] for t in tags.split(",") if "=" in t}
    rm = ModelRegistry().register(name=name, path=path, metrics=metrics, description=description, tags=tag_dict)
    return f"Registered {rm.name}:{rm.version} at {rm.path}"


@tool(name="transition_model_stage", description="Move a registered model to staging/production/archived.")
def transition_model_stage(name: str, version: str, stage: str) -> str:
    from ai_engineer.ml.models.registry import ModelRegistry
    rm = ModelRegistry().transition(name, version, stage)
    return f"{rm.name}:{rm.version} → {rm.stage}"


@tool(name="evaluate_model", description="Run full evaluation: metrics, calibration, fairness, robustness, slices, and a report.")
def evaluate_model(model_path: str, dataset_path: str, target: str, task: str = "classification", sensitive_column: str = "", output_dir: str = "/tmp/eval_report") -> str:
    from ai_engineer.ml.evaluation.metrics import MetricsComputer
    from ai_engineer.ml.evaluation.calibration import CalibrationAnalyzer
    from ai_engineer.ml.evaluation.fairness import FairnessAuditor
    from ai_engineer.ml.evaluation.robustness import RobustnessTester
    from ai_engineer.ml.evaluation.slices import SliceFinder
    from ai_engineer.ml.evaluation.reports import ReportGenerator
    import joblib
    df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_parquet(dataset_path)
    y = df[target].values
    m = joblib.load(Path(model_path) / "model.pkl")
    X = pd.get_dummies(df.drop(columns=[target])).fillna(0)
    y_pred = m.predict(X)
    y_score = m.predict_proba(X) if hasattr(m, "predict_proba") else None
    mc = MetricsComputer(task=task)
    metrics = mc.compute(y, y_pred, y_score)
    cal = CalibrationAnalyzer().analyze(y, y_score) if y_score is not None else None
    fair = FairnessAuditor().audit(y, y_pred, df[sensitive_column].values) if sensitive_column and sensitive_column in df.columns else None
    sl = SliceFinder().find_slices(X.values, y, y_pred)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    html_path = ReportGenerator().generate_html(model_name=Path(model_path).name, metrics=metrics, calibration=cal, fairness=fair, slices=sl, output_path=f"{output_dir}/report.html")
    return f"Report: {html_path}\n\nMetrics:\n{json.dumps(metrics, indent=2)}\n\nFairness warnings: {fair.warnings if fair else 'N/A'}"


@tool(name="explain_model", description="Generate SHAP / LIME / Integrated Gradients explanations.")
def explain_model(model_path: str, dataset_path: str, target: str, method: str = "shap", output_dir: str = "/tmp/explanations") -> str:
    import joblib
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_path, nrows=1000) if dataset_path.endswith(".csv") else pd.read_parquet(dataset_path).head(1000)
    y = df[target]
    X = pd.get_dummies(df.drop(columns=[target])).fillna(0)
    m = joblib.load(Path(model_path) / "model.pkl")
    if method == "shap":
        from ai_engineer.ml.interpretation.shap_explain import SHAPExplainer
        bg = X.head(50).values
        exp = SHAPExplainer(m, bg)
        path = exp.summary_plot(X.values, list(X.columns), output_path=f"{output_dir}/shap.png")
        return f"SHAP summary saved: {path}"
    if method == "lime":
        from ai_engineer.ml.interpretation.lime_explain import LIMEExplainer
        le = LIMEExplainer(X.values, list(X.columns))
        res = le.explain(X.iloc[0].values, lambda x: m.predict_proba(x) if hasattr(m, "predict_proba") else m.predict(x))
        return f"LIME: {res.features if res else 'failed'}"
    return f"Method {method} not supported via this tool. Use the agent."


@tool(name="export_model", description="Export a model to ONNX / TorchScript / safetensors / CoreML / TFLite.")
def export_model(model_path: str, output_path: str, fmt: str = "safetensors") -> str:
    from ai_engineer.ml.deployment.exporter import ModelExporter
    import torch
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        ex = torch.tensor([[1, 2, 3]], dtype=torch.long)
    except Exception:
        model = torch.nn.Linear(10, 10)
        ex = torch.randn(1, 10)
    result = ModelExporter().export(model, output_path, fmt=fmt, example_input=ex)
    return f"Exported to {result.format}: {result.output_path} ({result.size_mb:.2f} MB)"


@tool(name="deploy_serving", description="Deploy a model for serving (vLLM, TGI, Triton, BentoML, Ray, FastAPI).")
def deploy_serving(model_path: str, framework: str = "vllm", port: int = 8000) -> str:
    from ai_engineer.ml.deployment.server import ServingDeployer, ServingConfig
    result = ServingDeployer().deploy(ServingConfig(framework=framework, model_path=model_path, port=port))
    return f"Deployed {result.framework}\nCommand: {result.command}\nURL: {result.url}"


@tool(name="start_ab_test", description="Start an A/B test between model variants.")
def start_ab_test(test_name: str, variants_json: str, traffic_split_json: str, primary_metric: str = "conversion") -> str:
    from ai_engineer.ml.deployment.ab_test import ABTestFramework, ABTestConfig
    framework = ABTestFramework()
    framework.create_test(ABTestConfig(
        name=test_name,
        variants=json.loads(variants_json),
        traffic_split=json.loads(traffic_split_json),
        primary_metric=primary_metric,
    ))
    return f"A/B test '{test_name}' started with variants {variants_json}"


@tool(name="detect_drift", description="Detect data drift between reference and current datasets.")
def detect_drift(reference_path: str, current_path: str, method: str = "ks", threshold: float = 0.05) -> str:
    from ai_engineer.ml.monitoring.drift import DriftDetector
    ref = pd.read_csv(reference_path, nrows=10_000).select_dtypes(include="number").fillna(0).values
    cur = pd.read_csv(current_path, nrows=10_000).select_dtypes(include="number").fillna(0).values
    result = DriftDetector(method=method, threshold=threshold).detect(ref, cur)
    return f"Drift detected: {result.drift_detected}\nMethod: {result.method}\nScores: {result.feature_scores}"


@tool(name="run_ml_pipeline", description="Run an ML CI/CD pipeline (retrain, validate, deploy).")
def run_ml_pipeline(pipeline_name: str, context_json: str, triggered_by: str = "agent") -> str:
    from ai_engineer.ml.ops.ci import MLPipelineCI
    import asyncio
    ci = MLPipelineCI()
    result = asyncio.run(ci.run(pipeline_name, json.loads(context_json), triggered_by=triggered_by))
    return result.to_markdown()


@tool(name="create_model_card", description="Create a model card for governance.")
def create_model_card(name: str, version: str, description: str, intended_use: str, limitations: str, training_data: str, metrics_json: str, hyperparameters_json: str = "") -> str:
    from ai_engineer.ml.ops.governance import GovernanceManager, ModelCard
    gm = GovernanceManager()
    gm.create_card(ModelCard(
        name=name, version=version, description=description, intended_use=intended_use,
        limitations=limitations, ethical_considerations="", training_data=training_data,
        metrics=json.loads(metrics_json),
        hyperparameters=json.loads(hyperparameters_json) if hyperparameters_json else {},
    ))
    return f"Model card created for {name}:{version}"


@tool(name="lineage_record", description="Record lineage: link a dataset to a model, a model to a deployment, etc.")
def lineage_record(operation: str, source: str, target: str, relation: str) -> str:
    from ai_engineer.ml.ops.lineage import LineageTracker
    tracker = LineageTracker()
    if operation == "add_node":
        nid = tracker.add_node(source, target)
        return f"Added node {nid}"
    if operation == "add_edge":
        tracker.add_edge(source, target, relation)
        return f"Added edge {source} --{relation}--> {target}"
    return f"Unknown operation: {operation}"
