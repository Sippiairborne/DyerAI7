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

"""Comprehensive metrics: classification, regression, ranking, retrieval, generation, detection, segmentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Task = Literal["classification", "regression", "ranking", "retrieval", "generation", "detection", "segmentation"]


class MetricsComputer:
    def __init__(self, task: str = "classification") -> None:
        self.task = task

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
        if self.task == "classification":
            return self._classification(y_true, y_pred, y_score)
        if self.task == "regression":
            return self._regression(y_true, y_pred)
        if self.task in ("ranking", "retrieval"):
            return self._retrieval(y_true, y_score if y_score is not None else y_pred)
        if self.task == "generation":
            return self._generation(y_true, y_pred)
        if self.task == "detection":
            return self._detection(y_true, y_pred)
        if self.task == "segmentation":
            return self._segmentation(y_true, y_pred)
        raise AIEngineerError(f"Unknown task: {self.task}")

    def primary_metric(self, metrics: dict[str, float]) -> float:
        keys = {
            "classification": "roc_auc" if "roc_auc" in metrics else "f1",
            "regression": "r2",
            "ranking": "ndcg_at_10",
            "retrieval": "mrr_at_10",
            "generation": "bleu",
            "detection": "map",
            "segmentation": "miou",
        }
        return float(metrics.get(keys[self.task], next(iter(metrics.values()), 0.0)))

    def _classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score, log_loss, matthews_corrcoef, cohen_kappa_score, top_k_accuracy_score,
            average_precision_score,
        )
        m: dict[str, float] = {}
        m["accuracy"] = float(accuracy_score(y_true, y_pred))
        m["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        m["f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        m["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        m["precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        m["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        m["mcc"] = float(matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0)
        m["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        if y_score is not None and y_score.ndim > 1:
            try:
                m["roc_auc"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
            except Exception:
                pass
            try:
                m["log_loss"] = float(log_loss(y_true, y_score, labels=list(range(y_score.shape[1]))))
            except Exception:
                pass
            try:
                m["top_2_accuracy"] = float(top_k_accuracy_score(y_true, y_score, k=2))
                m["top_5_accuracy"] = float(top_k_accuracy_score(y_true, y_score, k=min(5, y_score.shape[1])))
            except Exception:
                pass
            try:
                m["pr_auc"] = float(average_precision_score(y_true.ravel(), y_score[:, 1] if y_score.shape[1] == 2 else y_score.max(1), average="macro"))
            except Exception:
                pass
        return m

    def _regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
        m: dict[str, float] = {}
        m["mae"] = float(mean_absolute_error(y_true, y_pred))
        m["mse"] = float(mean_squared_error(y_true, y_pred))
        m["rmse"] = float(np.sqrt(m["mse"]))
        m["r2"] = float(r2_score(y_true, y_pred))
        m["explained_variance"] = float(explained_variance_score(y_true, y_pred))
        try:
            m["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            pass
        # Residual stats
        residuals = y_true - y_pred
        m["residual_mean"] = float(np.mean(residuals))
        m["residual_std"] = float(np.std(residuals))
        return m

    def _retrieval(self, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import ndcg_score
        m: dict[str, float] = {}
        for k in (5, 10, 20):
            try:
                m[f"ndcg_at_{k}"] = float(ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k))
            except Exception:
                m[f"ndcg_at_{k}"] = 0.0
        # MRR
        mrr = 0.0
        for i, (t, s) in enumerate(zip(y_true, y_score)):
            order = np.argsort(-s)
            where = np.where(t[order] > 0)[0]
            if len(where):
                mrr += 1.0 / (where[0] + 1)
        m["mrr_at_10"] = float(mrr / max(len(y_true), 1))
        m["mrr"] = float(mrr / max(len(y_true), 1))
        return m

    def _generation(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        # y_true and y_pred are arrays of strings
        m: dict[str, float] = {}
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            refs = [[t.split()] for t in y_true]
            hyps = [t.split() for t in y_pred]
            smooth = SmoothingFunction().method1
            m["bleu"] = float(corpus_bleu(refs, hyps, smoothing_function=smooth))
        except Exception:
            m["bleu"] = 0.0
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            r1, r2, rl = 0.0, 0.0, 0.0
            for t, p in zip(y_true, y_pred):
                sc = scorer.score(t, p)
                r1 += sc["rouge1"].fmeasure
                r2 += sc["rouge2"].fmeasure
                rl += sc["rougeL"].fmeasure
            n = max(len(y_true), 1)
            m["rouge1"] = r1 / n
            m["rouge2"] = r2 / n
            m["rougeL"] = rl / n
        except Exception:
            pass
        try:
            from sacrebleu import corpus_ter
            m["ter"] = float(corpus_ter(list(y_pred), [list(y_true)]).score)
        except Exception:
            pass
        # Distinct-n
        for n in (1, 2, 3):
            ngrams = set()
            total = 0
            for s in y_pred:
                toks = s.split()
                for i in range(len(toks) - n + 1):
                    ngrams.add(tuple(toks[i:i + n]))
                    total += 1
            m[f"distinct_{n}"] = float(len(ngrams) / max(total, 1))
        return m

    def _detection(self, y_true: list[dict], y_pred: list[dict]) -> dict[str, float]:
        """Each entry: {'boxes': [N,4], 'scores': [N], 'labels': [N]}"""
        from collections import defaultdict
        aps = []
        for yt, yp in zip(y_true, y_pred):
            for label in set(list(yt.get("labels", [])) + list(yp.get("labels", []))):
                yt_b = np.array([b for b, l in zip(yt["boxes"], yt["labels"]) if l == label])
                yp_b = np.array([b for b, l in zip(yp["boxes"], yp["labels"]) if l == label])
                yp_s = np.array([s for s, l in zip(yp["scores"], yp["labels"]) if l == label])
                if len(yp_b) == 0:
                    continue
                order = np.argsort(-yp_s)
                tp = np.zeros(len(yp_b))
                fp = np.zeros(len(yp_b))
                for i, bb in enumerate(yp_b[order]):
                    if len(yt_b) == 0:
                        fp[i] = 1
                        continue
                    ious = self._box_iou(bb[None, :], yt_b)[0]
                    j = np.argmax(ious)
                    if ious[j] > 0.5:
                        tp[i] = 1
                    else:
                        fp[i] = 1
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recall = tp_cum / max(len(yt_b), 1)
                precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
                ap = float(np.trapz(precision, recall))
                aps.append(ap)
        return {"map": float(np.mean(aps)) if aps else 0.0}

    @staticmethod
    def _box_iou(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
        x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
        y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
        x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
        y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = a1[:, None] + a2[None, :] - inter
        return inter / np.maximum(union, 1e-8)

    def _segmentation(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """y_true, y_pred: [N, H, W] integer masks"""
        from collections import Counter
        labels = np.unique(np.concatenate([y_true.flatten(), y_pred.flatten()]))
        ious = []
        for l in labels:
            if l == 0:  # ignore background by convention
                continue
            i = ((y_pred == l) & (y_true == l)).sum()
            u = ((y_pred == l) | (y_true == l)).sum()
            ious.append(i / max(u, 1))
        return {
            "miou": float(np.mean(ious)) if ious else 0.0,
            "pixel_accuracy": float((y_pred == y_true).mean()),
            "per_class_iou": {int(l): float(i / max(((y_pred == l) | (y_true == l)).sum(), 1)) for l, i in zip(labels, [((y_pred == l) & (y_true == l)).sum() for l in labels])},
        }
