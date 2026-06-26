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

"""Text feature extraction: TF-IDF, embeddings, BM25."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ai_engineer.utils.errors import AIEngineerError


@dataclass
class TextVectorizer:
    kind: str  # tfidf | sentence | bm25
    vectorizer: Any = None
    dim: int = 0

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self.kind == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
            arr = self.vectorizer.fit_transform(texts).toarray()
        elif self.kind == "sentence":
            from sentence_transformers import SentenceTransformer
            self.vectorizer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            arr = np.array(self.vectorizer.encode(texts, show_progress_bar=False))
        elif self.kind == "bm25":
            from rank_bm25 import BM25Okapi
            tokenized = [t.lower().split() for t in texts]
            self.vectorizer = BM25Okapi(tokenized)
            arr = np.array([self.vectorizer.get_scores(t) for t in tokenized])
        else:
            raise AIEngineerError(f"Unknown vectorizer: {self.kind}")
        self.dim = arr.shape[1]
        return arr

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.kind == "tfidf":
            return self.vectorizer.transform(texts).toarray()
        if self.kind == "sentence":
            return np.array(self.vectorizer.encode(texts, show_progress_bar=False))
        if self.kind == "bm25":
            return np.array([self.vectorizer.get_scores(t.lower().split()) for t in texts])
        raise AIEngineerError(f"Unknown vectorizer: {self.kind}")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "TextVectorizer":
        with Path(path).open("rb") as f:
            return pickle.load(f)
