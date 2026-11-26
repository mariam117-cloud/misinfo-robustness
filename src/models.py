"""Classifiers. Every model exposes fit(texts, labels) and predict_proba(texts) -> (n, 2) array."""
from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TfidfLR:
    """TF-IDF + logistic regression. analyzer='word' mirrors a classic NLP baseline;
    analyzer='char' uses character n-grams, which should degrade more gracefully under typos."""

    def __init__(self, analyzer: str = "word", seed: int = 0):
        if analyzer == "word":
            self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        elif analyzer == "char":
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                       sublinear_tf=True)
        else:
            raise ValueError(analyzer)
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfLR":
        self.clf.fit(self.vec.fit_transform(texts), labels)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(list(texts)))


class TransformerClassifier:
    """Fine-tuned pretrained transformer (default DistilBERT). Needs torch + transformers;
    a GPU (e.g. free Colab T4) is strongly recommended."""

    def __init__(self, model_name: str = "distilbert-base-uncased", epochs: int = 3,
                 batch_size: int = 32, lr: float = 2e-5, max_len: int = 128, seed: int = 0):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.model_name, self.epochs, self.batch_size = model_name, epochs, batch_size
        self.lr, self.max_len, self.seed = lr, max_len, seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def _encode(self, texts: Sequence[str]):
        return self.tok(list(texts), truncation=True, max_length=self.max_len,
                        padding=True, return_tensors="pt")

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TransformerClassifier":
        torch = self.torch
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        n = len(texts)
        steps = self.epochs * ((n + self.batch_size - 1) // self.batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        y = torch.tensor(list(labels))
        gen = torch.Generator().manual_seed(self.seed)
        self.model.train()
        for epoch in range(self.epochs):
            order = torch.randperm(n, generator=gen).tolist()
            total = 0.0
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                batch = self._encode([texts[i] for i in idx]).to(self.device)
                out = self.model(**batch, labels=y[idx].to(self.device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                total += out.loss.item() * len(idx)
            print(f"    epoch {epoch + 1}/{self.epochs}  loss={total / n:.4f}")
        self.model.eval()
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), 128):
                batch = self._encode(texts[s:s + 128]).to(self.device)
                out.append(torch.softmax(self.model(**batch).logits, -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2))


class Preprocessed:
    """Wraps a model so that `preprocess` runs on every input, at training and test time."""

    def __init__(self, model, preprocess: Callable[[str], str]):
        self.model, self.preprocess = model, preprocess

    def fit(self, texts, labels):
        self.model.fit([self.preprocess(t) for t in texts], labels)
        return self

    def predict_proba(self, texts):
        return self.model.predict_proba([self.preprocess(t) for t in texts])


def build_model(name: str, seed: int, transformer_kwargs: Optional[dict] = None):
    if name == "tfidf_word":
        return TfidfLR("word", seed)
    if name == "tfidf_char":
        return TfidfLR("char", seed)
    if name == "distilbert":
        return TransformerClassifier(seed=seed, **(transformer_kwargs or {}))
    if name.startswith("hf:"):  # any Hugging Face model, e.g. hf:roberta-base
        return TransformerClassifier(model_name=name[3:], seed=seed, **(transformer_kwargs or {}))
    raise ValueError(f"unknown model {name}")
