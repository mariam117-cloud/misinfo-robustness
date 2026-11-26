"""Dataset loaders. Every loader returns a dict with 'train', 'val' and 'test'
splits, each a (texts, labels) tuple with labels 1 = misleading/false, 0 = credible."""
from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

Split = Tuple[List[str], List[int]]

# LIAR (Wang, 2017): six truthfulness labels collapsed to binary.
LIAR_FAKE = {"pants-fire", "false", "barely-true"}
LIAR_REAL = {"half-true", "mostly-true", "true"}


def _read_liar_tsv(path: str) -> Split:
    df = pd.read_csv(path, sep="\t", header=None, quoting=csv.QUOTE_NONE,
                     usecols=[1, 2], names=["label", "statement"], dtype=str)
    df = df.dropna()
    df = df[df["label"].isin(LIAR_FAKE | LIAR_REAL)]
    labels = df["label"].map(lambda l: 1 if l in LIAR_FAKE else 0).tolist()
    return df["statement"].str.strip().tolist(), labels


def load_liar(data_dir: str) -> Dict[str, Split]:
    """Expects train.tsv, valid.tsv and test.tsv from the original LIAR release."""
    return {
        "train": _read_liar_tsv(os.path.join(data_dir, "train.tsv")),
        "val": _read_liar_tsv(os.path.join(data_dir, "valid.tsv")),
        "test": _read_liar_tsv(os.path.join(data_dir, "test.tsv")),
    }


def load_csv(path: str, text_col: str = "text", label_col: str = "label",
             seed: int = 0, val_size: float = 0.1, test_size: float = 0.2) -> Dict[str, Split]:
    """Any CSV with a text column and a binary label column (e.g. your own tweet dataset).
    Label values must be 0/1, or strings where 'fake', 'false', 'misleading', '1' mean 1."""
    df = pd.read_csv(path)[[text_col, label_col]].dropna()
    raw = df[label_col].astype(str).str.strip().str.lower()
    positive = {"1", "fake", "false", "misleading", "misinformation", "rumor", "rumour"}
    y = raw.map(lambda v: 1 if v in positive else 0).tolist()
    x = df[text_col].astype(str).tolist()
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=val_size / (1 - test_size), random_state=seed, stratify=y_tmp)
    return {"train": (x_train, y_train), "val": (x_val, y_val), "test": (x_test, y_test)}


def load_synthetic(n: int = 1200, seed: int = 0) -> Dict[str, Split]:
    """Tiny artificial dataset, only for smoke-testing the pipeline."""
    rng = random.Random(seed)
    fake_cues = ["shocking", "secret", "hoax", "cover-up", "miracle", "banned", "exposed", "they"]
    real_cues = ["report", "according", "official", "study", "percent", "announced", "data", "agency"]
    filler = ["the", "government", "vaccine", "people", "new", "city", "health", "said",
              "will", "election", "today", "million", "workers", "president", "about"]
    texts, labels = [], []
    for _ in range(n):
        y = rng.randint(0, 1)
        cues = fake_cues if y else real_cues
        words = rng.sample(filler, 8) + rng.sample(cues, 3)
        if rng.random() < 0.15:  # label noise so the task is not trivial
            words += rng.sample(fake_cues if not y else real_cues, 2)
        rng.shuffle(words)
        texts.append(" ".join(words).capitalize() + ".")
        labels.append(y)
    a, b = int(0.7 * n), int(0.8 * n)
    return {"train": (texts[:a], labels[:a]), "val": (texts[a:b], labels[a:b]),
            "test": (texts[b:], labels[b:])}
