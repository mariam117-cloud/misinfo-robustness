"""Defenses against character-level evasion.

1. `normalize`  - input sanitisation applied before the model sees any text
                  (Unicode NFKC, zero-width removal, homoglyph and leetspeak reversal).
2. `augment`    - adversarial data augmentation: add perturbed copies of the
                  training data so the model learns to ignore the noise.
"""
from __future__ import annotations

import random
import re
import unicodedata
from typing import List, Sequence, Tuple

from .perturb import HOMOGLYPHS, PERTURBATIONS, perturb_text

_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")
_INV_HOMOGLYPHS = {v: k for k, v in HOMOGLYPHS.items()}
_INV_HOMOGLYPHS.update({v.upper(): k.upper() for k, v in HOMOGLYPHS.items() if v.upper() != v})
_INV_LEET = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s"}
_WORD_RE = re.compile(r"\S+")


_ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)\W*$", re.I)


def _deleet_word(m: re.Match) -> str:
    """Map a leet symbol back to a letter only when it touches a letter, so numbers such as
    "2016", "$5" or "COVID-19" and ordinals such as "4th" stay intact. Known limitation:
    genuine letter-digit tokens like "5G" are also rewritten."""
    w = m.group(0)
    if _ORDINAL_RE.match(w):
        return w
    # Repeat until stable so runs of symbols are decoded too ("53cret" -> "5ecret" -> "secret").
    while True:
        out = []
        for i, ch in enumerate(w):
            near_letter = (i > 0 and w[i - 1].isalpha()) or (i + 1 < len(w) and w[i + 1].isalpha())
            out.append(_INV_LEET[ch] if ch in _INV_LEET and near_letter else ch)
        new = "".join(out)
        if new == w:
            return w
        w = new


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = "".join(_INV_HOMOGLYPHS.get(ch, ch) for ch in text)
    text = _WORD_RE.sub(_deleet_word, text)
    return text


def augment(texts: Sequence[str], labels: Sequence[int], seed: int,
            kinds: Sequence[str] = tuple(PERTURBATIONS), rate: float = 0.2,
            copies: int = 1) -> Tuple[List[str], List[int]]:
    """Return the original data plus `copies` perturbed versions of every example.
    Each copy uses a randomly chosen perturbation type."""
    rng = random.Random(seed)
    out_t, out_y = list(texts), list(labels)
    for _ in range(copies):
        for t, y in zip(texts, labels):
            out_t.append(perturb_text(t, rng.choice(list(kinds)), rate, rng))
            out_y.append(y)
    return out_t, out_y
