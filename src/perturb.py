"""Character-level text perturbations that a misinformation spreader could use
to evade an automated detector while keeping the text readable to humans.

Every perturbation works on a single word. `perturb_text` applies one
perturbation type to a random fraction of the eligible words in a text;
`perturb_positions` applies it to specific word positions (used by the attack).
"""
from __future__ import annotations

import re
import random
from typing import Callable, Dict, List, Sequence

# Latin -> visually identical Cyrillic/other Unicode characters.
HOMOGLYPHS: Dict[str, str] = {
    "a": "а",  # Cyrillic a
    "c": "с",  # Cyrillic es
    "e": "е",  # Cyrillic ie
    "o": "о",  # Cyrillic o
    "p": "р",  # Cyrillic er
    "x": "х",  # Cyrillic ha
    "y": "у",  # Cyrillic u
    "i": "і",  # Cyrillic/Ukrainian i
    "s": "ѕ",  # Cyrillic dze
    "j": "ј",  # Cyrillic je
}

LEET: Dict[str, str] = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}

ZERO_WIDTH_SPACE = "​"

_KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
KEYBOARD_NEIGHBORS: Dict[str, str] = {}
for _r, _row in enumerate(_KEYBOARD_ROWS):
    for _c, _ch in enumerate(_row):
        _n = []
        for _dr in (-1, 0, 1):
            _rr = _r + _dr
            if 0 <= _rr < len(_KEYBOARD_ROWS):
                for _dc in (-1, 0, 1):
                    _cc = _c + _dc
                    if (_dr or _dc) and 0 <= _cc < len(_KEYBOARD_ROWS[_rr]):
                        _n.append(_KEYBOARD_ROWS[_rr][_cc])
        KEYBOARD_NEIGHBORS[_ch] = "".join(_n)

MIN_LETTERS = 3  # words with fewer letters are left untouched


def _letter_positions(word: str) -> List[int]:
    return [i for i, ch in enumerate(word) if ch.isalpha()]


def _match_case(src: str, repl: str) -> str:
    return repl.upper() if src.isupper() else repl


# ---------------------------------------------------------------- word level

def typo(word: str, rng: random.Random) -> str:
    """Swap, delete, insert or substitute (keyboard neighbour) one inner letter."""
    pos = _letter_positions(word)
    inner = pos[1:-1] if len(pos) > 2 else pos
    if not inner:
        return word
    i = rng.choice(inner)
    op = rng.choice(["swap", "delete", "insert", "substitute"])
    if op == "swap" and i + 1 < len(word) and word[i + 1].isalpha():
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "delete":
        return word[:i] + word[i + 1:]
    neighbours = KEYBOARD_NEIGHBORS.get(word[i].lower(), "")
    if not neighbours:
        return word[:i] + word[i + 1:]
    new = _match_case(word[i], rng.choice(neighbours))
    if op == "insert":
        return word[:i] + new + word[i:]
    return word[:i] + new + word[i + 1:]  # substitute (also fallback for swap)


def _replace_chars(word: str, table: Dict[str, str], rng: random.Random, k: int = 2) -> str:
    cands = [i for i, ch in enumerate(word) if ch.lower() in table]
    if not cands:
        return word
    chosen = rng.sample(cands, min(k, len(cands)))
    chars = list(word)
    for i in chosen:
        chars[i] = table[chars[i].lower()]
    return "".join(chars)


def homoglyph(word: str, rng: random.Random) -> str:
    """Replace up to two letters with look-alike Unicode characters."""
    return _replace_chars(word, HOMOGLYPHS, rng)


def leet(word: str, rng: random.Random) -> str:
    """Replace up to two letters with digits (e.g. 'vaccine' -> 'v4cc1ne')."""
    return _replace_chars(word, LEET, rng)


def zero_width(word: str, rng: random.Random) -> str:
    """Insert an invisible zero-width space inside the word."""
    if len(word) < 2:
        return word
    i = rng.randint(1, len(word) - 1)
    return word[:i] + ZERO_WIDTH_SPACE + word[i:]


PERTURBATIONS: Dict[str, Callable[[str, random.Random], str]] = {
    "typo": typo,
    "homoglyph": homoglyph,
    "leet": leet,
    "zero_width": zero_width,
}


# ---------------------------------------------------------------- text level

def split_words(text: str) -> List[str]:
    """Split keeping whitespace, so ''.join(parts) == text. Words sit at even indices."""
    return re.split(r"(\s+)", text)


def eligible_positions(parts: Sequence[str]) -> List[int]:
    return [
        i for i in range(0, len(parts), 2)
        if sum(ch.isalpha() for ch in parts[i]) >= MIN_LETTERS
    ]


def perturb_positions(parts: Sequence[str], positions: Sequence[int], kind: str,
                      rng: random.Random) -> List[str]:
    fn = PERTURBATIONS[kind]
    out = list(parts)
    for i in positions:
        out[i] = fn(out[i], rng)
    return out


def perturb_text(text: str, kind: str, rate: float, rng: random.Random) -> str:
    """Perturb round(rate * n_eligible) randomly chosen words (at least one if rate > 0)."""
    if rate <= 0:
        return text
    parts = split_words(text)
    elig = eligible_positions(parts)
    if not elig:
        return text
    k = min(len(elig), max(1, round(rate * len(elig))))
    chosen = rng.sample(elig, k)
    return "".join(perturb_positions(parts, chosen, kind, rng))


def perturb_corpus(texts: Sequence[str], kind: str, rate: float, seed: int) -> List[str]:
    rng = random.Random(seed)
    return [perturb_text(t, kind, rate, rng) for t in texts]
