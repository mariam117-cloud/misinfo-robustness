import random

import numpy as np

from src.attack import attack_one, run_attack
from src.data import load_synthetic
from src.defense import augment, normalize
from src.models import Preprocessed, TfidfLR
from src.perturb import PERTURBATIONS, ZERO_WIDTH_SPACE, perturb_text, split_words

TEXT = "The government secretly banned the vaccine report yesterday"


def test_split_roundtrip():
    t = "  Hello   world,\tthis is\na test "
    assert "".join(split_words(t)) == t


def test_each_perturbation_changes_text_and_keeps_word_count():
    for kind in PERTURBATIONS:
        out = perturb_text(TEXT, kind, 0.5, random.Random(1))
        assert out != TEXT, kind
        assert len(out.split()) == len(TEXT.split()), kind


def test_rate_zero_is_identity():
    assert perturb_text(TEXT, "typo", 0.0, random.Random(0)) == TEXT


def test_normalize_reverses_obfuscation():
    for kind in ["homoglyph", "zero_width", "leet"]:
        out = perturb_text(TEXT, kind, 1.0, random.Random(3))
        # leetspeak loses capitalisation ("7h3" -> "the"), so compare case-insensitively
        assert normalize(out).lower() == TEXT.lower(), (kind, out)
    assert ZERO_WIDTH_SPACE not in normalize("vac​cine")


def test_normalize_keeps_numbers():
    t = "In 2016 about 45 percent of COVID-19 cases, the 4th wave, cost $5"
    assert normalize(t) == t


def test_augment_doubles_data():
    x, y = augment(["a long sentence here", "another long sentence"], [0, 1], seed=0)
    assert len(x) == 4 and y == [0, 1, 0, 1]


class KeywordModel:
    """Toy model: predicts 'fake' iff the exact word 'hoax' appears."""
    def predict_proba(self, texts):
        p = np.array([0.9 if "hoax" in t.split() else 0.1 for t in texts])
        return np.stack([1 - p, p], axis=1)


def test_attack_flips_keyword_model():
    r = attack_one(KeywordModel().predict_proba, "this whole story is a hoax", 1,
                   list(PERTURBATIONS), budget=0.5, rng=random.Random(0))
    assert r["success"] and r["n_changed"] == 1


def test_end_to_end_tfidf():
    d = load_synthetic(n=600, seed=0)
    (xtr, ytr), (xte, yte) = d["train"], d["test"]
    m = Preprocessed(TfidfLR("char"), normalize).fit(xtr, ytr)
    acc = (m.predict_proba(xte).argmax(1) == np.array(yte)).mean()
    assert acc > 0.7
    res = run_attack(m, xte, yte, list(PERTURBATIONS), budget=0.3, n=30)
    assert 0.0 <= res["success_rate"] <= 1.0
