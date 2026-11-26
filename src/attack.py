"""Greedy black-box adversarial attack (in the spirit of DeepWordBug, Gao et al. 2018).

The attacker can only query predicted probabilities. For a correctly classified text:
1. rank words by importance = drop in true-class probability when the word is deleted;
2. walk down the ranking, perturbing one word at a time with whichever perturbation
   type lowers the true-class probability the most;
3. stop when the prediction flips (success) or the budget of perturbed words is used up.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from .perturb import PERTURBATIONS, eligible_positions, split_words


def attack_one(predict_proba, text: str, label: int, kinds: Sequence[str],
               budget: float, rng: random.Random) -> Optional[Dict]:
    p0 = predict_proba([text])[0]
    queries = 1
    if int(np.argmax(p0)) != label:
        return None  # already misclassified: not attacked
    parts = split_words(text)
    elig = eligible_positions(parts)
    if not elig:
        return {"success": False, "adv_text": text, "n_changed": 0, "n_words": 0,
                "queries": queries, "p_true_before": float(p0[label]), "p_true_after": float(p0[label])}

    # 1. word importance by deletion
    deleted = ["".join(parts[:i] + [""] + parts[i + 1:]) for i in elig]
    probs = predict_proba(deleted)
    queries += len(deleted)
    importance = p0[label] - probs[:, label]
    order = [elig[j] for j in np.argsort(-importance)]

    # 2-3. greedy perturbation
    max_changes = max(1, math.ceil(budget * len(elig)))
    cur, p_true, n_changed, success = list(parts), float(p0[label]), 0, False
    for pos in order[:max_changes]:
        cands = []
        for kind in kinds:
            new = list(cur)
            new[pos] = PERTURBATIONS[kind](cur[pos], rng)
            if new[pos] != cur[pos]:
                cands.append(new)
        if not cands:
            continue
        cp = predict_proba(["".join(c) for c in cands])
        queries += len(cands)
        best = int(np.argmin(cp[:, label]))
        if cp[best, label] >= p_true:
            continue  # no candidate helps the attacker; skip this word
        cur, p_true, n_changed = cands[best], float(cp[best, label]), n_changed + 1
        if int(np.argmax(cp[best])) != label:
            success = True
            break
    return {"success": success, "adv_text": "".join(cur), "n_changed": n_changed,
            "n_words": len(elig), "queries": queries,
            "p_true_before": float(p0[label]), "p_true_after": p_true}


def run_attack(model, texts: Sequence[str], labels: Sequence[int], kinds: Sequence[str],
               budget: float = 0.3, n: int = 300, seed: int = 0) -> Dict:
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    idx = idx[:n]
    results: List[Dict] = []
    for i in idx:
        r = attack_one(model.predict_proba, texts[i], labels[i], kinds, budget, rng)
        if r is not None:
            r.update({"orig_text": texts[i], "label": labels[i]})
            results.append(r)
    n_att = len(results)
    succ = [r for r in results if r["success"]]
    return {
        "n_sampled": len(idx),
        "n_attacked": n_att,                       # originally correct
        "success_rate": len(succ) / n_att if n_att else float("nan"),
        # accuracy on the sample once the attacker has done its best
        "adv_accuracy": (n_att - len(succ)) / len(idx) if idx else float("nan"),
        "mean_frac_changed": float(np.mean([r["n_changed"] / max(r["n_words"], 1) for r in succ]))
        if succ else float("nan"),
        "mean_queries": float(np.mean([r["queries"] for r in results])) if results else float("nan"),
        "examples": succ,
    }
