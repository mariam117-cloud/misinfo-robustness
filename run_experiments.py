"""Run the full robustness study and write tables, plots and a markdown summary to --out.

Examples
--------
Quick smoke test (CPU, seconds):
    python run_experiments.py --data synthetic --models tfidf_word tfidf_char

Main study on LIAR (GPU recommended for distilbert):
    python run_experiments.py --data liar --data-dir data/liar \
        --models tfidf_word tfidf_char distilbert --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.attack import run_attack
from src.data import load_csv, load_liar, load_synthetic
from src.defense import augment, normalize
from src.models import Preprocessed, build_model
from src.perturb import PERTURBATIONS, perturb_corpus

DEFENSES = ["none", "normalize", "augment", "normalize+augment"]


def metrics(model, texts, labels):
    pred = np.argmax(model.predict_proba(texts), axis=1)
    return {"accuracy": accuracy_score(labels, pred),
            "macro_f1": f1_score(labels, pred, average="macro")}


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def mean_std(series) -> str:
    s = pd.Series(series).dropna()
    if s.empty:
        return "n/a"
    return f"{s.mean():.3f} ± {s.std(ddof=0):.3f}" if len(s) > 1 else f"{s.mean():.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["liar", "csv", "synthetic"], default="synthetic")
    ap.add_argument("--data-dir", default="data/liar", help="folder with LIAR train/valid/test.tsv")
    ap.add_argument("--csv", help="path to CSV (with --data csv)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--models", nargs="+", default=["tfidf_word", "tfidf_char"],
                    help="tfidf_word, tfidf_char, distilbert, or hf:<model-name>")
    ap.add_argument("--defenses", nargs="+", default=DEFENSES, choices=DEFENSES)
    ap.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS), choices=list(PERTURBATIONS))
    ap.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--aug-rate", type=float, default=0.2)
    ap.add_argument("--attack-n", type=int, default=300, help="test examples to attack (0 = skip)")
    ap.add_argument("--attack-budget", type=float, default=0.3, help="max fraction of words changed")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    rob_rows, att_rows, examples = [], [], []
    tkw = {"epochs": args.epochs, "batch_size": args.batch_size}

    for seed in args.seeds:
        if args.data == "liar":
            data = load_liar(args.data_dir)
        elif args.data == "csv":
            data = load_csv(args.csv, args.text_col, args.label_col, seed=seed)
        else:
            data = load_synthetic(seed=seed)
        (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]
        print(f"\n=== seed {seed}: train={len(xtr)} val={len(xva)} test={len(xte)} "
              f"(fake share in train: {np.mean(ytr):.2f})")

        # identical perturbed test sets for every model/defense within a seed
        test_sets = {("clean", 0.0): xte}
        for k, kind in enumerate(args.perturbations):
            for r, rate in enumerate(args.rates):
                test_sets[(kind, rate)] = perturb_corpus(xte, kind, rate, seed=1000 * seed + 10 * k + r)

        for model_name in args.models:
            for defense in args.defenses:
                t0 = time.time()
                print(f"--- {model_name} | defense={defense}")
                tx, ty = list(xtr), list(ytr)
                if "augment" in defense:
                    tx, ty = augment(tx, ty, seed=seed, kinds=args.perturbations, rate=args.aug_rate)
                model = build_model(model_name, seed, tkw)
                if "normalize" in defense:
                    model = Preprocessed(model, normalize)
                model.fit(tx, ty)

                val = metrics(model, xva, yva)
                for (kind, rate), texts in test_sets.items():
                    m = metrics(model, texts, yte)
                    rob_rows.append({"seed": seed, "model": model_name, "defense": defense,
                                     "perturbation": kind, "rate": rate, **m,
                                     "val_macro_f1": val["macro_f1"]})
                clean = [r for r in rob_rows if r["seed"] == seed and r["model"] == model_name
                         and r["defense"] == defense and r["perturbation"] == "clean"][-1]
                print(f"    clean test macro-F1={clean['macro_f1']:.3f}  val macro-F1={val['macro_f1']:.3f}")

                if args.attack_n > 0:
                    a = run_attack(model, xte, yte, args.perturbations, args.attack_budget,
                                   args.attack_n, seed)
                    att_rows.append({"seed": seed, "model": model_name, "defense": defense,
                                     **{k: v for k, v in a.items() if k != "examples"}})
                    for e in a["examples"][:20]:
                        examples.append({"seed": seed, "model": model_name, "defense": defense,
                                         "label": e["label"], "orig_text": e["orig_text"],
                                         "adv_text": e["adv_text"].encode("unicode_escape").decode()
                                         if any(ord(c) > 127 for c in e["adv_text"]) else e["adv_text"],
                                         "n_changed": e["n_changed"],
                                         "p_true_before": round(e["p_true_before"], 3),
                                         "p_true_after": round(e["p_true_after"], 3)})
                    print(f"    attack success rate={a['success_rate']:.3f}  "
                          f"adversarial accuracy={a['adv_accuracy']:.3f}")
                print(f"    ({time.time() - t0:.1f}s)")

    rob = pd.DataFrame(rob_rows)
    rob.to_csv(os.path.join(args.out, "robustness_raw.csv"), index=False)
    att = pd.DataFrame(att_rows)
    if not att.empty:
        att.to_csv(os.path.join(args.out, "attack_raw.csv"), index=False)
    pd.DataFrame(examples).to_csv(os.path.join(args.out, "attack_examples.csv"), index=False)

    write_plots(rob, att, args)
    write_summary(rob, att, args)
    print(f"\nDone. Results in {args.out}/ (see summary.md)")


def write_plots(rob: pd.DataFrame, att: pd.DataFrame, args):
    agg = rob.groupby(["model", "defense", "perturbation", "rate"])["macro_f1"].mean().reset_index()
    kinds = args.perturbations
    for model_name in args.models:
        fig, axes = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 3.4), sharey=True, squeeze=False)
        for ax, kind in zip(axes[0], kinds):
            for defense in args.defenses:
                sub = agg[(agg.model == model_name) & (agg.defense == defense)]
                clean = sub[sub.perturbation == "clean"]["macro_f1"].tolist()
                pts = sub[sub.perturbation == kind].sort_values("rate")
                xs = [0.0] + pts["rate"].tolist()
                ys = clean[:1] + pts["macro_f1"].tolist()
                ax.plot(xs, ys, marker="o", label=defense)
            ax.set_title(kind)
            ax.set_xlabel("fraction of words perturbed")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("macro-F1 (test)")
        axes[0][-1].legend(fontsize=8)
        fig.suptitle(f"Robustness to random perturbations: {model_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"robustness_{model_name.replace(':', '_').replace('/', '_')}.png"), dpi=150)
        plt.close(fig)

    if not att.empty:
        g = att.groupby(["model", "defense"])["success_rate"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(1.6 * len(g) + 2, 3.6))
        labels = [f"{m}\n{d}" for m, d in zip(g.model, g.defense)]
        ax.bar(range(len(g)), g["mean"], yerr=g["std"].fillna(0), capsize=3, color="#4C72B0")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("attack success rate")
        ax.set_ylim(0, 1)
        ax.set_title(f"Greedy black-box attack (budget {args.attack_budget:.0%} of words)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "attack_success.png"), dpi=150)
        plt.close(fig)


def write_summary(rob: pd.DataFrame, att: pd.DataFrame, args):
    lines = ["# Results summary", "",
             f"Dataset: `{args.data}` · seeds: {args.seeds} · values are mean ± std over seeds.", ""]

    # Table 1: clean vs. worst-case random perturbation at the highest rate
    top = max(args.rates)
    rows = []
    for (m, d), g in rob.groupby(["model", "defense"], sort=False):
        row = {"model": m, "defense": d,
               "clean F1": mean_std(g[g.perturbation == "clean"]["macro_f1"])}
        for kind in args.perturbations:
            row[f"{kind}@{top:g}"] = mean_std(g[(g.perturbation == kind) & (g.rate == top)]["macro_f1"])
        rows.append(row)
    lines += [f"## Macro-F1 on clean and perturbed test sets ({top:.0%} of words perturbed)", "",
              md_table(pd.DataFrame(rows)), ""]

    if not att.empty:
        rows = []
        for (m, d), g in att.groupby(["model", "defense"], sort=False):
            rows.append({"model": m, "defense": d,
                         "attack success rate": mean_std(g["success_rate"]),
                         "adversarial accuracy": mean_std(g["adv_accuracy"]),
                         "words changed (succ.)": mean_std(g["mean_frac_changed"]),
                         "queries / example": mean_std(g["mean_queries"])})
        lines += [f"## Greedy black-box attack (budget {args.attack_budget:.0%} of words, "
                  f"{args.attack_n} test examples)", "", md_table(pd.DataFrame(rows)), ""]

    lines += ["Plots: `robustness_<model>.png`, `attack_success.png`. "
              "Successful adversarial examples: `attack_examples.csv`.", ""]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
