# How Robust Are Misinformation Detectors to Adversarial Text?

**Character-level evasion attacks and defenses for fake-news classifiers**

Automated misinformation detectors are deployed in an *adversarial* setting: the people spreading false claims have every incentive to slip past them. Simple tricks that humans barely notice, like typos, look-alike Unicode letters (`vаccine` with a Cyrillic `а`), leetspeak (`v4cc1ne`) or invisible zero-width characters, can change what a model "sees" completely.

This project measures **how much accuracy common misinformation detectors lose under such perturbations**, runs a **black-box adversarial attack** against them, and evaluates **two inexpensive defenses**.

> This extends my MS thesis, *Detection and Classification of Twitter Misleading Posts using Machine Learning and NLP* (COMSATS University Islamabad, 2023). The thesis asked how well we can detect misleading posts; this project asks **whether those detectors still work when someone is actively trying to fool them.**

---

## Research questions

1. **RQ1: Sensitivity.** How much does macro-F1 drop when 10–50% of the words in a claim are perturbed at random?
2. **RQ2: Worst case.** How often can a query-only attacker flip a correct prediction by changing at most 30% of the words?
3. **RQ3: Representation.** Do character n-gram and subword (transformer) models degrade more gracefully than word-level models?
4. **RQ4: Defenses.** How much robustness do input normalization and adversarial data augmentation buy, and at what cost to clean accuracy?

## Threat model

The attacker:
- **can** query the detector and observe its probability output (black-box);
- **can** apply character-level edits that keep the text readable to humans;
- **cannot** see model weights or gradients, or change the training data.

## Method

| Component | Details |
|---|---|
| **Data** | [LIAR](https://aclanthology.org/P17-2067/) (Wang, 2017), 12.8k fact-checked political statements from PolitiFact. Six truthfulness labels are collapsed to binary: *pants-fire / false / barely-true* → **misleading**; *half-true / mostly-true / true* → **credible**. Any CSV with text and label columns (e.g. a tweet dataset) also works. |
| **Models** | (a) TF-IDF word 1–2-grams + logistic regression (classic baseline, as in my thesis); (b) TF-IDF character 2–5-grams + logistic regression; (c) fine-tuned DistilBERT. Any Hugging Face model can be used via `hf:<name>`. |
| **Perturbations** (`src/perturb.py`) | `typo` (swap/delete/insert/keyboard-neighbour substitution) · `homoglyph` (Latin → identical-looking Cyrillic) · `leet` (letters → digits) · `zero_width` (invisible U+200B inside words). Applied to 10/20/30/50% of words. |
| **Attack** (`src/attack.py`) | Greedy black-box attack in the spirit of DeepWordBug (Gao et al., 2018): rank words by how much deleting them lowers the true-class probability, then perturb words in that order, picking the most damaging perturbation each time, until the label flips or the budget (30% of words) is used up. |
| **Defenses** (`src/defense.py`) | **Normalize:** Unicode NFKC + zero-width removal + homoglyph and leetspeak reversal before the model sees the input. **Augment:** add one perturbed copy of every training example (adversarial data augmentation). **Both** combined. |
| **Metrics** | Macro-F1 on clean and perturbed test sets; attack success rate (share of correctly classified examples that get flipped); adversarial accuracy; words changed per successful attack; model queries per example. |
| **Rigor** | Every perturbed test set is identical across models within a seed. Results are mean ± std over 3 seeds. |

## Results

> **To do:** run the experiments (see below), then paste the tables from `results/summary.md` and the plots here, followed by 3–5 sentences answering RQ1–RQ4 in your own words.

<!--
![Robustness](results/robustness_distilbert.png)
![Attack](results/attack_success.png)

### Key findings
1. ...
2. ...
-->

## How to run

### Option A: Google Colab (recommended, free GPU)

```python
# Runtime -> Change runtime type -> T4 GPU
!git clone https://github.com/<your-username>/misinfo-robustness.git
%cd misinfo-robustness
!pip install -q -r requirements.txt

# Download LIAR
!mkdir -p data/liar && wget -q https://www.cs.ucsb.edu/~william/data/liar_dataset.zip -O data/liar.zip
!unzip -o -q data/liar.zip -d data/liar && ls data/liar

# 1) Quick check (~1 min, CPU models only, 1 seed)
!python run_experiments.py --data liar --data-dir data/liar --models tfidf_word tfidf_char --seeds 0 --out results_quick

# 2) Full study (roughly 1-2 h on a T4)
!python run_experiments.py --data liar --data-dir data/liar \
    --models tfidf_word tfidf_char distilbert --seeds 0 1 2 --out results
```

If the LIAR link ever stops working, the dataset is also mirrored on Kaggle and Hugging Face. Put `train.tsv`, `valid.tsv` and `test.tsv` in `data/liar/`.

### Option B: local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiments.py --data synthetic          # smoke test, a few seconds
python -m pytest -q                                  # unit tests
```

### Using your own dataset

```bash
python run_experiments.py --data csv --csv my_tweets.csv --text-col tweet --label-col label
```
Labels may be `0/1` or strings such as `fake`/`real` or `misleading`/`credible`.

### Useful options

| Flag | Default | Meaning |
|---|---|---|
| `--models` | `tfidf_word tfidf_char` | add `distilbert` or `hf:roberta-base` |
| `--defenses` | all four | subset of `none normalize augment normalize+augment` |
| `--rates` | `0.1 0.2 0.3 0.5` | fraction of words perturbed |
| `--attack-n` | `300` | test examples attacked (`0` = skip attack) |
| `--attack-budget` | `0.3` | max fraction of words the attacker may change |
| `--seeds` | `0 1 2` | random seeds |
| `--epochs` | `3` | transformer fine-tuning epochs |

## Output files

| File | Content |
|---|---|
| `summary.md` | Main tables (mean ± std over seeds) |
| `robustness_<model>.png` | Macro-F1 vs. perturbation rate, one panel per perturbation type |
| `attack_success.png` | Attack success rate per model and defense |
| `attack_examples.csv` | Successful adversarial examples for qualitative analysis |
| `robustness_raw.csv`, `attack_raw.csv` | Per-seed raw numbers |
| `config.json` | Exact settings used |

## Project structure

```
├── run_experiments.py     # orchestrates the full study
├── src/
│   ├── data.py            # LIAR / CSV / synthetic loaders
│   ├── perturb.py         # character-level perturbations
│   ├── models.py          # TF-IDF + LR, transformer classifier
│   ├── attack.py          # greedy black-box attack
│   └── defense.py         # normalization + adversarial augmentation
└── tests/                 # unit tests (pytest)
```

## Limitations and future work

- Only character-level attacks; word-level (synonym substitution, paraphrasing with an LLM) attacks are a natural next step.
- Normalization reverses obfuscation (homoglyphs, leetspeak, zero-width) but **not typos**, and an adaptive attacker who knows the defense could use characters outside the mapping.
- LIAR statements are short political claims; results may differ for long articles or social-media posts.
- Certified robustness (e.g. randomized smoothing for text) and robustness of LLM-based detectors are open directions.

## References

- W. Y. Wang. *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.* ACL 2017.
- J. Gao, J. Lanchantin, M. L. Soffa, Y. Qi. *Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers (DeepWordBug).* IEEE S&P Workshops 2018.
- D. Pruthi, B. Dhingra, Z. C. Lipton. *Combating Adversarial Misspellings with Robust Word Recognition.* ACL 2019.
- V. Sanh et al. *DistilBERT, a distilled version of BERT.* 2019.

## Author

**Mariam Fazal Abbasi** · MS Computer Science, COMSATS University Islamabad · [LinkedIn](#) · mariamfa0348@gmail.com

## License

MIT
