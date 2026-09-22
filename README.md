# CSC1179 — Machine Translation Evaluation
### Evaluating Irish Idiom Translation with a Cross-Lingual Hybrid Metric

**Authors:** Vaseekaran Krishnan Vinodhan (A00050240) · Priyadharshini Dhanaraj Muthamil Selvi (A00050672)
**Course:** CSC1179 Machine Translation — DCU MSc Artificial Intelligence

---

## Overview

Standard n-gram overlap metrics like **chrF** compare a machine translation (MT) system's output to a single reference translation at the character level. This works reasonably well for literal text, but it breaks down for **idiomatic language**: two translations can express the exact same meaning using completely different words, and chrF penalises the "correct but different" one just as harshly as a wrong one.

This project targets that failure mode using a real dataset of Irish (*Gaeilge*) translations of English/Hiberno-English idioms and sayings (e.g. *"It's raining cats and dogs!"*, *"She's away with the fairies."*), produced by three MT systems — **Google Translate**, **Microsoft Translate**, and **ChatGPT** — and scored by a human annotator on a 1–5 scale.

We propose the **Cross-Lingual Semantic Hybrid Score**, a metric that evaluates *meaning* rather than surface character overlap, and show it correlates with human judgement far more strongly than the chrF baseline across all three systems.

## The Problem with chrF

**Example — "It's raining cats and dogs!"**

| | Text | |
|---|---|---|
| Reference Irish | *"Tá sé ag caitheamh sceana gréasaí!"* | (lit. "throwing shoemaker's knives") |
| Google's Irish output | *"Tá sé ag cur báistí go trom!"* | (lit. "raining heavily") |
| Human score | **5 / 5** | Both are valid Irish idioms for heavy rain |
| chrF score | **Very low** | Completely different characters — chrF fails |

chrF measures character n-gram overlap against a single reference. For idiomatic Irish, a correct translation can look completely different from the reference wording, so chrF penalises correct translations unfairly. What's needed is a metric that understands **meaning**, not just character patterns.

## Our Metric

```
Hybrid Score = 0.4 × Cross-Lingual Similarity
             + 0.3 × Literal BERTScore
             + 0.3 × Irish BERTScore
```

| Component | Weight | Compares | Model | Purpose |
|---|---|---|---|---|
| **Cross-Lingual Similarity** | 40% | System's Irish output ↔ English interpretation of the idiom | `paraphrase-multilingual-MiniLM-L12-v2` (cosine similarity of embeddings) | Directly measures whether the system understood the idiom's *intended meaning* — the hardest and most important aspect for this dataset |
| **Literal BERTScore** | 30% | System's English literal translation ↔ reference English literal | `roberta-large` via BERTScore F1 | Checks word-level fidelity in English |
| **Irish BERTScore** | 30% | System's Irish output ↔ reference Irish translation | `bert-base-multilingual-cased` via BERTScore F1 | Checks Irish surface phrasing quality |

**Why cross-lingual similarity carries the most weight:** most sentence-embedding models are monolingual and can't be used to directly compare an Irish translation against its English meaning. `paraphrase-multilingual-MiniLM-L12-v2` is trained specifically for cross-lingual tasks across 50+ languages, so it can map Irish and English text into the same vector space — enabling a direct "did this capture the idiom's meaning?" check, which is the critical fix for idiom evaluation. Weights are fixed (not learned) to avoid overfitting on a 6-sentence dataset.

### Model selection

Four multilingual sentence-embedding models were compared as candidates for the cross-lingual similarity component, scored by Pearson correlation with human judgement:

| Model | Google *r* | Microsoft *r* | ChatGPT *r* | Average *r* |
|---|---|---|---|---|
| **paraphrase-multilingual-MiniLM-L12-v2** ✅ | **0.9147** | **0.3552** | 0.6746 | **0.6482** |
| LaBSE | 0.7967 | 0.2257 | 0.5218 | 0.5147 |
| paraphrase-multilingual-mpnet-base-v2 | 0.8688 | -0.2948 | 0.7385 | 0.4375 |
| multilingual-e5-base | 0.6974 | -0.3919 | 0.5339 | 0.2798 |

`paraphrase-multilingual-MiniLM-L12-v2` was selected: highest average correlation, the only model with no negative correlations (i.e. most stable across systems), and lightweight (12 layers, ~66M parameters) for fast inference.

## Results

**Pearson correlation with human scores — Hybrid metric vs. chrF baseline:**

| System | chrF *r* | Hybrid *r* | Improvement | chrF ρ (Spearman) | Hybrid ρ (Spearman) |
|---|---|---|---|---|---|
| Google Translate | 0.4950 | **0.9147** | +0.4197 | 0.3586 | **0.9562** |
| Microsoft Translate | -0.5010 | **0.3552** | +0.8562 | -0.5002 | **0.2942** |
| ChatGPT | 0.2882 | **0.6746** | +0.3863 | 0.3086 | **0.4629** |

- The hybrid metric **outperforms chrF on all three MT systems**.
- On a sentence level, the hybrid score was closer to the human rating than chrF in **13 of 18** system–sentence pairs.
- Mean human scores by system: Google 3.0, Microsoft 3.83, ChatGPT 4.33.

## Repository Structure

```
.
├── data/
│   ├── groundtruth.csv            # English source, Irish reference, literal translation, interpretation
│   ├── google_translate.csv       # Google Translate outputs + literal translations
│   ├── microsoft_translate.csv    # Microsoft Translate outputs + literal translations
│   ├── chatgpt.csv                # ChatGPT outputs + literal translations
│   └── manual_evaluation.csv      # Human annotator scores (1-5) per system per sentence
├── MT_code.ipynb                  # Main notebook: baseline, metric implementation, evaluation, plots
├── presentation.pdf                # Project presentation slides
├── metric_comparison.png          # Generated: chrF vs. hybrid scatter plots against human scores
├── component_comparison.png       # Generated: per-component correlation bar chart
└── README.md
```

> Note: the `data/` CSVs are the course-provided dataset and are expected in a `data/` subfolder relative to the notebook — update the `data_dir` path in the notebook if your layout differs.

## Getting Started

### Requirements

- Python 3.9+
- `bert-score`, `sentence-transformers`, `nltk`, `scipy`, `pandas`, `numpy`, `matplotlib`, `torch`

### Installation

```bash
pip install bert-score sentence-transformers nltk scipy pandas numpy matplotlib torch
```

### Running

1. Place the course-provided CSVs in a `data/` folder alongside the notebook.
2. Open and run `MT_code.ipynb` top to bottom (Jupyter or JupyterLab).
3. The notebook will:
   - Load the data and compute the chrF baseline
   - Load the `paraphrase-multilingual-MiniLM-L12-v2` embedding model
   - Compute the three hybrid-score components and the combined score for each MT system
   - Compare Pearson/Spearman correlations against human scores, per system and per sentence
   - Generate `metric_comparison.png` and `component_comparison.png`

The first run downloads pretrained models from the Hugging Face Hub (`paraphrase-multilingual-MiniLM-L12-v2`, `roberta-large`, `bert-base-multilingual-cased`); an internet connection is required.

## Limitations

- **Dataset size:** only 6 sentences — correlation estimates are highly unstable, and a single sentence can shift Pearson *r* by ±0.20 or more.
- **Low-resource language:** Irish has limited representation in multilingual pretraining, so embeddings (and cross-lingual alignment) are weaker than for high-resource languages.
- **Single reference:** each sentence has one reference translation, so the metric cannot recognise alternative, equally valid idiomatic translations — this is the main reason Microsoft Translate's correlation (0.3552) is lower, since it sometimes uses valid but different idioms than the reference.
- **No task-specific fine-tuning:** the embedding model is general-purpose, not fine-tuned for translation evaluation or for Irish specifically.

## Future Work

- Expand the evaluation set to 50+ sentences and multiple references per sentence to stabilise correlation estimates.
- Fine-tune or adapt the embedding model on Irish-specific data.
- Learn component weights from a larger dataset instead of using fixed weights.
- Combine with complementary metrics (e.g. METEOR, TER) and explore human-in-the-loop evaluation.

## Acknowledgements

Dataset and project brief provided by Ellen Rushe, CSC1179 Machine Translation, DCU (March 2026).
