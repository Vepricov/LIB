## Rebuttal Experiments

Additional experimental results and analysis conducted during the review process (see `rebuttal/` folder):

### 1. Adapter Selection Stability (Jaccard Similarity)

We conduct a **new ablation study** to evaluate importance selection consistency across 5 random seeds. Using Jaccard similarity, we observe high stability: most seed pairs achieved a similarity of 1.0, with a minimum of 0.82 (representing a 9/10 adapter overlap). This confirms WeightLoRA identifies architecturally critical layers rather than being driven by initialization noise.

**Location:** `rebuttal/jaccard_similarity.pdf`

### 2. Llama-3.1 70B Experiments

We evaluate LoRA, WeightLoRA, and WeightLoRA+ across multiple ranks ($r \in \{2, 4, 8, 16\}$) on the GSM8K dataset, demonstrating that WeightLoRA+ consistently achieves superior performance, particularly at higher ranks and WeightLoRA keeps LoRA metrics. At the same time, WeightLoRA attains competitive results while using approximately three times fewer trainable parameters.

**Location:** `rebuttal/llama70b_gsm8k_rank_comparison.pdf`

## Experiments

This repository contains code to reproduce the experimental results for WeightLoRA and WeightLoRA+ on encoder-only, encoder–decoder, and decoder-only Transformer models.

### Methods
- **LoRA**: standard low-rank adaptation baseline.
- **WeightLoRA**: short warm-up phase estimates adapter importance and selects a subset of adapters; training then continues as standard LoRA on the selected subset.
- **WeightLoRA+**: budget-preserving variant; after warm-up, a subset of adapters is kept and its rank is increased while keeping the total LoRA parameter budget fixed.

### Benchmarks and models

#### Natural Language Understanding (GLUE)
- **Model**: DeBERTaV3-base.
- **Setup A (attention-only adapters)**: adapters attached to self-attention projections (fixed rank, default `r=8`) to enable per-layer adapter selection comparisons with dynamic/pruning-based baselines.
- **Setup B (adapters on all linear layers)**: tuned LoRA baseline across multiple ranks; compares LoRA vs WeightLoRA vs WeightLoRA+ under matched training protocol.
- **Model**: Llama3-7B (scaling experiment).
- **Metrics**: standard GLUE metrics (accuracy / MCC / correlation depending on task).

#### Classical LLM Benchmarks
- **Model**: Qwen3-8B.
- **Tasks**: MathQA, GSM8K, HellaSwag, BoolQ, ARC-Challenge.
- **Metrics**: accuracy.

#### Question Answering
- **Model**: DeBERTaV3-base.
- **Datasets**: SQuAD v1.1 and SQuAD v2.0.
- **Metric**: F1 score.

#### Natural Language Generation
- **Model**: BART-large.
- **Datasets**: XSum and CNN/DailyMail.
- **Metric**: ROUGE-1.

### Reporting
- For each benchmark, comparisons include standard LoRA, WeightLoRA, and WeightLoRA+.
- When applicable, results are reported across multiple ranks (e.g., `r ∈ {1,2,4,8,16}` depending on the setting) and averaged over multiple random seeds.
