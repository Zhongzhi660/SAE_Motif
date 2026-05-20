# Anonymous Motif Analysis Package

This repository provides a minimal, self-contained package for SAE-based motif analysis of attack and non-attack prompts.

---

## Core Insight

**Analyze jailbreaks in feature space, not only in prompt space.**

Jailbreak queries often look very different on the surface, but many of them are built from a much smaller set of reusable latent patterns. Instead of treating each jailbreak prompt as an isolated attack instance, this project identifies **motifs**: recurring co-activation patterns of sparse autoencoder (SAE) features.

By extracting sparse features from model activations and decomposing them with non-negative matrix factorization (NMF), we can:

- expose shared latent structure across superficially different jailbreaks;
- compare attack methods in a common motif space;
- distinguish attack-specific patterns from more general language features;
- support downstream analysis and synthetic data construction using interpretable motif structure.

### Why this perspective is useful

Traditional jailbreak analysis is usually organized around prompts, templates, or attack families. Those views are helpful, but they stay close to surface form. Our working hypothesis is that many attacks differ in wording while still relying on similar internal activation structure.

This package focuses on that internal structure:

- SAE features provide sparse, more interpretable activations than dense hidden states;
- NMF turns those activations into additive motifs that can be reused across samples;
- motif-level analysis helps compare methods and reveal which structures are shared, exclusive, or compositional.

---

## Getting Started

### Installation

```bash
cd anonymous_motif_jailbreak
pip install -r requirements.txt
```

If you run modules directly, set:

```bash
export PYTHONPATH=$(pwd)
```

Optional environment variables:

```bash
export HF_HOME=/path/to/huggingface_cache
export LLAMA_SAE_PATH=/path/to/llama_sae_checkpoint.pth
export MISTRAL_SAE_PATH=/path/to/mistral_sae_checkpoint.pth
export QWEN_SAE_PATH=/path/to/qwen_sae_checkpoint.pth
export API_KEY=your_api_key
export BASE_URL=https://api.example.com/v1/
export MODEL=gpt-4o-mini
```

### Quick Check

Run the preflight check before using the package:

```bash
python scripts/check_setup.py
```

### Toy Sample Run

A tiny toy example is bundled under `data/` for demonstrating file formats and pipeline structure:

```bash
bash scripts/run_sample_pipeline.sh
```

---

## Repository Structure

```text
anonymous_motif_jailbreak/
├── README.md
├── requirements.txt
├── configs/
│   ├── data_paths.example.yaml
│   └── run_defaults.yaml
│
├── data/
│   ├── README.md
│   ├── processed/                  # toy processed inputs
│   └── sae/                        # toy SAE-format outputs
│
├── scripts/
│   ├── check_setup.py              # dependency / data preflight check
│   ├── run_sae_collection.sh
│   ├── run_core_experiments.sh
│   └── run_sample_pipeline.sh
│
├── src/
│   ├── sae/                        # SAE loading, collection, preprocessing
│   ├── annotations/                # span grouping + explanation utilities
│   └── analysis/                   # motif structure + cross-method analysis
│
└── outputs/
```

---

## Core Pipeline

### 1. SAE Span Collection

Extract thresholded SAE activations and span summaries from processed text input:

```bash
python -m src.sae.collect_spans_llama3_safety \
  0 llama 0 1 \
  --data-path data/processed/sourceA_targetA/sae_input.txt \
  --threshold 2.0 \
  --sae-path "$LLAMA_SAE_PATH"
```

Equivalent wrapper:

```bash
bash scripts/run_sae_collection.sh \
  0 llama 0 1 \
  --data-path data/processed/sourceA_targetA/sae_input.txt \
  --threshold 2.0 \
  --sae-path "$LLAMA_SAE_PATH"
```

### 2. Group Top Activation Spans

Merge shard outputs and build one feature-level summary file:

```bash
python -m src.annotations.groupby_textspans sourceA_targetA 2.0 llama
```

This reads:

```text
data/sae/sourceA_targetA/threshold_2.0/textspans_*.tsv
```

and produces:

```text
data/sae/sourceA_targetA/threshold_2.0/features.tsv
```

### 3. Annotate Feature Explanations

Generate semantic summaries for grouped features:

```bash
python -m src.annotations.annotate_explanations \
  data/sae/sourceA_targetA/threshold_2.0/features.tsv \
  data/sae/sourceA_targetA/threshold_2.0/features_explained.tsv
```

### 4. Motif Structure Analysis

Run the main motif-structure analysis over attack and non-attack sparse feature matrices:

```bash
python -m src.analysis.motif_structure_analysis \
  --atk-base data/sae/sourceA_targetA \
  --non-attack-base data/sae/non_attack_reference \
  --save-dir outputs/motif_structure
```

Equivalent wrapper:

```bash
bash scripts/run_core_experiments.sh \
  --atk-base data/sae/sourceA_targetA \
  --non-attack-base data/sae/non_attack_reference \
  --save-dir outputs/motif_structure
```

### 5. Cross-Method Motif Analysis

Compare motif overlap and feature sharing across attack methods:

```bash
python -m src.analysis.cross_method_analysis \
  --atk-base data/sae/sourceA_targetA \
  --non-attack-base data/sae/non_attack_reference \
  --labels data/processed/sourceA_targetA/labels.tsv \
  --threshold 2.0 \
  --k 40 \
  --save-dir outputs/cross_method
```

If feature explanations are available, add:

```bash
--features-explained-tsv data/sae/sourceA_targetA/threshold_2.0/features_explained.tsv
```

---

## Data Format

The package includes a tiny toy dataset for demonstration, but not the full experimental data or pretrained checkpoints.

Expected layout:

```text
data/
├── processed/
│   ├── sourceA_targetA/
│   │   ├── sae_input.txt
│   │   └── labels.tsv
│   └── non_attack_reference/
└── sae/
    ├── sourceA_targetA/
    └── non_attack_reference/
```

For full file-format details, see:

```text
data/README.md
```

---

## Notes

- This package is intentionally minimal.
- It excludes extended experiments, logs, figures, checkpoints, and manuscript files.
- Local absolute paths and machine-specific cache paths were removed.
- The included sample data is only for illustrating formats and basic pipeline plumbing; it is not intended to reproduce the full experimental results.
