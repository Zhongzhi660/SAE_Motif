# Data Layout

This package does not include the full datasets or pretrained SAE checkpoints.  
To run the core pipeline, prepare the following directory structure under `data/`.

A tiny toy example is included in this package under:

```text
data/processed/sourceA_targetA/
data/sae/sourceA_targetA/threshold_2.0/
data/sae/non_attack_reference/threshold_2.0/
```

The toy files are only for illustrating file formats and basic plumbing. They are not intended to reproduce paper-scale results.

## Public Release Scope

In the public code release, we will include all collected attack data used by the core motif analysis pipeline.

The released attack set contains:

- `6 * 3 * 5 * 400` attack samples in total

This count corresponds to the full cross-product of:

- `6` attack-target settings
- `3` sampled attack outputs per prompt
- `5` attack methods
- `400` prompts per method

The toy files in this package are only format examples. The full released attack data will follow the same directory structure and file formats documented below.

## Overview

```text
data/
├── processed/
│   ├── sourceA_targetA/
│   │   ├── sae_input.txt
│   │   └── labels.tsv
│   └── non_attack_reference/
│       └── ...
└── sae/
    ├── sourceA_targetA/
    │   └── threshold_2.0/
    │       ├── textspans_*.tsv
    │       ├── features.tsv
    │       ├── features_explained.tsv
    │       └── full.tsv
    └── non_attack_reference/
        └── threshold_2.0/
            └── full.tsv
```

## File Formats

### `processed/<dataset_tag>/sae_input.txt`

Plain-text input file used by SAE collection.  
Each line is one sample.

Accepted content styles:

- single-turn text
- multi-turn dialogue serialized with `Human:` / `Assistant:` prefixes

Example:

```text
Human: Explain the history of cryptography.
Assistant: Sure, here is a short overview.
Human: Now focus on substitution ciphers.
```

### `processed/<dataset_tag>/labels.tsv`

Required by `src/analysis/cross_method_analysis.py`.

Expected columns:

- `TextID`
- `Method`

Recommended minimal format:

```tsv
TextID	Method
0	Crescendo
1	PAIR
2	FlipAttack
```

`TextID` should align with the row/sample order used in the corresponding SAE outputs.

### `sae/<dataset_tag>/threshold_<thr>/textspans_*.tsv`

Produced by SAE span collection.

Expected columns:

- `NeuronID`
- `TextID`
- `Score`
- `Span`

Example:

```tsv
NeuronID	TextID	Score	Span
123	0	3.14	Some decoded text span
456	0	2.71	Another high-activation span
```

Multiple `textspans_*.tsv` files can exist when collection is sharded by group.

### `sae/<dataset_tag>/threshold_<thr>/features.tsv`

Produced by `src.annotations.groupby_textspans`.

Expected columns:

- `FeatureID`
- `Words`

Example:

```tsv
FeatureID	Words
123	Span 1: ...\nSpan 2: ...
456	Span 1: ...\nSpan 2: ...
```

### `sae/<dataset_tag>/threshold_<thr>/features_explained.tsv`

Produced by `src.annotations.annotate_explanations`.

Expected columns:

- `FeatureID`
- `Verify`
- `Summary`
- `Words`

Example:

```tsv
FeatureID	Verify	Summary	Words
123	yes	Role-play framing	Span 1: ...\nSpan 2: ...
```

### `sae/<dataset_tag>/threshold_<thr>/full.tsv`

Required by the two core experiment scripts.

Expected columns:

- `NeuronID`
- `TextID`
- `Score`

Example:

```tsv
NeuronID	TextID	Score
123	0	3.14
456	0	2.71
123	1	1.88
```

This file represents sparse feature activations aggregated at the sample level.

## Minimal Requirements by Script

### `src.sae.collect_spans_llama3_safety`

Needs:

- `processed/<dataset_tag>/sae_input.txt`
- SAE checkpoint path via CLI or environment variable

Produces:

- `sae/<dataset_tag>/threshold_<thr>/textspans_*.tsv`

### `src.annotations.groupby_textspans`

Needs:

- `sae/<dataset_tag>/threshold_<thr>/textspans_*.tsv`

Produces:

- `sae/<dataset_tag>/threshold_<thr>/features.tsv`

### `src.annotations.annotate_explanations`

Needs:

- `sae/<dataset_tag>/threshold_<thr>/features.tsv`
- API credentials via environment variables

Produces:

- `sae/<dataset_tag>/threshold_<thr>/features_explained.tsv`

### `src.analysis.motif_structure_analysis`

Needs:

- `sae/<attack_dataset>/threshold_<thr>/full.tsv`
- `sae/<non_attack_dataset>/threshold_<thr>/full.tsv`

### `src.analysis.cross_method_analysis`

Needs:

- `sae/<attack_dataset>/threshold_<thr>/full.tsv`
- `sae/<non_attack_dataset>/threshold_<thr>/full.tsv`
- `processed/<attack_dataset>/labels.tsv`

Optional:

- `sae/<attack_dataset>/threshold_<thr>/features_explained.tsv`
