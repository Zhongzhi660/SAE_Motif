"""
Step 1: Group SAE text spans by neuron and generate top-k activation summary.

Usage:
    python groupby_textspans.py <DATASET_TAG> [THRESHOLD] [MODEL_KEY]

Examples:
    python groupby_textspans.py sourceA_targetA 2.0 llama
    python groupby_textspans.py sourceB_targetB 2.0 mistral
    python groupby_textspans.py sourceC_targetC 2.0 qwen

Input:  data/sae/{dataset_tag}/threshold_{threshold}/textspans_*.tsv
Output: data/sae/{dataset_tag}/threshold_{threshold}/features.tsv

This script:
1. Merges and deduplicates spans across groups
2. For each of 65536 neurons, collects top-K text spans
3. Outputs a single TSV ready for annotate_explanations.py
"""
import time
import re
import os
import sys
import string
import json
import concurrent
import multiprocessing

import pandas as pd
import transformers as trf
import tqdm


CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Auto-detect model_key from attacker_target name
TARGET_TO_MODEL = {
    "llama": "llama",
    "mistral": "mistral",
    "qwen2": "qwen",
}

# Model map for tokenizer selection
TOKENIZER_MAP = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen": "Qwen/Qwen2-7B-Instruct",
}


class Reader:

    def __init__(self, fpath, model_key="llama"):
        self.df = pd.read_csv(fpath, engine="python",
            on_bad_lines="skip", encoding="utf8", sep="\t")
        print(f"Loaded {len(self.df)} records.")
        self.df.sort_values("NeuronID", inplace=True)

        tokenizer_name = TOKENIZER_MAP.get(model_key, TOKENIZER_MAP["llama"])
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = trf.AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=False, padding_side="right", cache_dir=CACHE_DIR)

    def select(self, idx, topK=5, key="Span"):
        i = self.df.NeuronID.searchsorted(idx, side="left")
        j = self.df.NeuronID.searchsorted(idx, side="right")
        if not i <= j - 1:
            return []
        df = self.df.iloc[i:j]
        df = df.sort_values(by="Score", ascending=False)
        return df[key].tolist()[:topK]

    def truncate(self, span, topN=10):
        if not isinstance(span, str):
            span = ''
        ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(span))[-topN:]
        return self.tokenizer.batch_decode([ids])[0]

    def get_neuron_spans(self, idx, topK, topN=10):
        spans = [self.truncate(_, topN) for _ in self.select(idx, topK)]
        return "\n".join("Span %d: %s" % pair
                         for pair in enumerate(spans, 1))


def detect_model_key(dataset_tag):
    """Auto-detect model_key from dataset tag suffix like 'source_target'."""
    target = dataset_tag.split("_", 1)[1]
    model_key = TARGET_TO_MODEL.get(target)
    if model_key is None:
        print(f"WARNING: Cannot auto-detect model_key from '{attacker_target}', defaulting to 'llama'")
        model_key = "llama"
    return model_key


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python groupby_textspans.py <DATASET_TAG> [THRESHOLD] [MODEL_KEY]")
        print()
        print("  DATASET_TAG:     source_target or any <name>_<target> style tag")
        print("  THRESHOLD:       (optional) default 2.0")
        print("  MODEL_KEY:       (optional) auto-detected from DATASET_TAG")
        print()
        print("Examples:")
        print("  python groupby_textspans.py sourceA_targetA")
        print("  python groupby_textspans.py sourceB_targetB 2.0 mistral")
        sys.exit(1)

    dataset_tag = sys.argv[1]
    threshold = sys.argv[2] if len(sys.argv) >= 3 else "2.0"
    model_key = sys.argv[3] if len(sys.argv) >= 4 else detect_model_key(dataset_tag)

    # Paths
    folder = os.path.join(PROJECT_ROOT, "data", "sae", dataset_tag, f"threshold_{threshold}")
    output_file = os.path.join(folder, "features.tsv")

    print(f"===== Groupby Text Spans =====")
    print(f"  Attacker-Target: {attacker_target}")
    print(f"  Threshold:       {threshold}")
    print(f"  Model:           {model_key}")
    print(f"  Input folder:    {folder}")
    print(f"  Output:          {output_file}")
    print(f"==============================")

    if not os.path.isdir(folder):
        print(f"ERROR: Input folder not found: {folder}")
        print(f"  Run SAE extraction first:")
        print(f"  bash run_collect_spans_llama3_safety.sh {attacker_target} {model_key} 0 {threshold}")
        sys.exit(1)

    # Step 1: Merge and deduplicate all group TSVs
    full_path = os.path.join(folder, "full.tsv")
    dedup_path = os.path.join(folder, "full_deduplicated.tsv")

    group_files = sorted([f for f in os.listdir(folder) if f.startswith("textspans_") and f.endswith(".tsv")])
    if not group_files:
        print(f"ERROR: No textspans_*.tsv files found in {folder}")
        sys.exit(1)
    print(f"Found {len(group_files)} group files, merging...")

    with open(full_path, "w", encoding="utf8") as out:
        header_written = False
        for gf in group_files:
            with open(os.path.join(folder, gf), encoding="utf8") as f:
                header = f.readline()
                if not header_written:
                    out.write(header)
                    header_written = True
                for line in f:
                    out.write(line)

    # Deduplicate
    duplicated = set()
    with open(dedup_path, "w", encoding="utf8") as f:
        with open(full_path, encoding="utf8") as g:
            f.write(g.readline())
            for row in g:
                temp = row.split("\t")
                key = (temp[0], temp[-1])
                if key in duplicated:
                    continue
                f.write(row)
                duplicated.add(key)

    # Step 2: Generate per-neuron feature summary
    reader = Reader(full_path, model_key)

    bar = tqdm.tqdm(total=2**16, desc="Generating features")
    activated = 0
    with open(output_file, "w", encoding="utf8") as f:
        f.write("FeatureID\tWords\n")
        for idx in range(2 ** 16):
            span = reader.get_neuron_spans(idx, topK=10)
            if "Span" in span:
                activated += 1
            f.write("%d\t%s\n" % (idx, span.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "")))
            bar.update(1)

    print(f"Totally {activated} neurons are activated.")
    print(f"Saved to {output_file}")
