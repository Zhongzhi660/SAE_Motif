#!/usr/bin/env python3
import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PACKAGES = [
    "accelerate",
    "datasets",
    "filelock",
    "numpy",
    "openai",
    "pandas",
    "pyparsing",
    "scipy",
    "sklearn",
    "matplotlib",
    "tqdm",
    "transformers",
    "torch",
]


CORE_FILES = [
    "src/sae/autoencoder.py",
    "src/sae/corpus.py",
    "src/sae/generator.py",
    "src/sae/llm_surgery.py",
    "src/sae/collect_spans_llama3_safety.py",
    "src/annotations/groupby_textspans.py",
    "src/annotations/annotate_explanations.py",
    "src/analysis/motif_structure_analysis.py",
    "src/analysis/cross_method_analysis.py",
]


def status(ok: bool, msg: str):
    prefix = "[OK]" if ok else "[MISSING]"
    print(f"{prefix} {msg}")
    return ok


def check_packages():
    print("== Python packages ==")
    ok_all = True
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            status(True, pkg)
        except Exception as e:
            ok_all &= status(False, f"{pkg} ({e})")
    return ok_all


def check_core_files():
    print("\n== Core repository files ==")
    ok_all = True
    for rel in CORE_FILES:
        ok_all &= status((ROOT / rel).exists(), rel)
    return ok_all


def check_env():
    print("\n== Optional environment variables ==")
    env_keys = [
        "HF_HOME",
        "LLAMA_SAE_PATH",
        "MISTRAL_SAE_PATH",
        "QWEN_SAE_PATH",
        "API_KEY",
        "BASE_URL",
        "MODEL",
    ]
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            status(True, f"{key} is set")
        else:
            print(f"[INFO] {key} is not set")
    return True


def check_data_layout():
    print("\n== Data directories ==")
    data_dir = ROOT / "data"
    processed = data_dir / "processed"
    sae = data_dir / "sae"
    outputs = ROOT / "outputs"
    status(data_dir.exists(), "data/")
    status(processed.exists(), "data/processed/")
    status(sae.exists(), "data/sae/")
    status(outputs.exists(), "outputs/")
    return True


def print_examples():
    print("\n== Minimal expected files for the core pipeline ==")
    print("Attack dataset:")
    print("  data/processed/sourceA_targetA/sae_input.txt")
    print("  data/processed/sourceA_targetA/labels.tsv")
    print("  data/sae/sourceA_targetA/threshold_2.0/full.tsv")
    print("Non-attack dataset:")
    print("  data/sae/non_attack_reference/threshold_2.0/full.tsv")
    print("Optional:")
    print("  data/sae/sourceA_targetA/threshold_2.0/features.tsv")
    print("  data/sae/sourceA_targetA/threshold_2.0/features_explained.tsv")


def main():
    print(f"Repository root: {ROOT}")
    ok_pkg = check_packages()
    ok_files = check_core_files()
    check_env()
    check_data_layout()
    print_examples()
    print("\n== Summary ==")
    if ok_pkg and ok_files:
        print("[OK] Core package structure looks usable.")
    else:
        print("[WARN] Fix missing packages or files before running experiments.")
        sys.exit(1)


if __name__ == "__main__":
    main()
