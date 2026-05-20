"""
The mixing ratio is chosen to match a target sparsity regime while avoiding a strong
NNZ-only separation confound.

For each threshold that has data in both non-attack sources, produces:
  data/processed/non_attack_mixed/sae_input.txt
  data/sae/non_attack_mixed/threshold_X/textspans_llama_group0.tsv

TextIDs in the output TSV are new sequential indices 0..N-1.
The corresponding lines in sae_input.txt map to the same indices.

Usage:
  python build_non_attack_mixed.py [--n-source-a N] [--n-source-b N] [--dry-run]
"""
import os, sys, argparse
import numpy as np
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────
SEED = 42

BASE    = os.path.join(os.path.dirname(__file__), "../../data")
PROC    = os.path.join(BASE, "processed")
SAE     = os.path.join(BASE, "sae")
# ─────────────────────────────────────────────────────────────────────────────


def load_nnz_map(tsv_path):
    """Return {text_id_str: set_of_neuron_ids} from a textspans TSV."""
    d = defaultdict(set)
    with open(tsv_path) as f:
        header = next(f)
        assert header.startswith("NeuronID"), f"Unexpected header: {header!r}"
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                d[parts[1]].add(parts[0])
    return d


def load_text_lines(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def sample_ids(nnz_map, n, rng):
    """Randomly sample n text IDs from nnz_map (without replacement)."""
    all_ids = list(nnz_map.keys())
    chosen  = rng.choice(len(all_ids), size=n, replace=False)
    return [all_ids[i] for i in chosen]


def verify_and_report(mixed_tsv, source_a_txt, source_b_txt, sel_a, sel_b):
    """Sanity checks on the merged output."""
    errors = []

    # 1. All selected source TextIDs should be valid line indices
    a_lines = load_text_lines(source_a_txt)
    b_lines = load_text_lines(source_b_txt)
    for tid in sel_a:
        if int(tid) >= len(a_lines):
            errors.append(f"Source-A TextID {tid} out of range (file has {len(a_lines)} lines)")
    for tid in sel_b:
        if int(tid) >= len(b_lines):
            errors.append(f"Source-B TextID {tid} out of range (file has {len(b_lines)} lines)")

    # 2. Output TSV: check row count and TextID range
    new_tids = set()
    with open(mixed_tsv) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                errors.append(f"Short row: {line!r}")
                continue
            new_tids.add(parts[1])

    expected_tids = set(str(i) for i in range(len(sel_a) + len(sel_b)))
    missing  = expected_tids - new_tids
    extra    = new_tids - expected_tids
    if missing:
        preview = sorted(missing)[:10]
        print(f"  [INFO] {len(missing)} selected TextIDs have 0 activations in TSV: {preview}")
    if extra:
        errors.append(f"Unexpected TextIDs in TSV: {sorted(extra)[:5]}...")

    return errors


def build_for_threshold(threshold, rng, sel_a_ids, sel_b_ids, dry_run):
    """
    Merge two non-attack source TSVs for one threshold.
    sel_a_ids / sel_b_ids: list of string TextIDs selected from each source.
    """
    a_tsv = f"{SAE}/non_attack_source_a/threshold_{threshold}/textspans_llama_group0.tsv"
    b_tsv = f"{SAE}/non_attack_source_b/threshold_{threshold}/textspans_llama_group0.tsv"
    a_txt = f"{PROC}/non_attack_source_a/sae_input.txt"
    b_txt = f"{PROC}/non_attack_source_b/sae_input.txt"

    for p in [a_tsv, b_tsv, a_txt, b_txt]:
        if not os.path.exists(p):
            print(f"  SKIP threshold={threshold}: missing {p}")
            return False

    sel_a_set = set(sel_a_ids)
    sel_b_set = set(sel_b_ids)

    # New TextID mapping: source A first, then source B.
    a_newid = {old: str(i)             for i, old in enumerate(sel_a_ids)}
    b_newid = {old: str(len(sel_a_ids) + i) for i, old in enumerate(sel_b_ids)}

    out_dir = f"{SAE}/non_attack_mixed/threshold_{threshold}"
    out_tsv = f"{out_dir}/textspans_llama_group0.tsv"

    if dry_run:
        print(f"  [dry-run] would write: {out_tsv}")
        return True

    os.makedirs(out_dir, exist_ok=True)

    row_count = 0
    with open(out_tsv, "w", encoding="utf-8") as fout:
        fout.write("NeuronID\tTextID\tScore\tSpan\n")

        for tsv_path, sel_set, newid_map in [
            (a_tsv, sel_a_set, a_newid),
            (b_tsv, sel_b_set, b_newid),
        ]:
            with open(tsv_path, encoding="utf-8") as fin:
                next(fin)  # skip header
                for line in fin:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 4:
                        continue
                    old_tid = parts[1]
                    if old_tid not in sel_set:
                        continue
                    parts[1] = newid_map[old_tid]
                    fout.write("\t".join(parts) + "\n")
                    row_count += 1

    print(f"  threshold={threshold}: wrote {row_count:,} rows → {out_tsv}")
    return True


def build_sae_input(sel_a_ids, sel_b_ids):
    """Write combined sae_input.txt for non_attack_mixed."""
    a_lines = load_text_lines(f"{PROC}/non_attack_source_a/sae_input.txt")
    b_lines = load_text_lines(f"{PROC}/non_attack_source_b/sae_input.txt")

    out_dir = f"{PROC}/non_attack_mixed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/sae_input.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for tid in sel_a_ids:
            f.write(a_lines[int(tid)] + "\n")
        for tid in sel_b_ids:
            f.write(b_lines[int(tid)] + "\n")

    print(f"Wrote {len(sel_a_ids)+len(sel_b_ids)} lines to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-source-a", type=int, default=None)
    parser.add_argument("--n-source-b", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    # ── select TextIDs from threshold=2.0 (ground truth for NNZ check) ────────
    print("Loading NNZ maps (threshold=2.0)...")
    a_nnz = load_nnz_map(f"{SAE}/non_attack_source_a/threshold_2.0/textspans_llama_group0.tsv")
    b_nnz = load_nnz_map(f"{SAE}/non_attack_source_b/threshold_2.0/textspans_llama_group0.tsv")

    n_source_a = args.n_source_a if args.n_source_a is not None else len(a_nnz)
    n_source_b = args.n_source_b if args.n_source_b is not None else len(b_nnz)

    if len(a_nnz) < n_source_a:
        print(f"ERROR: source A has only {len(a_nnz)} texts, need {n_source_a}")
        sys.exit(1)
    if len(b_nnz) < n_source_b:
        print(f"ERROR: source B has only {len(b_nnz)} texts, need {n_source_b}")
        sys.exit(1)

    sel_a = sample_ids(a_nnz, n_source_a, rng)
    sel_b = sample_ids(b_nnz, n_source_b, rng)

    # ── NNZ check ─────────────────────────────────────────────────────────────
    a_vals = np.array([len(a_nnz[t]) for t in sel_a])
    b_vals = np.array([len(b_nnz[t]) for t in sel_b])
    mixed = np.concatenate([a_vals, b_vals])

    print(f"\nNNZ check (threshold=2.0):")
    print(f"  Source-A (n={len(a_vals)}): median={np.median(a_vals):.0f}  "
          f"p25={np.percentile(a_vals,25):.0f}  p75={np.percentile(a_vals,75):.0f}")
    print(f"  Source-B (n={len(b_vals)}): median={np.median(b_vals):.0f}  "
          f"p25={np.percentile(b_vals,25):.0f}  p75={np.percentile(b_vals,75):.0f}")
    print(f"  Mixed   (n={len(mixed)}):  median={np.median(mixed):.0f}  "
          f"p25={np.percentile(mixed,25):.0f}  p75={np.percentile(mixed,75):.0f}  "
          f"p95={np.percentile(mixed,95):.0f}")

    # Optional NNZ-only comparison against a reference attack split if present
    try:
        ref_nnz_map = load_nnz_map(f"{SAE}/reference_attack/threshold_2.0/textspans_llama_group0.tsv")
        ref_vals = np.array([len(v) for v in ref_nnz_map.values()])
        from sklearn.metrics import roc_auc_score
        y = np.array([1]*len(ref_vals) + [0]*len(mixed))
        s = np.array(list(ref_vals) + list(mixed))
        auc = max(roc_auc_score(y, s), roc_auc_score(y, -s))
        print(f"  NNZ-only AUC vs reference attack split: {auc:.3f}  (target ≈ 0.50-0.60)")
    except Exception as e:
        print(f"  [WARN] Could not compute AUC: {e}")

    if args.dry_run:
        print("\n[dry-run] Stopping before writing files.")
        return

    # ── write sae_input.txt ───────────────────────────────────────────────────
    print("\nWriting sae_input.txt...")
    build_sae_input(sel_a, sel_b)

    # ── write TSVs for each available threshold ───────────────────────────────
    print("\nMerging TSV files...")
    thresholds = ["0.0", "0.5", "1.0", "1.5", "2.0", "4.0"]
    built = []
    for thr in thresholds:
        ok = build_for_threshold(thr, rng, sel_a, sel_b, dry_run=False)
        if ok:
            built.append(thr)

    # ── verify threshold=2.0 output ──────────────────────────────────────────
    print("\nVerifying threshold=2.0 output...")
    out_tsv = f"{SAE}/non_attack_mixed/threshold_2.0/textspans_llama_group0.tsv"
    errors = verify_and_report(
        out_tsv,
        f"{PROC}/non_attack_source_a/sae_input.txt",
        f"{PROC}/non_attack_source_b/sae_input.txt",
        sel_a, sel_b
    )
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("  All checks passed.")

    # ── final NNZ re-check on written data ───────────────────────────────────
    print("\nFinal NNZ re-check on written TSV...")
    written_nnz = load_nnz_map(out_tsv)
    nnz_arr = np.array([len(v) for v in written_nnz.values()])
    print(f"  n={len(nnz_arr)}  median={np.median(nnz_arr):.0f}  "
          f"p25={np.percentile(nnz_arr,25):.0f}  p75={np.percentile(nnz_arr,75):.0f}")
    expected_n = len(sel_a) + len(sel_b)
    if len(nnz_arr) < expected_n:
        print(f"  WARNING: {expected_n - len(nnz_arr)} selected texts have 0 activations in the written TSV.")

    print(f"\nDone. Thresholds built: {built}")


if __name__ == "__main__":
    main()
