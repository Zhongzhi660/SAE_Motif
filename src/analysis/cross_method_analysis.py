from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.font_manager as mfont_manager
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


_PAPER_SERIF_FAMILY: Optional[str] = None


def _paper_serif_fontfamily() -> str:
    global _PAPER_SERIF_FAMILY
    if _PAPER_SERIF_FAMILY is not None:
        return _PAPER_SERIF_FAMILY
    preferred = (
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "Nimbus Roman No9 L",
        "TeX Gyre Termes",
        "DejaVu Serif",
    )
    names = {font.name for font in mfont_manager.fontManager.ttflist}
    for fam in preferred:
        if fam in names:
            _PAPER_SERIF_FAMILY = fam
            return _PAPER_SERIF_FAMILY
    for font in mfont_manager.fontManager.ttflist:
        base = os.path.basename(font.fname).lower()
        b = base.replace(" ", "").replace("_", "").replace("-", "")
        if "timesnewroman" in b or b.startswith("timesn"):
            _PAPER_SERIF_FAMILY = font.name
            return _PAPER_SERIF_FAMILY
    _PAPER_SERIF_FAMILY = "DejaVu Serif"
    return _PAPER_SERIF_FAMILY

N_FEATURES = 65536
FLIP_METHOD = "FlipAttack"
CORE_SEMANTIC_METHODS = ("Crescendo", "PAIR", "XTeaming")


def _flip_attack_span_mechanism_score(span_raw: str) -> float:
    """
    Heuristic match to the reversal-style attack format: user message is TASK is '<reversed text>';
    system text mentions FLIPPING each character / NEVER SAY IT / KEEP IN YOUR MIND.
    Down-weight static Rule 1–4 / JUST SOLVE IT blocks (appear in many methods' jailbreak shells).
    """
    t = str(span_raw).lower().replace("\n", " ")
    s = 0.0
    if "task is '" in t or 'task is "' in t or "task is `" in t:
        s += 10.0
    if "flipping each character" in t:
        s += 7.0
    if "never say it" in t or "keep in your mind" in t:
        s += 5.0
    if "step 1." in t and "step 2." in t:
        s += 2.0
    # Generic template shared with many attacks — not flip-mechanism specific
    for needle in (
        "rule 1.",
        "rule 2.",
        "rule 3.",
        "rule 4.",
        "never never never say the words",
        "just solve it",
        "original task",
    ):
        if needle in t:
            s -= 2.5
    return s


def _span_is_flip_consistent(span_raw: str, min_mech: float = 4.0) -> bool:
    """True if span looks like FlipAttack pipeline (TASK wrapper / high mechanism score)."""
    t = str(span_raw).lower()
    if "task is'" in t or 'task is "' in t or "task is `" in t:
        return True
    return _flip_attack_span_mechanism_score(span_raw) >= min_mech


def _final_decision_from_task(task_raw: str) -> str:
    """Parse the safety explainer verdict embedded in the Task column."""
    t = (task_raw or "").replace("[[yes]]", "[[ Yes ]]")
    if "[[ Yes ]]" in t:
        return "yes"
    if "[[ No ]]" in t:
        return "no"
    if "[[ Maybe ]]" in t:
        return "maybe"
    return ""


def _clip_text(s: str, max_len: int) -> str:
    s = str(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _explain_score_shared_semantic(summary: str, task: str, words: str) -> float:
    """Prefer explainer-aligned *harmful / jailbreak* semantics; down-rank benign writing aids."""
    blob = f"{summary} {task} {words}".lower()
    s = 0.0
    fd = _final_decision_from_task(task)
    if fd == "yes":
        s += 14.0
    elif fd == "maybe":
        s += 5.0
    elif fd == "no":
        s -= 10.0
    sum_l = (summary or "").strip().lower()
    if sum_l.startswith("cannot tell") or sum_l == "cannot tell.":
        s -= 8.0
    for kw in (
        "jailbreak",
        "harmful",
        "malicious",
        "bypass",
        "unsafe",
        "attack",
        "obfuscat",
        "injection",
        "manipulat",
        "deceive",
        "dangerous",
        "evade",
        "illegitimate",
    ):
        if kw in blob:
            s += 1.2
    for kw in (
        "persuasive writing",
        "blog post",
        "historical fiction",
        "screenplay",
        "academic purposes",
        "ethical",
        "i'd be happy",
        "happy to help",
        "benign",
        "constructive assistance",
        "professional communication",
        "organizational psychology",
    ):
        if kw in blob:
            s -= 2.2
    return s


def _explain_score_flip_tsv(summary: str, task: str, words: str) -> float:
    """TSV-side cues for FlipAttack (reversal / character flip / TASK wrapper)."""
    blob = f"{summary} {task} {words}".lower()
    s = 0.0
    for kw in (
        "revers",
        "flip",
        "backward",
        "left-to-right",
        "character",
        "obfuscat",
        "mirrored",
        "read backwards",
    ):
        if kw in blob:
            s += 2.2
    if "task is" in blob:
        s += 4.0
    if _final_decision_from_task(task) == "yes":
        s += 3.0
    return s


def _explain_score_universal(summary: str, task: str, words: str) -> float:
    """Prefer clear harmful verdict + substantive cross-cutting wording; penalise toy reversal demos."""
    sum_s = (summary or "").strip()
    blob = f"{sum_s} {task} {words}".lower()
    s = 0.0
    fd = _final_decision_from_task(task)
    if fd == "yes":
        s += 14.0
    elif fd == "maybe":
        s += 6.0
    elif fd == "no":
        s -= 8.0
    sum_l = sum_s.lower()
    if sum_l.startswith("cannot tell") or sum_l == "cannot tell.":
        s -= 7.0
    # Down-rank trivial "X is Y reversed" one-liners (e.g. "taht tse" / test-that demos)
    if "taht " in sum_l or "tset " in sum_l:
        s -= 16.0
    if "which is \"" in sum_l and "reversed" in sum_l and len(sum_s) < 170:
        s -= 14.0
    if "test that" in sum_l and "reversed" in sum_l:
        s -= 14.0
    if "specific text pattern" in sum_l and len(sum_s) < 140:
        s -= 8.0
    if len(sum_s) < 55:
        s -= 6.0
    if len(sum_s) >= 200:
        s += 4.0
    elif len(sum_s) >= 120:
        s += 2.0
    for kw in ("various", "multiple", "different types", "pattern across", "systematic", "common"):
        if kw in blob:
            s += 0.9
    for kw in ("jailbreak", "harmful", "bypass", "unsafe", "attack", "malicious"):
        if kw in blob:
            s += 0.8
    for kw in (
        "requests",
        "procedural",
        "detailed",
        "illicit",
        "sensitive",
        "dangerous",
        "instruction",
        "harmful topics",
    ):
        if kw in blob:
            s += 0.65
    return s


def load_features_explained_df(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path or not os.path.isfile(path):
        return None
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "FeatureID" not in df.columns:
        return None
    df = df.drop_duplicates(subset=["FeatureID"], keep="first").set_index("FeatureID")
    return df


def _load_mat(base: str, thr: str) -> Tuple[sparse.csr_matrix, np.ndarray]:
    path = f"{base}/threshold_{thr}/full.tsv"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", usecols=["NeuronID", "TextID", "Score"])
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    text_ids = np.sort(df["TextID"].unique())
    ridx = pd.Index(text_ids).get_indexer(df["TextID"])
    sp = sparse.csr_matrix(
        (df["Score"].values.astype(np.float32), (ridx, df["NeuronID"].values)),
        shape=(len(text_ids), N_FEATURES),
        dtype=np.float32,
    )
    return sp, text_ids


def load_combined(atk_base: str, nrm_base: str, thr: str):
    atk, atk_tids = _load_mat(atk_base, thr)
    nrm, _ = _load_mat(nrm_base, thr)
    fids = np.array(sorted(set(atk.nonzero()[1]) | set(nrm.nonzero()[1])), dtype=np.int64)
    return atk[:, fids].tocsr(), nrm[:, fids].tocsr(), fids, atk_tids


def load_labels_aligned(labels_path: str, text_ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(labels_path, sep="\t")
    method_col = None
    for col in ["attack_method", "method", "label"]:
        if col in df.columns:
            method_col = col
            break
    if method_col is None:
        raise ValueError("No method column in labels.tsv")
    id_col = next((c for c in ["TextID", "text_id", "id", "ID"] if c in df.columns), None)
    if id_col is not None:
        lmap = df[[id_col, method_col]].drop_duplicates(subset=[id_col])
        merged = pd.DataFrame({id_col: text_ids}).merge(lmap, on=id_col, how="left")
        if merged[method_col].isna().any():
            raise ValueError("Missing labels after TextID alignment")
        return merged[method_col].astype(str).values
    if len(df) != len(text_ids):
        raise ValueError("Row count mismatch and no TextID column")
    return df[method_col].astype(str).values


def load_spans_df(atk_base: str, thr: str, atk_tids: np.ndarray, method_arr: np.ndarray) -> pd.DataFrame:
    """
    Load full.tsv including the Span column and attach attack_method labels.

    Returns a DataFrame with columns:
        NeuronID, TextID, Score, Span, attack_method
    Only rows whose TextID belongs to the attack set are kept.
    """
    path = f"{atk_base}/threshold_{thr}/full.tsv"
    df = pd.read_csv(path, sep="\t", usecols=["NeuronID", "TextID", "Score", "Span"],
                     dtype={"NeuronID": np.int32, "TextID": np.int64, "Score": np.float32})
    tid_to_method = {int(tid): m for tid, m in zip(atk_tids, method_arr)}
    df["attack_method"] = df["TextID"].map(tid_to_method)
    # Keep only rows belonging to attack samples (drop non-attack / unmatched)
    df = df[df["attack_method"].notna()].copy()
    df["attack_method"] = df["attack_method"].astype(str)
    return df.reset_index(drop=True)


def run_combined_nmf(atk_mat: sparse.csr_matrix, nrm_mat: sparse.csr_matrix, k: int):
    combined = sparse.vstack([atk_mat, nrm_mat]).tocsr()
    nmf = NMF(n_components=k, init="nndsvda", random_state=42, max_iter=500)
    W = nmf.fit_transform(combined)
    W_atk = W[: atk_mat.shape[0]]
    W_nrm = W[atk_mat.shape[0] :]
    ratio = W_atk.mean(axis=0) / (W_nrm.mean(axis=0) + 1e-10)
    return W_atk, ratio


def _method_profiles(W_atk: np.ndarray, method_arr: np.ndarray, methods: List[str], atk_motifs: np.ndarray) -> np.ndarray:
    prof = np.zeros((len(methods), len(atk_motifs)), dtype=np.float32)
    for i, m in enumerate(methods):
        mask = method_arr == m
        if mask.sum() == 0:
            continue
        prof[i] = W_atk[mask][:, atk_motifs].mean(axis=0)
    return prof


def method_sets_from_profiles(
    method_profiles: np.ndarray,
    methods: List[str],
    atk_motifs: np.ndarray,
    set_mode: str = "topk",
    top_k: int = 10,
    act_thr: float = 0.01,
) -> Dict[str, set]:
    method_sets: Dict[str, set] = {}
    for i, m in enumerate(methods):
        vec = method_profiles[i]
        if set_mode == "topk":
            pos_idx = np.where(vec > 0)[0]
            if len(pos_idx) == 0:
                chosen = np.array([], dtype=int)
            else:
                order = pos_idx[np.argsort(-vec[pos_idx])]
                chosen = order[: min(top_k, len(order))]
        elif set_mode == "threshold":
            chosen = np.where(vec > act_thr)[0]
        else:
            raise ValueError(f"Unknown set_mode: {set_mode}")
        method_sets[m] = set(int(atk_motifs[j]) for j in chosen)
    return method_sets


def jaccard_matrix(method_sets: Dict[str, set], methods: List[str]) -> np.ndarray:
    n = len(methods)
    jac = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            s1, s2 = method_sets[methods[i]], method_sets[methods[j]]
            union = len(s1 | s2)
            jac[i, j] = len(s1 & s2) / union if union > 0 else 0.0
    return jac


def weighted_jaccard_matrix(method_profiles: np.ndarray) -> np.ndarray:
    """Weighted Jaccard on non-negative mean-activation profiles.

    For two profile vectors a,b:
        sim(a,b) = sum_i min(a_i, b_i) / sum_i max(a_i, b_i)
    This keeps the "overlap" semantics while using continuous activations.
    """
    X = method_profiles.astype(np.float64, copy=False)
    # L1 normalization reduces scale effects across methods.
    X = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    n = X.shape[0]
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            num = np.minimum(X[i], X[j]).sum()
            den = np.maximum(X[i], X[j]).sum()
            sim[i, j] = float(num / den) if den > 0 else 0.0
    return sim


def grouped_stats(jac: np.ndarray, methods: List[str], semantic_group: Tuple[str, ...]):
    sem_idx = [i for i, m in enumerate(methods) if m in semantic_group]
    flip_idx = methods.index(FLIP_METHOD)
    sem_pairs, flip_pairs = [], []
    for a in sem_idx:
        for b in sem_idx:
            if a < b:
                sem_pairs.append(float(jac[a, b]))
    for si in sem_idx:
        flip_pairs.append(float(jac[flip_idx, si]))
    sem_mean = float(np.mean(sem_pairs)) if sem_pairs else float("nan")
    flip_mean = float(np.mean(flip_pairs)) if flip_pairs else float("nan")
    return sem_mean, flip_mean, sem_pairs, flip_pairs


def permutation_test_vector(
    W_atk: np.ndarray,
    method_arr: np.ndarray,
    methods: List[str],
    atk_motifs: np.ndarray,
    semantic_group: Tuple[str, ...],
    n_perm: int,
    seed: int = 42,
):
    """Permutation test for vector-based weighted-Jaccard gap (semantic - flip)."""
    rng = np.random.default_rng(seed)
    obs_prof = _method_profiles(W_atk, method_arr, methods, atk_motifs)
    sim_obs = weighted_jaccard_matrix(obs_prof)
    sem_obs, flip_obs, _, _ = grouped_stats(sim_obs, methods, semantic_group)
    observed_gap = sem_obs - flip_obs

    perm_gaps = []
    for _ in range(n_perm):
        perm_labels = rng.permutation(method_arr)
        prof = _method_profiles(W_atk, perm_labels, methods, atk_motifs)
        sim = weighted_jaccard_matrix(prof)
        s, f, _, _ = grouped_stats(sim, methods, semantic_group)
        perm_gaps.append(s - f)
    perm_gaps = np.array(perm_gaps, dtype=np.float64)
    p_value = float((perm_gaps >= observed_gap).mean())
    return observed_gap, p_value, perm_gaps


def bootstrap_ci_vector(
    W_atk: np.ndarray,
    method_arr: np.ndarray,
    methods: List[str],
    atk_motifs: np.ndarray,
    semantic_group: Tuple[str, ...],
    n_boot: int,
    seed: int = 99,
):
    """Bootstrap CI for vector-based weighted-Jaccard group means."""
    rng = np.random.default_rng(seed)
    sem_boots, flip_boots = [], []
    method_indices = {m: np.where(method_arr == m)[0] for m in methods}
    n_total = len(method_arr)
    for _ in range(n_boot):
        boot_W = np.zeros_like(W_atk)
        boot_labels = np.empty(n_total, dtype=object)
        for m, idx in method_indices.items():
            chosen = rng.choice(idx, size=len(idx), replace=True)
            for orig, new in zip(idx, chosen):
                boot_W[orig] = W_atk[new]
                boot_labels[orig] = m
        prof = _method_profiles(boot_W, boot_labels, methods, atk_motifs)
        sim = weighted_jaccard_matrix(prof)
        s, f, _, _ = grouped_stats(sim, methods, semantic_group)
        sem_boots.append(s)
        flip_boots.append(f)
    sem_boots = np.array(sem_boots, dtype=np.float64)
    flip_boots = np.array(flip_boots, dtype=np.float64)
    return (
        (float(np.percentile(sem_boots, 2.5)), float(np.percentile(sem_boots, 97.5))),
        (float(np.percentile(flip_boots, 2.5)), float(np.percentile(flip_boots, 97.5))),
        sem_boots,
        flip_boots,
    )


def permutation_test(
    W_atk: np.ndarray,
    method_arr: np.ndarray,
    methods: List[str],
    atk_motifs: np.ndarray,
    semantic_group: Tuple[str, ...],
    set_mode: str,
    top_k: int,
    act_thr: float,
    n_perm: int,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    obs_prof = _method_profiles(W_atk, method_arr, methods, atk_motifs)
    obs_sets = method_sets_from_profiles(obs_prof, methods, atk_motifs, set_mode, top_k, act_thr)
    jac_obs = jaccard_matrix(obs_sets, methods)
    sem_obs, flip_obs, _, _ = grouped_stats(jac_obs, methods, semantic_group)
    observed_gap = sem_obs - flip_obs

    perm_gaps = []
    for _ in range(n_perm):
        perm_labels = rng.permutation(method_arr)
        prof = _method_profiles(W_atk, perm_labels, methods, atk_motifs)
        sets = method_sets_from_profiles(prof, methods, atk_motifs, set_mode, top_k, act_thr)
        jac = jaccard_matrix(sets, methods)
        s, f, _, _ = grouped_stats(jac, methods, semantic_group)
        perm_gaps.append(s - f)
    perm_gaps = np.array(perm_gaps, dtype=np.float64)
    p_value = float((perm_gaps >= observed_gap).mean())
    return observed_gap, p_value, perm_gaps


def bootstrap_ci(
    W_atk: np.ndarray,
    method_arr: np.ndarray,
    methods: List[str],
    atk_motifs: np.ndarray,
    semantic_group: Tuple[str, ...],
    set_mode: str,
    top_k: int,
    act_thr: float,
    n_boot: int,
    seed: int = 99,
):
    rng = np.random.default_rng(seed)
    sem_boots, flip_boots = [], []
    method_indices = {m: np.where(method_arr == m)[0] for m in methods}
    n_total = len(method_arr)
    for _ in range(n_boot):
        boot_W = np.zeros_like(W_atk)
        boot_labels = np.empty(n_total, dtype=object)
        for m, idx in method_indices.items():
            chosen = rng.choice(idx, size=len(idx), replace=True)
            for orig, new in zip(idx, chosen):
                boot_W[orig] = W_atk[new]
                boot_labels[orig] = m
        prof = _method_profiles(boot_W, boot_labels, methods, atk_motifs)
        sets = method_sets_from_profiles(prof, methods, atk_motifs, set_mode, top_k, act_thr)
        jac = jaccard_matrix(sets, methods)
        s, f, _, _ = grouped_stats(jac, methods, semantic_group)
        sem_boots.append(s)
        flip_boots.append(f)
    sem_boots = np.array(sem_boots, dtype=np.float64)
    flip_boots = np.array(flip_boots, dtype=np.float64)
    return (
        (float(np.percentile(sem_boots, 2.5)), float(np.percentile(sem_boots, 97.5))),
        (float(np.percentile(flip_boots, 2.5)), float(np.percentile(flip_boots, 97.5))),
        sem_boots,
        flip_boots,
    )


def build_method_feature_stats(
    atk_mat: sparse.csr_matrix,
    method_arr: np.ndarray,
    methods: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Method x feature summary statistics on raw activated features."""
    n_methods, n_feat = len(methods), atk_mat.shape[1]
    mean_mat = np.zeros((n_methods, n_feat), dtype=np.float64)
    rate_mat = np.zeros((n_methods, n_feat), dtype=np.float64)
    count_mat = np.zeros((n_methods, n_feat), dtype=np.float64)
    for i, m in enumerate(methods):
        idx = np.where(method_arr == m)[0]
        if len(idx) == 0:
            continue
        sub = atk_mat[idx]
        mean_mat[i] = np.asarray(sub.mean(axis=0)).ravel()
        cnt = np.asarray((sub > 0).sum(axis=0)).ravel().astype(np.float64)
        count_mat[i] = cnt
        rate_mat[i] = cnt / float(len(idx))
    return mean_mat, rate_mat, count_mat


def span_based_jaccard(
    spans_df: pd.DataFrame,
    methods: List[str],
    rate_thr: float = 0.05,
    save_dir: str = None,
) -> np.ndarray:
    """
    Span-level method×method matrices: Jaccard, Overlap coef, and raw |A∩B| counts.

    Steps:
      1. For each (NeuronID, attack_method) pair, count how many unique TextIDs
         activated that feature.
      2. Divide by total samples per method → per-method activation rate per feature.
      3. Binarise: method M "has" feature F iff rate >= rate_thr.
      4. Jaccard / Overlap / raw intersection; save CSV + PNG for each.

    Returns:
      overlap matrix (primary for grouped stats), method_feature_sets
    """
    # Count per-method samples (denominator)
    n_per_method = spans_df.groupby("attack_method")["TextID"].nunique().to_dict()

    # Count unique TextIDs activating each feature, per method
    act_counts = (
        spans_df.groupby(["attack_method", "NeuronID"])["TextID"]
        .nunique()
        .reset_index()
        .rename(columns={"TextID": "n_texts"})
    )
    act_counts["rate"] = act_counts.apply(
        lambda r: r["n_texts"] / n_per_method.get(r["attack_method"], 1), axis=1
    )

    # Build binary feature sets per method
    method_feature_sets: Dict[str, set] = {}
    for m in methods:
        sub = act_counts[(act_counts["attack_method"] == m) & (act_counts["rate"] >= rate_thr)]
        method_feature_sets[m] = set(sub["NeuronID"].tolist())
        print(f"  {m:<15}: {len(method_feature_sets[m])} features activated (rate>={rate_thr})")

    # Jaccard, Overlap coef, and raw |A∩B| counts (GeoNorm removed)
    n = len(methods)
    jac      = np.zeros((n, n), dtype=np.float32)
    overlap  = np.zeros((n, n), dtype=np.float32)
    inter_m  = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        for j in range(n):
            s1, s2 = method_feature_sets[methods[i]], method_feature_sets[methods[j]]
            inter  = len(s1 & s2)
            union  = len(s1 | s2)
            mn     = min(len(s1), len(s2))
            jac[i, j]      = inter / union if union > 0 else 0.0
            overlap[i, j]  = inter / mn   if mn   > 0 else 0.0
            inter_m[i, j]  = inter

    print_matrix(f"  Jaccard        (|A∩B|/|A∪B|,  rate_thr={rate_thr})", jac,      methods)
    print_matrix(f"  Overlap coef   (|A∩B|/min,     rate_thr={rate_thr})", overlap,  methods)
    print_matrix(f"  Raw intersection |A∩B|",                               inter_m.astype(float), methods)

    # Grouped stats on Overlap coefficient (primary metric)
    sem_idx  = [i for i, m in enumerate(methods) if m in CORE_SEMANTIC_METHODS]
    flip_idx = methods.index(FLIP_METHOD) if FLIP_METHOD in methods else None
    sem_pairs  = [overlap[a, b] for a in sem_idx for b in sem_idx if a < b]
    flip_pairs = [overlap[flip_idx, si] for si in sem_idx] if flip_idx is not None else []
    sem_mean   = float(np.mean(sem_pairs))  if sem_pairs  else float("nan")
    flip_mean  = float(np.mean(flip_pairs)) if flip_pairs else float("nan")
    print(f"\n  [Overlap coef grouped]")
    print(f"  Semantic-intra mean : {sem_mean:.3f}")
    print(f"  FlipAttack cross    : {flip_mean:.3f}")
    print(f"  Gap                 : {sem_mean - flip_mean:+.3f}")

    # Save CSVs + heatmaps (Jaccard, Overlap, raw intersection counts)
    if save_dir:
        for name, mat in [
            ("jaccard", jac),
            ("overlap", overlap),
            ("intersection", inter_m.astype(np.float32)),
        ]:
            pd.DataFrame(mat, index=methods, columns=methods).to_csv(
                os.path.join(save_dir, f"cross_method_span_{name}_rate{rate_thr}.csv")
            )
        for name, mat, title, heat_kwargs in [
            ("jaccard", jac, f"Span Jaccard (rate≥{rate_thr})", {}),
            ("overlap", overlap, f"Span Overlap coef (rate≥{rate_thr})", {}),
            (
                "intersection",
                inter_m.astype(np.float32),
                f"Span raw |A∩B| (rate≥{rate_thr})",
                {"vmin": 0.0, "vmax": None, "cell_int": True},
            ),
        ]:
            out_path = os.path.join(save_dir, f"cross_method_span_{name}_rate{rate_thr}.png")
            plot_similarity_heatmap(mat, methods, out_path, title=title, **heat_kwargs)
        print(f"  Saved CSVs and heatmaps for rate_thr={rate_thr}")

    return overlap, method_feature_sets  # return overlap as primary


def analyze_feature_spans(
    spans_df: pd.DataFrame,
    fids: np.ndarray,
    mean_mat: np.ndarray,
    rate_mat: np.ndarray,
    methods: List[str],
    semantic_group: Tuple[str, ...],
    rate_thr: float = 0.05,
    global_rate_thr: float = 0.01,
    top_n_spans: int = 3,
    max_span_len: int = 120,
    save_path: str = None,
    features_explained_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Qualitative report with up to three examples.

    When ``features_explained_path`` points to ``features_explained_*_safety_*.tsv`` under the same
    ``threshold_*`` folder, picks are ranked by explainer Task/Summary/Words (harmful
    Yes/Maybe, jailbreak keywords; penalise Cannot Tell / benign writing), then by activation stats.

    1. *shared_semantic*: trio >= rate_thr, Flip < rate_thr.
    2. *flip_plus_1or2_flip_related*: Flip >= rate_thr and 1-2 other methods >= rate_thr; each co-method
       must have at least one flip-consistent span on that neuron. Co-method spans are
       flip-filtered. Ranked by TSV + Flip span mechanism scores.
    3. *universal_all_methods*: all methods >= rate_thr; TSV harmful + broad wording, then min rate.

    (*flip_exclusive_strict* is a fallback if the stricter flip_plus pool is empty.)
    """
    expl_df = load_features_explained_df(features_explained_path)
    if expl_df is not None:
        print(f"  [features_explained] loaded {features_explained_path!r}  rows={len(expl_df)}")
    elif features_explained_path:
        print(f"  [WARN] features_explained TSV missing or invalid: {features_explained_path}")

    flip_idx    = methods.index(FLIP_METHOD) if FLIP_METHOD in methods else None
    sem_indices = [i for i, m in enumerate(methods) if m in semantic_group]
    all_indices = list(range(len(methods)))

    # Global activation rate filter (drop extremely rare features)
    global_rate = rate_mat.mean(axis=0)
    valid = global_rate >= global_rate_thr
    valid_local = np.where(valid)[0]

    def _all_methods_active(li: int) -> bool:
        return all(rate_mat[i, li] >= rate_thr for i in all_indices)

    # Categorise features (universal = truly all methods above threshold)
    shared_semantic_idx = []          # trio on, flip off (ReNeLLM arbitrary)
    flip_exclusive_strict_idx = []    # only Flip on (strict; pool / fallback)
    flip_plus_1or2_idx = []           # Flip on + exactly 1-2 other methods on
    universal_idx = []               # all methods on

    for li in valid_local:
        sem_active = all(rate_mat[i, li] >= rate_thr for i in sem_indices)
        flip_active = (flip_idx is not None) and (rate_mat[flip_idx, li] >= rate_thr)

        if _all_methods_active(li):
            universal_idx.append(li)
        elif sem_active and not flip_active:
            shared_semantic_idx.append(li)
        elif flip_active and flip_idx is not None:
            others_on = [i for i in all_indices if i != flip_idx and rate_mat[i, li] >= rate_thr]
            n_oth = len(others_on)
            if n_oth == 0:
                flip_exclusive_strict_idx.append(li)
            elif 1 <= n_oth <= 2:
                # Co-methods must fire on *flip-like* spans, not unrelated weak hits (PAIR pharma, etc.)
                fid_li = int(fids[li])
                co_ok = True
                for oi in others_on:
                    sub_co = spans_df[
                        (spans_df["NeuronID"] == fid_li)
                        & (spans_df["attack_method"] == methods[oi])
                    ]
                    if sub_co.empty or not any(
                        _span_is_flip_consistent(str(r["Span"]), min_mech=4.0)
                        for _, r in sub_co.iterrows()
                    ):
                        co_ok = False
                        break
                if co_ok:
                    flip_plus_1or2_idx.append(li)

    def _sort_by_mean(local_indices: List[int], method_indices: List[int]) -> List[int]:
        if not local_indices:
            return []
        scores = [mean_mat[method_indices, li].mean() for li in local_indices]
        order = np.argsort(-np.array(scores))
        return [local_indices[o] for o in order]

    def _expl_parts(fid: int) -> Tuple[str, str, str]:
        if expl_df is None or fid not in expl_df.index:
            return "", "", ""
        r = expl_df.loc[fid]
        return (
            str(r.get("Summary", "") or ""),
            str(r.get("Task", "") or ""),
            str(r.get("Words", "") or ""),
        )

    def _sort_shared_semantic_for_appendix(local_indices: List[int]) -> List[int]:
        if not local_indices:
            return []
        rows: List[Tuple[float, float, float, int]] = []
        for li in local_indices:
            fid = int(fids[li])
            sm, tk, wd = _expl_parts(fid)
            ex = _explain_score_shared_semantic(sm, tk, wd) if expl_df is not None else 0.0
            sem_mean = float(mean_mat[sem_indices, li].mean())
            trio_min_r = float(min(float(rate_mat[i, li]) for i in sem_indices))
            rows.append((ex, sem_mean, trio_min_r, li))
        rows.sort(key=lambda t: (-t[0], -t[1], -t[2]))
        return [t[3] for t in rows]

    def _sort_flip_strict_for_appendix(local_indices: List[int]) -> List[int]:
        if not local_indices or flip_idx is None:
            return []
        rows: List[Tuple[float, float, float, float, int, int]] = []
        for li in local_indices:
            fid = int(fids[li])
            sm, tk, wd = _expl_parts(fid)
            ex_tsv = _explain_score_flip_tsv(sm, tk, wd) if expl_df is not None else 0.0
            sub = spans_df[
                (spans_df["NeuronID"] == fid) & (spans_df["attack_method"] == FLIP_METHOD)
            ]
            if sub.empty:
                continue
            peak = float(sub["Score"].max())
            n_txt = int(sub["TextID"].nunique())
            mean_f = float(mean_mat[flip_idx, li])
            best_combo = 0.0
            for _, row in sub.iterrows():
                mech = _flip_attack_span_mechanism_score(str(row["Span"]))
                sc = float(row["Score"])
                best_combo = max(best_combo, (mech + 1.0) * float(np.log1p(sc)))
            rows.append((ex_tsv, best_combo, peak, mean_f, n_txt, li))
        rows.sort(key=lambda t: (-t[0], -t[1], -t[2], -t[3], -t[4]))
        return [t[5] for t in rows]

    def _sort_universal_for_appendix(local_indices: List[int]) -> List[int]:
        if not local_indices:
            return []
        rows: List[Tuple[float, float, float, int, int]] = []
        for li in local_indices:
            fid = int(fids[li])
            sm, tk, wd = _expl_parts(fid)
            ex = _explain_score_universal(sm, tk, wd) if expl_df is not None else 0.0
            min_r = float(min(float(rate_mat[i, li]) for i in all_indices))
            mean_all = float(mean_mat[:, li].mean())
            sum_len = len(sm.strip())
            rows.append((ex, min_r, mean_all, sum_len, li))
        rows.sort(key=lambda t: (-t[0], -t[1], -t[2], -t[3]))
        return [t[4] for t in rows]

    shared_semantic_sorted = (
        _sort_shared_semantic_for_appendix(shared_semantic_idx)
        if expl_df is not None
        else _sort_by_mean(shared_semantic_idx, sem_indices)
    )
    flip_exclusive_sorted = _sort_flip_strict_for_appendix(flip_exclusive_strict_idx)
    flip_plus_sorted = _sort_flip_strict_for_appendix(flip_plus_1or2_idx)
    universal_sorted = (
        _sort_universal_for_appendix(universal_idx)
        if expl_df is not None
        else _sort_by_mean(universal_idx, all_indices)
    )

    # Flip + 1-2 other methods, flip-meaning-weighted (not strict-exclusive)
    appendix_specs: List[Tuple[str, int]] = []
    if shared_semantic_sorted:
        appendix_specs.append(("shared_semantic", shared_semantic_sorted[0]))
    if flip_plus_sorted:
        appendix_specs.append(("flip_plus_1or2_flip_related", flip_plus_sorted[0]))
    elif flip_exclusive_sorted:
        appendix_specs.append(("flip_exclusive_strict", flip_exclusive_sorted[0]))
    if universal_sorted:
        appendix_specs.append(("universal_all_methods", universal_sorted[0]))
    appendix_specs = appendix_specs[:3]

    # ── Span rows: text + activation score (from this row) ────────────────────
    def _get_span_items(
        neuron_id: int,
        method: str,
        n: int,
        *,
        only_flip_consistent_for_non_flip: bool = False,
        flip_consistent_min_mech: float = 3.5,
    ) -> List[Dict[str, object]]:
        sub = spans_df[(spans_df["NeuronID"] == neuron_id) &
                       (spans_df["attack_method"] == method)]
        if sub.empty:
            return []
        sub = sub.sort_values("Score", ascending=False)
        # Same span string often repeats across TextIDs — keep one row (max Score).
        span_key = sub["Span"].astype(str).str.strip().str.replace("\n", " ", regex=False)
        sub = sub.assign(_span_key=span_key).drop_duplicates(subset="_span_key", keep="first")
        if only_flip_consistent_for_non_flip and method != FLIP_METHOD:
            mask = sub["Span"].astype(str).map(
                lambda x: _span_is_flip_consistent(x, min_mech=flip_consistent_min_mech)
            )
            sub = sub.loc[mask]
            if sub.empty:
                return []
        if method == FLIP_METHOD:
            mech = sub["Span"].astype(str).map(_flip_attack_span_mechanism_score)
            sub = sub.assign(
                _flip_pick=(mech.astype(np.float64) + 1.0)
                * np.log1p(sub["Score"].astype(np.float64))
            ).sort_values("_flip_pick", ascending=False)
        else:
            sub = sub.sort_values("Score", ascending=False)
        sub = sub.head(n)
        out: List[Dict[str, object]] = []
        for _, row in sub.iterrows():
            raw = str(row["Span"]).strip().replace("\n", " ")
            clipped = raw[:max_span_len] + ("…" if len(raw) > max_span_len else "")
            out.append({"text": clipped, "score": float(row["Score"])})
        return out

    def _feature_entry(
        li: int,
        methods_with_spans: Optional[Set[str]] = None,
        *,
        co_methods_flip_consistent_spans: bool = False,
    ) -> Dict[str, object]:
        """
        If ``methods_with_spans`` is set, ``per_method`` only lists those methods (e.g. Flip-only,
        or Flip + the 1–2 co-active attacks for ``flip_plus_1or2_flip_related``).
        When ``co_methods_flip_consistent_spans``, non-Flip rows only show spans that pass
        ``_span_is_flip_consistent`` (avoids PAIR/XTeaming generic text on spurious co-fires).
        """
        real_fid = int(fids[li])
        entry: Dict[str, object] = {
            "feature_id": real_fid,
            "global_rate": float(global_rate[li]),
            "per_method": {},
        }
        if expl_df is not None and real_fid in expl_df.index:
            r = expl_df.loc[real_fid]
            tk = str(r.get("Task", "") or "")
            entry["sae_explanation"] = {
                "summary": _clip_text(str(r.get("Summary", "") or ""), 900),
                "task_excerpt": _clip_text(tk, 700),
                "verify": str(r.get("Verify", "") or ""),
                "final_decision_parse": _final_decision_from_task(tk),
            }
        else:
            entry["sae_explanation"] = None
        m_show: Set[str] = methods_with_spans if methods_with_spans is not None else set(methods)
        for i, m in enumerate(methods):
            if m not in m_show:
                continue
            use_fc = co_methods_flip_consistent_spans and m != FLIP_METHOD
            entry["per_method"][m] = {
                "mean_score": float(mean_mat[i, li]),
                "activation_rate": float(rate_mat[i, li]),
                "top_spans": _get_span_items(
                    real_fid,
                    m,
                    top_n_spans,
                    only_flip_consistent_for_non_flip=use_fc,
                ),
            }
        return entry

    def _methods_active_at_thr(li: int) -> Set[str]:
        return {methods[i] for i in all_indices if float(rate_mat[i, li]) >= rate_thr}

    appendix_examples: List[Dict[str, object]] = []
    for cat, li in appendix_specs:
        if cat == "flip_exclusive_strict":
            appendix_examples.append(
                {"category": cat, "feature": _feature_entry(li, methods_with_spans={FLIP_METHOD})}
            )
        elif cat == "flip_plus_1or2_flip_related":
            appendix_examples.append(
                {
                    "category": cat,
                    "feature": _feature_entry(
                        li,
                        methods_with_spans=_methods_active_at_thr(li),
                        co_methods_flip_consistent_spans=True,
                    ),
                }
            )
        else:
            appendix_examples.append({"category": cat, "feature": _feature_entry(li)})

    for ex in appendix_examples:
        cat = ex["category"]
        if cat != "flip_exclusive_strict" and cat != "flip_plus_1or2_flip_related":
            continue
        fid = int(ex["feature"]["feature_id"])
        fsub = spans_df[
            (spans_df["NeuronID"] == fid) & (spans_df["attack_method"] == FLIP_METHOD)
        ]
        best_combo = -1.0
        best_mech = 0.0
        best_span_snip = ""
        for _, row in fsub.iterrows():
            raw = str(row["Span"])
            mech = _flip_attack_span_mechanism_score(raw)
            sc = float(row["Score"])
            combo = (mech + 1.0) * float(np.log1p(sc))
            if combo > best_combo:
                best_combo = combo
                best_mech = mech
                best_span_snip = raw.replace("\n", " ")[:200]
        ex["flip_mechanism_selection"] = {
            "best_row_combo": float(best_combo),
            "best_span_mechanism_score": float(best_mech),
            "best_span_prefix": best_span_snip,
            "note": "Aligned with the reversal-style attack format: user content is "
            "TASK is '<reversed target>'; system template lines (Rule 1–4, JUST SOLVE IT) are down-weighted.",
        }
        if cat == "flip_plus_1or2_flip_related":
            ex["flip_mechanism_selection"]["co_active_methods"] = sorted(
                m for m in ex["feature"]["per_method"] if m != FLIP_METHOD
            )

    report = {
        "appendix_examples": appendix_examples,
        "counts": {
            "shared_semantic_pool": len(shared_semantic_idx),
            "flip_exclusive_strict_pool": len(flip_exclusive_strict_idx),
            "flip_plus_1or2_pool": len(flip_plus_1or2_idx),
            "universal_all_methods_pool": len(universal_idx),
            "valid_features": int(valid.sum()),
            "features_explained_used": expl_df is not None,
        },
    }

    # Terminal: red ANSI for span text, always print all methods for each example
    _RED = "\033[91m"
    _RST = "\033[0m"

    print(
        f"\n[SPAN] Feature-span appendix (≤3 examples)  "
        f"rate_thr={rate_thr}, global_rate_thr={global_rate_thr}"
    )
    print(f"  Valid features (global_rate >= {global_rate_thr}): {int(valid.sum())}")
    print(
        f"  Pool sizes — shared_semantic: {len(shared_semantic_idx)}, "
        f"flip_plus_1or2: {len(flip_plus_1or2_idx)}, "
        f"flip_exclusive_strict: {len(flip_exclusive_strict_idx)}, "
        f"universal_all_methods: {len(universal_idx)}"
    )

    for ex in appendix_examples:
        cat = ex["category"]
        entry = ex["feature"]
        print(f"\n  ── {cat} (appendix) ──")
        fid = entry["feature_id"]
        print(f"    Feature {fid}  (global_rate={entry['global_rate']:.3f})")
        sx = entry.get("sae_explanation")
        if sx:
            print(
                f"    [features_explained] decision={sx.get('final_decision_parse')!r}  "
                f"verify={sx.get('verify')!r}"
            )
            print(f"    Summary: {_clip_text(str(sx.get('summary', '')), 220)}")
        if cat in ("flip_exclusive_strict", "flip_plus_1or2_flip_related"):
            sel = ex.get("flip_mechanism_selection") or {}
            if sel:
                extra = ""
                if cat == "flip_plus_1or2_flip_related" and sel.get("co_active_methods"):
                    extra = f"  co_active={sel['co_active_methods']}"
                print(
                    f"    [FlipMechanismRank] combo={sel['best_row_combo']:.3f}  "
                    f"span_mech={sel['best_span_mechanism_score']:.2f}{extra}"
                )
        disp_methods = [m for m in methods if m in entry["per_method"]]
        for m in disp_methods:
            pm = entry["per_method"][m]
            print(
                f"      [{m}]  activation_rate={pm['activation_rate']:.3f}  "
                f"mean_score={pm['mean_score']:.4f}"
            )
            for item in pm["top_spans"]:
                txt = str(item["text"])
                sc = float(item["score"])
                print(f"        · score={sc:.4f}  {_RED}{txt}{_RST}")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Saved span report → {save_path}")

    return report


def print_matrix(title: str, mat: np.ndarray, methods: List[str]):
    print(f"\n{title}")
    col_w = max(len(m) for m in methods) + 2
    header = f"  {'':>{col_w}}" + "".join(f"  {m[:8]:>8}" for m in methods)
    print(header)
    print("  " + "-" * len(header))
    for i, m1 in enumerate(methods):
        row = f"  {m1:>{col_w}}"
        for j in range(len(methods)):
            row += f"  {mat[i, j]:>8.3f}"
        print(row)


def save_figure_png_and_pdf(fig: plt.Figure, out_path: str, **savefig_kwargs) -> None:
    """Save figure to PNG path and mirrored PDF path."""
    fig.savefig(out_path, **savefig_kwargs)
    base, ext = os.path.splitext(out_path)
    if ext.lower() == ".png":
        fig.savefig(f"{base}.pdf", **savefig_kwargs)


def plot_similarity_heatmap(
    jac: np.ndarray,
    methods: List[str],
    out_path: str,
    title: Optional[str] = None,
    *,
    vmin: float = 0.0,
    vmax: float | None = 1.0,
    cell_int: bool = False,
    cmap: str = "Blues",
):
    """
    Clean publication-style heatmap with sequential Blues and serif text if available.
    Default ``title=None`` omits the figure title (NMF outputs call without title).
    """
    ff = _paper_serif_fontfamily()
    display_order = [m for m in ["Crescendo", "PAIR", "XTeaming", "ReNeLLM", "FlipAttack"] if m in methods]
    idx = [methods.index(m) for m in display_order]
    mat = jac[np.ix_(idx, idx)]

    rc = {
        "font.family": ff,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
    imshow_kw: Dict[str, object] = {"cmap": cmap, "aspect": "equal", "origin": "upper"}

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        vn = float(vmin) if vmin is not None else float(np.nanmin(mat))
        vx = float(vmax) if vmax is not None else (float(np.nanmax(mat)) if mat.size else 1.0)
        if not np.isfinite(vx) or vx == vn:
            vx = vn + 1e-6
        norm = mcolors.Normalize(vmin=vn, vmax=vx)
        cmap_obj = plt.get_cmap(cmap)

        im = ax.imshow(mat, norm=norm, **imshow_kw)
        ax.set_xticks(np.arange(len(display_order)))
        ax.set_yticks(np.arange(len(display_order)))
        ax.set_xticklabels(display_order, rotation=35, ha="right", fontfamily=ff, fontsize=10)
        ax.set_yticklabels(display_order, fontfamily=ff, fontsize=10)
        ax.tick_params(axis="both", which="major", length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = float(mat[i, j])
                txt = f"{val:.0f}" if cell_int else f"{val:.2f}"
                rgba = cmap_obj(norm(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "#f8f8f8" if lum < 0.52 else "#1a1a1a"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=12,
                    fontfamily=ff,
                )

        if title:
            ax.set_title(title, fontsize=11, pad=8, fontfamily=ff)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.82)
        cbar.ax.tick_params(labelsize=10, width=0.6, length=3)
        for t in cbar.ax.get_yticklabels():
            t.set_fontfamily(ff)
        cbar.outline.set_visible(False)
        fig.tight_layout()
        save_figure_png_and_pdf(
            fig,
            out_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    plt.close(fig)


def plot_grouped_comparison(sem_boots: np.ndarray, flip_boots: np.ndarray, sem_mean: float, flip_mean: float, p_val: float, out_path: str):
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    data = [sem_boots, flip_boots]
    x = np.arange(2)

    parts = ax.violinplot(data, positions=x, widths=0.6, showmeans=False, showmedians=False, showextrema=False)
    colors = ["#1f77b4", "#ff7f0e"]
    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
        pc.set_edgecolor("none")

    sem_ci = (float(np.percentile(sem_boots, 2.5)), float(np.percentile(sem_boots, 97.5)))
    flip_ci = (float(np.percentile(flip_boots, 2.5)), float(np.percentile(flip_boots, 97.5)))
    means = [sem_mean, flip_mean]
    ci_low = [sem_ci[0], flip_ci[0]]
    ci_high = [sem_ci[1], flip_ci[1]]
    ax.errorbar(
        x,
        means,
        yerr=[np.array(means) - np.array(ci_low), np.array(ci_high) - np.array(means)],
        fmt="o",
        color="black",
        capsize=4,
        zorder=3,
    )

    rng = np.random.default_rng(0)
    n_show = int(min(250, len(sem_boots), len(flip_boots)))
    if n_show > 0:
        sem_show = rng.choice(sem_boots, size=n_show, replace=False)
        flip_show = rng.choice(flip_boots, size=n_show, replace=False)
        jitter = 0.06
        ax.scatter(x[0] + rng.uniform(-jitter, jitter, size=n_show), sem_show, s=8, alpha=0.18, color="black", linewidths=0)
        ax.scatter(x[1] + rng.uniform(-jitter, jitter, size=n_show), flip_show, s=8, alpha=0.18, color="black", linewidths=0)

    ax.set_xticks(x)
    ax.set_xticklabels(["Core semantic\n(intra-group)", "FlipAttack\n(cross-group)"])
    ax.set_ylabel("Jaccard overlap")
    ax.set_ylim(0, 1.0)
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
    if p_val == 0.0:
        p_txt = "p < 1/n_perm"
    elif p_val < 0.001:
        p_txt = "p < 0.001"
    else:
        p_txt = f"p = {p_val:.3f}"
    ax.set_title(f"Grouped motif overlap comparison ({sig}, {p_txt})")

    y = 0.9
    h = 0.03
    ax.plot([x[0], x[0], x[1], x[1]], [y, y + h, y + h, y], lw=1.2)
    ax.text((x[0] + x[1]) / 2, y + h + 0.01, sig, ha="center", va="bottom", fontsize=12)

    fig.tight_layout()
    save_figure_png_and_pdf(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atk-base", required=True)
    parser.add_argument("--non-attack-base", dest="nrm_base", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--threshold", required=True)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--set-mode", choices=["topk", "threshold"], default="topk")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--act-thr", type=float, default=0.01)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--feature-rate-thr", type=float, default=0.01)
    parser.add_argument("--feature-global-rate-thr", type=float, default=0.005)
    parser.add_argument("--top-texts-per-feature", type=int, default=3)
    parser.add_argument(
        "--features-explained-tsv",
        default=None,
        help="SAE explainer TSV (FeatureID,Task,Verify,Summary,Words). "
        "Default: <atk-base>/threshold_<thr>/features_explained.tsv",
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--skip-nmf-ablation",
        action="store_true",
        help="Skip combined NMF + attack/non-attack motif-ratio ablation (vector + set Jaccard, heatmaps, JSON).",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    thr, K = args.threshold, args.k

    print(f"[1] Loading (δ={thr})...")
    atk_mat, nrm_mat, fids, atk_tids = load_combined(args.atk_base, args.nrm_base, thr)
    method_arr = load_labels_aligned(args.labels, atk_tids)
    methods = sorted(set(method_arr))
    print(f"  Attack: {atk_mat.shape}  Normal: {nrm_mat.shape}")
    print(f"  Methods: {methods}")
    for m in methods:
        print(f"    {m}: {(method_arr == m).sum()}")

    # ── Span-based Jaccard: directly from span activations ────────────────────
    print(f"\n[1] Span-based feature Jaccard (δ={thr})...")
    spans_df = load_spans_df(args.atk_base, thr, atk_tids, method_arr)
    print(f"  Loaded {len(spans_df):,} span rows for attack samples")

    for rate_thr in [0.01, 0.02, 0.05]:
        print(f"\n  -- rate_thr={rate_thr} --")
        span_jac, method_feature_sets = span_based_jaccard(
            spans_df=spans_df,
            methods=methods,
            rate_thr=rate_thr,
            save_dir=args.save_dir,
        )

    # ── Feature-span text report (qualitative) ────────────────────────────────
    print(f"\n[2] Feature-span text analysis...")
    mean_mat_full, rate_mat_full, _ = build_method_feature_stats(atk_mat, method_arr, methods)
    fe_default = os.path.join(
        args.atk_base,
        f"threshold_{args.threshold}",
        "features_explained.tsv",
    )
    fe_path = args.features_explained_tsv or fe_default
    analyze_feature_spans(
        spans_df=spans_df,
        fids=fids,
        mean_mat=mean_mat_full,
        rate_mat=rate_mat_full,
        methods=methods,
        semantic_group=CORE_SEMANTIC_METHODS,
        rate_thr=args.feature_rate_thr,
        global_rate_thr=args.feature_global_rate_thr,
        top_n_spans=args.top_texts_per_feature,
        save_path=os.path.join(args.save_dir, "cross_method_feature_span_report.json"),
        features_explained_path=fe_path,
    )

    # ── NMF / motif ablation
    if not args.skip_nmf_ablation:
        print(f"\n[2] Combined NMF (K={K})...")
        W_atk, ratio = run_combined_nmf(atk_mat, nrm_mat, K)

        ratio_list = [1.5, 2.0, 2.5, 3.0]
        topk_list = [5, 10, 15, 20]

        for ratio_thr in ratio_list:
            atk_motifs = np.where(ratio > ratio_thr)[0]
            print(f"\n=== Ablation: ratio>{ratio_thr} ===")
            print(f"  Attack-specific motifs: {len(atk_motifs)} / {K}")
            if len(atk_motifs) == 0:
                print("  [SKIP] No attack-specific motifs for this ratio.")
                continue

            # === Vector-based output: weighted Jaccard on mean activation profiles ===
            print("\n[3v] Vector similarity (mean activation weighted-Jaccard)")
            profiles_v = _method_profiles(W_atk, method_arr, methods, atk_motifs)
            sim_v = weighted_jaccard_matrix(profiles_v)
            print_matrix("  Weighted-Jaccard Similarity Matrix", sim_v, methods)

            print("\n[4v] Grouped Comparison (vector weighted-Jaccard): core-semantic vs. FlipAttack")
            sem_mean_v, flip_mean_v, sem_pairs_v, flip_pairs_v = grouped_stats(sim_v, methods, CORE_SEMANTIC_METHODS)
            gap_v = sem_mean_v - flip_mean_v
            print(f"  Core-semantic intra (n={len(sem_pairs_v):>2} pairs):  mean={sem_mean_v:.3f}")
            print(f"  FlipAttack cross    (n={len(flip_pairs_v):>2} pairs):  mean={flip_mean_v:.3f}")
            print(f"  Gap (semantic - flip):                     {gap_v:+.3f}")

            print(f"\n[5v] Bootstrap 95% CI (vector; n_boot={args.n_boot})...")
            sem_ci_v, flip_ci_v, sem_boots_v, flip_boots_v = bootstrap_ci_vector(
                W_atk, method_arr, methods, atk_motifs, CORE_SEMANTIC_METHODS, args.n_boot
            )
            print(f"  Core-semantic 95% CI: [{sem_ci_v[0]:.3f}, {sem_ci_v[1]:.3f}]")
            print(f"  Flip-cross    95% CI: [{flip_ci_v[0]:.3f}, {flip_ci_v[1]:.3f}]")

            print(f"\n[6v] Permutation Test (vector; n_perm={args.n_perm})...")
            obs_gap_v, p_val_v, perm_gaps_v = permutation_test_vector(
                W_atk, method_arr, methods, atk_motifs, CORE_SEMANTIC_METHODS, args.n_perm
            )
            sig_v = "***" if p_val_v < 0.001 else ("**" if p_val_v < 0.01 else ("*" if p_val_v < 0.05 else "n.s."))
            print(f"  Observed gap (core_semantic - flip_cross): {obs_gap_v:+.3f}")
            print(f"  Permutation p-value:                       {p_val_v:.4f}  {sig_v}")
            print(f"  Permuted gap  mean={perm_gaps_v.mean():.3f}  std={perm_gaps_v.std():.3f}")

            suffix_v = f"ratio{ratio_thr}_vector"
            # Vector similarity heatmap output is disabled by default.
            # heatmap_v_path = os.path.join(args.save_dir, f"cross_method_similarity_heatmap_{suffix_v}.png")
            # plot_similarity_heatmap(sim_v, methods, heatmap_v_path)
            heatmap_v_path = None
            # grouped_v_path = os.path.join(args.save_dir, f"cross_method_grouped_comparison_{suffix_v}.png")
            # plot_grouped_comparison(sem_boots_v, flip_boots_v, sem_mean_v, flip_mean_v, p_val_v, grouped_v_path)

            summary_v = {
                "threshold": thr,
                "K": K,
                "mode": "vector_weighted_jaccard",
                "ratio_threshold": ratio_thr,
                "n_attack_specific_motifs": int(len(atk_motifs)),
                "methods": methods,
                "core_semantic_mean": sem_mean_v,
                "flip_cross_mean": flip_mean_v,
                "gap": obs_gap_v,
                "p_value": p_val_v,
                "semantic_ci": sem_ci_v,
                "flip_ci": flip_ci_v,
                "heatmap_path": heatmap_v_path,
                "grouped_plot_path": None,
            }
            with open(os.path.join(args.save_dir, f"cross_method_summary_{suffix_v}.json"), "w") as f:
                json.dump(summary_v, f, indent=2)

            for top_k in topk_list:
                print(f"\n[3] Method overlap ({args.set_mode}, top_k={top_k}, act_thr={args.act_thr})")
                profiles = _method_profiles(W_atk, method_arr, methods, atk_motifs)
                method_sets = method_sets_from_profiles(profiles, methods, atk_motifs, args.set_mode, top_k, args.act_thr)
                jac = jaccard_matrix(method_sets, methods)
                print_matrix("  Jaccard Similarity Matrix", jac, methods)

                print("\n  Selected motifs per method:")
                for m in methods:
                    tag = "(lexical)" if m == FLIP_METHOD else ("(core semantic)" if m in CORE_SEMANTIC_METHODS else "(intermediate)")
                    print(f"    {m:<15} {len(method_sets[m]):>3} / {len(atk_motifs)}  {tag}")

                print("\n[4] Grouped Comparison: core-semantic vs. FlipAttack")
                sem_mean, flip_mean, sem_pairs, flip_pairs = grouped_stats(jac, methods, CORE_SEMANTIC_METHODS)
                gap = sem_mean - flip_mean
                print(f"  Core-semantic intra (n={len(sem_pairs):>2} pairs):  mean={sem_mean:.3f}")
                print(f"  FlipAttack cross    (n={len(flip_pairs):>2} pairs):  mean={flip_mean:.3f}")
                print(f"  Gap (semantic - flip):                     {gap:+.3f}")

                print(f"\n[5] Bootstrap 95% CI (n_boot={args.n_boot})...")
                sem_ci, flip_ci, sem_boots, flip_boots = bootstrap_ci(
                    W_atk, method_arr, methods, atk_motifs, CORE_SEMANTIC_METHODS,
                    args.set_mode, top_k, args.act_thr, args.n_boot,
                )
                print(f"  Core-semantic 95% CI: [{sem_ci[0]:.3f}, {sem_ci[1]:.3f}]")
                print(f"  Flip-cross    95% CI: [{flip_ci[0]:.3f}, {flip_ci[1]:.3f}]")

                print(f"\n[6] Permutation Test (H0: method labels exchangeable, n_perm={args.n_perm})...")
                obs_gap, p_val, perm_gaps = permutation_test(
                    W_atk, method_arr, methods, atk_motifs, CORE_SEMANTIC_METHODS,
                    args.set_mode, top_k, args.act_thr, args.n_perm,
                )
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
                print(f"  Observed gap (core_semantic - flip_cross): {obs_gap:+.3f}")
                print(f"  Permutation p-value:                       {p_val:.4f}  {sig}")
                print(f"  Permuted gap  mean={perm_gaps.mean():.3f}  std={perm_gaps.std():.3f}")

                if "ReNeLLM" in methods:
                    idx_r = methods.index("ReNeLLM")
                    core_idx = [methods.index(m) for m in CORE_SEMANTIC_METHODS if m in methods]
                    flip_idx = methods.index(FLIP_METHOD)
                    r_core = float(np.mean([jac[idx_r, i] for i in core_idx]))
                    r_flip = float(jac[idx_r, flip_idx])
                    print("\n[7] ReNeLLM as intermediate case")
                    print(f"  Mean Jaccard(ReNeLLM, core semantic): {r_core:.3f}")
                    print(f"  Jaccard(ReNeLLM, FlipAttack):         {r_flip:.3f}")

                suffix = f"ratio{ratio_thr}_topk{top_k}"
                heatmap_path = os.path.join(args.save_dir, f"cross_method_similarity_heatmap_{suffix}.png")
                plot_similarity_heatmap(
                    jac,
                    methods,
                    heatmap_path,
                    title=f"Motif Jaccard (ratio>{ratio_thr}, top_k={top_k})",
                )

                summary = {
                    "threshold": thr,
                    "K": K,
                    "set_mode": args.set_mode,
                    "top_k": top_k,
                    "ratio_threshold": ratio_thr,
                    "act_thr": args.act_thr,
                    "n_attack_specific_motifs": int(len(atk_motifs)),
                    "methods": methods,
                    "core_semantic_mean": sem_mean,
                    "flip_cross_mean": flip_mean,
                    "gap": obs_gap,
                    "p_value": p_val,
                    "semantic_ci": sem_ci,
                    "flip_ci": flip_ci,
                    "heatmap_path": heatmap_path,
                    "grouped_plot_path": None,
                }
                summary_path = os.path.join(args.save_dir, f"cross_method_summary_{suffix}.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)

                print(f"\nSaved figure 1 → {heatmap_path}")
                print(f"Saved summary  → {summary_path}")

                print("\n" + "=" * 72)
                print(f"Conclusion (δ={thr}, K={K}, ratio>{ratio_thr}, top_k={top_k}):")
                print(f"  Core-semantic Jaccard:   {sem_mean:.3f}  {sem_ci}")
                print(f"  FlipAttack cross-Jacc.:  {flip_mean:.3f}  {flip_ci}")
                print(f"  Gap: {obs_gap:+.3f}   p={p_val:.4f} {sig}")
                if p_val < 0.05 and gap > 0:
                    print("  → Core semantic jailbreaks share significantly more motif structure")
                    print("    than FlipAttack, supporting a semantic–lexical split.")
                else:
                    print("  → Gap not statistically significant at p<0.05.")
                print("=" * 72)

    print("\n[Done]")



if __name__ == "__main__":
    main()
