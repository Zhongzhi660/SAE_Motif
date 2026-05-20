from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

N_FEATURES = 65536
NULL_SEEDS = 5
K_DIAG_LIST = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
NNZ_PENALTY_LAMBDA = 2.0


@dataclass
class SweepRow:
    delta: str
    atk_err: float
    nrm_err: float
    atk_train_atk_err: float
    atk_train_nrm_err: float
    atk_train_gap: float
    nrm_train_nrm_err: float
    nrm_train_atk_err: float
    nrm_train_gap: float
    sym_gap: float
    sym_nnz_gap: float
    null_atk_err: float
    atk_spec_pct: float
    atk_nnz_med: int
    nrm_nnz_med: int
    active_feat_union: int
    matched_pairs: int
    motif_auc: float
    null_group_auc: float
    auc_gain: float
    atk_stable_rank: float
    nrm_stable_rank: float
    stable_rank_gap: float
    atk_topk_energy: float
    nrm_topk_energy: float
    topk_energy_gap: float


def load_matrix(base_path: str, threshold: str):
    path = f"{base_path}/threshold_{threshold}/full.tsv"
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, sep="\t", usecols=["NeuronID", "TextID", "Score"])
    if df.empty:
        return None, None
    text_ids = np.sort(df["TextID"].unique())
    ridx = pd.Index(text_ids).get_indexer(df["TextID"])
    sp = sparse.csr_matrix(
        (df["Score"].values.astype(np.float32), (ridx, df["NeuronID"].values)),
        shape=(len(text_ids), N_FEATURES),
        dtype=np.float32,
    )
    return sp, text_ids


def align_to_union(mat_a: sparse.csr_matrix, mat_b: sparse.csr_matrix):
    fids = np.array(sorted(set(mat_a.nonzero()[1]) | set(mat_b.nonzero()[1])), dtype=np.int64)
    return mat_a[:, fids].tocsr(), mat_b[:, fids].tocsr(), fids


def frobenius(x):
    if sparse.issparse(x):
        return float(np.sqrt(x.power(2).sum()))
    return float(np.linalg.norm(x, ord="fro"))


def nmf_fit_relerr(mat: sparse.csr_matrix, k: int, seed: int = 42):
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=500)
    model.fit(mat)
    err = float(model.reconstruction_err_ / (frobenius(mat) + 1e-10))
    return err, model


def nmf_cross_relerr(train_mat: sparse.csr_matrix, eval_mat: sparse.csr_matrix, k: int, seed: int = 42):
    """Fit NMF on train_mat, then evaluate relative reconstruction error on both sets."""
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=500)
    W_train = model.fit_transform(train_mat)
    H = model.components_
    train_recon = W_train @ H
    train_diff = train_mat.toarray() - train_recon
    train_err = float(np.linalg.norm(train_diff, ord="fro") / (frobenius(train_mat) + 1e-10))

    W_eval = model.transform(eval_mat)
    eval_recon = W_eval @ H
    eval_diff = eval_mat.toarray() - eval_recon
    eval_err = float(np.linalg.norm(eval_diff, ord="fro") / (frobenius(eval_mat) + 1e-10))
    return train_err, eval_err, model


def null_permute_rowwise(mat: sparse.csr_matrix, rng: np.random.Generator):
    mat = mat.tocsr().copy().astype(np.float32)
    n_feat = mat.shape[1]
    ptr = mat.indptr
    new_indices = mat.indices.copy()
    for i in range(mat.shape[0]):
        s, e = ptr[i], ptr[i + 1]
        if e > s:
            new_indices[s:e] = rng.choice(n_feat, size=e - s, replace=False)
    mat.indices = new_indices
    mat.has_sorted_indices = False
    mat.sort_indices()
    return mat


def atk_specific_fraction(atk_mat: sparse.csr_matrix, nrm_mat: sparse.csr_matrix, k: int, ratio_thresh: float = 3.0):
    combined = sparse.vstack([atk_mat, nrm_mat]).tocsr()
    nmf = NMF(n_components=k, init="nndsvda", random_state=42, max_iter=500)
    W = nmf.fit_transform(combined)
    atk_mean = W[: atk_mat.shape[0]].mean(axis=0)
    nrm_mean = W[atk_mat.shape[0] :].mean(axis=0)
    ratio = atk_mean / (nrm_mean + 1e-10)
    return float((ratio > ratio_thresh).sum()) / k, ratio


def spectral_lowrank_metrics(mat: sparse.csr_matrix, topk: int) -> tuple[float, float]:
    """
    Direct low-rankness diagnostics from singular spectrum.
      - stable_rank = ||X||_F^2 / ||X||_2^2 (smaller -> more concentrated spectrum)
      - topk_energy = sum_{i<=k} s_i^2 / ||X||_F^2 (larger -> stronger low-rank concentration)
    """
    mat = mat.tocsr()
    n_rows, n_cols = mat.shape
    max_rank = min(n_rows, n_cols) - 1
    if max_rank < 1:
        return np.nan, np.nan
    k_eff = max(1, min(int(topk), max_rank))
    try:
        s = svds(mat, k=k_eff, return_singular_vectors=False)
    except Exception:
        return np.nan, np.nan
    s = np.sort(np.asarray(s, dtype=np.float64))[::-1]
    if s.size == 0 or s[0] <= 0:
        return np.nan, np.nan
    fro2 = float(mat.power(2).sum())
    if fro2 <= 0:
        return np.nan, np.nan
    stable_rank = float(fro2 / (float(s[0] ** 2) + 1e-10))
    topk_energy = float(np.sum(s ** 2) / (fro2 + 1e-10))
    return stable_rank, topk_energy


def random_group_columns(mat: sparse.csr_matrix, out_dim: int, rng: np.random.Generator) -> sparse.csr_matrix:
    """
    Dimension-matched null: random feature grouping that preserves non-negativity.
    """
    mat = mat.tocoo()
    n_in = mat.shape[1]
    bucket = rng.integers(0, out_dim, size=n_in, endpoint=False)
    new_col = bucket[mat.col]
    grouped = sparse.csr_matrix((mat.data, (mat.row, new_col)), shape=(mat.shape[0], out_dim), dtype=np.float32)
    return grouped


def binary_auc_from_rep(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    test_size: float = 0.3,
) -> float:
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    clf.fit(x_tr, y_tr)
    prob = clf.predict_proba(x_te)[:, 1]
    return float(roc_auc_score(y_te, prob))


def separability_auc_suite(
    atk_mat: sparse.csr_matrix,
    nrm_mat: sparse.csr_matrix,
    k: int,
    repeats: int = 3,
) -> tuple[float, float]:
    """
    Direct structure evidence:
      - motif_auc: NMF representation separability (attack vs non-attack)
      - null_group_auc: dimension-matched random-grouping null
    """
    combined = sparse.vstack([atk_mat, nrm_mat]).tocsr()
    y = np.concatenate([np.ones(atk_mat.shape[0], dtype=int), np.zeros(nrm_mat.shape[0], dtype=int)])

    motif_aucs = []
    null_aucs = []
    for rep in range(repeats):
        seed = 42 + rep
        # Motif representation (NMF-W as latent features)
        nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=500)
        w = nmf.fit_transform(combined)
        motif_aucs.append(binary_auc_from_rep(w, y, seed=seed))

        # Strong null: same output dimension, random grouped columns then linear classifier
        rng = np.random.default_rng(1000 + seed)
        grouped = random_group_columns(combined, out_dim=k, rng=rng).toarray()
        null_aucs.append(binary_auc_from_rep(grouped, y, seed=seed))

    return float(np.mean(motif_aucs)), float(np.mean(null_aucs))


def make_nnz_bins(atk_nnz: np.ndarray, nrm_nnz: np.ndarray) -> np.ndarray:
    combined = np.concatenate([atk_nnz, nrm_nnz])
    max_nnz = int(combined.max())
    if max_nnz <= 1:
        return np.array([0, 2], dtype=int)
    # log-spaced bins with extra density near small nnz values
    raw = np.unique(np.rint(np.geomspace(1, max_nnz + 1, num=12)).astype(int))
    edges = np.unique(np.concatenate([[0], raw, [max_nnz + 1]])).astype(int)
    edges.sort()
    return edges


def stratified_shared_range_subsets(
    atk_mat: sparse.csr_matrix,
    nrm_mat: sparse.csr_matrix,
    atk_nnz: np.ndarray,
    nrm_nnz: np.ndarray,
    rng: np.random.Generator,
    min_bin_count: int = 25,
) -> Tuple[sparse.csr_matrix | None, sparse.csr_matrix | None, int]:
    """
    Build attack/non-attack subsets matched by nnz bins over the shared activity range.
    This preserves real samples and avoids damaging either side.
    """
    lo = max(int(atk_nnz.min()), int(nrm_nnz.min()))
    hi = min(int(atk_nnz.max()), int(nrm_nnz.max()))
    if lo >= hi:
        return None, None, 0

    atk_mask = (atk_nnz >= lo) & (atk_nnz <= hi)
    nrm_mask = (nrm_nnz >= lo) & (nrm_nnz <= hi)
    if atk_mask.sum() == 0 or nrm_mask.sum() == 0:
        return None, None, 0

    atk_idx_all = np.where(atk_mask)[0]
    nrm_idx_all = np.where(nrm_mask)[0]
    atk_shared = atk_nnz[atk_idx_all]
    nrm_shared = nrm_nnz[nrm_idx_all]
    bins = make_nnz_bins(atk_shared, nrm_shared)

    atk_sel: List[int] = []
    nrm_sel: List[int] = []
    for left, right in zip(bins[:-1], bins[1:]):
        a_local = atk_idx_all[(atk_nnz[atk_idx_all] >= left) & (atk_nnz[atk_idx_all] < right)]
        n_local = nrm_idx_all[(nrm_nnz[nrm_idx_all] >= left) & (nrm_nnz[nrm_idx_all] < right)]
        if len(a_local) < min_bin_count or len(n_local) < min_bin_count:
            continue
        m = min(len(a_local), len(n_local))
        atk_sel.extend(rng.choice(a_local, size=m, replace=False).tolist())
        nrm_sel.extend(rng.choice(n_local, size=m, replace=False).tolist())

    if len(atk_sel) == 0 or len(nrm_sel) == 0:
        return None, None, 0

    atk_sel = np.array(sorted(atk_sel), dtype=int)
    nrm_sel = np.array(sorted(nrm_sel), dtype=int)
    return atk_mat[atk_sel], nrm_mat[nrm_sel], len(atk_sel)


def threshold_sweep(atk_base: str, nrm_base: str, thresholds: Iterable[str], k: int):
    rng = np.random.default_rng(42)
    rows: list[SweepRow] = []

    print("\n" + "=" * 108)
    print(f"Threshold Sweep Summary (K={k})")
    print("=" * 108)
    print(
        f"  {'δ':>5} {'Atk_err':>9} {'Nrm_err':>9} {'Gap':>9} "
        f"{'SR_atk':>8} {'SR_nrm':>8} {'ΔSR':>8} "
        f"{'E@K_atk':>8} {'E@K_nrm':>8} {'ΔE@K':>8} "
        f"{'A->A':>9} {'A->N':>9} {'A_gap':>9} "
        f"{'N->N':>9} {'N->A':>9} {'N_gap':>9} {'SymGap':>9} "
        f"{'nnz_gap':>8} "
        f"{'NullAtk':>9} {'Atk-spec%':>10} {'AUC_motif':>10} {'AUC_null':>10} {'ΔAUC':>8} "
        f"{'nnz_atk':>8} {'nnz_nrm':>8} {'|F|':>6} {'Pairs':>7}"
    )
    print("  " + "-" * 222)

    for thr in thresholds:
        atk_raw, _ = load_matrix(atk_base, thr)
        nrm_raw, _ = load_matrix(nrm_base, thr)
        if atk_raw is None or nrm_raw is None:
            print(f"  {thr:>5} [SKIP - missing data]")
            continue

        atk_mat, nrm_mat, fids = align_to_union(atk_raw, nrm_raw)
        atk_nnz = np.asarray(atk_mat.getnnz(axis=1)).ravel()
        nrm_nnz = np.asarray(nrm_mat.getnnz(axis=1)).ravel()

        atk_err, _ = nmf_fit_relerr(atk_mat, k)
        nrm_err, _ = nmf_fit_relerr(nrm_mat, k)
        atk_sr, atk_e = spectral_lowrank_metrics(atk_mat, topk=k)
        nrm_sr, nrm_e = spectral_lowrank_metrics(nrm_mat, topk=k)
        sr_gap = atk_sr - nrm_sr if np.isfinite(atk_sr) and np.isfinite(nrm_sr) else np.nan
        e_gap = atk_e - nrm_e if np.isfinite(atk_e) and np.isfinite(nrm_e) else np.nan

        # Shared-range matched subsets: fit attack motifs, then compare attack vs non-attack
        # under the same dictionary. Average across multiple resamples for stability.
        atk_train_atk_list = []
        atk_train_nrm_list = []
        nrm_train_nrm_list = []
        nrm_train_atk_list = []
        nnz_gap_list = []
        pair_counts = []
        for rep in range(5):
            rep_rng = np.random.default_rng(42 + rep)
            atk_match, nrm_match, n_pairs = stratified_shared_range_subsets(
                atk_mat, nrm_mat, atk_nnz, nrm_nnz, rep_rng
            )
            if n_pairs == 0:
                continue
            # nnz mismatch diagnostic (0 means perfectly matched activity)
            a_med = float(np.median(np.asarray(atk_match.getnnz(axis=1)).ravel()))
            n_med = float(np.median(np.asarray(nrm_match.getnnz(axis=1)).ravel()))
            nnz_gap = abs(a_med - n_med) / (a_med + n_med + 1e-10)
            atk_train_atk_err_rep, atk_train_nrm_err_rep, _ = nmf_cross_relerr(
                atk_match, nrm_match, k, seed=42 + rep
            )
            nrm_train_nrm_err_rep, nrm_train_atk_err_rep, _ = nmf_cross_relerr(
                nrm_match, atk_match, k, seed=142 + rep
            )
            atk_train_atk_list.append(atk_train_atk_err_rep)
            atk_train_nrm_list.append(atk_train_nrm_err_rep)
            nrm_train_nrm_list.append(nrm_train_nrm_err_rep)
            nrm_train_atk_list.append(nrm_train_atk_err_rep)
            nnz_gap_list.append(nnz_gap)
            pair_counts.append(n_pairs)

        if atk_train_atk_list:
            atk_train_atk_err = float(np.mean(atk_train_atk_list))
            atk_train_nrm_err = float(np.mean(atk_train_nrm_list))
            atk_train_gap = atk_train_atk_err - atk_train_nrm_err
            nrm_train_nrm_err = float(np.mean(nrm_train_nrm_list))
            nrm_train_atk_err = float(np.mean(nrm_train_atk_list))
            nrm_train_gap = nrm_train_atk_err - nrm_train_nrm_err
            # SymGap (attack-advantage oriented):
            #   A_gap = A->A - A->N   (more negative => attack dictionary favors attack)
            #   N_gap = N->A - N->N   (more positive => non-attack dictionary favors non-attack)
            # We combine them as A_gap - N_gap, so expected attack-dominant regions
            # become clearly negative and easier to compare with Atk_err trend.
            sym_raw = 0.5 * (atk_train_gap - nrm_train_gap)
            sym_nnz_gap = float(np.mean(nnz_gap_list)) if nnz_gap_list else 0.0
            sym_gap = float(sym_raw / (1.0 + NNZ_PENALTY_LAMBDA * sym_nnz_gap))
            n_pairs = int(np.mean(pair_counts))
        else:
            atk_train_atk_err = np.nan
            atk_train_nrm_err = np.nan
            atk_train_gap = np.nan
            nrm_train_nrm_err = np.nan
            nrm_train_atk_err = np.nan
            nrm_train_gap = np.nan
            sym_gap = np.nan
            sym_nnz_gap = np.nan
            n_pairs = 0

        null_errs = []
        for seed in range(NULL_SEEDS):
            rng_null = np.random.default_rng(100 + seed)
            null_mat = null_permute_rowwise(atk_mat, rng_null)
            ne, _ = nmf_fit_relerr(null_mat, k)
            null_errs.append(ne)
        null_atk_err = float(np.mean(null_errs))

        atk_spec_pct, _ = atk_specific_fraction(atk_mat, nrm_mat, k)
        motif_auc, null_group_auc = separability_auc_suite(atk_mat, nrm_mat, k)
        auc_gain = motif_auc - null_group_auc

        row = SweepRow(
            delta=str(thr),
            atk_err=atk_err,
            nrm_err=nrm_err,
            atk_train_atk_err=atk_train_atk_err,
            atk_train_nrm_err=atk_train_nrm_err,
            atk_train_gap=atk_train_gap,
            nrm_train_nrm_err=nrm_train_nrm_err,
            nrm_train_atk_err=nrm_train_atk_err,
            nrm_train_gap=nrm_train_gap,
            sym_gap=sym_gap,
            sym_nnz_gap=sym_nnz_gap,
            null_atk_err=null_atk_err,
            atk_spec_pct=atk_spec_pct,
            atk_nnz_med=int(np.median(atk_nnz)),
            nrm_nnz_med=int(np.median(nrm_nnz)),
            active_feat_union=len(fids),
            matched_pairs=n_pairs,
            motif_auc=motif_auc,
            null_group_auc=null_group_auc,
            auc_gain=auc_gain,
            atk_stable_rank=atk_sr,
            nrm_stable_rank=nrm_sr,
            stable_rank_gap=sr_gap,
            atk_topk_energy=atk_e,
            nrm_topk_energy=nrm_e,
            topk_energy_gap=e_gap,
        )
        rows.append(row)
        sr_a = f"{row.atk_stable_rank:>8.1f}" if np.isfinite(row.atk_stable_rank) else f"{'NA':>8}"
        sr_n = f"{row.nrm_stable_rank:>8.1f}" if np.isfinite(row.nrm_stable_rank) else f"{'NA':>8}"
        sr_d = f"{row.stable_rank_gap:>+8.1f}" if np.isfinite(row.stable_rank_gap) else f"{'NA':>8}"
        e_a = f"{row.atk_topk_energy:>8.3f}" if np.isfinite(row.atk_topk_energy) else f"{'NA':>8}"
        e_n = f"{row.nrm_topk_energy:>8.3f}" if np.isfinite(row.nrm_topk_energy) else f"{'NA':>8}"
        e_d = f"{row.topk_energy_gap:>+8.3f}" if np.isfinite(row.topk_energy_gap) else f"{'NA':>8}"
        aa = f"{row.atk_train_atk_err:>9.4f}" if np.isfinite(row.atk_train_atk_err) else f"{'NA':>9}"
        an = f"{row.atk_train_nrm_err:>9.4f}" if np.isfinite(row.atk_train_nrm_err) else f"{'NA':>9}"
        ag = f"{row.atk_train_gap:>+9.4f}" if np.isfinite(row.atk_train_gap) else f"{'NA':>9}"
        nn = f"{row.nrm_train_nrm_err:>9.4f}" if np.isfinite(row.nrm_train_nrm_err) else f"{'NA':>9}"
        na = f"{row.nrm_train_atk_err:>9.4f}" if np.isfinite(row.nrm_train_atk_err) else f"{'NA':>9}"
        ng = f"{row.nrm_train_gap:>+9.4f}" if np.isfinite(row.nrm_train_gap) else f"{'NA':>9}"
        sg = f"{row.sym_gap:>+9.4f}" if np.isfinite(row.sym_gap) else f"{'NA':>9}"
        zg = f"{row.sym_nnz_gap:>8.3f}" if np.isfinite(row.sym_nnz_gap) else f"{'NA':>8}"
        gap = row.atk_err - row.nrm_err
        print(
            f"  {row.delta:>5} {row.atk_err:>9.4f} {row.nrm_err:>9.4f} {gap:>+9.4f} "
            f"{sr_a} {sr_n} {sr_d} {e_a} {e_n} {e_d} "
            f"{aa} {an} {ag} {nn} {na} {ng} {sg} "
            f"{zg} "
            f"{row.null_atk_err:>9.4f} {row.atk_spec_pct:>10.3f} {row.motif_auc:>10.3f} {row.null_group_auc:>10.3f} {row.auc_gain:>+8.3f} "
            f"{row.atk_nnz_med:>8} {row.nrm_nnz_med:>8} {row.active_feat_union:>6} {row.matched_pairs:>7}",
            flush=True,
        )

    return rows


def summarize_stable_band(rows: list[SweepRow]):
    valid = []
    for r in rows:
        try:
            d = float(r.delta)
        except ValueError:
            continue
        if 0.5 <= d <= 2.0 and np.isfinite(r.sym_gap):
            valid.append((d, r))
    if not valid:
        print("\n[Stable-band summary] No valid rows in delta in [0.5, 2.0].")
        return

    valid.sort(key=lambda x: x[0])
    deltas = np.array([x[0] for x in valid], dtype=float)
    sym = np.array([x[1].sym_gap for x in valid], dtype=float)
    gains = np.array([x[1].auc_gain for x in valid], dtype=float)
    motif_auc = np.array([x[1].motif_auc for x in valid], dtype=float)
    null_auc = np.array([x[1].null_group_auc for x in valid], dtype=float)

    slope = float(np.polyfit(deltas, sym, deg=1)[0]) if len(deltas) >= 2 else float("nan")
    neg_ratio = float((sym < 0).mean())

    print("\n" + "=" * 92)
    print("Stable-band summary (delta in [0.5, 2.0])")
    print("=" * 92)
    print(f"  SymGap mean+/-std   : {sym.mean():+.4f} +/- {sym.std(ddof=0):.4f}")
    print(f"  SymGap negative rate: {neg_ratio:.1%}")
    print(f"  SymGap slope vs δ   : {slope:+.4f}")
    print(f"  AUC motif mean      : {motif_auc.mean():.3f}")
    print(f"  AUC null-group mean : {null_auc.mean():.3f}")
    print(f"  DeltaAUC mean+/-std : {gains.mean():+.3f} +/- {gains.std(ddof=0):.3f}")


def detailed_diagnostics(atk_base: str, nrm_base: str, delta_fixed: str, k: int, save_dir: str):
    atk_raw, _ = load_matrix(atk_base, delta_fixed)
    nrm_raw, _ = load_matrix(nrm_base, delta_fixed)
    if atk_raw is None or nrm_raw is None:
        raise FileNotFoundError(f"Could not load attack/non-attack data for delta={delta_fixed}")

    atk_mat, nrm_mat, fids = align_to_union(atk_raw, nrm_raw)
    print("\n" + "=" * 92)
    print(f"Detailed diagnostics at delta={delta_fixed}")
    print("=" * 92)
    print(f"  Attack: {atk_mat.shape}   nnz_median={int(np.median(atk_mat.getnnz(axis=1)))}")
    print(f"  Non-attack: {nrm_mat.shape}   nnz_median={int(np.median(nrm_mat.getnnz(axis=1)))}")
    print(f"  Union active features: {len(fids)}")

    print("\n  [K sweep]")
    print(f"  {'K':>5} {'Atk_err':>9} {'Nrm_err':>9} {'Gap':>9}")
    print("  " + "-" * 38)
    k_rows = []
    for kk in K_DIAG_LIST:
        a_err, _ = nmf_fit_relerr(atk_mat, kk)
        n_err, _ = nmf_fit_relerr(nrm_mat, kk)
        gap = a_err - n_err
        k_rows.append({"K": kk, "Atk_err": a_err, "Nrm_err": n_err, "Gap": gap})
        print(f"  {kk:>5} {a_err:>9.4f} {n_err:>9.4f} {gap:>+9.4f}")

    atk_spec_pct, ratio = atk_specific_fraction(atk_mat, nrm_mat, k)
    breakdown = {
        "atk_specific": int((ratio > 3.0).sum()),
        "atk_lean": int(((ratio > 1.5) & (ratio <= 3.0)).sum()),
        "shared": int(((ratio >= 0.67) & (ratio <= 1.5)).sum()),
        "nrm_lean": int(((ratio >= 0.33) & (ratio < 0.67)).sum()),
        "nrm_specific": int((ratio < 0.33).sum()),
    }

    print("\n  [Motif breakdown]")
    for key, val in breakdown.items():
        print(f"    {key:<14}: {val:>3} ({val / k:.1%})")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(k_rows).to_csv(os.path.join(save_dir, f"k_sweep_delta_{delta_fixed}.csv"), index=False)
        with open(os.path.join(save_dir, f"motif_breakdown_delta_{delta_fixed}.json"), "w") as f:
            json.dump({"delta": delta_fixed, "K": k, "atk_spec_pct": atk_spec_pct, **breakdown}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atk-base", required=True)
    parser.add_argument("--non-attack-base", dest="nrm_base", required=True)
    parser.add_argument("--thresholds", nargs="+", default=["0.5", "1.0", "1.5", "2.0"])
    parser.add_argument("--delta-fixed", default="2.0")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    rows = threshold_sweep(args.atk_base, args.nrm_base, args.thresholds, args.k)
    summarize_stable_band(rows)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out_df = pd.DataFrame([r.__dict__ for r in rows])
        out_df.to_csv(os.path.join(args.save_dir, "main_threshold_sweep.csv"), index=False)
        print(f"\nSaved main-table CSV to {os.path.join(args.save_dir, 'main_threshold_sweep.csv')}")


    detailed_diagnostics(args.atk_base, args.nrm_base, args.delta_fixed, args.k, args.save_dir)
    print("\n[Done]")


if __name__ == "__main__":
    main()
