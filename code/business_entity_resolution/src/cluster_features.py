"""Phase 6 - cluster-consistency (second-stage) features.

Idea: the true matches of one S1 entity are records of the SAME business (typically one from S2 and one/more from S3), so they resemble
each other, while look-alike decoys form a separate group. Given first-stage match probabilities p for every candidate of a query,
these features describe how a candidate relates to the other candidates of the same query.

cluster_features(...) is label-free given p; p must be out-of-fold for training rows (see the Phase 6 notebook).
"""
import os
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normalization_v2 as nv

GROUPS = {
    "listwise": ["p_logit", "p_rank", "p_margin", "p_sum", "n_gt_half"],
    "support": ["sup_sum", "sup_max", "nbr_hi", "nbr_hi_p", "best_other_p_cluster", "cross_src_sup"],
    "competitor": ["comp_max_p", "comp_cnt_hi"],
}
ALL = [c for g in GROUPS.values() for c in g]


def prep_strings(df, cfg):
    """Normalized candidate name and (unsorted) address for every row, via unique-string maps."""
    nr = cfg["normalization_config"]["name_rules"]
    ar = [r for r in cfg["normalization_config"]["addr_rules"] if r != "token_sort"]
    un = df["c_name"].unique(); ua = df["c_addr"].unique()
    mn = dict(zip(un, [nv.normalize_name(x, rules=nr) for x in un]))
    ma = dict(zip(ua, [nv.normalize_addr(x, rules=ar) for x in ua]))
    return df["c_name"].map(mn).values, df["c_addr"].map(ma).values


def affinity(names, addrs):
    """Candidate-candidate 'same business' affinity matrix in [0,1] (diagonal 0)."""
    sn = process.cdist(names, names, scorer=fuzz.ratio, dtype=np.uint8) / 100.0
    has = np.array([bool(a.strip()) for a in addrs])
    both = has[:, None] & has[None, :]
    if both.any():
        sa = process.cdist(addrs, addrs, scorer=fuzz.token_sort_ratio, dtype=np.uint8) / 100.0
        aff = np.where(both, 0.5 * sn + 0.5 * sa, 0.8 * sn)
    else:
        aff = 0.8 * sn
    np.fill_diagonal(aff, 0.0)
    return aff


def cluster_features(s1_ids, name_n, addr_n, src_is_s2, p):
    """Row-aligned cluster features (float32 DataFrame, columns = ALL)."""
    n_rows = len(p)
    out = np.zeros((n_rows, len(ALL)), np.float32)
    col = {c: i for i, c in enumerate(ALL)}
    groups = pd.Series(np.arange(n_rows)).groupby(np.asarray(s1_ids)).indices
    for q, ix in groups.items():
        n = len(ix)
        pv = np.asarray(p)[ix].astype(float)
        aff = affinity(list(name_n[ix]), list(addr_n[ix]))
        P = np.broadcast_to(pv[None, :], (n, n))
        W = aff * P
        hi = aff >= 0.8
        off = ~np.eye(n, dtype=bool)
        cross = (src_is_s2[ix][:, None] != src_is_s2[ix][None, :])
        comp = (aff < 0.5) & off
        pc = np.clip(pv, 1e-4, 1 - 1e-4)
        other_max = np.array([np.max(np.delete(pv, i)) if n > 1 else 0.0 for i in range(n)])
        r = np.empty((n, len(ALL)), np.float32)
        r[:, col["p_logit"]] = np.log(pc / (1 - pc))
        r[:, col["p_rank"]] = (-pv).argsort().argsort() + 1
        r[:, col["p_margin"]] = pv - other_max
        r[:, col["p_sum"]] = pv.sum()
        r[:, col["n_gt_half"]] = (pv > 0.5).sum()
        r[:, col["sup_sum"]] = W.sum(1)
        r[:, col["sup_max"]] = W.max(1)
        r[:, col["nbr_hi"]] = hi.sum(1)
        r[:, col["nbr_hi_p"]] = (hi * P).sum(1)
        r[:, col["best_other_p_cluster"]] = (hi * P).max(1)
        r[:, col["cross_src_sup"]] = (W * cross).max(1)
        r[:, col["comp_max_p"]] = np.where(comp, P, 0.0).max(1)
        r[:, col["comp_cnt_hi"]] = (comp & (P > 0.5)).sum(1)
        out[ix] = r
    return pd.DataFrame(out, columns=ALL)
