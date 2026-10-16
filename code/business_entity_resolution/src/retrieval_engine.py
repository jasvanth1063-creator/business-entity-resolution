"""Sparse TF-IDF top-K retrieval engine (shared by Phase 3 experiments and Phase 4 candidate generation).

Memory is bounded: query batches x pool chunks are scored densely (batch*chunk floats),
the full query x pool similarity matrix is never materialised.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def make_vectorizer(kind="char", ngram=(2, 4), max_df=1.0, sublinear=False):
    common = dict(ngram_range=tuple(ngram), min_df=1, max_df=max_df,
                  sublinear_tf=sublinear, dtype=np.float32)
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", **common)
    return TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", **common)


def build_chunks(P, pchunk=100_000):
    """Pre-transpose the pool matrix into chunks (list of (row_offset, V x chunk CSR)). Free P afterwards."""
    P = sp.csr_matrix(P)
    return [(p0, P[p0:p0 + pchunk].T.tocsr()) for p0 in range(0, P.shape[0], pchunk)], P.shape[0]


def topk_sparse(Q, P, k=100, qbatch=256, pchunk=100_000):
    """Top-k pool rows per query row by dot product. Returns (idx, score); idx=-1 where score<=0."""
    chunks, n = build_chunks(P, pchunk)
    return search_chunks(Q, chunks, n, k=k, qbatch=qbatch)


def search_chunks(Q, chunks, n, k=100, qbatch=256):
    Q = sp.csr_matrix(Q)
    nq = Q.shape[0]
    k_eff = min(k, n)
    best_s = np.full((nq, k_eff), -1.0, np.float32)
    best_i = np.full((nq, k_eff), -1, np.int64)
    for q0 in range(0, nq, qbatch):
        Qb = Q[q0:q0 + qbatch]
        bs, bi = best_s[q0:q0 + qbatch], best_i[q0:q0 + qbatch]
        for p0, Pt in chunks:
            S = (Qb @ Pt).toarray()
            kk = min(k_eff, S.shape[1])
            idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            val = np.take_along_axis(S, idx, 1)
            cs = np.concatenate([bs, val], 1)
            ci = np.concatenate([bi, idx + p0], 1)
            sel = np.argpartition(-cs, k_eff - 1, axis=1)[:, :k_eff]
            bs[:] = np.take_along_axis(cs, sel, 1)
            bi[:] = np.take_along_axis(ci, sel, 1)
    order = np.argsort(-best_s, axis=1)
    best_s = np.take_along_axis(best_s, order, 1)
    best_i = np.take_along_axis(best_i, order, 1)
    best_i[best_s <= 0] = -1
    return best_i, best_s


def topk_by_country(Q, P, q_country, p_country, k=100, **kw):
    """Same-country retrieval. Returns idx into P (global rows), -1 padded, shape (nq, k)."""
    q_country = np.asarray(q_country)
    p_country = np.asarray(p_country)
    out = np.full((Q.shape[0], k), -1, np.int64)
    Q = sp.csr_matrix(Q)
    P = sp.csr_matrix(P)
    for c in np.unique(q_country):
        qi = np.flatnonzero(q_country == c)
        pi = np.flatnonzero(p_country == c)
        if len(qi) == 0 or len(pi) == 0:
            continue
        idx, _ = topk_sparse(Q[qi], P[pi], k=k, **kw)
        glob = np.where(idx >= 0, pi[np.clip(idx, 0, None)], -1)
        out[qi, :glob.shape[1]] = glob
    return out


def search_chunks_sparse(Q, chunks, n, k=100, qbatch=256):
    """Same result as search_chunks (identical scores; ties may order differently) but ~7-15x faster:
    the sparse product Q @ P^T is never densified, top-k is taken from each row's non-zero entries only."""
    Q = sp.csr_matrix(Q)
    nq = Q.shape[0]
    best_i = np.full((nq, k), -1, np.int64)
    best_s = np.zeros((nq, k), np.float32)
    for q0 in range(0, nq, qbatch):
        Qb = Q[q0:q0 + qbatch]
        ci = [[] for _ in range(Qb.shape[0])]
        cs = [[] for _ in range(Qb.shape[0])]
        for p0, Pt in chunks:
            S = (Qb @ Pt).tocsr()
            ip, ind, dat = S.indptr, S.indices, S.data
            for r in range(S.shape[0]):
                a, b = ip[r], ip[r + 1]
                if a == b:
                    continue
                d, ix = dat[a:b], ind[a:b]
                if b - a > k:
                    sel = np.argpartition(-d, k - 1)[:k]
                    d, ix = d[sel], ix[sel]
                ci[r].append(ix.astype(np.int64) + p0)
                cs[r].append(d)
        for r in range(Qb.shape[0]):
            if cs[r]:
                d = np.concatenate(cs[r]); ix = np.concatenate(ci[r])
                o = np.argsort(-d, kind="stable")[:k]
                best_i[q0 + r, :len(o)] = ix[o]; best_s[q0 + r, :len(o)] = d[o]
    best_i[best_s <= 0] = -1
    return best_i, best_s
