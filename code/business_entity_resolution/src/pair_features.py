"""Phase 5 - pair feature engineering (label-free; reused unchanged for validation and test).

compute_features(df, stats, cfg) takes a candidate table with columns
    s1_id, s1_name, s1_addr, cand_id, c_name, c_addr, name_rank, name_score, addr_rank, addr_score, country
and returns a float32 feature DataFrame (one row per candidate pair, same order).

`stats` (token document-frequencies + TF-IDF vectorizers) is fitted on an *unlabelled* S2/S3 pool sample by build_stats().
"""
import math
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import jellyfish
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normalization_v2 as nv

LEGAL = {"ltd", "pvt", "llc", "inc", "corp", "co", "llp", "lp", "pc", "pllc", "plc", "incorporated", "company"}
_DIGIT = re.compile(r"\d")
_NONWORD = re.compile(r"[^\w\s]")
_SDX = {}

GROUPS = {
    "name_basic": ["n_ratio", "n_partial", "n_tsort", "n_tset", "n_jw", "n_jacc", "n_exact", "n_tfidf_cos",
                   "n_len_s", "n_len_c", "n_len_ratio", "n_ntok_s", "n_ntok_c"],
    "name_structural": ["n_idf_jacc", "n_unm_s", "n_unm_c", "n_unm_s_idf", "n_unm_c_idf", "n_unm_s_max_idf", "n_unm_c_max_idf",
                        "n_contain", "n_soft_s", "n_soft_c", "n_soft_min", "n_first_eq", "n_first_jw", "n_first_sdx",
                        "n_suf_equal", "n_suf_conflict", "n_suf_onesided", "c_domain_style", "c_nonlatin"],
    "addr_basic": ["a_c_missing", "a_ratio", "a_tsort", "a_tset", "a_partial", "a_jacc", "a_idf_jacc", "a_tfidf_cos", "a_part_jacc"],
    "addr_numbers": ["a_s_has_num", "a_c_has_num", "a_num_jacc", "a_num_shared", "a_num_conflict", "a_first_num_eq", "a_num_min_edit"],
    "retrieval": ["name_rank", "name_score", "addr_rank", "addr_score", "in_name", "in_addr", "c_is_s2", "is_us"],
    "relative": [],     # filled below
}
REL_BASE = ["n_tfidf_cos", "n_jw", "n_soft_min", "n_idf_jacc", "a_tsort", "a_tfidf_cos", "a_num_jacc"]
for _b in REL_BASE:
    GROUPS["relative"] += [f"{_b}__rank", f"{_b}__dmax", f"{_b}__margin"]
ALL_FEATURES = [c for g in GROUPS.values() for c in g]


# ---------------------------------------------------------------------------------------------
def build_stats(names_n, addrs_n, fit_n=500_000, seed=42):
    """Document-frequency tables + TF-IDF vectorizers from an unlabelled pool sample (normalized strings)."""
    rng = np.random.default_rng(seed)
    names_n = list(names_n); addrs_n = list(addrs_n)
    ndf, adf = Counter(), Counter()
    for s in names_n:
        ndf.update(set(s.split()))
    for s in addrs_n:
        adf.update(set(s.split()))
    sel = rng.choice(len(names_n), min(fit_n, len(names_n)), replace=False)
    vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, dtype=np.float32).fit([names_n[i] for i in sel])
    va = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", min_df=1, dtype=np.float32).fit([addrs_n[i] for i in sel])
    return dict(N=len(names_n), name_df=dict(ndf), addr_df=dict(adf), name_vec=vn, addr_vec=va)


_ST = {}


def _idf(t, which):
    st = _ST["stats"]
    return math.log((st["N"] + 1) / (st[which].get(t, 0) + 1)) + 1.0


def _sdx(t):
    v = _SDX.get(t)
    if v is None:
        try:
            v = jellyfish.soundex(t) if t else ""
        except Exception:
            v = ""
        _SDX[t] = v
    return v


def _norm_num(t):
    return re.sub(r"^0+(?=\d)", "", t)


NAN = float("nan")


def _row(sn, cn, sa, ca, s_parts, c_parts, raw_c_name):
    f = {}
    # ---------------- name ----------------
    ta, tb = sn.split(), cn.split()
    A, B = set(ta), set(tb)
    inter, uni = A & B, A | B
    f["n_ratio"] = fuzz.ratio(sn, cn) / 100
    f["n_partial"] = fuzz.partial_ratio(sn, cn) / 100
    f["n_tsort"] = fuzz.token_sort_ratio(sn, cn) / 100
    f["n_tset"] = fuzz.token_set_ratio(sn, cn) / 100
    f["n_jw"] = JaroWinkler.similarity(sn, cn)
    f["n_jacc"] = len(inter) / len(uni) if uni else 0.0
    f["n_exact"] = float(sn == cn)
    f["n_len_s"], f["n_len_c"] = len(sn), len(cn)
    f["n_len_ratio"] = min(len(sn), len(cn)) / max(len(sn), len(cn), 1)
    f["n_ntok_s"], f["n_ntok_c"] = len(ta), len(tb)
    wi = sum(_idf(t, "name_df") for t in inter); wu = sum(_idf(t, "name_df") for t in uni)
    f["n_idf_jacc"] = wi / wu if wu else 0.0
    ua, ub = A - B, B - A
    ia = [_idf(t, "name_df") for t in ua]; ib = [_idf(t, "name_df") for t in ub]
    f["n_unm_s"], f["n_unm_c"] = len(ua), len(ub)
    f["n_unm_s_idf"], f["n_unm_c_idf"] = sum(ia), sum(ib)
    f["n_unm_s_max_idf"], f["n_unm_c_max_idf"] = max(ia, default=0.0), max(ib, default=0.0)
    f["n_contain"] = float(bool(A) and bool(B) and (A <= B or B <= A))
    if ta and tb:
        ba = [max(JaroWinkler.similarity(x, y) for y in tb) for x in ta]
        bb = [max(JaroWinkler.similarity(y, x) for x in ta) for y in tb]
        f["n_soft_s"], f["n_soft_c"] = sum(ba) / len(ba), sum(bb) / len(bb)
        f["n_soft_min"] = min(f["n_soft_s"], f["n_soft_c"])
        f["n_first_eq"] = float(ta[0] == tb[0]); f["n_first_jw"] = JaroWinkler.similarity(ta[0], tb[0])
        f["n_first_sdx"] = float(_sdx(ta[0]) == _sdx(tb[0]) and _sdx(ta[0]) != "")
    else:
        f["n_soft_s"] = f["n_soft_c"] = f["n_soft_min"] = f["n_first_eq"] = f["n_first_jw"] = f["n_first_sdx"] = 0.0
    ls, lc = A & LEGAL, B & LEGAL
    f["n_suf_equal"] = float(bool(ls) and bool(lc) and ls == lc)
    f["n_suf_conflict"] = float(bool(ls) and bool(lc) and ls != lc)
    f["n_suf_onesided"] = float(bool(ls) != bool(lc))
    f["c_domain_style"] = float(bool(re.search(r"\.(?:com|net|org|co|in|biz|io)\b", raw_c_name, re.I)))
    f["c_nonlatin"] = float(bool(re.search(r"[^\x00-\x7F]", raw_c_name)))
    # ---------------- address ----------------
    present = bool(sa.strip()) and bool(ca.strip())
    f["a_c_missing"] = float(not ca.strip())
    nums_s = [_norm_num(t) for t in sa.split() if _DIGIT.search(t)]
    nums_c = [_norm_num(t) for t in ca.split() if _DIGIT.search(t)]
    f["a_s_has_num"] = float(bool(nums_s)); f["a_c_has_num"] = float(bool(nums_c))
    if present:
        xa, xb = set(sa.split()), set(ca.split())
        f["a_ratio"] = fuzz.ratio(sa, ca) / 100
        f["a_tsort"] = fuzz.token_sort_ratio(sa, ca) / 100
        f["a_tset"] = fuzz.token_set_ratio(sa, ca) / 100
        f["a_partial"] = fuzz.partial_ratio(sa, ca) / 100
        u = xa | xb
        f["a_jacc"] = len(xa & xb) / len(u) if u else 0.0
        wi = sum(_idf(t, "addr_df") for t in xa & xb); wu = sum(_idf(t, "addr_df") for t in u)
        f["a_idf_jacc"] = wi / wu if wu else 0.0
        pu = s_parts | c_parts
        f["a_part_jacc"] = len(s_parts & c_parts) / len(pu) if pu else 0.0
        if nums_s and nums_c:
            ss, sc = set(nums_s), set(nums_c)
            f["a_num_jacc"] = len(ss & sc) / len(ss | sc)
            f["a_num_shared"] = float(len(ss & sc)); f["a_num_conflict"] = float(not (ss & sc))
            f["a_first_num_eq"] = float(nums_s[0] == nums_c[0])
            f["a_num_min_edit"] = float(min(Levenshtein.distance(x, y) for x in ss for y in sc))
        else:
            f["a_num_jacc"] = f["a_num_shared"] = f["a_num_conflict"] = f["a_first_num_eq"] = f["a_num_min_edit"] = NAN
    else:
        for k in ["a_ratio", "a_tsort", "a_tset", "a_partial", "a_jacc", "a_idf_jacc", "a_part_jacc",
                  "a_num_jacc", "a_num_shared", "a_num_conflict", "a_first_num_eq", "a_num_min_edit"]:
            f[k] = NAN
    return f


def _chunk(args):
    cols = ["sn", "cn", "sa", "ca", "sp", "cp", "raw"]
    rows = [_row(*t) for t in zip(*[args[c] for c in cols])]
    return pd.DataFrame(rows).astype(np.float32)


def _parts(raw):
    return frozenset(p for p in (_NONWORD.sub(" ", x).strip().lower() for x in str(raw).split(",")) if p)


def _cos_rows(Xa, Xb, ia, ib, chunk=200_000):
    out = np.zeros(len(ia), np.float32)
    for s in range(0, len(ia), chunk):
        e = s + chunk
        out[s:e] = np.asarray(Xa[ia[s:e]].multiply(Xb[ib[s:e]]).sum(axis=1)).ravel()
    return out


def _second_largest(vals, g):
    """Second-largest non-NaN value of each row's group (NaN if the group has < 2 non-NaN values). Vectorised; same result as
    groupby.transform(lambda x: x.nlargest(2).iloc[-1] if x.notna().sum() >= 2 else nan) but ~30x faster."""
    v = np.asarray(vals, dtype=float); n = len(v)
    if n == 0:
        return v
    codes = pd.factorize(np.asarray(g))[0]; key = np.where(np.isnan(v), -np.inf, v)
    order = np.lexsort((-key, codes)); cs = codes[order]; sv = key[order]
    start = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1]; size = np.diff(np.r_[start, n])
    nn = np.add.reduceat((~np.isnan(v[order])).astype(np.int64), start)
    sec = np.where(nn >= 2, sv[np.minimum(start + 1, n - 1)], np.nan)
    out = np.empty(n); out[order] = np.repeat(sec, size)
    return out


def _init_worker(stats):
    _ST["stats"] = stats


def make_pool(stats, workers):
    """Persistent worker pool that works with fork (Linux/macOS) and spawn (Windows): stats are passed once via the initializer."""
    import multiprocessing as mp
    method = os.environ.get("PAIR_START_METHOD") or ("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    return mp.get_context(method).Pool(workers, initializer=_init_worker, initargs=(stats,))


def compute_features(df, stats, cfg, workers=2, log=print, pool=None):
    _ST["stats"] = stats
    nr = cfg["normalization_config"]["name_rules"]
    ar_unsorted = [r for r in cfg["normalization_config"]["addr_rules"] if r != "token_sort"]
    ar_full = cfg["normalization_config"]["addr_rules"]

    def norm_map(series, fn):
        u = series.unique()
        return dict(zip(u, [fn(x) for x in u]))

    mn_s = norm_map(df["s1_name"], lambda x: nv.normalize_name(x, rules=nr))
    mn_c = norm_map(df["c_name"], lambda x: nv.normalize_name(x, rules=nr))
    ma_s = norm_map(df["s1_addr"], lambda x: nv.normalize_addr(x, rules=ar_unsorted))
    ma_c = norm_map(df["c_addr"], lambda x: nv.normalize_addr(x, rules=ar_unsorted))
    mp_s = norm_map(df["s1_addr"], _parts); mp_c = norm_map(df["c_addr"], _parts)
    log(f"   normalized {len(mn_c):,} unique candidate names / {len(ma_c):,} addresses")
    cols = dict(sn=df["s1_name"].map(mn_s).tolist(), cn=df["c_name"].map(mn_c).tolist(),
                sa=df["s1_addr"].map(ma_s).tolist(), ca=df["c_addr"].map(ma_c).tolist(),
                sp=df["s1_addr"].map(mp_s).tolist(), cp=df["c_addr"].map(mp_c).tolist(), raw=df["c_name"].tolist())
    n = len(df); step = 25_000
    jobs = [{k: v[s:s + step] for k, v in cols.items()} for s in range(0, n, step)]
    import multiprocessing as mp
    parts = []
    def _progress(i):
        if (i + 1) % 10 == 0 or i + 1 == len(jobs):
            log(f"   pair features {min((i+1)*step, n):,}/{n:,}")
    if pool is not None:
        for i, r in enumerate(pool.imap(_chunk, jobs, chunksize=1)):
            parts.append(r); _progress(i)
    elif workers > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(workers) as pool:        # workers inherit _ST (stats) through fork
            for i, r in enumerate(pool.imap(_chunk, jobs)):
                parts.append(r); _progress(i)
    else:                                                          # Windows: no fork -> single process
        log("   computing in a single process (no fork on this OS)")
        for i, j in enumerate(jobs):
            parts.append(_chunk(j)); _progress(i)
    F = pd.concat(parts, ignore_index=True)

    # vectorised TF-IDF cosines on unique strings
    for key, src_s, src_c, vec, mapper_s, mapper_c in [
        ("n_tfidf_cos", "s1_name", "c_name", stats["name_vec"], mn_s, mn_c)]:
        us, uc = df[src_s].unique(), df[src_c].unique()
        Xs = vec.transform([mapper_s[x] for x in us]); Xc = vec.transform([mapper_c[x] for x in uc])
        ia = pd.Series(np.arange(len(us)), index=us)[df[src_s]].values; ib = pd.Series(np.arange(len(uc)), index=uc)[df[src_c]].values
        F[key] = _cos_rows(Xs, Xc, ia, ib)
    nsorted_s = norm_map(df["s1_addr"], lambda x: nv.normalize_addr(x, rules=ar_full))
    nsorted_c = norm_map(df["c_addr"], lambda x: nv.normalize_addr(x, rules=ar_full))
    us, uc = df["s1_addr"].unique(), df["c_addr"].unique()
    vec = stats["addr_vec"]
    Xs = vec.transform([nsorted_s[x] for x in us]); Xc = vec.transform([nsorted_c[x] for x in uc])
    ia = pd.Series(np.arange(len(us)), index=us)[df["s1_addr"]].values; ib = pd.Series(np.arange(len(uc)), index=uc)[df["c_addr"]].values
    F["a_tfidf_cos"] = _cos_rows(Xs, Xc, ia, ib)
    miss = F["a_c_missing"].values > 0
    F.loc[miss, "a_tfidf_cos"] = np.nan

    # retrieval features
    for c in ["name_rank", "name_score", "addr_rank", "addr_score"]:
        F[c] = df[c].astype(np.float32).values
    F["in_name"] = df["name_rank"].notna().astype(np.float32).values
    F["in_addr"] = df["addr_rank"].notna().astype(np.float32).values
    F["c_is_s2"] = df["cand_id"].str.startswith("S2").astype(np.float32).values
    F["is_us"] = (df["country"] == "US").astype(np.float32).values

    # within-query relative features
    g = df["s1_id"].values
    for b in REL_BASE:
        s = pd.Series(F[b].values, index=F.index)
        grp = s.groupby(g)
        F[f"{b}__rank"] = grp.rank(ascending=False, method="min").astype(np.float32).values
        mx = grp.transform("max")
        F[f"{b}__dmax"] = (s - mx).astype(np.float32).values
        # margin: this candidate minus the best OTHER candidate in the same query
        second = pd.Series(_second_largest(s.values, g), index=s.index)
        other_best = np.where(s.values >= mx.values, second.values, mx.values)
        F[f"{b}__margin"] = (s.values - other_best).astype(np.float32)
    return F[ALL_FEATURES].astype(np.float32)
