"""Phase 4 - full-scale candidate generation (used for the train/val sample now and for the test set later).

For every query S1 record: same-country top-K1 by name TF-IDF  UNION  top-K2 by address TF-IDF
(settings come from retrieval_config.json, written by Phase 3). Both indexes are built over the *full*
S2+S3 pool, one country at a time, so memory stays bounded.

CLI:
    python candidate_generation.py --queries q.pkl --out cand.pkl --s2 <S2.tsv> --s3 <S3.tsv>
`q.pkl` is a pickled DataFrame with columns s1_id, s1_name, s1_addr, country.
"""
import argparse
import gc
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normalization_v2 as nv
import retrieval_engine as RE

HERE = os.path.dirname(os.path.abspath(__file__))
PCHUNK = 400_000
QBATCH = 64


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg():
    with open(os.path.join(HERE, "retrieval_config.json")) as f:
        return json.load(f)


def _norm_chunk(a):
    names, addrs, nr, ar = a
    return ([nv.normalize_name(x, rules=nr) for x in names],
            [nv.normalize_addr(x, rules=ar) for x in addrs])


def read_country(paths, country, name_rules, addr_rules, workers=2, chunksize=400_000, max_chunks=None):
    """Stream the pool files, keep one country, normalize in parallel. Returns ids, names_n, addrs_n (lists)."""
    ids, names, addrs = [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for path in paths:
            for ci, ch in enumerate(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize)):
                if max_chunks is not None and ci >= max_chunks:
                    break
                ch = ch[ch["country"] == country]
                n = len(ch)
                if not n:
                    continue
                halves = np.array_split(np.arange(n), workers)
                futs = [ex.submit(_norm_chunk, (ch["business_name"].values[h].tolist(),
                                                ch["business_address"].values[h].tolist(), name_rules, addr_rules))
                        for h in halves]
                for f in futs:
                    nn, aa = f.result()
                    names += nn; addrs += aa
                ids += ch["entity_id"].tolist()
    return ids, names, addrs


def fit_transform_chunks(spec, texts, fit_n, seed, chunk=500_000):
    rng = np.random.default_rng(seed)
    n = len(texts)
    sel = rng.choice(n, min(fit_n, n), replace=False)
    vec = RE.make_vectorizer(spec["kind"], spec["ngram"], spec["max_df"], spec.get("sublinear_tf", False))
    vec.fit([texts[i] for i in sel])
    vec.stop_words_ = None          # can be huge; not needed for transform
    mats = [vec.transform(texts[i:i + chunk]) for i in range(0, n, chunk)]
    return vec, sp.vstack(mats).tocsr()


_G = {}


def _work(a):
    route, q0, q1 = a
    idx, sc = RE.search_chunks_sparse(_G[route + "_Q"][q0:q1], _G[route + "_chunks"], _G[route + "_n"],
                               k=_G[route + "_k"], qbatch=QBATCH)
    return route, q0, idx, sc


def _search(routes, nq, workers):
    tasks = [(r, q0, min(q0 + 512, nq)) for r in routes for q0 in range(0, nq, 512)]
    out = {r: (np.full((nq, _G[r + "_k"]), -1, np.int64), np.zeros((nq, _G[r + "_k"]), np.float32)) for r in routes}
    t0, done = time.time(), 0

    def consume(res):
        nonlocal done
        r, q0, idx, sc = res
        out[r][0][q0:q0 + idx.shape[0], :idx.shape[1]] = idx
        out[r][1][q0:q0 + idx.shape[0], :idx.shape[1]] = sc
        done += 1
        if done % 5 == 0 or done == len(tasks):
            log(f"   search {done}/{len(tasks)} tasks, {time.time()-t0:.0f}s")

    # Worker processes can only share the big in-memory index through fork (Linux/macOS).
    # Windows has no fork (workers would start with an empty _G), so search in this process instead.
    if workers > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(workers) as pool:
            for res in pool.imap_unordered(_work, tasks):
                consume(res)
    else:
        log("   searching in a single process (no fork on this OS, or workers=1)")
        for t in tasks:
            consume(_work(t))
    return out


def generate(queries, paths, cfg, out_path, fit_n=2_000_000, workers=2, seed=42, max_chunks=None):
    nr, ar = cfg["normalization_config"]["name_rules"], cfg["normalization_config"]["addr_rules"]
    nspec = cfg["name_vectorizer"]; aspec = cfg["address_route"]
    K1, K2 = cfg["K_name"], (aspec["K_addr"] if aspec else 0)
    parts = []
    for country in sorted(queries["country"].unique()):
        part_path = f"{out_path}.{country}.pkl"
        if os.path.exists(part_path):
            log(f"{country}: cached part found, skipping"); parts.append(pickle.load(open(part_path, "rb"))); continue
        qc = queries[queries["country"] == country].reset_index(drop=True)
        log(f"=== {country}: {len(qc):,} queries; reading + normalizing pool")
        t0 = time.time()
        ids, names, addrs = read_country(paths, country, nr, ar, workers=workers, max_chunks=max_chunks)
        ids = np.array(ids, dtype=object)
        log(f"   pool rows: {len(ids):,} ({time.time()-t0:.0f}s)")
        qn = [nv.normalize_name(x, rules=nr) for x in qc["s1_name"]]
        qa = [nv.normalize_addr(x, rules=ar) for x in qc["s1_addr"]]
        routes = ["name"] + (["addr"] if aspec else [])

        t0 = time.time()
        vec, Pn = fit_transform_chunks(nspec, names, fit_n, seed)
        _G["name_Q"] = vec.transform(qn)
        log(f"   name index: nnz/row {Pn.nnz/Pn.shape[0]:.1f}, vocab {len(vec.vocabulary_):,} ({time.time()-t0:.0f}s)")
        _G["name_chunks"], _G["name_n"] = RE.build_chunks(Pn, PCHUNK); _G["name_k"] = K1
        del Pn, names, vec; gc.collect()
        if aspec:
            t0 = time.time()
            vec, Pa = fit_transform_chunks(aspec, addrs, fit_n, seed)
            _G["addr_Q"] = vec.transform(qa)
            log(f"   addr index: nnz/row {Pa.nnz/Pa.shape[0]:.1f}, vocab {len(vec.vocabulary_):,} ({time.time()-t0:.0f}s)")
            _G["addr_chunks"], _G["addr_n"] = RE.build_chunks(Pa, PCHUNK); _G["addr_k"] = K2
            del Pa, vec
        del addrs; gc.collect()

        res = _search(routes, len(qc), workers)
        frames = []
        qid = qc["s1_id"].values
        for r in routes:
            idx, sc = res[r]
            rows, ranks = np.nonzero(idx >= 0)
            frames.append(pd.DataFrame({"s1_id": qid[rows], "cand_id": ids[idx[rows, ranks]],
                                        f"{r}_rank": ranks + 1, f"{r}_score": sc[rows, ranks]}))
        df = frames[0]
        for f in frames[1:]:
            df = df.merge(f, on=["s1_id", "cand_id"], how="outer")
        df["country"] = country
        pickle.dump(df, open(part_path, "wb"))
        parts.append(df)
        log(f"   {country}: {len(df):,} candidate pairs")
        _G.clear(); del ids; gc.collect()
    out = pd.concat(parts, ignore_index=True)
    pickle.dump(out, open(out_path, "wb"))
    return out


def fetch_raw(ids, paths, chunksize=500_000):
    """Raw name/address/country for a set of S2/S3 entity ids (chunked scan)."""
    wanted, found = set(ids), []
    for path in paths:
        for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize):
            h = ch[ch["entity_id"].isin(wanted)]
            if len(h):
                found.append(h)
    return pd.concat(found).rename(columns={"entity_id": "cand_id", "business_name": "c_name",
                                            "business_address": "c_addr", "country": "c_country"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--s2", required=True); ap.add_argument("--s3", required=True)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--fit-n", type=int, default=2_000_000)
    a = ap.parse_args()
    q = pickle.load(open(a.queries, "rb"))
    generate(q, [a.s2, a.s3], load_cfg(), a.out, fit_n=a.fit_n, max_chunks=a.max_chunks)
    log("done")
