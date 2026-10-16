"""Phase 13 (final) - test-set inference, built for the full-size test files (about 1.8M S1 queries against ~1 GB of S2+S3).

    python predict_phase13.py --test-dir ../../../dataset/test --out-dir ../../../output [--workers 4]

Inputs (names fixed by the competition): test_source1.tsv, test_source2.tsv, test_source3.tsv in --test-dir.
Outputs: <out-dir>/candidate_pairs.tsv and <out-dir>/matching_results.tsv (every S1 entity has a row; empty = no match).

Design for size
  * one country at a time; the TF-IDF indexes of a country exist only while that country is searched;
  * queries are processed in blocks (--block, default 20,000 queries) and every finished block is written to <work-dir>;
    re-running the same command resumes where it stopped (delete the work-dir to start over);
  * stage A (retrieval) -> stage B (features + model scores) are separate, so stage B can be repeated without redoing retrieval;
  * only pairs scoring >= --cand-min-score (default 0.01) are kept for the final files. candidate_pairs.tsv lists those pairs
    (the retrieved candidates that the classifier did not rule out); every matched ID is also in its candidate list;
  * memory: roughly 8 GB peak for the larger country; Windows uses spawn-based workers for features and one process for retrieval.
"""
import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import candidate_generation as CG
import normalization_v2 as nv
import pair_features as PF
import retrieval_engine as RE
from cleanup import apply_cleanup
import listwise_model as LM

log = CG.log
S1_COLS = {"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"}


def read_s1(path, limit=None):
    q = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).rename(columns=S1_COLS)
    return q.head(limit) if limit else q


def read_raw_pool(paths, country, chunksize=500_000):
    """Raw ids / names / addresses of one country's S2+S3 records, in the same order as CG.read_country."""
    ids, names, addrs = [], [], []
    for path in paths:
        for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize):
            ch = ch[ch["country"] == country]
            if len(ch):
                ids += ch["entity_id"].tolist(); names += ch["business_name"].tolist(); addrs += ch["business_address"].tolist()
    return np.array(ids, dtype=object), np.array(names, dtype=object), np.array(addrs, dtype=object)


def stage_a(qc, paths, cfg, country, blocks, work, block, workers, fit_n=2_000_000, seed=42):
    """Retrieval for all missing blocks of one country; writes cand_<country>_<b>.pkl (pool row indices + ranks + scores)."""
    todo = [b for b in blocks if not os.path.exists(f"{work}/cand_{country}_{b:06d}.pkl")]
    if not todo:
        return
    nr, ar = cfg["normalization_config"]["name_rules"], cfg["normalization_config"]["addr_rules"]
    nspec, aspec = cfg["name_vectorizer"], cfg["address_route"]
    K1, K2 = cfg["K_name"], (aspec["K_addr"] if aspec else 0)
    log(f"=== {country}: retrieval for {len(todo)}/{len(blocks)} blocks; reading + normalizing pool")
    t0 = time.time()
    ids, names, addrs = CG.read_country(paths, country, nr, ar, workers=max(1, workers))
    log(f"   pool rows {len(ids):,} ({time.time()-t0:.0f}s)")
    del ids
    t0 = time.time(); vec_n, Pn = CG.fit_transform_chunks(nspec, names, fit_n, seed)
    G = CG._G; G["name_chunks"], G["name_n"] = RE.build_chunks(Pn, CG.PCHUNK); G["name_k"] = K1; del Pn, names
    log(f"   name index built ({time.time()-t0:.0f}s)")
    routes = ["name"]; vec_a = None
    if aspec:
        t0 = time.time(); vec_a, Pa = CG.fit_transform_chunks(aspec, addrs, fit_n, seed)
        G["addr_chunks"], G["addr_n"] = RE.build_chunks(Pa, CG.PCHUNK); G["addr_k"] = K2; del Pa
        routes.append("addr"); log(f"   address index built ({time.time()-t0:.0f}s)")
    del addrs; gc.collect()
    t_start = time.time()
    for done, b in enumerate(todo, 1):
        qb = qc.iloc[b:b + block]
        G["name_Q"] = vec_n.transform([nv.normalize_name(x, rules=nr) for x in qb["s1_name"]])
        if aspec:
            G["addr_Q"] = vec_a.transform([nv.normalize_addr(x, rules=ar) for x in qb["s1_addr"]])
        res = CG._search(routes, len(qb), workers)
        frames = []
        for r in routes:
            idx, sc = res[r]; rows, ranks = np.nonzero(idx >= 0)
            frames.append(pd.DataFrame({"q": (b + rows).astype(np.int32), "pool": idx[rows, ranks].astype(np.int32),
                                        f"{r}_rank": (ranks + 1).astype(np.int16), f"{r}_score": sc[rows, ranks]}))
        df = frames[0]
        for f in frames[1:]:
            df = df.merge(f, on=["q", "pool"], how="outer")
        df.to_pickle(f"{work}/cand_{country}_{b:06d}.pkl.tmp"); os.replace(f"{work}/cand_{country}_{b:06d}.pkl.tmp", f"{work}/cand_{country}_{b:06d}.pkl")
        el = time.time() - t_start; log(f"   {country} retrieval block {done}/{len(todo)} ({len(df):,} pairs) | elapsed {el/60:.1f} min, ETA {el/done*(len(todo)-done)/60:.0f} min")
    G.clear(); del vec_n, vec_a; gc.collect()


_W = {}      # per-process state of a scoring worker


def _score_init(paths, country, cfg, stats, blend, feats, cand_min, work, block, serial):
    """Runs once in every scoring process: loads the raw pool of the country, the model and the feature statistics."""
    import joblib as _jl
    t0 = time.time()
    _W["ids"], _W["names"], _W["addrs"] = read_raw_pool(paths, country)
    _W.update(country=country, cfg=cfg, stats=stats, blend=blend, feats=feats, cand_min=cand_min, work=work, block=block,
              model=_jl.load(os.path.join(HERE, "model_pair.joblib")))
    if not serial:                       # several processes share the machine: listwise net on CPU, one thread each
        
        if LM.HAVE_TORCH: LM.DEV = LM.torch.device("cpu"); LM.torch.set_num_threads(1)
    if os.environ.get("PREDICT_VERBOSE"):
        print(f"   scoring worker ready ({time.time()-t0:.0f}s)", flush=True)


def _score_block(task):
    """Features + model score for one block of queries; writes score2_<country>_<b>.pkl."""
    b, qb = task; w = _W; country = w["country"]
    c = pd.read_pickle(f"{w['work']}/cand_{country}_{b:06d}.pkl").sort_values(["q", "pool"], kind="stable").reset_index(drop=True)
    q_local = c["q"].values - b
    df = pd.DataFrame({"s1_id": qb["s1_id"].values[q_local], "s1_name": qb["s1_name"].values[q_local], "s1_addr": qb["s1_addr"].values[q_local],
                       "cand_id": w["ids"][c["pool"].values], "c_name": w["names"][c["pool"].values], "c_addr": w["addrs"][c["pool"].values], "country": country})
    for col in ("name_rank", "name_score", "addr_rank", "addr_score"):
        df[col] = c[col].astype(np.float32).values if col in c else np.nan
    F = PF.compute_features(df, w["stats"], w["cfg"], workers=1, log=lambda m: None)
    Xf = F[w["feats"]].values.astype(np.float32)
    p = w["model"].predict_proba(Xf)[:, 1]
    if w["blend"]["adopt"]:      # Phase 7d: average with the listwise-attention score (all candidates of a query are in this block)
        p = w["blend"]["weight_gbdt"] * p + w["blend"]["weight_listwise"] * LM.predict_listwise(Xf, df["s1_id"].values)
    out = pd.DataFrame({"s1_id": df["s1_id"].values, "cand_id": df["cand_id"].values, "score": p.astype(np.float32)})
    out = out[out["score"] >= w["cand_min"]]
    path = f"{w['work']}/score2_{country}_{b:06d}.pkl"; out.to_pickle(path + ".tmp"); os.replace(path + ".tmp", path)
    return b, len(df), len(out)


def stage_b(qc, paths, cfg, country, blocks, work, block, workers, feats, stats, cand_min, blend):
    """Scoring of every block that has candidates but no score file. workers > 1 = that many processes, each scoring whole blocks
    (each process holds its own copy of the raw pool of the country: about 1-2 GB each)."""
    todo = [b for b in blocks if not os.path.exists(f"{work}/score2_{country}_{b:06d}.pkl")]
    if not todo:
        return
    log(f"=== {country}: scoring {len(todo)}/{len(blocks)} blocks with {workers} process(es)")
    initargs = (paths, country, cfg, stats, blend, feats, cand_min, work, block, workers <= 1)
    tasks = [(b, qc.iloc[b:b + block][["s1_id", "s1_name", "s1_addr"]]) for b in todo]
    t_start = time.time()

    def report(done, r):
        el = time.time() - t_start; log(f"   {country} scoring block {done}/{len(todo)} (query offset {r[0]}: {r[1]:,} pairs, {r[2]:,} kept) | elapsed {el/60:.1f} min, ETA {el/done*(len(todo)-done)/60:.0f} min")
    if workers <= 1:
        _score_init(*initargs)
        for done, t in enumerate(tasks, 1):
            report(done, _score_block(t))
    else:
        import multiprocessing as mp
        method = os.environ.get("PAIR_START_METHOD") or ("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        with mp.get_context(method).Pool(workers, initializer=_score_init, initargs=initargs) as pool:
            for done, r in enumerate(pool.imap_unordered(_score_block, tasks), 1):
                report(done, r)
    _W.clear()


def preflight():
    """Stop immediately (instead of hours later) if the src/ folder mixes old and new files or the environment differs from training."""
    problems = []
    if not hasattr(RE, "search_chunks_sparse"): problems.append("retrieval_engine.py is an OLD version (no search_chunks_sparse; retrieval would be ~10x slower)")
    if not hasattr(PF, "make_pool") or not hasattr(PF, "_second_largest"): problems.append("pair_features.py is an OLD version (missing make_pool/_second_largest)")
    if "search_chunks_sparse" not in open(os.path.join(HERE, "candidate_generation.py"), encoding="utf-8").read(): problems.append("candidate_generation.py is an OLD version (still uses the slow dense search)")
    for f in ("model_pair.joblib", "feature_stats.pkl", "feature_list.json", "threshold_config.json", "cleanup_config.json", "blend_config.json", "listwise_model.pt", "listwise_prep.joblib", "listwise_model.py", "cleanup.py"):
        if not os.path.isfile(os.path.join(HERE, f)): problems.append(f"missing file in src/: {f}")
    if problems:
        raise SystemExit("PRE-FLIGHT CHECK FAILED - copy the complete src/ folder from the latest zip over yours:\n  - " + "\n  - ".join(problems))
    import sklearn
    want = json.load(open(os.path.join(HERE, "env_info.json"))).get("scikit-learn") if os.path.isfile(os.path.join(HERE, "env_info.json")) else None
    if want and sklearn.__version__ != want:
        log(f"WARNING: scikit-learn {sklearn.__version__} installed, the model files were saved with {want}. Run:  pip install scikit-learn=={want}   (results may differ or loading may fail otherwise)")


def write_lists(path, header2, s1_ids, lists):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{header2}\n")
        for s, l in zip(s1_ids, lists):
            f.write(f"{s}\t{','.join(l)}\n")


def run(test_dir, out_dir, work_dir=None, limit=None, block=20_000, workers=2, cand_min=0.01, countries=None, score_workers=None):
    preflight(); os.makedirs(out_dir, exist_ok=True)
    s1p, s2p, s3p = [os.path.join(test_dir, f"test_source{i}.tsv") for i in (1, 2, 3)]
    for p in (s1p, s2p, s3p):
        if not os.path.isfile(p):
            raise SystemExit(f"missing input file: {p}")
    q = read_s1(s1p, limit)
    sig = hashlib.md5(f"{len(q)}|{os.path.getsize(s1p)}|{os.path.getsize(s2p)}|{os.path.getsize(s3p)}|{block}|{limit}".encode()).hexdigest()[:8]
    work = work_dir or os.path.join(HERE, "..", "..", "..", "dataset", "interim", "predict_work")
    work = os.path.join(work, sig); os.makedirs(work, exist_ok=True)
    log(f"{len(q):,} S1 queries | countries {q['country'].value_counts().to_dict()} | work dir {work}")
    cfg = CG.load_cfg(); thr = json.load(open(os.path.join(HERE, "threshold_config.json")))["decision"]; assert thr["rule"] == "global", thr
    ccfg = json.load(open(os.path.join(HERE, "cleanup_config.json")))
    blend = json.load(open(os.path.join(HERE, "blend_config.json"))); thr = {"rule": "global", "threshold": blend["threshold"]} if blend["adopt"] else thr
    log(f"decision rule: {'gradient boosting + listwise attention blend' if blend['adopt'] else 'gradient boosting only'}, threshold {thr['threshold']}")
    feats = json.load(open(os.path.join(HERE, "feature_list.json")))["final_features"]
    stats = pickle.load(open(os.path.join(HERE, "feature_stats.pkl"), "rb"))

    for country in sorted(q["country"].unique()):
        if countries and country not in countries:
            continue
        qc = q[q["country"] == country].reset_index(drop=True)
        blocks = list(range(0, len(qc), block))
        stage_a(qc, [s2p, s3p], cfg, country, blocks, work, block, workers)
        stage_b(qc, [s2p, s3p], cfg, country, blocks, work, block, score_workers or workers, feats, stats, cand_min, blend)

    log("assembling final files")
    parts = [pd.read_pickle(os.path.join(work, f)) for f in sorted(os.listdir(work)) if f.startswith("score2_") and f.endswith(".pkl")]
    S = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"s1_id": [], "cand_id": [], "score": []})
    keep = apply_cleanup(S, thr["threshold"], ccfg)
    M = S[keep]
    log(f"{len(S):,} candidate pairs kept (score >= {cand_min}); {int((S['score'] >= thr['threshold']).sum()):,} above threshold -> {len(M):,} matches after cleanup")
    cl = S.groupby("s1_id")["cand_id"].apply(lambda x: sorted(x)); ml = M.groupby("s1_id")["cand_id"].apply(lambda x: sorted(x))
    s1_all = q["s1_id"].tolist()
    write_lists(os.path.join(out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", s1_all, [cl.get(s, []) for s in s1_all])
    write_lists(os.path.join(out_dir, "matching_results.tsv"), "matched_entity_ids", s1_all, [ml.get(s, []) for s in s1_all])
    log(f"wrote {len(M):,} matches for {ml.shape[0]:,}/{len(s1_all):,} entities -> {out_dir}")
    # per-country sanity summary (labels are unknown at test time: compare countries with each other and with validation: ~6 % of entities without a match, ~2-4 matches otherwise)
    cmap = q.set_index("s1_id")["country"]; nm = pd.Series([len(ml.get(s, [])) for s in s1_all], index=s1_all); nc = pd.Series([len(cl.get(s, [])) for s in s1_all], index=s1_all)
    summ = {}
    for c in sorted(q["country"].unique()):
        ix = cmap.index[cmap.values == c]; m_, c_ = nm.reindex(ix), nc.reindex(ix)
        summ[c] = {"entities": int(len(ix)), "no_match_share": round(float((m_ == 0).mean()), 4), "mean_matches_when_any": round(float(m_[m_ > 0].mean()), 3) if (m_ > 0).any() else None,
                   "mean_candidates": round(float(c_.mean()), 2), "share_with_candidates_but_no_match": round(float(((m_ == 0) & (c_ > 0)).mean()), 4)}
    log("per-country summary: " + json.dumps(summ)); json.dump(summ, open(os.path.join(work, "run_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir"); ap.add_argument("--limit", type=int, help="only the first N S1 rows (smoke test)")
    ap.add_argument("--block", type=int, default=20_000); ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--score-workers", type=int, default=2, help="processes for the feature+scoring stage (each needs ~1-2 GB RAM)"); ap.add_argument("--cand-min-score", type=float, default=0.01); ap.add_argument("--countries", nargs="*")
    a = ap.parse_args()
    run(a.test_dir, a.out_dir, a.work_dir, a.limit, a.block, a.workers, a.cand_min_score, a.countries, a.score_workers)
