"""Phase 12 - Phase 9 inference + consistency cleanup (cleanup.py / cleanup_config.json): record exclusivity.

    python predict_phase12.py --test-dir ../../../dataset/test --out-dir ../../../output

Steps: read S1 queries -> candidate generation (same-country name TF-IDF top-K1 U address TF-IDF top-K2) -> raw text of candidates ->
69 pair features -> gradient-boosted classifier -> global probability threshold (threshold_config.json) and the Phase 12 consistency rules -> two TSV files.
Every S1 entity gets a row in both files; an empty list means "no match".
In-memory version: fine for validation-size runs; for the full test files use predict_phase13.py.
"""
import argparse
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
import pair_features as PF
from cleanup import apply_cleanup

log = CG.log


def write_lists(path, header2, s1_ids, lists):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{header2}\n")
        for s, l in zip(s1_ids, lists):
            f.write(f"{s}\t{','.join(l)}\n")


def run(test_dir, out_dir, work_dir=None, limit=None, q_block=100_000, workers=2):
    os.makedirs(out_dir, exist_ok=True)
    work_dir = work_dir or os.path.join(HERE, "..", "..", "..", "dataset", "interim", "predict_work_p12"); os.makedirs(work_dir, exist_ok=True)
    s1p, s2p, s3p = [os.path.join(test_dir, f"test_source{i}.tsv") for i in (1, 2, 3)]
    q = pd.read_csv(s1p, sep="\t", dtype=str, keep_default_na=False).rename(
        columns={"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"})
    if limit: q = q.head(limit)
    log(f"{len(q):,} S1 queries")
    cfg = CG.load_cfg(); thr = json.load(open(os.path.join(HERE, "threshold_config.json")))["decision"]
    assert thr["rule"] == "global", thr
    ccfg = json.load(open(os.path.join(HERE, "cleanup_config.json")))
    model = joblib.load(os.path.join(HERE, "model_pair.joblib")); feats = json.load(open(os.path.join(HERE, "feature_list.json")))["final_features"]
    stats = pickle.load(open(os.path.join(HERE, "feature_stats.pkl"), "rb"))

    cand_path = os.path.join(work_dir, "candidates_raw.pkl")
    cand = pickle.load(open(cand_path, "rb")) if os.path.exists(cand_path) else CG.generate(q, [s2p, s3p], cfg, cand_path, workers=workers)
    log(f"{len(cand):,} candidate pairs")
    raw = CG.fetch_raw(cand["cand_id"].unique(), [s2p, s3p])[["cand_id", "c_name", "c_addr"]].drop_duplicates("cand_id")
    cand = cand.merge(raw, on="cand_id", how="left").merge(q[["s1_id", "s1_name", "s1_addr"]], on="s1_id", how="left")
    cand[["c_name", "c_addr"]] = cand[["c_name", "c_addr"]].fillna("")
    cand = cand.sort_values(["s1_id", "cand_id"], kind="stable").reset_index(drop=True)

    ids = cand["s1_id"].unique(); blocks = np.array_split(ids, max(1, int(np.ceil(len(ids) / q_block))))
    scores = np.zeros(len(cand), np.float32); pos = cand.groupby("s1_id").indices
    for bi, b in enumerate(blocks):
        ix = np.concatenate([pos[i] for i in b]); sub = cand.iloc[ix]
        log(f"features block {bi+1}/{len(blocks)} ({len(sub):,} pairs)")
        F = PF.compute_features(sub.reset_index(drop=True), stats, cfg, workers=workers, log=log)
        scores[ix] = model.predict_proba(F[feats].values.astype(np.float32))[:, 1]
    cand["score"] = scores
    cand[["s1_id", "cand_id", "score"]].to_pickle(os.path.join(work_dir, "scores.pkl"))

    keep = cand[apply_cleanup(cand, thr["threshold"], ccfg)]
    log(f"cleanup: {int((cand['score'] >= thr['threshold']).sum()):,} thresholded -> {len(keep):,} after consistency rules")
    cl = cand.groupby("s1_id")["cand_id"].apply(list); ml = keep.groupby("s1_id")["cand_id"].apply(list)
    s1_all = q["s1_id"].tolist()
    write_lists(os.path.join(out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", s1_all, [sorted(cl.get(s, [])) for s in s1_all])
    write_lists(os.path.join(out_dir, "matching_results.tsv"), "matched_entity_ids", s1_all, [sorted(ml.get(s, [])) for s in s1_all])
    log(f"wrote {len(keep):,} matches for {ml.shape[0]:,}/{len(s1_all):,} entities -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir"); ap.add_argument("--limit", type=int); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(); run(a.test_dir, a.out_dir, a.work_dir, a.limit, workers=a.workers)
