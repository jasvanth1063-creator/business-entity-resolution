"""Phase 12 - consistency cleanup applied to thresholded predictions (used by predict.py and the Phase 12 notebook).

df: DataFrame with columns s1_id, cand_id, score (one row per candidate pair). Returns a boolean numpy array 'keep' aligned with df.
Rules (each switched on/off in cleanup_config.json):
  1. exclusivity - an S2/S3 record belongs to at most one S1 entity in the training truth, so if several S1 entities claim a record,
     only the highest-scoring claimant keeps it (optionally only if it leads the next claimant by `margin`).
  2. lone-entity thresholds - entities with no prediction at the global threshold may still get their best candidate if it scores >= t_rescue;
     entities with exactly one prediction keep it only if it scores >= t_lone.
"""
import numpy as np
import pandas as pd


def apply_cleanup(df, thr, cfg):
    s = df["score"].values; q = df["s1_id"].values
    keep = s >= thr
    r = cfg.get("lone", {})
    if r.get("enabled"):
        d = pd.DataFrame({"q": q, "s": s, "k": keep})
        n = d.groupby("q")["k"].transform("sum").values
        top = d.groupby("q")["s"].transform("max").values
        if r.get("t_rescue") is not None:
            keep = keep | ((n == 0) & (s >= top) & (s >= r["t_rescue"]))
        if r.get("t_lone") is not None:
            keep = keep & ~((n == 1) & (s < r["t_lone"]))
    e = cfg.get("exclusivity", {})
    if e.get("enabled"):
        d = pd.DataFrame({"c": df["cand_id"].values, "s": s, "k": keep, "i": np.arange(len(df))})
        k = d[d.k].sort_values(["c", "s"], ascending=[True, False])
        first = k.groupby("c")["s"].transform("first"); second = k.groupby("c")["s"].transform(lambda x: x.iloc[1] if len(x) > 1 else -1.0)
        multi = k.groupby("c")["s"].transform("size") > 1
        lose = multi & ~((k["s"] == first) & ((first - second) >= e.get("margin", 0.0)))
        # losers drop the record; if the leader fails the margin test nobody keeps it
        keep = keep.copy(); keep[k.loc[lose, "i"].values] = False
    return keep
