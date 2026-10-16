"""Competition-style evaluation: macro F_0.5 over S1 entities (singletons score 1.0 for an empty prediction)."""
import numpy as np
import pandas as pd


def macro_f05(entities, ntrue, q, label, pred):
    """entities: iterable of all query ids; ntrue: Series (index s1_id) of TRUE match counts incl. matches retrieval missed;
    q/label/pred: per-candidate arrays (query id, 0/1 truth, 0/1 predicted). Returns (macro score, per-entity Series)."""
    d = pd.DataFrame({"q": q, "tp": np.asarray(label) * np.asarray(pred), "p": np.asarray(pred)})
    g = d.groupby("q").agg(tp=("tp", "sum"), npred=("p", "sum"))
    e = pd.DataFrame(index=pd.Index(list(entities), name="s1_id"))
    e = e.join(g).fillna(0)
    e["ntrue"] = ntrue.reindex(e.index).fillna(0).values
    P = np.where(e["npred"] > 0, e["tp"] / e["npred"].replace(0, np.nan), 0.0)
    R = np.where(e["ntrue"] > 0, e["tp"] / e["ntrue"].replace(0, np.nan), 0.0)
    f = np.where((P + R) > 0, 1.25 * P * R / (0.25 * P + R + 1e-12), 0.0)
    f = np.where(e["ntrue"] == 0, (e["npred"] == 0).astype(float), f)
    e["f05"] = f
    return float(e["f05"].mean()), e["f05"]


def best_threshold(entities, ntrue, q, label, score, grid=None):
    grid = np.arange(0.05, 0.96, 0.05) if grid is None else grid
    res = [(t, macro_f05(entities, ntrue, q, label, (score >= t).astype(int))[0]) for t in grid]
    t, v = max(res, key=lambda x: x[1])
    return t, v, res
