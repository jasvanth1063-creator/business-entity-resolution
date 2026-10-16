"""Listwise attention scorer (Phase 7b/7d): scores every candidate of a query *together* with its competitors.

Usage at inference:  p = predict_listwise(F, q)  where F is the (n_pairs x 69) feature matrix in feature_list.json order and q the S1 id of each row.
Files: listwise_prep.joblib (quantile transform + missing indicators), listwise_model.pt (state dicts of the seed ensemble).
"""
import os

import joblib
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception as _e:          # e.g. DLL blocked by Windows Application Control -> numpy fallback
    torch = None; nn = None; HAVE_TORCH = False
    print("[listwise_model] torch unavailable (%s) -> using numpy inference" % str(_e)[:80])

HERE = os.path.dirname(os.path.abspath(__file__))
LMAX = 40                      # max candidates per query (K_name 20 + K_addr 20)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAVE_TORCH else None


class SetNet(nn.Module if HAVE_TORCH else object):
    def __init__(self, d, dm=128, heads=4, layers=2, drop=0.1):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(d, dm), nn.ReLU()); self.skip = nn.Linear(d, 1)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, heads, 2 * dm, drop, batch_first=True, activation="relu"), layers)
        self.out = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, 1))

    def forward(self, x, mask):
        h = self.enc(self.emb(x), src_key_padding_mask=~mask)
        return (self.out(h) + self.skip(x)).squeeze(-1)


class Prep:
    """Quantile-normal transform + missing-value indicators, fitted on training rows only."""
    def __init__(self, qt, nan_cols): self.qt, self.nan_cols = qt, nan_cols

    def __call__(self, X):
        out = []
        for i in range(0, len(X), 200_000):
            A = X[i:i + 200_000]; Z = np.nan_to_num(self.qt.transform(A), nan=0.0)
            out.append(np.concatenate([Z, np.isnan(A[:, self.nan_cols]).astype(np.float32)], 1).astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def group_rows(q):
    return list(pd.Series(np.arange(len(q))).groupby(np.asarray(q)).indices.values())


def pad(Z, rows, y=None, L=LMAX):
    n, d = len(rows), Z.shape[1]
    Xq = np.zeros((n, L, d), np.float32); Mq = np.zeros((n, L), bool); Yq = np.zeros((n, L), np.float32)
    for i, r in enumerate(rows):
        assert len(r) <= L, f"query with {len(r)} candidates (> {L})"
        Xq[i, :len(r)] = Z[r]; Mq[i, :len(r)] = True
        if y is not None: Yq[i, :len(r)] = y[r]
    return torch.from_numpy(Xq), torch.from_numpy(Yq), torch.from_numpy(Mq)


def scores_from_model(model, Xq, Mq, rows, n, bs=256):
    model.eval(); out = np.zeros(n, np.float32)
    with torch.no_grad():
        for i in range(0, len(Xq), bs):
            p = torch.sigmoid(model(Xq[i:i + bs].to(DEV), Mq[i:i + bs].to(DEV))).cpu().numpy()
            for j, r in enumerate(rows[i:i + bs]): out[r] = p[j, :len(r)]
    return out


_CACHE = {}
def load():
    if "m" not in _CACHE:
        prep = joblib.load(os.path.join(HERE, "listwise_prep.joblib"))
        states = torch.load(os.path.join(HERE, "listwise_model.pt"), map_location="cpu")
        models = []
        for s in states:
            m = SetNet(s["d"]); m.load_state_dict(s["state"]); models.append(m.to(DEV).eval())
        _CACHE["m"] = (prep, models)
    return _CACHE["m"]


def predict_listwise(F, q):
    """Average probability of the seed ensemble for every row of F (rows may be in any order; queries need not be contiguous)."""
    if not HAVE_TORCH:
        import listwise_numpy as LN
        prep = joblib.load(os.path.join(HERE, "listwise_prep.joblib")); Z = prep(np.asarray(F, np.float32)); return LN.predict(Z, group_rows(q), LMAX)
    prep, models = load(); Z = prep(np.asarray(F, np.float32)); rows = group_rows(q); Xq, _, Mq = pad(Z, rows)
    return np.mean([scores_from_model(m, Xq, Mq, rows, len(Z)) for m in models], 0)
