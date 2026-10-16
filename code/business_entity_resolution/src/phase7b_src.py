# %% [markdown]
# # Phase 7b — Deep-Learning Experiments vs the Gradient-Boosting Classifier
#
# **Question.** Phase 7 picked `HistGradientBoostingClassifier` (best hold-out F0.5). Can a **deep neural network trained by back-propagation** do better? Three families are tried, all PyTorch (runs on the **GPU automatically when CUDA is available**, otherwise on the CPU):
# 1. **MLP** — ReLU hidden layers + BatchNorm + dropout, 2-way **softmax** output, cross-entropy loss, AdamW (back-propagation), on the same 69 features (quantile-normalised, missing-value indicators).
# 2. **Listwise attention model** — all candidates of one query are processed *together*: a self-attention (Transformer encoder) layer lets every candidate "see" its competitors, then a per-candidate output. This is the neural version of the cluster-consistency idea that failed as hand-made rules.
# 3. **Hybrid character-CNN** — a 1-D convolutional network reads the raw characters of the names and addresses (both sides, shared weights), compares the two embeddings (|u−v|, u·v) and joins them with the 69 features. This can learn spelling/transliteration patterns the handcrafted features miss.
#
# **Fair comparison.** Same 80/20 hold-out of **train queries** as Phases 5–7 (the real validation set is not touched). Every network trains on the 80 % fit queries only, with a 10 % inner split for early stopping. The baseline is the Phase 7 gradient-boosting configuration retrained on the same fit queries.
# **Adoption rule (fixed beforehand):** a network (or a blend) replaces gradient boosting only if ΔF0.5 ≥ **+0.003** and the paired-bootstrap 95 % interval of ΔF0.5 is entirely above 0 — a new model costs test-time complexity, so it must clearly win. PR-AUC is also reported.
#
# **Sections:** 1 Load + split → 2 Baseline → 3 MLP → 4 Listwise attention → 5 Hybrid char-CNN → 6 Blends → 7 Decision

# %%
import os, sys, json, time, pickle, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as Fn
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import QuantileTransformer
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
import eval_utils as EU
torch.set_num_threads(os.cpu_count() or 2); torch.manual_seed(42); np.random.seed(42)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # uses the GPU automatically when CUDA is available
GPU = DEV.type == "cuda"
ROOT = "../../../"; INTERIM = ROOT + "dataset/interim"
FEATS = json.load(open("feature_list.json"))["final_features"]; MC = json.load(open("model_config.json"))
pd.set_option("display.width", 170)
RES = {}
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
log(f"torch {torch.__version__} | device {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if GPU else "") + f" | {len(FEATS)} features")

# %% [markdown]
# ## 1. Load and split (identical hold-out to Phases 5–7)

# %%
ctr = pd.read_pickle(INTERIM + "/candidates_train.pkl")
Ftr = pd.read_pickle(INTERIM + "/features_train.pkl")[FEATS]
P4 = pickle.load(open(INTERIM + "/phase4_truth_and_queries.pkl", "rb"))
ntrue = P4["truth"]["train"].groupby("s1_id").size()
ents = P4["queries"][P4["queries"]["split"] == "train"]["s1_id"].tolist()
SMOKE = bool(os.environ.get("SMOKE"))      # quick plumbing test on 4,000 queries
if SMOKE:
    ents = ents[:4000]; keep = ctr["s1_id"].isin(set(ents)).values; ctr = ctr[keep].reset_index(drop=True); Ftr = Ftr[keep].reset_index(drop=True)
y = ctr["label"].values.astype(np.int64); qid = ctr["s1_id"].values
rng = np.random.default_rng(42); hold = set(rng.choice(np.array(ents), int(0.2 * len(ents)), replace=False))
is_hold = ctr["s1_id"].isin(hold).values; ent_hold = [e for e in ents if e in hold]
fit_ents = np.array([e for e in ents if e not in hold]); r2 = np.random.default_rng(7); inner = set(r2.choice(fit_ents, int(0.1 * len(fit_ents)), replace=False))
is_inner = ctr["s1_id"].isin(inner).values & ~is_hold; is_train = ~is_hold & ~is_inner
X = Ftr.values.astype(np.float32); yH, qH = y[is_hold], qid[is_hold]
print(f"train {is_train.sum():,} pairs | inner-val {is_inner.sum():,} | hold-out {is_hold.sum():,} ({len(ent_hold):,} queries) | positives {y.mean():.2%}")

# preprocessing for the networks: missing indicators + quantile-normal transform fitted on training rows only
nan_cols = np.where(np.isnan(X[is_train]).any(0))[0]
qt = QuantileTransformer(n_quantiles=256, output_distribution="normal", subsample=200_000, random_state=0).fit(X[is_train])
def prep(A):
    Z = qt.transform(A); ind = np.isnan(A[:, nan_cols]).astype(np.float32); Z = np.nan_to_num(Z, nan=0.0)
    return np.concatenate([Z, ind], 1).astype(np.float32)
Z = np.concatenate([prep(X[i:i + 200_000]) for i in range(0, len(X), 200_000)]); D = Z.shape[1]
print(f"network input width {D} ({len(nan_cols)} missing-value indicators)")

def evaluate(p, name):
    thr, f, _ = EU.best_threshold(ent_hold, ntrue, qH, yH, p, grid=np.arange(0.3, 0.96, 0.025))
    out = {"pr_auc": float(average_precision_score(yH, p)), "f05": float(f), "thr": float(thr)}; RES[name] = out
    print(f"{name:28s} PR-AUC {out['pr_auc']:.4f} | best-threshold macro F0.5 {out['f05']:.4f} @ {thr:.3f}"); return out
SCORES = {}
def ent_f(p, name): return EU.macro_f05(ent_hold, ntrue, qH, yH, (p >= RES[name]["thr"]).astype(int))[1].values
def boot(a, b, n=1000):
    r = np.random.default_rng(0); d = a - b; m = [d[r.integers(0, len(d), len(d))].mean() for _ in range(n)]; return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

# %% [markdown]
# ## 2. Baseline — gradient boosting retrained on the same fit queries

# %%
t0 = time.time(); fitm = ~is_hold
hgb = HistGradientBoostingClassifier(random_state=42, **MC["gbdt"]["params"]).fit(X[fitm], y[fitm])
SCORES["gbdt"] = hgb.predict_proba(X[is_hold])[:, 1]; evaluate(SCORES["gbdt"], "gbdt (baseline)"); RES["gbdt (baseline)"]["train_s"] = time.time() - t0
SCORES["gbdt (baseline)"] = SCORES["gbdt"]

# %% [markdown]
# ## 3. MLP — ReLU + BatchNorm + dropout, softmax output, back-propagation (AdamW, one-cycle LR, early stopping on inner PR-AUC)

# %%
class MLP(nn.Module):
    def __init__(self, d, hidden=(512, 256, 128), drop=0.1):
        super().__init__(); L = []; k = d
        for h in hidden: L += [nn.Linear(k, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]; k = h
        self.body = nn.Sequential(*L); self.out = nn.Linear(k, 2)          # 2 logits -> softmax
    def forward(self, x): return self.out(self.body(x))
def predict_proba(model, A, bs=8192):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(A), bs): out.append(torch.softmax(model(torch.from_numpy(A[i:i + bs]).to(DEV)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)
def train_mlp(hidden, epochs=(1 if SMOKE else 14), bs=(8192 if GPU else 2048), lr=3e-3, drop=0.1, tag="mlp"):
    m = MLP(D, hidden, drop).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(Z[is_train]).to(DEV), torch.from_numpy(y[is_train]).to(DEV); n = len(Xt); steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps); best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=DEV); t0 = time.time(); tot = 0.0
        for i in range(0, n, bs):
            ix = perm[i:i + bs]; loss = Fn.cross_entropy(m(Xt[ix]), yt[ix]); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(ix)
        pr = average_precision_score(y[is_inner], predict_proba(m, Z[is_inner]))
        log(f"  {tag} epoch {ep+1}/{epochs} loss {tot/n:.4f} | inner PR-AUC {pr:.4f} | {time.time()-t0:.0f}s")
        if pr > best: best, bad = pr, 0; best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 3: break
    m.load_state_dict(best_state); return m
for name, hid in (("mlp 512-256-128", (512, 256, 128)), ("mlp 1024-512-256-128", (1024, 512, 256, 128))):
    t0 = time.time(); m = train_mlp(hid, tag=name); SCORES[name] = predict_proba(m, Z[is_hold]); evaluate(SCORES[name], name); RES[name]["train_s"] = time.time() - t0

# %% [markdown]
# ## 4. Listwise attention model — candidates of a query attend to each other
# Each query is a sequence of its candidates (≤ 40, padded). Linear embedding → 2 Transformer-encoder layers (multi-head self-attention + ReLU feed-forward) → one logit per candidate; binary cross-entropy over valid candidates.

# %%
def to_lists(mask):
    idx = np.flatnonzero(mask); g = pd.Series(idx).groupby(qid[idx]).indices; keys = list(g.keys()); L = max(len(v) for v in g.values())
    rows = [idx[g[k]] for k in keys]; return rows, L
def pad(rows, L):
    n = len(rows); Xq = np.zeros((n, L, D), np.float32); Yq = np.zeros((n, L), np.float32); Mq = np.zeros((n, L), bool)
    for i, r in enumerate(rows): Xq[i, :len(r)] = Z[r]; Yq[i, :len(r)] = y[r]; Mq[i, :len(r)] = True
    return torch.from_numpy(Xq), torch.from_numpy(Yq), torch.from_numpy(Mq)
rows_tr, L_tr = to_lists(is_train); rows_in, _ = to_lists(is_inner); rows_ho, L_ho = to_lists(is_hold); LMAX = max(L_tr, L_ho)
T_tr, T_in, T_ho = [tuple(t.to(DEV) for t in pad(r, LMAX)) for r in (rows_tr, rows_in, rows_ho)]
print(f"queries: train {len(rows_tr):,} | inner {len(rows_in):,} | hold-out {len(rows_ho):,} | max candidates {LMAX}")
class SetNet(nn.Module):
    def __init__(self, d, dm=128, heads=4, layers=2, drop=0.1):
        super().__init__(); self.emb = nn.Sequential(nn.Linear(d, dm), nn.ReLU()); self.skip = nn.Linear(d, 1)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, heads, 2 * dm, drop, batch_first=True, activation="relu"), layers)
        self.out = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, 1))
    def forward(self, x, mask):
        h = self.enc(self.emb(x), src_key_padding_mask=~mask); return (self.out(h) + self.skip(x)).squeeze(-1)    # residual path from the raw features
def set_scores(m, T, rows, bs=256):
    m.eval(); Xq, _, Mq = T; out = np.zeros(len(y), np.float32)
    with torch.no_grad():
        for i in range(0, len(Xq), bs):
            lg = torch.sigmoid(m(Xq[i:i + bs], Mq[i:i + bs])).cpu().numpy()
            for j, r in enumerate(rows[i:i + bs]): out[r] = lg[j, :len(r)]
    return out
def train_set(epochs=(1 if SMOKE else 25), bs=64, lr=1e-3, tag="listwise"):
    m = SetNet(D).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4); Xq, Yq, Mq = T_tr; n = len(Xq)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((n + bs - 1) // bs)); best, bad, state = -1, 0, None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=DEV); t0 = time.time(); tot = 0.0
        for i in range(0, n, bs):
            ix = perm[i:i + bs]; lg = m(Xq[ix], Mq[ix]); loss = Fn.binary_cross_entropy_with_logits(lg[Mq[ix]], Yq[ix][Mq[ix]]); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        pr = average_precision_score(y[is_inner], set_scores(m, T_in, rows_in)[is_inner])
        if ep % 3 == 0 or ep == epochs - 1: log(f"  {tag} epoch {ep+1}/{epochs} loss {tot/((n+bs-1)//bs):.4f} | inner PR-AUC {pr:.4f} | {time.time()-t0:.0f}s")
        if pr > best: best, bad, state = pr, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 5: break
    m.load_state_dict(state); return m
t0 = time.time(); sm = train_set(); SCORES["listwise attention"] = set_scores(sm, T_ho, rows_ho)[is_hold]; evaluate(SCORES["listwise attention"], "listwise attention"); RES["listwise attention"]["train_s"] = time.time() - t0

# %% [markdown]
# ## 5. Hybrid character-CNN — raw characters of names and addresses + the 69 features
# Characters → embedding → two 1-D convolutions (ReLU) → global max-pool, shared between the S1 side and the candidate side. Pair vector = [|u−v|, u·v] for names and for addresses, concatenated with a feature branch, then a 2-way softmax head.

# %%
def build_vocab(strings, size=300):
    from collections import Counter
    c = Counter(); [c.update(s) for s in strings]; return {ch: i + 2 for i, (ch, _) in enumerate(c.most_common(size))}     # 0 pad, 1 unknown
def encode_unique(series, vocab, L):
    u, inv = np.unique(series.values.astype(str), return_inverse=True); T = np.zeros((len(u), L), np.int16)
    for i, s in enumerate(u):
        ids = [vocab.get(ch, 1) for ch in s.lower()[:L]]; T[i, :len(ids)] = ids
    return torch.from_numpy(T.astype(np.int64)).to(DEV), torch.from_numpy(inv.astype(np.int64)).to(DEV)
sample = ctr.loc[is_train, ["s1_name", "c_name", "s1_addr", "c_addr"]].sample(100_000, random_state=0)
vocab = build_vocab(pd.concat([sample[c] for c in sample.columns]).str.lower().tolist()); NV = len(vocab) + 2; LN, LA = 48, 96
sn_T, sn_i = encode_unique(ctr["s1_name"], vocab, LN); cn_T, cn_i = encode_unique(ctr["c_name"], vocab, LN)
sa_T, sa_i = encode_unique(ctr["s1_addr"], vocab, LA); ca_T, ca_i = encode_unique(ctr["c_addr"], vocab, LA)
print(f"vocab {NV} chars | unique strings: s1 names {len(sn_T):,}, cand names {len(cn_T):,}, s1 addr {len(sa_T):,}, cand addr {len(ca_T):,}")
class CharEnc(nn.Module):
    def __init__(self, nv, e=32, c=64):
        super().__init__(); self.emb = nn.Embedding(nv, e, padding_idx=0); self.c1 = nn.Conv1d(e, c, 3, padding=1); self.c2 = nn.Conv1d(c, c, 3, padding=1)
    def forward(self, ids):
        h = self.emb(ids).transpose(1, 2); h = Fn.relu(self.c1(h)); h = Fn.relu(self.c2(h)); return h.max(2).values
class Hybrid(nn.Module):
    def __init__(self, d, nv, c=64):
        super().__init__(); self.name_enc, self.addr_enc = CharEnc(nv, c=c), CharEnc(nv, c=c)
        self.feat = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1))
        self.head = nn.Sequential(nn.Linear(256 + 4 * c, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))
    def forward(self, z, sn, cn, sa, ca):
        a, b = self.name_enc(sn), self.name_enc(cn); u, v = self.addr_enc(sa), self.addr_enc(ca)
        return self.head(torch.cat([self.feat(z), (a - b).abs(), a * b, (u - v).abs(), u * v], 1))
Zt = torch.from_numpy(Z).to(DEV)
def batch(ix):
    ix = torch.as_tensor(ix, device=DEV); return (Zt[ix], sn_T[sn_i[ix]], cn_T[cn_i[ix]], sa_T[sa_i[ix]], ca_T[ca_i[ix]])
def hybrid_proba(m, mask, bs=4096):
    m.eval(); idx = np.flatnonzero(mask); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs): out.append(torch.softmax(m(*batch(idx[i:i + bs])), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)
def train_hybrid(epochs=(1 if SMOKE else 3), bs=(2048 if GPU else 1024), lr=2e-3, max_train=None):
    m = Hybrid(D, NV).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4); idx = np.flatnonzero(is_train)
    if max_train: idx = np.random.default_rng(0).choice(idx, max_train, replace=False)
    yt = torch.from_numpy(y).to(DEV); steps = epochs * ((len(idx) + bs - 1) // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps); best, state = -1, None
    for ep in range(epochs):
        m.train(); perm = np.random.default_rng(ep).permutation(idx); t0 = time.time(); tot = 0.0
        for k, i in enumerate(range(0, len(perm), bs)):
            ix = perm[i:i + bs]; loss = Fn.cross_entropy(m(*batch(ix)), yt[torch.as_tensor(ix, device=DEV)]); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(ix)
            if k % 200 == 0: log(f"    hybrid ep {ep+1} step {k}/{len(perm)//bs} loss {loss.item():.4f}")
        pr = average_precision_score(y[is_inner], hybrid_proba(m, is_inner)); log(f"  hybrid epoch {ep+1}/{epochs} loss {tot/len(perm):.4f} | inner PR-AUC {pr:.4f} | {time.time()-t0:.0f}s")
        if pr > best: best, state = pr, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); return m
t0 = time.time(); hm = train_hybrid(); SCORES["hybrid char-CNN"] = hybrid_proba(hm, is_hold); evaluate(SCORES["hybrid char-CNN"], "hybrid char-CNN"); RES["hybrid char-CNN"]["train_s"] = time.time() - t0
# inference cost per 100k pairs (matters for 60M+ test pairs)
t0 = time.time(); _ = hybrid_proba(hm, np.isin(np.arange(len(y)), np.flatnonzero(is_hold)[:100_000])); RES["hybrid char-CNN"]["infer_s_per_100k"] = time.time() - t0
t0 = time.time(); _ = hgb.predict_proba(X[is_hold][:100_000]); RES["gbdt (baseline)"]["infer_s_per_100k"] = time.time() - t0
print({k: round(v.get("infer_s_per_100k", np.nan), 1) for k, v in RES.items() if "infer_s_per_100k" in v})

# %% [markdown]
# ## 6. Blends with gradient boosting (simple average of probabilities)

# %%
names_nn = [k for k in SCORES if k not in ("gbdt", "gbdt (baseline)")]
for k in names_nn: SCORES[f"gbdt + {k}"] = 0.5 * SCORES["gbdt"] + 0.5 * SCORES[k]; evaluate(SCORES[f"gbdt + {k}"], f"gbdt + {k}")
best_nn = max(names_nn, key=lambda k: RES[k]["f05"]); SCORES["gbdt + all nets"] = 0.4 * SCORES["gbdt"] + 0.6 * np.mean([SCORES[k] for k in names_nn], 0); evaluate(SCORES["gbdt + all nets"], "gbdt + all nets")

# %% [markdown]
# ## 7. Decision — paired bootstrap against the baseline

# %%
base = ent_f(SCORES["gbdt"], "gbdt (baseline)"); rows = []
for k in [k for k in SCORES if k not in ("gbdt", "gbdt (baseline)")]:
    d, lo, hi = boot(ent_f(SCORES[k], k), base); rows.append((k, RES[k]["pr_auc"], RES[k]["f05"], d, lo, hi, d >= 0.003 and lo > 0))
T = pd.DataFrame(rows, columns=["model", "PR-AUC", "F0.5", "dF0.5 vs gbdt", "CI low", "CI high", "adopt"]).set_index("model")
print(f"baseline gbdt: PR-AUC {RES['gbdt (baseline)']['pr_auc']:.4f}, F0.5 {RES['gbdt (baseline)']['f05']:.4f}"); print(T.round(4).to_string())
winner = T[T.adopt]["dF0.5 vs gbdt"].idxmax() if T.adopt.any() else None
print("\nWINNER over gradient boosting:", winner or "none - gradient boosting stays the final model")
if SMOKE: sys.exit(0)
json.dump({"results": RES, "table": T.reset_index().to_dict("records"), "winner": winner}, open("phase7b_results.json", "w"), indent=2, default=float)
pickle.dump({k: v for k, v in SCORES.items()}, open(INTERIM + "/phase7b_hold_scores.pkl", "wb"))
