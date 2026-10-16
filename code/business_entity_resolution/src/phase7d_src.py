# %% [markdown]
# # Phase 7d — Confirming the Listwise-Attention Blend on the Real Validation Set
#
# **Why.** Phase 7b found, on the internal hold-out of train queries, that averaging the gradient-boosting probability with a **listwise attention network** (every candidate of a query attends to its competitors; ReLU layers, back-propagation, sigmoid output) beats gradient boosting alone:
# ΔF0.5 = +0.0049, bootstrap 95 % interval [+0.0026, +0.0071], PR-AUC 0.9980 vs 0.9972. That met the adoption rule, but hold-out numbers are what the model was *selected* on, so the final word comes from the validation set, which neither model has seen.
#
# **Protocol (fixed beforehand).**
# 1. Train the attention network on **all** train queries (**2 seeds**, averaged), 8 % of queries held back for early stopping.
# 2. Score the 6,000 validation queries; blend = 0.5 · gradient-boosting score (Phase 7 final model) + 0.5 · attention score.
# 3. Threshold: global, chosen by **5-fold cross-fitting** over validation entities exactly as in Phase 8; compare with the gradient-boosting-only cross-fit estimate **0.9505**.
# 4. **Adopt** if the cross-fit macro F0.5 is at least **+0.002** above gradient boosting alone **and** the paired-bootstrap 95 % interval of the per-entity difference (both at their cross-fit thresholds) is above 0.
#
# **Sections:** 1 Load + preprocessing → 2 Train the ensemble → 3 Validation scores → 4 Threshold + cross-fit → 5 Decision + save

# %%
import os, sys, json, time, pickle, warnings
import numpy as np
import pandas as pd
import torch, torch.nn.functional as Fn
import joblib
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import QuantileTransformer
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
import eval_utils as EU
import listwise_model as LM
ROOT = "../../../"; INTERIM = ROOT + "dataset/interim"; DEV = LM.DEV
FEATS = json.load(open("feature_list.json"))["final_features"]
torch.set_num_threads(os.cpu_count() or 2)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
log(f"device {DEV}")
ctr = pd.read_pickle(INTERIM + "/candidates_train.pkl"); X = pd.read_pickle(INTERIM + "/features_train.pkl")[FEATS].values.astype(np.float32)
y = ctr.label.values.astype(np.float32); q = ctr.s1_id.values
qt = QuantileTransformer(n_quantiles=256, output_distribution="normal", subsample=200_000, random_state=0).fit(X)
nan_cols = np.where(np.isnan(X).any(0))[0]; prep = LM.Prep(qt, nan_cols); Z = prep(X); D = Z.shape[1]
rows = LM.group_rows(q); rs = np.random.default_rng(11); perm = rs.permutation(len(rows)); n_in = int(0.08 * len(rows))
rows_in = [rows[i] for i in perm[:n_in]]; rows_tr = [rows[i] for i in perm[n_in:]]
T_tr, T_in = LM.pad(Z, rows_tr, y), LM.pad(Z, rows_in, y)
print(f"{len(rows_tr):,} training queries, {len(rows_in):,} early-stopping queries | input width {D}")

# %% [markdown]
# ## 2. Train the ensemble (2 seeds)
# Binary cross-entropy over the valid candidates of each query; AdamW with a one-cycle learning-rate schedule; early stopping on the PR-AUC of the early-stopping queries.

# %%
def train(seed, epochs=25, bs=64, lr=1e-3):
    torch.manual_seed(seed); m = LM.SetNet(D).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    Xq, Yq, Mq = [t.to(DEV) for t in T_tr]; n = len(Xq); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((n + bs - 1) // bs))
    yin = np.concatenate([y[r] for r in rows_in]); best, bad, state = -1, 0, None
    for ep in range(epochs):
        m.train(); p_ = torch.randperm(n, device=DEV); t0 = time.time()
        for i in range(0, n, bs):
            ix = p_[i:i + bs]; lg = m(Xq[ix], Mq[ix]); loss = Fn.binary_cross_entropy_with_logits(lg[Mq[ix]], Yq[ix][Mq[ix]]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        sc = LM.scores_from_model(m, T_in[0], T_in[2], rows_in, len(y)); pr = average_precision_score(yin, np.concatenate([sc[r] for r in rows_in]))
        if ep % 3 == 0 or ep == epochs - 1: log(f"  seed {seed} epoch {ep+1}/{epochs} | early-stop PR-AUC {pr:.4f} | {time.time()-t0:.0f}s")
        if pr > best: best, bad, state = pr, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 5: break
    log(f"  seed {seed}: best early-stop PR-AUC {best:.4f}"); return {"d": D, "state": state}
STATES = [train(s) for s in (1, 2)]
torch.save(STATES, "listwise_model.pt"); joblib.dump(prep, "listwise_prep.joblib"); print("saved listwise_model.pt, listwise_prep.joblib")

# %% [markdown]
# ## 3. Validation scores (first use of validation labels in this phase)

# %%
LM._CACHE.clear()
cv = pd.read_pickle(INTERIM + "/candidates_val.pkl"); sv = pd.read_pickle(INTERIM + "/val_scores.pkl"); Fv = pd.read_pickle(INTERIM + "/features_val.pkl")[FEATS].values.astype(np.float32)
qv, yv = cv.s1_id.values, cv.label.values; g = sv.score.values; a = LM.predict_listwise(Fv, qv); b = 0.5 * g + 0.5 * a
P4 = pickle.load(open(INTERIM + "/phase4_truth_and_queries.pkl", "rb")); ents = cv.s1_id.unique().tolist(); tv = P4["truth"]["val"]; nt = tv[tv.s1_id.isin(set(ents))].groupby("s1_id").size()
for nm, p in (("gradient boosting (final)", g), ("listwise attention (2 seeds)", a), ("blend 0.5/0.5", b)): print(f"{nm:30s} validation PR-AUC {average_precision_score(yv, p):.4f}")

# %% [markdown]
# ## 4. Global threshold, cross-fitted over validation entities (as in Phase 8)

# %%
grid = np.round(np.arange(0.3, 0.96, 0.025), 3)
def per_entity_table(p): return pd.DataFrame({t: EU.macro_f05(ents, nt, qv, yv, (p >= t).astype(int))[1] for t in grid})
TB = {"gbdt": per_entity_table(g), "blend": per_entity_table(b), "listwise": per_entity_table(a)}
folds = np.array_split(np.random.default_rng(42).permutation(len(ents)), 5)
def crossfit(M):
    out = np.zeros(len(M)); ch = []
    for f in folds:
        tr = np.setdiff1d(np.arange(len(M)), f); t = M.iloc[tr].mean().idxmax(); out[f] = M.iloc[f][t].values; ch.append(t)
    return out, ch
CF = {k: crossfit(v) for k, v in TB.items()}
for k, v in TB.items(): print(f"{k:9s} in-sample best {v.mean().max():.4f} @ {v.mean().idxmax():.3f} | cross-fit {CF[k][0].mean():.4f} | thresholds per fold {CF[k][1]}")
d = CF["blend"][0] - CF["gbdt"][0]; r = np.random.default_rng(0); bs_ = [d[r.integers(0, len(d), len(d))].mean() for _ in range(2000)]
lo, hi = np.percentile(bs_, [2.5, 97.5]); print(f"\nblend minus gradient boosting (cross-fit): {d.mean():+.4f}, 95 % bootstrap [{lo:+.4f}, {hi:+.4f}]")

# %% [markdown]
# ## 5. Decision + save

# %%
adopt = bool(d.mean() >= 0.002 and lo > 0)
curve = TB["blend"].mean(); near = curve[curve >= curve.max() - 0.001].index; t_blend = float(near[np.abs(near.values - np.median(near.values)).argmin()])
print(f"ADOPT blend: {adopt} | smoothed global threshold {t_blend} (in-sample {curve.loc[t_blend]:.4f})")
bc = {"adopt": adopt, "weight_gbdt": 0.5, "weight_listwise": 0.5, "threshold": t_blend, "val_crossfit_blend": round(float(CF['blend'][0].mean()), 4), "val_crossfit_gbdt": round(float(CF['gbdt'][0].mean()), 4),
      "gain": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "val_pr_auc_blend": round(float(average_precision_score(yv, b)), 4)}
json.dump(bc, open("blend_config.json", "w"), indent=2); print(json.dumps(bc, indent=2))
