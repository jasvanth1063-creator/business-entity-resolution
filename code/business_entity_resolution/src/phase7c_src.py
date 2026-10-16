# %% [markdown]
# # Phase 7c — Cascade Speed-Up: skip the expensive features for easy negatives
#
# **Why.** On the full test set (~1.8M queries × ~38 candidates ≈ 68M pairs) the run time is dominated by two things: retrieval (already made ~7× faster) and the **per-pair similarity features** (Python string metrics: Jaro-Winkler, edit distances, token sets, soundex, number matching ...).
# About 87 % of retrieved candidates end with a model probability below 0.001 — they are easy negatives. A **cheap first-stage model** that sees only features that come almost free from retrieval and vectorised TF-IDF can discard most of them before the expensive features are computed.
#
# **Cheap features (stage 1):** retrieval ranks/scores and route flags (`name_rank, name_score, addr_rank, addr_score, in_name, in_addr, c_is_s2, is_us`), the two TF-IDF cosines (`n_tfidf_cos, a_tfidf_cos`), `a_c_missing`, and the four length features.
#
# **Experiment.**
# 1. Measure where the feature time goes (profile).
# 2. Train stage 1 on the 80 % fit queries; on the hold-out choose the cut-off τ that keeps **≥ 99.9 % of the pairs the full model accepts** (label-free target — no labels used to choose τ).
# 3. End-to-end effect: prune, **recompute the within-query relative features on the surviving candidates** (they depend on the competitors), rescore, compare macro F0.5 with the unpruned pipeline — (a) with the frozen final model, (b) with a model retrained on pruned-context features.
# 4. Repeat on the real validation set with the final model and with the adopted gradient-boosting + listwise-attention blend (Phase 7d; its attention context also shrinks when candidates are pruned).
#
# **Adoption rule (fixed beforehand):** macro-F0.5 loss ≤ 0.001 on both hold-out and validation **and** at most 50 % of pairs kept.
#
# **Sections:** 1 Load → 2 Where the time goes → 3 Stage-1 model and cut-off → 4 End-to-end (hold-out) → 5 Validation → 6 Decision

# %%
import os, sys, json, time, pickle, warnings, cProfile, pstats, io
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
import eval_utils as EU
import pair_features as PF
ROOT = "../../../"; INTERIM = ROOT + "dataset/interim"
FEATS = json.load(open("feature_list.json"))["final_features"]; MC = json.load(open("model_config.json")); THR = json.load(open("threshold_config.json"))["decision"]["threshold"]
CHEAP = ["name_rank", "name_score", "addr_rank", "addr_score", "in_name", "in_addr", "c_is_s2", "is_us", "n_tfidf_cos", "a_tfidf_cos", "a_c_missing", "n_len_s", "n_len_c", "n_len_ratio"]
REL = PF.REL_BASE; RELCOLS = [f"{b}__{k}" for b in REL for k in ("rank", "dmax", "margin")]
BASE_FEATS = [f for f in FEATS if f not in RELCOLS]; print(f"{len(FEATS)} model features = {len(BASE_FEATS)} pair-level + {len(FEATS)-len(BASE_FEATS)} within-query relative | cheap stage-1 features: {len(CHEAP)}")
FULL = pd.read_pickle(INTERIM + "/features_train.pkl")      # all 77 columns incl. the dropped retrieval group
ctr = pd.read_pickle(INTERIM + "/candidates_train.pkl")
P4 = pickle.load(open(INTERIM + "/phase4_truth_and_queries.pkl", "rb"))
ntr = P4["truth"]["train"].groupby("s1_id").size(); ents = P4["queries"][P4["queries"].split == "train"].s1_id.tolist()
y = ctr["label"].values; qid = ctr["s1_id"].values
rng = np.random.default_rng(42); hold = set(rng.choice(np.array(ents), int(0.2 * len(ents)), replace=False)); is_hold = ctr["s1_id"].isin(hold).values; ent_hold = [e for e in ents if e in hold]

def relative(B, q):
    """Recompute the 21 within-query relative features from the pair-level columns of B (same definition as pair_features.compute_features)."""
    out = {}
    for b in REL:
        s = pd.Series(B[b].values); g = s.groupby(q); mx = g.transform("max")
        out[f"{b}__rank"] = g.rank(ascending=False, method="min").astype(np.float32).values
        out[f"{b}__dmax"] = (s - mx).astype(np.float32).values
        second = g.transform(lambda x: x.nlargest(2).iloc[-1] if x.notna().sum() >= 2 else np.nan)
        other = np.where(s.values >= mx.values, second.values, mx.values); out[f"{b}__margin"] = (s.values - other).astype(np.float32)
    return pd.DataFrame(out)
chk = relative(FULL.iloc[:20000], qid[:20000]); first_q = pd.Series(qid[:20000]).drop_duplicates().iloc[:-1]; ok = pd.Series(qid[:20000]).isin(first_q).values
print("sanity: recomputed relative features equal the stored ones:", all(np.allclose(chk[c].values[ok], FULL[c].values[:20000][ok], equal_nan=True) for c in RELCOLS))

# %% [markdown]
# ## 2. Where does the feature time go?
# Profile `compute_features` on 20,000 validation pairs, single process.

# %%
cv = pd.read_pickle(INTERIM + "/candidates_val.pkl"); cfg = json.load(open("retrieval_config.json")); stats = pickle.load(open("feature_stats.pkl", "rb"))
samp = cv.sort_values("s1_id", kind="stable").iloc[:20000].reset_index(drop=True)
pr = cProfile.Profile(); t0 = time.time(); pr.enable(); _ = PF.compute_features(samp.drop(columns=["label"]), stats, cfg, workers=1, log=lambda m: None); pr.disable(); tot = time.time() - t0
st = pstats.Stats(pr); fn = {k[2]: v for k, v in st.stats.items()}
def cum(name): return next((v[3] for k, v in st.stats.items() if k[2] == name), 0.0)
t_row = cum("_chunk"); print(f"compute_features total {tot:.1f}s for 20,000 pairs ({20000/tot:,.0f} pairs/s) | per-pair string metrics (_chunk) {t_row:.1f}s = {t_row/tot:.0%} | rest (normalisation, TF-IDF cosines, relative) {(tot-t_row)/tot:.0%}")
SHARE = t_row / tot

# %% [markdown]
# ## 3. Stage-1 model and cut-off τ (hold-out)

# %%
fit = ~is_hold; Xc = FULL[CHEAP].values.astype(np.float32)
s1 = HistGradientBoostingClassifier(random_state=42, max_iter=150, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200).fit(Xc[fit], y[fit]); p1 = s1.predict_proba(Xc[is_hold])[:, 1]
print(f"stage-1 PR-AUC on hold-out {average_precision_score(y[is_hold], p1):.4f}")
Xf = FULL[FEATS].values.astype(np.float32)
full = HistGradientBoostingClassifier(random_state=42, **MC["gbdt"]["params"]).fit(Xf[fit], y[fit]); pf = full.predict_proba(Xf[is_hold])[:, 1]
acc = pf >= THR; yH, qH = y[is_hold], qid[is_hold]
rows = []
for tau in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1):
    kp = p1 >= tau; rows.append((tau, kp.mean(), 1 - (kp & acc).sum() / acc.sum(), 1 - (kp & (yH == 1)).sum() / (yH == 1).sum()))
T = pd.DataFrame(rows, columns=["tau", "pairs kept", "accepted pairs lost", "true positives lost"]); print(T.round(5).to_string(index=False))
ok = T[T["accepted pairs lost"] <= 0.001]; TAU = float(ok.tau.max()) if len(ok) else float(T.tau.min()); print(f"\nchosen tau = {TAU:g} (largest cut-off losing <= 0.1 % of the pairs the full model accepts)")

# %% [markdown]
# ## 4. End-to-end on the hold-out: prune, recompute relative features, rescore

# %%
def macro(ent, nt, q, lab, keep): return EU.macro_f05(ent, nt, q, lab, keep.astype(int))[0]
f_full = macro(ent_hold, ntr, qH, yH, pf >= THR); kp = p1 >= TAU
H = FULL.loc[is_hold, BASE_FEATS].reset_index(drop=True)
def pruned_view(Bdf, q, keepmask):
    sub = Bdf[keepmask].reset_index(drop=True); R = relative(sub, q[keepmask]); return pd.concat([sub, R], axis=1)[FEATS]
Hp = pruned_view(H, qH, kp)
pfp = np.zeros(len(yH)); pfp[kp] = full.predict_proba(Hp.values.astype(np.float32))[:, 1]
f_frozen = macro(ent_hold, ntr, qH, yH, pfp >= THR)
# (b) retrain on pruned-context training rows (stage-1 scores for train rows are out-of-fold)
from sklearn.model_selection import GroupKFold
oof = np.zeros(len(y)); idx = np.flatnonzero(fit)
for a, b in GroupKFold(3).split(idx, groups=qid[idx]):
    m = HistGradientBoostingClassifier(random_state=42, max_iter=150, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200).fit(Xc[idx[a]], y[idx[a]]); oof[idx[b]] = m.predict_proba(Xc[idx[b]])[:, 1]
ktr = (oof >= TAU) & fit; Btr = FULL.loc[ktr, BASE_FEATS].reset_index(drop=True); Ptr = pd.concat([Btr, relative(Btr, qid[ktr])], axis=1)[FEATS]
re_m = HistGradientBoostingClassifier(random_state=42, **MC["gbdt"]["params"]).fit(Ptr.values.astype(np.float32), y[ktr])
pfr = np.zeros(len(yH)); pfr[kp] = re_m.predict_proba(Hp.values.astype(np.float32))[:, 1]; f_retrain = macro(ent_hold, ntr, qH, yH, pfr >= THR)
print(f"hold-out macro F0.5 at t={THR}: unpruned {f_full:.4f} | pruned, frozen model {f_frozen:.4f} ({f_frozen-f_full:+.4f}) | pruned, retrained {f_retrain:.4f} ({f_retrain-f_full:+.4f}) | pairs kept {kp.mean():.1%}")
print(f"training pairs kept for the retrained model: {ktr.sum()/fit.sum():.1%}")

# %% [markdown]
# ## 5. Validation set (final model, stage 1 trained on all training pairs)

# %%
Fv = pd.read_pickle(INTERIM + "/features_val.pkl"); cvv = pd.read_pickle(INTERIM + "/candidates_val.pkl"); sv = pd.read_pickle(INTERIM + "/val_scores.pkl")
yv, qv = cvv.label.values, cvv.s1_id.values; ev = cvv.s1_id.unique().tolist(); tv = P4["truth"]["val"]; ntv = tv[tv.s1_id.isin(set(ev))].groupby("s1_id").size()
s1_all = HistGradientBoostingClassifier(random_state=42, max_iter=150, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200).fit(Xc, y)
pv1 = s1_all.predict_proba(Fv[CHEAP].values.astype(np.float32))[:, 1]; kv = pv1 >= TAU
final = joblib.load("model_pair.joblib"); Bv = Fv[BASE_FEATS].reset_index(drop=True); Vp = pruned_view(Bv, qv, kv)
pvp = np.zeros(len(yv)); pvp[kv] = final.predict_proba(Vp.values.astype(np.float32))[:, 1]
f_val_full = macro(ev, ntv, qv, yv, sv.score.values >= THR); f_val_pruned = macro(ev, ntv, qv, yv, pvp >= THR)
acc_v = sv.score.values >= THR
import listwise_model as LM
bl = json.load(open("blend_config.json")); TB = bl["threshold"]
a_full = LM.predict_listwise(Fv[FEATS].values.astype(np.float32), qv); a_pr = np.zeros(len(yv)); a_pr[kv] = LM.predict_listwise(Vp.values.astype(np.float32), qv[kv])
blend_full = 0.5 * sv.score.values + 0.5 * a_full; blend_pr = 0.5 * pvp + 0.5 * a_pr
f_bl_full = macro(ev, ntv, qv, yv, blend_full >= TB); f_bl_pruned = macro(ev, ntv, qv, yv, blend_pr >= TB)
print(f"validation macro F0.5 of the ADOPTED blend (t={TB}): unpruned {f_bl_full:.4f} | pruned {f_bl_pruned:.4f} ({f_bl_pruned-f_bl_full:+.4f})")
print(f"validation macro F0.5 at t={THR}: unpruned {f_val_full:.4f} | pruned (frozen final model) {f_val_pruned:.4f} ({f_val_pruned-f_val_full:+.4f}) | pairs kept {kv.mean():.1%} | accepted pairs lost {1-(kv&acc_v).sum()/acc_v.sum():.4%}")

# %% [markdown]
# ## 6. Decision and saving estimate

# %%
loss_h = min(f_frozen, f_retrain) - f_full; loss_v = min(f_val_pruned - f_val_full, f_bl_pruned - f_bl_full); kept = max(kp.mean(), kv.mean())
adopt = bool(loss_h >= -0.001 and loss_v >= -0.001 and kept <= 0.5)
feat_time_saved = SHARE * (1 - kept); print(f"adopt cascade: {adopt} | pairs kept {kept:.1%} | per-pair feature time cut to ~{kept:.0%} of its original ({SHARE:.0%} of feature time is per-pair metrics) -> total feature time x{1-feat_time_saved:.2f}")
json.dump({"adopt": adopt, "tau": TAU, "cheap_features": CHEAP, "pairs_kept_holdout": float(kp.mean()), "pairs_kept_val": float(kv.mean()), "f05_holdout": {"unpruned": f_full, "pruned_frozen": f_frozen, "pruned_retrained": f_retrain},
           "f05_val": {"unpruned": f_val_full, "pruned": f_val_pruned, "blend_unpruned": f_bl_full, "blend_pruned": f_bl_pruned}, "per_pair_share_of_feature_time": SHARE}, open("cascade_config.json", "w"), indent=2, default=float)
