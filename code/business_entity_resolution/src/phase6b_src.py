# %% [markdown]
# # Phase 6b — Cluster-Consistency Rules on the Final Lists (completes Phase 6)
#
# **Where Phase 6 stands.** Phase 6 tested second-stage *stacking* (cluster features fed to a retrained classifier): it did **not** clear the adoption rule
# (ΔF0.5 −0.0002, bootstrap interval spanning 0), so that layer stays **off** (`cluster_config.json: adopt=false`).
# This part completes the phase with the other, cheaper form of cluster consistency: **label-free rules applied to the finished match lists** — no retraining, no extra model.
#
# **Idea.** The true matches of one S1 entity are records of the same business, so they should resemble *each other*.
# * **Prune:** a predicted match that does not resemble the entity's strongest prediction (low name+address affinity) is suspicious → drop it unless the classifier is very confident.
# * **Rescue:** a candidate just below the threshold that closely resembles an already-confident match is probably the same business → add it.
# Affinity = 0.5·name similarity + 0.5·address similarity between two candidates (`cluster_features.affinity`, 0.8·name only if an address is missing).
#
# **Evidence and rule (fixed beforehand).** Same two datasets as Phase 12: the 6,000 validation queries (final model scores) and the 4,800 train hold-out queries (80 %-model scores).
# Settings are chosen by 5-fold cross-fitting on validation; a rule is adopted only if the cross-fit gain is **> +0.002** and the hold-out replication gain is **> 0**.
#
# **Sections:** 1 Load → 2 Affinity features → 3 Prune → 4 Rescue → 5 Combined (cross-fit + replication) → 6 Decision + save

# %%
import os, sys, json, pickle, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
import eval_utils as EU
import cluster_features as CF
ROOT = "../../../"; INTERIM = ROOT + "dataset/interim"; THR = json.load(open("threshold_config.json"))["decision"]["threshold"]
CFG = json.load(open("retrieval_config.json")); LOW = 0.3          # only candidates with p >= LOW take part in the rules
P4 = pickle.load(open(INTERIM + "/phase4_truth_and_queries.pkl", "rb"))
cv = pd.read_pickle(INTERIM + "/candidates_val.pkl"); sv = pd.read_pickle(INTERIM + "/val_scores.pkl")
VAL = cv[["s1_id", "cand_id", "c_name", "c_addr", "label"]].assign(score=sv.score.values).reset_index(drop=True)
ctr = pd.read_pickle(INTERIM + "/candidates_train.pkl")
tr_ents = P4["queries"][P4["queries"].split == "train"].s1_id.tolist()
rng = np.random.default_rng(42); hold = set(rng.choice(np.array(tr_ents), int(0.2 * len(tr_ents)), replace=False)); m = ctr.s1_id.isin(hold).values
hs = pickle.load(open(INTERIM + "/phase7_hold_scores.pkl", "rb"))["hist_gbm"]
HOLD = ctr[["s1_id", "cand_id", "c_name", "c_addr", "label"]][m].assign(score=np.asarray(hs, float)).reset_index(drop=True)
def tc(split, ents): t = P4["truth"][split]; return t[t.s1_id.isin(set(ents))].groupby("s1_id").size()
SETS = {"val": (VAL, VAL.s1_id.unique().tolist(), tc("val", VAL.s1_id.unique())), "hold-out": (HOLD, sorted(hold), tc("train", hold))}
def score(name, keep): D, E, nt = SETS[name]; return EU.macro_f05(E, nt, D.s1_id.values, D.label.values, np.asarray(keep).astype(int))
for k in SETS: print(f"{k}: {len(SETS[k][1]):,} queries | baseline macro F0.5 at t={THR}: {score(k, SETS[k][0].score.values >= THR)[0]:.4f}")

# %% [markdown]
# ## 2. Affinity features (label-free)
# For every candidate with p ≥ 0.3: its affinity to the **top-scored other candidate** of the query (`aff_top`) and its **best affinity to any confident candidate** (p ≥ 0.9, other than itself; `aff_anchor`, −1 if there is none).

# %%
def add_aff(D):
    D = D.copy(); n = len(D); aff_top = np.full(n, np.nan); aff_anc = np.full(n, np.nan)
    sel = np.flatnonzero(D.score.values >= LOW); sub = D.iloc[sel]
    nm, ad = CF.prep_strings(sub, CFG)
    grp = pd.Series(np.arange(len(sub))).groupby(sub.s1_id.values).indices
    for q, ix in grp.items():
        if len(ix) < 2: continue
        A = CF.affinity(list(nm[ix]), list(ad[ix])); p = sub.score.values[ix]; k = len(ix)
        order = np.argsort(-p)
        top = np.where(order[0] == np.arange(k), order[1], order[0])       # best OTHER candidate
        aff_top[sel[ix]] = A[np.arange(k), top]
        hi = p >= 0.9; B = np.where(hi[None, :], A, -1.0); aff_anc[sel[ix]] = B.max(1)
    D["aff_top"] = aff_top; D["aff_anc"] = aff_anc
    return D
t0 = time.time()
for k in SETS: SETS[k] = (add_aff(SETS[k][0]), SETS[k][1], SETS[k][2])
print(f"affinities computed in {time.time()-t0:.0f}s")
V = SETS["val"][0]; pos = V[(V.score >= THR)]
print("affinity of accepted candidates to the query's top other candidate (val):")
print(pd.DataFrame({"true": pos[pos.label == 1].aff_top.describe(), "false": pos[pos.label == 0].aff_top.describe()}).round(3).T)

# %% [markdown]
# ## 3–4. Prune and rescue rules
# * **prune(a_min, p_keep):** among entities with ≥ 2 predictions, drop a prediction with `aff_top < a_min` unless `p ≥ p_keep`.
# * **rescue(t_low, a_add):** add a candidate with `t_low ≤ p < t` whose `aff_anc ≥ a_add`.

# %%
def rules(D, a_min=None, p_keep=1.01, t_low=None, a_add=0.9):
    s = D.score.values; keep = s >= THR
    if t_low is not None:
        keep = keep | ((s >= t_low) & (s < THR) & (D.aff_anc.values >= a_add))
    if a_min is not None:
        npred = pd.Series(keep).groupby(D.s1_id.values).transform("sum").values
        drop = keep & (npred >= 2) & (D.aff_top.values < a_min) & (s < p_keep)
        keep = keep & ~drop
    return keep
PR = [(None, 1.01)] + [(a, pk) for a in (0.3, 0.4, 0.5, 0.6) for pk in (0.9, 0.95, 0.98, 1.01)]
RS = [(None, 0.9)] + [(tl, aa) for tl in (0.3, 0.4, 0.5, 0.6) for aa in (0.8, 0.9)]
def per_entity(name, pr, rs): return score(name, rules(SETS[name][0], pr[0], pr[1], rs[0], rs[1]))[1]
base = per_entity("val", PR[0], RS[0]).mean()
print("PRUNE only (val in-sample macro F0.5; baseline %.4f):" % base)
print(pd.Series({f"a_min={a} p_keep={pk}": per_entity("val", (a, pk), RS[0]).mean() for a, pk in PR[1:]}).round(4).to_string())
print("\nRESCUE only:")
print(pd.Series({f"t_low={tl} a_add={aa}": per_entity("val", PR[0], (tl, aa)).mean() for tl, aa in RS[1:]}).round(4).to_string())

# %% [markdown]
# ## 5. Combined settings: cross-fit on validation, replication on the hold-out

# %%
G = pd.DataFrame({(pr, rs): per_entity("val", pr, rs) for pr in PR for rs in RS})
print(f"{G.shape[1]} combined settings scored per entity")
folds = np.array_split(np.random.default_rng(42).permutation(len(G)), 5); out = np.zeros(len(G)); chosen = []
for f in folds:
    tr = np.setdiff1d(np.arange(len(G)), f); k = G.iloc[tr].mean().idxmax(); out[f] = G.iloc[f][k].values; chosen.append(k)
b0 = G[(PR[0], RS[0])].mean(); print(f"baseline {b0:.4f} | cross-fit {out.mean():.4f} | gain {out.mean()-b0:+.4f}"); print("chosen per fold:", chosen)
best = G.mean().idxmax(); print("best in-sample setting:", best, f"{G[best].mean():.4f}")
hb, hb0 = per_entity("hold-out", *best).mean(), per_entity("hold-out", PR[0], RS[0]).mean(); print(f"hold-out replication: {hb0:.4f} -> {hb:.4f} ({hb-hb0:+.4f})")
# which part helps? prune-only / rescue-only best on both sets
for nm_, sub in (("prune only", [(p, RS[0]) for p in PR]), ("rescue only", [(PR[0], r) for r in RS])):
    kb = max(sub, key=lambda k: G[k].mean()); print(f"{nm_}: best {kb} val {G[kb].mean():.4f} ({G[kb].mean()-b0:+.4f}) | hold-out {per_entity('hold-out', *kb).mean()-hb0:+.4f}")

# %% [markdown]
# ## 6. Decision and save

# %%
adopt = bool((out.mean() - b0 > 0.002) and (hb - hb0 > 0))
print("adopt cluster post-rules:", adopt)
cfgc = json.load(open("cluster_config.json"))
cfgc["stacking"] = {"adopt": cfgc.get("adopt", False), "note": "second-stage stacking (Phase 6): not adopted"}
cfgc["post_rules"] = {"adopt": adopt, "crossfit_gain": round(float(out.mean() - b0), 4), "holdout_gain": round(float(hb - hb0), 4),
                      "best_setting": {"prune": best[0], "rescue": best[1]} if adopt else None, "rule": "crossfit gain > 0.002 and hold-out gain > 0"}
json.dump(cfgc, open("cluster_config.json", "w"), indent=2); print(json.dumps(cfgc["post_rules"], indent=2))
