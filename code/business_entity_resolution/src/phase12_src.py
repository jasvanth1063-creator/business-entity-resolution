# %% [markdown]
# # Phase 12 — Consistency Cleanup
#
# **Purpose.** After thresholding (Phase 8) apply label-free consistency rules to the final match lists and keep a rule **only if** it measurably raises macro F0.5.
# Cluster-consistency *features* were tested in Phase 6 and not adopted; this phase is limited to simple post-processing rules. (Cluster-style pruning is the next step.)
#
# **Rules tested**
# 1. **Exclusivity** — in the training truth no S2/S3 record belongs to more than one S1 entity. If several S1 entities claim the same record, keep only the best-scoring claimant.
# 2. **Lone-entity thresholds** — entities with *no* prediction may still receive their best candidate if its score ≥ `t_rescue`; entities with *exactly one* prediction keep it only if score ≥ `t_lone`.
#
# **Evidence.** Two datasets: the 6,000 validation queries (final model scores) and the 4,800 train hold-out queries (Phase 7 `hist_gbm` scores, never trained on). Parameters for rule 2 are chosen by **5-fold cross-fitting** on validation and replicated on the hold-out.
# **Adoption rule (fixed beforehand):** cross-fit gain > +0.002 on validation **and** gain > 0 on the hold-out. Exclusivity is a hard property of the data, so it is adopted if it does not reduce either score.
#
# **Sections:** 1 Load → 2 Constraint facts → 3 Exclusivity → 4 Lone-entity thresholds → 5 Freeze → 6 Check of `predict_phase12.py`
#
# **Script for this phase:** `predict_phase12.py` = the Phase 9 inference script (`predict.py`) plus these rules. The Phase 9 script is left unchanged; the full-size test run uses `predict_phase13.py`.

# %%
import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
import eval_utils as EU
from cleanup import apply_cleanup
ROOT = "../../../"; INTERIM = ROOT + "dataset/interim"; THR = json.load(open("threshold_config.json"))["decision"]["threshold"]
P4 = pickle.load(open(INTERIM + "/phase4_truth_and_queries.pkl", "rb"))
cv = pd.read_pickle(INTERIM + "/candidates_val.pkl"); sv = pd.read_pickle(INTERIM + "/val_scores.pkl")
VAL = pd.DataFrame({"s1_id": cv.s1_id.values, "cand_id": cv.cand_id.values, "score": sv.score.values, "label": cv.label.values})
ctr = pd.read_pickle(INTERIM + "/candidates_train.pkl")
tr_ents = P4["queries"][P4["queries"].split == "train"].s1_id.tolist()
rng = np.random.default_rng(42); hold = set(rng.choice(np.array(tr_ents), int(0.2 * len(tr_ents)), replace=False))
m = ctr.s1_id.isin(hold).values
hs = pickle.load(open(INTERIM + "/phase7_hold_scores.pkl", "rb"))["hist_gbm"]
HOLD = pd.DataFrame({"s1_id": ctr.s1_id.values[m], "cand_id": ctr.cand_id.values[m], "score": np.asarray(hs, float), "label": ctr.label.values[m]})
def truth_counts(split, ents): t = P4["truth"][split]; return t[t.s1_id.isin(set(ents))].groupby("s1_id").size()
SETS = {"val": (VAL, VAL.s1_id.unique().tolist(), truth_counts("val", VAL.s1_id.unique())),
        "hold-out": (HOLD, sorted(hold), truth_counts("train", hold))}
def score(name, keep): D, E, nt = SETS[name]; return EU.macro_f05(E, nt, D.s1_id.values, D.label.values, keep.astype(int))
for k in SETS: print(f"{k}: {len(SETS[k][1]):,} queries | baseline macro F0.5 at t={THR}: {score(k, SETS[k][0].score.values >= THR)[0]:.4f}")

# %% [markdown]
# ## 2. Is exclusivity really a property of the data?

# %%
for split in ("train", "val"):
    t = P4["truth"][split]; c = t.groupby("cand_id").s1_id.nunique()
    print(f"{split}: {len(t):,} true pairs, {int((c > 1).sum())} records matched to more than one S1 entity")

# %% [markdown]
# ## 3. Rule 1 — exclusivity
# The 6,000 + 4,800 queries are a **sample** of all S1 entities, so few records are claimed twice here. To see how conflicts grow with the number of queries, count them on random subsets.

# %%
for k in SETS:
    D = SETS[k][0]; kp = D.score.values >= THR
    cl = D[kp].groupby("cand_id").s1_id.nunique(); print(f"{k}: predicted records claimed by >1 S1: {int((cl > 1).sum())} of {len(cl):,}")
Dall = pd.concat([VAL, HOLD]); kp = Dall.score.values >= THR; rs = np.random.default_rng(1); qs = Dall.s1_id.unique()
for f in (0.25, 0.5, 1.0):
    sel = set(rs.choice(qs, int(f * len(qs)), replace=False)); sub = Dall[Dall.s1_id.isin(sel).values & kp]; cl = sub.groupby("cand_id").s1_id.nunique()
    print(f"  {f:.0%} of {len(qs):,} queries: {int((cl > 1).sum())} double-claimed records")
cfg_e = {"exclusivity": {"enabled": True, "margin": 0.0}}
for k in SETS:
    D = SETS[k][0]; a = score(k, D.score.values >= THR)[0]; b = score(k, apply_cleanup(D, THR, cfg_e))[0]; print(f"{k}: exclusivity {a:.4f} -> {b:.4f} ({b - a:+.4f})")
# pairs of the combined set: apply across val+hold together so records can collide between them
cb = apply_cleanup(Dall.reset_index(drop=True), THR, cfg_e); print("dropped predictions across combined set:", int((kp & ~cb).sum()), "| of which true matches:", int((kp & ~cb & (Dall.label.values == 1)).sum()))
excl_ok = all(score(k, apply_cleanup(SETS[k][0], THR, cfg_e))[0] >= score(k, SETS[k][0].score.values >= THR)[0] - 1e-9 for k in SETS) and int((kp & ~cb & (Dall.label.values == 1)).sum()) == 0
print("exclusivity adopted:", excl_ok)

# %% [markdown]
# ## 4. Rule 2 — lone-entity thresholds (cross-fitted)

# %%
RES = [None, 0.3, 0.4, 0.5, 0.6]; LONE = [None, 0.75, 0.8, 0.85, 0.9]
def per_entity(name, tr, tl):
    D = SETS[name][0]; cfg = {"lone": {"enabled": True, "t_rescue": tr, "t_lone": tl}}; return score(name, apply_cleanup(D, THR, cfg))[1]
grid = {(a, b): per_entity("val", a, b) for a in RES for b in LONE}
G = pd.DataFrame(grid); base = G[(None, None)]
print("in-sample mean per setting (val):"); print(G.mean().unstack().round(4))
folds = np.array_split(np.random.default_rng(42).permutation(len(G)), 5); out = np.zeros(len(G))
for f in folds:
    tr = np.setdiff1d(np.arange(len(G)), f); k = G.iloc[tr].mean().idxmax(); out[f] = G.iloc[f][k].values
print(f"\ncross-fit macro F0.5: {out.mean():.4f} vs baseline {base.mean():.4f} -> gain {out.mean()-base.mean():+.4f}")
best = G.mean().idxmax(); print("best in-sample setting:", best, f"({G[best].mean():.4f})")
hb = per_entity("hold-out", *best).mean(); hb0 = per_entity("hold-out", None, None).mean(); print(f"replication on hold-out with that setting: {hb0:.4f} -> {hb:.4f} ({hb-hb0:+.4f})")
lone_ok = (out.mean() - base.mean() > 0.002) and (hb - hb0 > 0)
print("lone-entity rule adopted:", lone_ok)

# %% [markdown]
# ## 5. Freeze

# %%
cfg = {"exclusivity": {"enabled": bool(excl_ok), "margin": 0.0},
       "lone": {"enabled": bool(lone_ok), "t_rescue": best[0] if lone_ok else None, "t_lone": best[1] if lone_ok else None},
       "val_crossfit_gain_lone": round(float(out.mean() - base.mean()), 4)}
json.dump(cfg, open("cleanup_config.json", "w"), indent=2); print(json.dumps(cfg, indent=2))

# %% [markdown]
# ## 6. Check of `predict_phase12.py`
# Same 300 validation queries as the Phase 9 check (`sim_out/` from `predict.py`, `sim_out_p12/` from `predict_phase12.py`). Because the adopted rule (exclusivity) found no conflicts in this sample, the two result files must be identical.
# To reproduce: `python predict_phase12.py --test-dir ../../../sim_test --out-dir ../../../sim_out_p12`

# %%
a, b = ROOT + "sim_out/matching_results.tsv", ROOT + "sim_out_p12/matching_results.tsv"
if os.path.exists(a) and os.path.exists(b):
    A = pd.read_csv(a, sep="\t", dtype=str, keep_default_na=False); B = pd.read_csv(b, sep="\t", dtype=str, keep_default_na=False)
    print(f"Phase 9 predict.py: {int((A.matched_entity_ids != '').sum())} non-empty rows | Phase 12 predict_phase12.py: {int((B.matched_entity_ids != '').sum())} non-empty rows | identical: {A.equals(B)}")
else:
    print("run predict.py and predict_phase12.py on ../../../sim_test first")
