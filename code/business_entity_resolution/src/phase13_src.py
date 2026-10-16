# %% [markdown]
# # Phase 13 — Test Inference and Submission Validation
#
# **Purpose.** Run the frozen pipeline on `test_source1/2/3.tsv`, write `output/matching_results.tsv` and `output/candidate_pairs.tsv`, and validate them with the organisers' `validate_submission.py`.
# Nothing is tuned here: the model (Phase 7), threshold (Phase 8) and cleanup rules (Phase 12) are fixed files in `src/`.
#
# **Run this notebook from `src/`.** The heavy work is done by `predict_phase13.py` (a subprocess), because it can run for many hours:
# * it processes one country at a time in blocks of 20,000 queries and saves every finished block to `dataset/interim/predict_work/<id>/`;
# * if the run is stopped (or the computer sleeps), run the same cell again — it resumes from the saved blocks;
# * set `SCORE_WORKERS` to the number of processes for the feature + scoring stage (the slow part; it scales almost linearly, each process needs ~1-2 GB RAM). Retrieval is single-process on Windows.
# * **Use the complete `src/` folder of the latest zip** — the script stops at start-up (pre-flight check) if it finds old copies of `pair_features.py`, `retrieval_engine.py` or `candidate_generation.py`.
# * **Countries:** the test set contains **France**, which does not exist in the training data (India and US only). Check `run_summary.json` in the work folder (printed at the end) and compare the three countries.
#
# **Sections:** 1 Settings and input check → 2 Run → 3 Output summary → 4 Official validator

# %%
import os, sys, subprocess, time
import pandas as pd
ROOT = "../../../"; TEST_DIR = ROOT + "dataset/test"; OUT_DIR = ROOT + "output"
WORKERS = 4            # processes for retrieval (Linux/macOS only; Windows searches in one process)
SCORE_WORKERS = 2      # processes for feature + model scoring (each needs ~1-2 GB RAM; raise to 3-4 if you have 16 GB+)
BLOCK = 20_000         # queries per block
LIMIT = None           # set e.g. 2000 for a quick smoke test (first N S1 rows); None = full test set
for f in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
    p = os.path.join(TEST_DIR, f); print(f"{f}: {'OK' if os.path.isfile(p) else 'MISSING'}" + (f" ({os.path.getsize(p)/1e6:,.0f} MB)" if os.path.isfile(p) else ""))
s1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id", "country"])
print(f"{len(s1):,} S1 queries | {s1.country.value_counts().to_dict()}")

# %% [markdown]
# ## 2. Run inference
# Progress and ETA are printed per block. Leave the notebook open (or run the same command in a terminal: `python predict_phase13.py --test-dir ../../../dataset/test --out-dir ../../../output --workers 4 --score-workers 2`).

# %%
cmd = [sys.executable, "-u", "predict_phase13.py", "--test-dir", TEST_DIR, "--out-dir", OUT_DIR, "--workers", str(WORKERS), "--score-workers", str(SCORE_WORKERS), "--block", str(BLOCK)]
if LIMIT: cmd += ["--limit", str(LIMIT)]
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout: print(line, end="")
proc.wait(); print(f"\nexit code {proc.returncode} after {(time.time()-t0)/60:.1f} min")
assert proc.returncode == 0, "inference failed - fix the error above and re-run this cell (finished blocks are kept)"

# %% [markdown]
# ## 3. Output summary

# %%
M = pd.read_csv(OUT_DIR + "/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
C = pd.read_csv(OUT_DIR + "/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
n_m = M.matched_entity_ids.map(lambda x: 0 if x == "" else x.count(",") + 1); n_c = C.candidate_entity_ids.map(lambda x: 0 if x == "" else x.count(",") + 1)
print(f"matching_results: {len(M):,} rows | empty {int((n_m == 0).sum()):,} ({(n_m == 0).mean():.1%}) | mean matches per non-empty row {n_m[n_m > 0].mean():.2f}")
print(f"candidate_pairs : {len(C):,} rows | empty {int((n_c == 0).sum()):,} | mean candidates per row {n_c.mean():.2f}")
print("matches per entity:", n_m.clip(upper=6).value_counts().sort_index().to_dict(), "(6 = 6+)")
print("sanity (validation had ~6% entities with no match and mostly 2-4 matches): compare the two lines above")
mm = M.set_index("source1_entity_id").matched_entity_ids; cc = C.set_index("source1_entity_id").candidate_entity_ids
bad = sum(1 for s in mm.index[:200_000] if mm[s] and not set(mm[s].split(",")) <= set(cc[s].split(",")))
print("rows (first 200k) whose matches are not inside their candidate list:", bad)

# %% [markdown]
# ## 4. Official validator
# Point `VALIDATOR` at the organisers' `validate_submission.py` (it is not part of the submission zip). Add `--check-ids` only if you have the memory for it.

# %%
cands = [ROOT + "validate_submission.py", ROOT + "utils/validate_submission.py", "validate_submission.py"]
VALIDATOR = next((c for c in cands if os.path.isfile(c)), None)
if VALIDATOR is None: print("validate_submission.py not found - set VALIDATOR to its path")
else:
    r = subprocess.run([sys.executable, VALIDATOR, "--matching", OUT_DIR + "/matching_results.tsv", "--candidate", OUT_DIR + "/candidate_pairs.tsv", "--test-dir", TEST_DIR], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-500:])
