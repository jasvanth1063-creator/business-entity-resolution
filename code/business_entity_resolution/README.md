# Business Entity Resolution — ML Challenge 2026

## Project Details

**Project Title:** Business Entity Resolution — Blocking + Gradient Boosting + Listwise Attention Blend for Multi-Source Record Matching

**Project Category:** Machine Learning Project · Data Matching / Record Linkage · Interpretable Project

### Problem Statement
Businesses appear under different names and addresses in three sources. Given every Source-1 (S1) entity, the task is to find all of its records in Source-2 (S2) and Source-3 (S3). The metric is macro F0.5 per S1 entity (precision weighted above recall); an entity with no true match scores 1.0 for an empty list.

Rules followed: models are trained from scratch (no pretrained weights), licences are MIT/Apache-2.0/BSD, far fewer than 8 B parameters, and no external lookups.

The pipeline emphasises correctness, reproducibility and interpretability: all analysis uses the training split only, and validation labels never influence upstream steps (retrieval, features, model choice rules were fixed before they were evaluated).

## Dataset
Competition files (train and test TSVs): **https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip**

| File | Content |
|---|---|
| `train_source1/2/3.tsv`, `train_ground_truth.tsv` | Training entities, records and true S1 → S2/S3 matches (India and US) |
| `test_source1/2/3.tsv` | 1,732,544 test S1 queries and the S2/S3 record pools (India, US and **France**) |

The data is **not** included in the submission zip. Download and unzip it, then place the files as:
```
dataset/train/   train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/    test_source1.tsv   test_source2.tsv   test_source3.tsv
```

## Architecture
```
 S1 query (name, address, country)
        |
        v
 [1] Normalisation          case / punctuation / abbreviation rules (normalization_config.json)
        |
        v
 [2] Retrieval (blocking)   same country only, over the whole S2+S3 pool
        |                     - char 2-4-gram TF-IDF on names  -> top 20
        |                     - word TF-IDF on addresses       -> top 20     (union, <= 40 candidates)
        v
 [3] Pair features          69 features per (query, candidate): string similarities, token overlap,
        |                     phonetic, address parts, rank/score, and 21 within-query relative features
        +--------------------------+
        v                          v
 [4a] HistGradientBoosting   [4b] Listwise attention network
      (per-pair probability)       all <=40 candidates of a query together:
                                   Linear+ReLU(128) -> 2 x Transformer encoder (4 heads)
                                   + linear skip -> per-candidate logit, 2-seed ensemble
        +--------------------------+
        v
 [5] Blend  p = 0.5 * p_gbdt + 0.5 * p_attention
        v
 [6] Decision rule  p >= 0.975  (global threshold, tuned on validation)
        v
 [7] Cleanup  record exclusivity: a record is kept only by its best-scoring S1 entity
        v
 output/matching_results.tsv   (matched S2/S3 ids per S1 entity)
 output/candidate_pairs.tsv    (retrieved candidates with score >= 0.01)
```
The attention network scores a candidate in the context of its competitors, which is where the gain over the pair-only gradient boosting comes from. Both models were trained on the training split only. If PyTorch cannot be imported (for example when blocked by an application-control policy) the attention network runs from `listwise_numpy.py` + `listwise_weights.npz` (NumPy only, same scores to ~1e-6).

## Pipeline (one notebook per phase; `phaseN_src.py` is the same code as a plain script)
| Phase | Notebook | Description |
|---|---|---|
| 0 | `phase0_split` | Entity-level 80/20 train/validation split, stratified by country × match count |
| 2 | `phase2_normalization_blocking` | Normalisation rules validated on hard negatives |
| 3 | `phase3_retrieval_mechanism` | Retrieval design (name char n-grams + address words) |
| 4 | `phase4_candidate_generation` | Full-scale candidate generation, recall and ceiling |
| 5 | `phase5_feature_engineering` | 77 pair features pruned to 69 by ablation |
| 6 / 6b | `phase6_cluster_consistency`, `phase6b_cluster_rules` | Cluster-level stacking and rules — tested, **not adopted** |
| 7 | `phase7_classifier` | Model families compared and tuned (gradient boosting chosen) |
| 7b / 7c / 7d | `phase7b_deep_experiments`, `phase7c_cascade_speedup`, `phase7d_listwise_blend_final` | Deep-learning experiments; cascade rejected; **blend adopted** |
| 8 | `phase8_threshold_validation` | Cross-fitted threshold tuning and evaluation |
| 9 | `phase9_interpretation_inference` | Permutation importance, error analysis, inference check |
| 11 / 11b | `phase11_shap_interpretation`, `phase11b_shap_blend` | SHAP for gradient boosting and for the blend, country comparison |
| 12 | `phase12_consistency_cleanup` | Record-exclusivity cleanup (adopted); lone-entity rule rejected |
| 13 | `phase13_test_inference` | Resumable full-size test run (`predict_phase13.py`) |

## Key Results
| Metric | Value |
|---|---|
| Final model | Gradient boosting + listwise attention blend, threshold 0.575 |
| Validation macro F0.5 (5-fold cross-fitted, 6,000 held-out queries) | **0.9559** |
| Gradient boosting alone | 0.9506 (blend gain +0.0053, 95 % CI [+0.0035, +0.0071]) |
| Pair-level precision / PR-AUC | 0.991 / 0.9986 |
| Retrieval recall (true matches reaching the classifier) | 92.9 % (ceiling macro F0.5 = 0.972) |
| Test queries processed | 1,732,544 |
| Test entities with no match | 118,611 (6.8 %; validation about 6 %) |
| Mean matches per non-empty test row | 3.36 |
| Mean candidates per row in `candidate_pairs.tsv` | 4.62 |
| Organisers' `validate_submission.py` | PASS |

Test labels are hidden, so the F0.5 above is a **validation** figure. **France** (about 15 % of test queries) does not occur in the training data, so its accuracy could not be validated; only distribution checks (share of empty rows, matches per entity) were possible.

## Repository / Zip Structure
```
output/
    matching_results.tsv
    candidate_pairs.tsv
code/business_entity_resolution/
    README.md
    requirements.txt
    src/
        predict_phase13.py          full-size test inference (blocked, resumable)
        predict.py / predict_phase12.py   in-memory versions for validation-size runs
        normalization*.py, retrieval_engine.py, candidate_generation.py, pair_features.py
        listwise_model.py, listwise_numpy.py, sample_scoring.py, cleanup.py, eval_utils.py
        model_pair.joblib, listwise_model.pt, listwise_prep.joblib, listwise_weights.npz
        *_config.json, feature_list.json, env_info.json
        phase*.ipynb, phase*_src.py
Documentation_template.md
```

## How to Run
```
pip install -r requirements.txt
cd code/business_entity_resolution/src
python predict_phase13.py --test-dir ../../../dataset/test --out-dir ../../../output --workers 4 --score-workers 2
# organisers' validator (provided with the competition files)
python validate_submission.py --matching ../../../output/matching_results.tsv --candidate ../../../output/candidate_pairs.tsv --test-dir ../../../dataset/test
```
* The run is block-wise and **resumable**: re-run the same command after an interruption (delete `dataset/interim/predict_work/` to start clean).
* Expect several hours for the full test files (the S2+S3 pool is read and indexed once per country); about 8 GB RAM is advisable on Windows.
* `scikit-learn` must be exactly 1.9.1 (the saved models depend on it); a pre-flight check stops the run on mismatched or stale files.
* To rerun the notebooks, run them from `src/` in phase order with the data placed as shown under *Dataset*.

## Tech Stack
Python 3.10+ (developed on 3.13) · pandas, numpy, scipy · scikit-learn (HistGradientBoosting, TF-IDF) · PyTorch (listwise attention; NumPy fallback) · rapidfuzz, jellyfish, unidecode (string similarity) · SHAP (interpretation) · joblib (serialisation)

## Licences
All dependencies are permissive (BSD / MIT). No pretrained models; the models have far fewer than 8 B parameters.
