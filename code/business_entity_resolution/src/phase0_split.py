"""
Phase 0 - Train/Validation Split

Splits Source-1 entities (and their ground-truth labels) into a train and
validation set, stratified by country and match-count bucket, so the
validation set mirrors the real distribution (including the singleton rate).

IMPORTANT: This only splits the S1 / ground-truth side. Source2 and Source3
files are NOT split - both train and val S1 entities search across the full
S2/S3 pool, exactly like the real test set will.

Usage:
    python src/split.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_DIR = "dataset/train"
SPLIT_DIR = "dataset/split"
RANDOM_SEED = 42
VAL_FRACTION = 0.2


def load_ground_truth_with_country():
    gt = pd.read_csv(
        f"{TRAIN_DIR}/train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    s1 = pd.read_csv(
        f"{TRAIN_DIR}/train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "country"],
    )

    gt["match_count"] = gt["matched_entity_ids"].apply(
        lambda x: 0 if x == "" else len(x.split(","))
    )
    gt["bucket"] = pd.cut(
        gt["match_count"],
        bins=[-1, 0, 1, 3, 5, 100],
        labels=["0", "1", "2-3", "4-5", "6+"],
    )

    gt = gt.merge(s1, left_on="source1_entity_id", right_on="entity_id", how="left")
    gt["strata"] = gt["country"].astype(str) + "_" + gt["bucket"].astype(str)
    return gt


def main():
    gt = load_ground_truth_with_country()
    print(f"Total S1 entities: {len(gt):,}")
    print("\nStrata distribution (full set):")
    print(gt["strata"].value_counts().sort_index())

    train_ids, val_ids = train_test_split(
        gt["source1_entity_id"],
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=gt["strata"],
    )

    train_ids.to_csv(f"{SPLIT_DIR}/train_s1_ids.txt", index=False, header=False)
    val_ids.to_csv(f"{SPLIT_DIR}/val_s1_ids.txt", index=False, header=False)

    print(f"\nTrain S1 entities: {len(train_ids):,}")
    print(f"Val S1 entities:   {len(val_ids):,}")

    # Sanity check: confirm no overlap, confirm strata proportions preserved
    assert set(train_ids).isdisjoint(set(val_ids)), "Train/val overlap detected!"

    gt_indexed = gt.set_index("source1_entity_id")
    train_strata = gt_indexed.loc[train_ids, "strata"].value_counts(normalize=True)
    val_strata = gt_indexed.loc[val_ids, "strata"].value_counts(normalize=True)
    print("\nStrata proportions - train vs val (should closely match):")
    comparison = pd.DataFrame({"train_pct": train_strata, "val_pct": val_strata}).sort_index()
    print(comparison)

    # Also save the ground-truth rows for each split, for convenience
    gt_train = gt[gt["source1_entity_id"].isin(set(train_ids))].drop(
        columns=["entity_id", "bucket", "strata", "match_count", "country"]
    )
    gt_val = gt[gt["source1_entity_id"].isin(set(val_ids))].drop(
        columns=["entity_id", "bucket", "strata", "match_count", "country"]
    )
    gt_train.to_csv(f"{SPLIT_DIR}/train_ground_truth.tsv", sep="\t", index=False)
    gt_val.to_csv(f"{SPLIT_DIR}/val_ground_truth.tsv", sep="\t", index=False)
    print(f"\nSaved: {SPLIT_DIR}/train_s1_ids.txt, val_s1_ids.txt, "
          f"train_ground_truth.tsv, val_ground_truth.tsv")


if __name__ == "__main__":
    main()
