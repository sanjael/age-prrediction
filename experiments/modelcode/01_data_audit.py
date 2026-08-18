import os
import glob
import hashlib
import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config import Config

def compute_md5(filepath: str) -> str:
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        return f"ERROR_{str(e)}"

def process_file(record):
    fpath, age, raw_split = record
    h = compute_md5(fpath)
    return {
        "filepath": fpath,
        "filename": os.path.basename(fpath),
        "age": int(age),
        "raw_split": raw_split,
        "hash": h
    }

def main(val_ratio: float = 0.10, seed: int = 42, fast_hash: bool = False):
    cfg = Config()
    print("=" * 60)
    print("FACIAL AGE ESTIMATION — DATASET AUDIT & MANIFEST CREATION")
    print("=" * 60)
    
    records = []
    
    # 1. Scan Train and Test directories
    for split_name, split_dir in [("train", cfg.train_dir), ("test", cfg.test_dir)]:
        if not os.path.exists(split_dir):
            print(f"Error: Directory not found: {split_dir}")
            continue
            
        print(f"Scanning '{split_name}' directory: {split_dir}")
        age_dirs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        
        for age_str in tqdm(age_dirs, desc=f"Scanning {split_name} folders"):
            try:
                age_val = int(age_str)
            except ValueError:
                continue
                
            age_folder_path = os.path.join(split_dir, age_str)
            for fname in os.listdir(age_folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(age_folder_path, fname)
                    records.append((full_path, age_val, split_name))
                    
    print(f"Total image files found: {len(records):,}")
    
    if len(records) == 0:
        print("No images found. Please check paths in config.py!")
        return

    # 2. Compute MD5 Hashes in parallel
    print("\nComputing MD5 hashes for duplicate & leakage detection...")
    if fast_hash:
        # Quick hash of first 1MB or size+name for preview
        print("Note: Running in fast mode...")
    
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for res in tqdm(executor.map(process_file, records), total=len(records), desc="Hashing"):
            results.append(res)
            
    df = pd.DataFrame(results)
    
    # 3. Analyze duplicates and leakage
    print("\n--- Duplicate & Contamination Audit ---")
    total_samples = len(df)
    unique_hashes = df["hash"].nunique()
    duplicates = df[df.duplicated("hash", keep=False)]
    
    print(f"Total Samples:       {total_samples:,}")
    print(f"Unique Hashes:       {unique_hashes:,}")
    print(f"Duplicates Detected: {len(duplicates):,} ({len(duplicates)/total_samples*100:.2f}%)")
    
    # Check cross-split leakage between raw train and test
    train_hashes = set(df[df["raw_split"] == "train"]["hash"])
    test_hashes = set(df[df["raw_split"] == "test"]["hash"])
    leaked_hashes = train_hashes.intersection(test_hashes)
    print(f"Cross-split leaked image hashes (Train & Test intersection): {len(leaked_hashes):,}")
    
    # Check if duplicates have conflicting age labels
    if len(duplicates) > 0:
        conflicting_labels = duplicates.groupby("hash")["age"].nunique()
        conflict_hashes = conflicting_labels[conflicting_labels > 1].index
        print(f"Hashes with conflicting age labels: {len(conflict_hashes):,}")
    
    # 4. Clean and Split
    print("\n--- Creating Stratified Splits ---")
    # For exact duplicate hashes, keep only the first occurrence within the training set to prevent overfitting
    # But preserve original test set evaluation intact
    train_df = df[df["raw_split"] == "train"].copy()
    test_df = df[df["raw_split"] == "test"].copy()
    
    # Remove leaked samples from training set if any (train-side deduplication against test)
    if len(leaked_hashes) > 0:
        print(f"Purging {len(train_df[train_df['hash'].isin(leaked_hashes)]):,} leaked samples from train set...")
        train_df = train_df[~train_df["hash"].isin(leaked_hashes)].copy()
        
    # Deduplicate within train set
    train_df = train_df.drop_duplicates(subset=["hash"], keep="first")
    
    # Perform Stratified 90/10 Train/Validation Split on cleaned train data
    # Filter ages with at least 2 samples for stratification
    age_counts = train_df["age"].value_counts()
    valid_ages = age_counts[age_counts >= 2].index
    
    train_strat = train_df[train_df["age"].isin(valid_ages)]
    train_singleton = train_df[~train_df["age"].isin(valid_ages)]
    
    train_sub, val_sub = train_test_split(
        train_strat,
        test_size=val_ratio,
        stratify=train_strat["age"],
        random_state=seed
    )
    
    # Add singletons to train split
    train_sub = pd.concat([train_sub, train_singleton], ignore_index=True)
    
    train_sub["split"] = "train"
    val_sub["split"] = "val"
    test_df["split"] = "test"
    
    manifest_df = pd.concat([train_sub, val_sub, test_df], ignore_index=True)
    manifest_df = manifest_df[["filepath", "filename", "age", "split", "raw_split", "hash"]]
    
    # Save clean manifest
    manifest_path = cfg.manifest_path
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nClean Manifest saved to: {manifest_path}")
    
    # 5. Summary Statistics
    print("\n" + "=" * 60)
    print("FINAL MANIFEST SUMMARY")
    print("=" * 60)
    split_summary = manifest_df.groupby("split").agg(
        total_images=("age", "count"),
        min_age=("age", "min"),
        max_age=("age", "max"),
        mean_age=("age", "mean"),
        std_age=("age", "std")
    ).reset_index()
    print(split_summary.to_string(index=False))
    
    # Age bin distribution summary
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
    labels = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]
    manifest_df["age_bin"] = pd.cut(manifest_df["age"], bins=bins, labels=labels, right=True)
    
    bin_table = manifest_df.pivot_table(index="age_bin", columns="split", values="age", aggfunc="count", observed=False, fill_value=0)
    print("\nAge Bin Distribution across Splits:")
    print(bin_table)
    
    print("\nAudit and Manifest Generation Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset audit and manifest generator")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Validation ratio from train set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--fast", action="store_true", help="Fast mode")
    args = parser.parse_args()
    
    main(val_ratio=args.val_ratio, seed=args.seed, fast_hash=args.fast)
