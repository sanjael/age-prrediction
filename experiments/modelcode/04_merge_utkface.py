"""
04_merge_utkface.py
Fast UTKFace dataset merger.
"""
import os
import pandas as pd
import numpy as np

def main():
    external_dir = "externaldata"
    base_manifest_path = "manifest_p2_320.csv"
    output_manifest_path = "manifest_p2_320_plus_utkface.csv"
    
    print(f"Loading base manifest: {base_manifest_path}...")
    base_df = pd.read_csv(base_manifest_path)
    print(f"Base manifest: Train={len(base_df[base_df['split']=='train'])}, Val={len(base_df[base_df['split']=='val'])}, Test={len(base_df[base_df['split']=='test'])}")
    
    files = [f for f in os.listdir(external_dir) if f.endswith('.jpg')]
    print(f"Found {len(files)} UTKFace files.")
    
    records = []
    for f in files:
        parts = f.split("_")
        if len(parts) < 3:
            continue
        try:
            age = int(parts[0])
        except ValueError:
            continue
            
        if age < 1 or age > 116:
            continue
            
        age = int(np.clip(age, 1, 100))
        full_path = os.path.abspath(os.path.join(external_dir, f))
        
        records.append({
            "filepath": full_path,
            "original_filepath": full_path,
            "filename": f,
            "age": age,
            "split": "train",
            "hash": f"utk_{f}"
        })
        
    df_utk = pd.DataFrame(records)
    print(f"Parsed {len(df_utk)} UTKFace training images.")
    
    merged_df = pd.concat([base_df, df_utk], ignore_index=True)
    print(f"\nFinal Merged Split Summary:")
    print(merged_df["split"].value_counts())
    
    merged_df.to_csv(output_manifest_path, index=False)
    print(f"\nSaved combined manifest to {output_manifest_path}")

if __name__ == "__main__":
    main()
