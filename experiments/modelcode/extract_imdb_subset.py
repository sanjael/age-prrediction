"""
extract_imdb_subset.py
Extract and preprocess high-quality targeted demographic faces from imdb_crop.tar
"""
import os
import tarfile
import scipy.io as sio
import numpy as np
from datetime import datetime
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm

def main():
    tar_path = "imdb_crop.tar"
    mat_path = "imdb_extracted/imdb_crop/imdb.mat"
    output_dir = "data_processed/imdb_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Loading IMDB metadata...")
    mat = sio.loadmat(mat_path)
    imdb = mat["imdb"][0, 0]
    
    dob = imdb["dob"][0]
    photo_taken = imdb["photo_taken"][0]
    full_path = [p[0] for p in imdb["full_path"][0]]
    face_score = imdb["face_score"][0]
    second_face_score = imdb["second_face_score"][0]
    
    # 1. Quality Filter: Valid single face detection with high confidence
    valid_quality = ~np.isnan(face_score) & np.isnan(second_face_score) & (face_score > 1.2)
    
    # 2. Age Computation
    birth_years = []
    for d in dob:
        try:
            by = int(datetime.fromordinal(max(1, int(d)) - 366).year) if not np.isnan(d) and d > 366 else -1
            birth_years.append(by)
        except:
            birth_years.append(-1)
            
    birth_years = np.array(birth_years)
    ages = photo_taken - birth_years
    
    valid_mask = valid_quality & (ages >= 1) & (ages <= 100) & (birth_years > 1850)
    
    valid_indices = np.where(valid_mask)[0]
    print(f"[*] Found {len(valid_indices)} high-confidence single face records in 1-100 age range.")
    
    # 3. Targeted Demographic Selection:
    # Prioritize under-represented brackets: 1-19 and 46-100, plus moderate sample from 20-45
    selected_indices = []
    
    records = []
    for idx in valid_indices:
        a = int(ages[idx])
        p = full_path[idx]
        score = float(face_score[idx])
        records.append({"idx": idx, "age": a, "path": p, "score": score})
        
    df_all = pd.DataFrame(records)
    print("\nAvailable Valid IMDB Samples by Bracket:")
    bins = [0, 12, 19, 30, 45, 60, 75, 100]
    labels = ['1-12', '13-19', '20-30', '31-45', '46-60', '61-75', '76-100']
    df_all['bracket'] = pd.cut(df_all['age'], bins=bins, labels=labels)
    print(df_all['bracket'].value_counts())
    
    # Target quotas per bracket to create a balanced dataset:
    quotas = {
        '1-12': 2500,     # Take all available children
        '13-19': 6000,    # Take strong teen representation
        '20-30': 5000,    # Small selective sample
        '31-45': 5000,    # Small selective sample
        '46-60': 15000,   # Strong middle age supplement
        '61-75': 8000,    # Strong senior supplement
        '76-100': 2000    # Take all available elderly
    }
    
    selected_dfs = []
    for bracket, quota in quotas.items():
        sub = df_all[df_all['bracket'] == bracket].sort_values("score", ascending=False)
        selected_dfs.append(sub.head(quota))
        
    df_selected = pd.concat(selected_dfs).reset_index(drop=True)
    print(f"\n[+] Total Selected IMDB Target Supplement: {len(df_selected)} images.")
    print(df_selected['bracket'].value_counts())
    
    # Build lookup dict of tar member name to output info
    # In tar, names start with 'imdb_crop/' + full_path
    tar_lookup = {}
    for _, row in df_selected.iterrows():
        member_name = "imdb_crop/" + row["path"].replace("\\", "/")
        tar_lookup[member_name] = row["age"]
        
    print(f"[*] Extracting and standardizing {len(tar_lookup)} images from {tar_path}...")
    
    extracted_records = []
    saved_count = 0
    
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if member.name in tar_lookup:
                age = tar_lookup[member.name]
                f = tar.extractfile(member)
                if f is not None:
                    try:
                        img_bytes = f.read()
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        # Standardize to 320x320
                        img = img.resize((320, 320), Image.BICUBIC)
                        
                        out_filename = f"imdb_{saved_count:06d}_age_{age}.jpg"
                        out_filepath = os.path.abspath(os.path.join(output_dir, out_filename))
                        img.save(out_filepath, "JPEG", quality=95)
                        
                        extracted_records.append({
                            "filepath": out_filepath,
                            "original_filepath": out_filepath,
                            "filename": out_filename,
                            "age": age,
                            "split": "train",
                            "hash": f"imdb_{saved_count:06d}"
                        })
                        saved_count += 1
                        
                        if saved_count % 5000 == 0:
                            print(f" [+] Extracted and standardized {saved_count}/{len(tar_lookup)} images...")
                    except Exception as e:
                        pass
                        
    print(f"\n[+] Extraction complete! Successfully processed {saved_count} IMDB faces.")
    df_extracted = pd.DataFrame(extracted_records)
    
    # 4. Merge with base training split
    base_manifest_path = "manifest_p2_320_plus_utkface.csv"
    base_df = pd.read_csv(base_manifest_path)
    print(f"[*] Base Manifest: {len(base_df)} images (Train={len(base_df[base_df['split']=='train'])}, Val={len(base_df[base_df['split']=='val'])}, Test={len(base_df[base_df['split']=='test'])})")
    
    # Validation and Test MUST remain 100% untouched
    combined_df = pd.concat([base_df, df_extracted], ignore_index=True)
    out_manifest = "manifest_master_imdb_augmented.csv"
    combined_df.to_csv(out_manifest, index=False)
    
    print(f"\n[+] Final Augmented Manifest Saved to: {out_manifest}")
    print("Partition Summary:")
    print(combined_df["split"].value_counts())

if __name__ == "__main__":
    main()
