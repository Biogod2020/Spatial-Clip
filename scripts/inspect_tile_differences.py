import pandas as pd
import webdataset as wds
import tarfile
import io
import json
import numpy as np
from pathlib import Path
import rootutils

# Setup root
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def get_hugo_data(sample_id, split="train"):
    parquet_path = root / f"data/hest_hugo_6nei_correct_parquet_data/{split}/nodes.parquet"
    df = pd.read_parquet(parquet_path)
    sample_df = df[df["sample_id"] == sample_id].copy()
    return sample_df

def get_medium_data(sample_id):
    shard_dir = root / f"data/processed/hest_v1_human_medium/{sample_id}"
    # Find all tar files
    tars = list(shard_dir.glob("*.tar"))
    
    data = []
    for tar_path in tars:
        ds = wds.WebDataset(str(tar_path)).decode()
        for sample in ds:
            if isinstance(sample["json"], str) or isinstance(sample["json"], bytes):
                meta = json.loads(sample["json"])
            else:
                meta = sample["json"]
            
            txt = sample["txt"]
            data.append({
                "x": meta["x"],
                "y": meta["y"],
                "tile_id": meta.get("tile_id"),
                "txt_len": len(txt.strip()),
                "txt": txt
            })
    return pd.DataFrame(data)

def analyze_sample(sample_id):
    print(f"\nAnalyzing Sample: {sample_id}")
    
    # Load Hugo
    hugo_df = get_hugo_data(sample_id)
    print(f"Hugo Tiles: {len(hugo_df)}")
    
    # Load Medium
    medium_df = get_medium_data(sample_id)
    print(f"Medium Tiles: {len(medium_df)}")
    
    # Compare Spatial Extent
    print("\nSpatial Extent (X, Y):")
    print(f"Hugo:   X[{hugo_df['x'].min():.1f}, {hugo_df['x'].max():.1f}], Y[{hugo_df['y'].min():.1f}, {hugo_df['y'].max():.1f}]")
    print(f"Medium: X[{medium_df['x'].min():.1f}, {medium_df['x'].max():.1f}], Y[{medium_df['y'].min():.1f}, {medium_df['y'].max():.1f}]")
    
    hugo_coords = set(zip(hugo_df['x'].round(0).astype(int), hugo_df['y'].round(0).astype(int)))

    # Check for overlap with SWAPPED coordinates for Medium
    print("\nChecking Overlap with SWAPPED Medium Coordinates (X <-> Y)...")
    # Swap X and Y for Medium
    medium_coords_swapped = set(zip(medium_df['y'].round(0).astype(int), medium_df['x'].round(0).astype(int)))
    
    common_swapped = hugo_coords.intersection(medium_coords_swapped)
    print(f"Common (Swapped): {len(common_swapped)}")
    
    if len(common_swapped) > 0:
        print("Found overlap after swapping axes!")
    else:
        print("Still no overlap after swapping axes.")

    # Text Length Statistics
    print("\nMedium Dataset Text Length Statistics:")
    print(medium_df['txt_len'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

if __name__ == "__main__":
    # Analyze another TEST sample
    analyze_sample("TENX99")
