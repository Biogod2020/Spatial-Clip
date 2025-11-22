
import sys
from pathlib import Path
import torch
import numpy as np
from src.data.datasets.shard_backend import ShardedSpatialDataset

def inspect_neighbors():
    # Configuration matching smoke_shards.yaml
    data_dir = Path("/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/git_repo/Spatial-Clip/data/processed/hest_v1_smoke")
    # The smoke config uses these splits
    train_ids = ["NCBI883", "TENX157", "TENX158"]
    
    print(f"Loading dataset from {data_dir}...")
    
    # Mock preprocess/tokenizer
    def identity(x): return x
    
    dataset = ShardedSpatialDataset(
        dataset_root=data_dir,
        split="train",
        sample_ids=train_ids,
        k_neighbors=6,
        preprocess_fn=identity,
        tokenizer=identity,
        rebuild_cache=True # Force rebuild to be sure
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check first few items
    has_neighbors = False
    for i in range(min(10, len(dataset))):
        item = dataset[i]
        anchor_id = item['anchor_tile_id']
        nbr_ids = item['neighbor_tile_ids']
        nbr_alphas = item['neighbor_alphas']
        
        # Filter out padding (-1)
        valid_nbrs = [n for n in nbr_ids if n != -1]
        
        print(f"Sample {i}: Anchor {anchor_id}")
        print(f"  Neighbors: {nbr_ids}")
        print(f"  Alphas: {nbr_alphas}")
        
        if len(valid_nbrs) > 0:
            has_neighbors = True

    if has_neighbors:
        print("\nCONCLUSION: The dataset DOES contain spatial neighbor information.")
    else:
        print("\nCONCLUSION: The dataset DOES NOT contain spatial neighbor information (or only self-loops/padding).")

if __name__ == "__main__":
    inspect_neighbors()
