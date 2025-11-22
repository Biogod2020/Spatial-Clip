import yaml
import pandas as pd
import os

# 1. Get Medium IDs
with open("configs/preprocess/hest_human_medium.yaml", "r") as f:
    medium_config = yaml.safe_load(f)

medium_ids = set(medium_config["params"]["samples_allowlist"])
print(f"Found {len(medium_ids)} IDs in Medium dataset.")

# 2. Get Hugo IDs
hugo_train_path = "data/hest_hugo_6nei_correct_parquet_data/train/nodes.parquet"
hugo_val_path = "data/hest_hugo_6nei_correct_parquet_data/val/nodes.parquet"

hugo_ids = set()

if os.path.exists(hugo_train_path):
    df_train = pd.read_parquet(hugo_train_path, columns=["sample_id"])
    hugo_ids.update(df_train["sample_id"].unique())

if os.path.exists(hugo_val_path):
    df_val = pd.read_parquet(hugo_val_path, columns=["sample_id"])
    hugo_ids.update(df_val["sample_id"].unique())

print(f"Found {len(hugo_ids)} IDs in Hugo dataset.")

# 3. Find Overlap
overlap_ids = medium_ids.intersection(hugo_ids)
print(f"Found {len(overlap_ids)} overlapping IDs.")
print("Overlapping IDs:", sorted(list(overlap_ids)))
