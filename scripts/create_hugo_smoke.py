import pandas as pd
import numpy as np
import os
from pathlib import Path
import shutil

# Define paths
PROJECT_ROOT = Path("/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/git_repo/Spatial-Clip")
SOURCE_DIR = PROJECT_ROOT / "data/hest_hugo_6nei_correct_parquet_data"
TARGET_DIR = PROJECT_ROOT / "data/processed/hest_hugo_smoke"
TARGET_SAMPLES = ['TENX158', 'TENX157', 'NCBI883']

def process_split(split_name):
    src_path = SOURCE_DIR / split_name
    dst_path = TARGET_DIR / split_name
    
    if not src_path.exists():
        print(f"Source split {split_name} not found at {src_path}")
        return

    print(f"Processing {split_name}...")
    
    # Read nodes
    nodes_path = src_path / "nodes.parquet"
    if not nodes_path.exists():
        print(f"nodes.parquet not found in {src_path}")
        return
        
    nodes_df = pd.read_parquet(nodes_path)
    
    # Filter nodes
    filtered_nodes = nodes_df[nodes_df['sample_id'].isin(TARGET_SAMPLES)].copy()
    
    if filtered_nodes.empty:
        print(f"No target samples found in {split_name}.")
        return

    print(f"Found {len(filtered_nodes)} nodes for target samples in {split_name}.")
    
    # Get indices (assuming the index of nodes_df corresponds to the row in embeddings)
    # We need to be careful here. If nodes_df index is not 0..N-1, we might have issues if we just take .index
    # But usually read_parquet preserves index.
    # Let's assume the embeddings correspond to the rows of nodes_df.
    indices = filtered_nodes.index.to_numpy()
    
    # Read and filter embeddings
    img_embeds = np.load(src_path / "image_embeds.npy")
    txt_embeds = np.load(src_path / "text_embeds.npy")
    
    filtered_img_embeds = img_embeds[indices]
    filtered_txt_embeds = txt_embeds[indices]
    
    # Read edges
    edges_df = pd.read_parquet(src_path / "edges.parquet")
    
    # Remap tile_ids
    # 1. Save original tile_id (which is the index in the original full dataset)
    # In the source data, 'tile_id' column usually holds the index.
    filtered_nodes['original_global_id'] = filtered_nodes['tile_id'] 
    
    # 2. Reset index to get new 0-based indices
    filtered_nodes = filtered_nodes.reset_index(drop=True)
    filtered_nodes['tile_id'] = filtered_nodes.index
    
    # 3. Create map: old_tile_id -> new_tile_id
    valid_old_ids = set(filtered_nodes['original_global_id'])
    
    # Filter edges where BOTH ends are in our new set
    filtered_edges = edges_df[
        edges_df['src_tile_id'].isin(valid_old_ids) & 
        edges_df['nbr_tile_id'].isin(valid_old_ids)
    ].copy()
    
    # Build map
    id_map = dict(zip(filtered_nodes['original_global_id'], filtered_nodes['tile_id']))
    
    filtered_edges['src_tile_id'] = filtered_edges['src_tile_id'].map(id_map)
    filtered_edges['nbr_tile_id'] = filtered_edges['nbr_tile_id'].map(id_map)
    
    # Create output dir
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Save
    # Drop the temporary column
    filtered_nodes.drop(columns=['original_global_id'], inplace=True, errors='ignore')
    
    filtered_nodes.to_parquet(dst_path / "nodes.parquet")
    filtered_edges.to_parquet(dst_path / "edges.parquet")
    np.save(dst_path / "image_embeds.npy", filtered_img_embeds)
    np.save(dst_path / "text_embeds.npy", filtered_txt_embeds)
    
    print(f"Saved split {split_name} to {dst_path}")
    print(f"  Nodes: {len(filtered_nodes)}")
    print(f"  Edges: {len(filtered_edges)}")
    print(f"  Embeddings: {filtered_img_embeds.shape}")

if __name__ == "__main__":
    if TARGET_DIR.exists():
        print(f"Cleaning existing target directory: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
    
    process_split("train")
    process_split("val")
