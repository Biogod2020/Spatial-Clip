import os
import sys
import json
import io
import logging
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import webdataset as wds
from PIL import Image
from tqdm import tqdm
import rootutils

# Setup root
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_full_hugo_dataset():
    # Paths
    INPUT_ROOT = root / "data/hest_hugo_6nei_correct_parquet_data"
    OUTPUT_ROOT = root / "data/processed/hest_hugo_full"
    
    if not INPUT_ROOT.exists():
        logging.error(f"Input directory not found: {INPUT_ROOT}")
        return

    # Splits to process
    splits = ['train', 'val']
    
    for split in splits:
        logging.info(f"Processing split: {split}")
        
        input_dir = INPUT_ROOT / split
        output_dir = OUTPUT_ROOT / split
        
        if not input_dir.exists():
            logging.warning(f"Split directory not found: {input_dir}, skipping.")
            continue
            
        # Load Data
        nodes_path = input_dir / "nodes.parquet"
        if not nodes_path.exists():
            logging.warning(f"Nodes file not found: {nodes_path}, skipping.")
            continue
            
        logging.info(f"Loading {nodes_path}...")
        nodes_df = pd.read_parquet(nodes_path)
        
        # Clean output directory
        if output_dir.exists():
            logging.info(f"Cleaning existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        
        # Shard Writer
        # We use a pattern that allows for many shards
        pattern = str(output_dir / "shard-%06d.tar")
        max_count = 1000 # Tiles per shard
        
        logging.info(f"Writing shards to {output_dir}...")
        
        with wds.ShardWriter(pattern, maxcount=max_count) as sink:
            for idx, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc=f"Writing {split}"):
                sample_id = row['sample_id']
                tile_id = row['tile_id']
                
                # Unique key for this tile
                key = f"{sample_id}_{tile_id}"
                
                # Metadata
                meta = {
                    "sample_id": sample_id,
                    "tile_id": tile_id,
                    "x": row['x'],
                    "y": row['y']
                }
                
                sample = {
                    "__key__": key,
                    "json": json.dumps(meta).encode('utf-8')
                }
                
                # Image
                img_path = row['image_path']
                try:
                    # Check if image path is absolute or relative
                    p = Path(img_path)
                    if not p.exists():
                        # Try relative to project root if absolute fails
                        p_rel = root / img_path
                        if p_rel.exists():
                            p = p_rel
                    
                    with open(p, 'rb') as f:
                        img_bytes = f.read()
                    sample["png"] = img_bytes
                except Exception as e:
                    logging.warning(f"Failed to read image {img_path}: {e}")
                    continue

                # Text
                text = row['gene_sentence']
                sample["txt"] = text.encode('utf-8')
                
                sink.write(sample)

        logging.info(f"Finished split: {split}")

    logging.info(f"All done! Full dataset saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    create_full_hugo_dataset()
