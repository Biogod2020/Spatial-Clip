import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import webdataset as wds
from tqdm import tqdm

import rootutils

# Setup root
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OVERLAPPING_IDS = [
    'INT19', 'INT21', 'NCBI673', 'NCBI674', 'NCBI675', 'NCBI681', 'NCBI683', 'NCBI684', 
    'NCBI762', 'NCBI763', 'NCBI764', 'NCBI776', 'SPA125', 'SPA126', 'TENX122', 'TENX124', 
    'TENX125', 'TENX128', 'TENX138', 'TENX144', 'TENX153', 'TENX154', 'TENX155', 'TENX156', 
    'TENX27', 'TENX53', 'TENX56', 'TENX70', 'TENX71', 'TENX72', 'TENX73', 'TENX90', 
    'TENX92', 'TENX99'
]

def _write_sample_shards(sample_id: str, sample_df: pd.DataFrame, output_dir: Path) -> int:
    """Write shard tarballs for a single sample."""
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(sample_dir / "shard-%06d.tar")
    written = 0
    with wds.ShardWriter(pattern, maxcount=1000) as sink:
        for _, row in tqdm(
            sample_df.iterrows(),
            total=len(sample_df),
            desc=f"{sample_id}",
            leave=False,
        ):
            tile_id = row["tile_id"]
            key = f"{sample_id}_{tile_id}"
            meta = {
                "sample_id": sample_id,
                "tile_id": int(tile_id),
                "x": float(row["x"]),
                "y": float(row["y"]),
            }

            sample = {
                "__key__": key,
                "json": json.dumps(meta).encode("utf-8"),
            }

            img_path = row["image_path"]
            try:
                with open(img_path, "rb") as handle:
                    sample["png"] = handle.read()
            except Exception as exc:
                logging.warning("Failed to read image %s (%s)", img_path, exc)
                continue

            text = row["gene_sentence"]
            sample["txt"] = text.encode("utf-8")
            sink.write(sample)
            written += 1

    return written


def create_hugo_overlap_shards():
    INPUT_DIR_TRAIN = root / "data/hest_hugo_6nei_correct_parquet_data/train"
    INPUT_DIR_VAL = root / "data/hest_hugo_6nei_correct_parquet_data/val"
    OUTPUT_DIR = root / "data/processed/hest_hugo_overlap"

    if OUTPUT_DIR.exists():
        logging.info("Cleaning existing output dir %s", OUTPUT_DIR)
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    manifest = {"selected_samples": [], "total_tiles": 0}
    processed_ids = set()

    # Process Train
    if INPUT_DIR_TRAIN.exists():
        logging.info("Loading parquet files from %s", INPUT_DIR_TRAIN)
        nodes_df = pd.read_parquet(INPUT_DIR_TRAIN / "nodes.parquet")
        
        for sample_id in OVERLAPPING_IDS:
            if sample_id in processed_ids:
                continue
                
            sample_df = nodes_df[nodes_df["sample_id"] == sample_id].copy()
            if not sample_df.empty:
                logging.info("Writing shards for %s (%d tiles) from TRAIN", sample_id, len(sample_df))
                count = _write_sample_shards(sample_id, sample_df, OUTPUT_DIR)
                manifest["selected_samples"].append(
                    {"sample_id": sample_id, "tiles": int(count)}
                )
                manifest["total_tiles"] += count
                processed_ids.add(sample_id)

    # Process Val
    if INPUT_DIR_VAL.exists():
        logging.info("Loading parquet files from %s", INPUT_DIR_VAL)
        nodes_df = pd.read_parquet(INPUT_DIR_VAL / "nodes.parquet")
        
        for sample_id in OVERLAPPING_IDS:
            if sample_id in processed_ids:
                continue
                
            sample_df = nodes_df[nodes_df["sample_id"] == sample_id].copy()
            if not sample_df.empty:
                logging.info("Writing shards for %s (%d tiles) from VAL", sample_id, len(sample_df))
                count = _write_sample_shards(sample_id, sample_df, OUTPUT_DIR)
                manifest["selected_samples"].append(
                    {"sample_id": sample_id, "tiles": int(count)}
                )
                manifest["total_tiles"] += count
                processed_ids.add(sample_id)

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logging.info("Wrote manifest to %s", manifest_path)
    logging.info(
        "Done! Generated shards for %d samples (%d tiles)",
        len(manifest["selected_samples"]),
        manifest["total_tiles"],
    )
    
    missing = set(OVERLAPPING_IDS) - processed_ids
    if missing:
        logging.warning("Missing IDs: %s", missing)

if __name__ == "__main__":
    create_hugo_overlap_shards()
