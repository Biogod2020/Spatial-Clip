# Creating the Hugo Smoke Dataset

This document details the process of creating the `hest_hugo_smoke` dataset used for smoke testing the Spatial-Clip pipeline.

## 1. Source Data
The source data is located in:
`data/hest_hugo_6nei_correct_parquet_data/`

It consists of Parquet files (`nodes.parquet`, `edges.parquet`) containing spatial transcriptomics data, where each row represents a tile (patch) of a tissue sample.

## 2. Dataset Creation Logic
The creation process is handled by the script `scripts/create_hugo_shards.py`.

### Selection Criteria
To ensure a fair comparison with the existing `multitech` smoke dataset, we selected the **exact same samples** that are present in the `hest_v1_multitech_smoke` dataset.

**Selected Samples:**
*   `MISC52`
*   `NCBI461`
*   `NCBI759`
*   `NCBI858`
*   `TENX158`

This allows for a direct performance comparison between the two dataset versions using identical biological samples.

### Processing Steps
1.  **Load Parquet**: Read `nodes.parquet` from the source directory.
2.  **Filter**: Select rows corresponding to the 5 chosen `sample_id`s.
3.  **Shard Generation**:
    *   The data is converted into the **WebDataset** format (shards).
    *   Each record in a shard corresponds to a single tile and contains:
        *   `__key__`: Unique identifier (`{sample_id}_{tile_id}`).
        *   `.json`: Metadata (sample_id, tile_id, coordinates).
        *   `.png`: The image patch (read from the path specified in the parquet).
        *   `.txt`: The gene sentence (text representation of gene expression).
    *   Output Structure: `data/processed/hest_hugo_smoke/hugo_combined/shard-{000000..N}.tar`.

## 3. Usage
This dataset is used in the `smoke_hugo` experiment configuration (`configs/experiment/smoke_hugo.yaml`) to verify the training pipeline functionality.
