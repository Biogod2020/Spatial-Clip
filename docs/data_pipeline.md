# Data Pipeline

This document collects everything you need to ingest raw slides, generate processed datasets, and prove how each artifact was built.

## 1. Directory layout (raw → intermediate → processed)

Every dataset lives under `data/` using a consistent three-tier layout:

```
data/
  raw/<dataset_key>/                 # Original slides, metadata CSVs, HGNC/HVG auxiliaries
  processed_intermediate/<dataset_key>/  # AnnData checkpoints between pipeline stages
  processed/<dataset_key>/           # Final WebDataset or parquet shards + manifest.json
```

- `dataset_key` is declared in each Hydra config (see `configs/preprocess/*.yaml`).
- You can keep the raw source wherever you like (local disk, mounted share, or symlink). The important part is that the config points to the canonical folder inside `data/raw/<dataset_key>`.
- Intermediate + processed folders are safe to delete and regenerate at any time; the manifest captures enough provenance to rebuild them.

## 2. Configuring datasets with Hydra

- Start from `configs/preprocess/default.yaml`. It already encodes the layout above through `dataset.key` and `paths_layout.*`.
- Override whatever is specific to your source by adding a new file in the same folder (see `configs/preprocess/hest_mouse.yaml` for an example). Typical overrides:
  - `dataset.key` and `dataset.description` so logs stay readable.
  - `source.*` paths (raw location, HVG list, metadata CSV, HGNC analog).
  - `params.general.species_filter`, tiling size, or sharding knobs.
- Need a tiny, real-data smoke run? Use `configs/preprocess/hest_human_smoke.yaml`. It inherits `default`, keeps the canonical HEST raw folder, switches `dataset.key` to `hest_v1_smoke`, and pins `params.samples_allowlist` to three Homo sapiens slides (`TENX158`, `TENX157`, `NCBI883`).
- Point the pipeline to that config via `CFG=preprocess/<name>.yaml`.

## 3. Running the Hydra entrypoint (no DVC needed)

The Make targets now call the Hydra-native entrypoint exposed via `python -m src.data.preprocessing`:

```bash
make preprocess-hest-v1                                # Full three-stage run for the default HEST release
make preprocess CFG=preprocess/hest_mouse.yaml RUN_STAGE=stage-1
make preprocess CFG=preprocess/hest_human_smoke.yaml RUN_STAGE=full
make preprocess RUN_STAGE=stage-2                      # Resume from aligned AnnData
make preprocess CFG=preprocess/custom.yaml RUN_STAGE=full

# Direct Hydra invocation (identical to the Make targets):
python -m src.data.preprocessing --config-name preprocess/hest_human_smoke.yaml run.stage=full

# Optional legacy compatibility shim (same flags as before):
python -m src.data.preprocessing.cli stage-2 --config-name preprocess/hest_human_smoke.yaml
```

- `run.stage` accepts `stage-1`, `stage-2`, `stage-3`, or `full` (the default). Comma-separated lists like `run.stage="stage-2,stage-3"` also work.
- The Typer shim simply shells out to Hydra; use it only if downstream scripts still expect the older interface.
- All stages are idempotent: if an output already exists it will be skipped.
- Logs and artifacts land under `data/processed_intermediate/<dataset_key>` and `data/processed/<dataset_key>`.

## 4. What each stage does (plain language)

1. **Stage 1 – merge & align**: loads every sample from the raw directory, filters out excluded IDs, merges AnnData objects, and harmonizes gene names via HGNC metadata. Output: `adata_aligned_unfiltered.h5ad`.
2. **Stage 2 – normalize & HVG filter**: reads the aligned AnnData, keeps only the provided HVG list, runs normalization, and stores `adata_final_for_sharding.h5ad`.
3. **Stage 3 – sharding**: walks the filtered AnnData, grabs corresponding tissue tiles, builds WebDataset shards per sample, and finally records a manifest with hashes and runtime metadata.

## 5. Manifests and provenance

After Stage 3 a `manifest.json` is written next to the shards. It contains:

- The resolved Hydra config (`hydra_config.resolved`) so you can replay the run.
- Fingerprints and SHA256 hashes for raw data folders, the HVG list, and HGNC resources.
- Git commit, working tree status, CLI invocation, host/user info, and timing data.
- Output statistics (number of shards, total bytes, per-sample spot counts).

To inspect or validate a manifest:

```bash
python scripts/inspect_manifest.py data/processed/hest_v1
python scripts/inspect_manifest.py data/processed/custom_ds/manifest.json --no-check-files
```

Use this when registering a dataset, sharing it with teammates, or before uploading to remote storage.

## 6. Smoke tests and CI guardrails

- `tests/test_preprocess.py` monkeypatches the heavy I/O pieces and runs Stage 1→3 on a synthetic dataset. It asserts that manifests exist and that shard counts stay in sync.
- `pytest -k preprocess` (or the default `make test`) now exercises this smoke path so regressions are caught early.
- The optional `scripts/inspect_manifest.py` validation plus the smoke test give you two automated checks before training consumes a dataset.

## 7. Helpful habits

1. Keep raw data read-only and rely on the manifest when sharing results.
2. Commit new `configs/preprocess/*.yaml` files whenever you onboard a dataset.
3. If you tune tiling/sharding parameters for a dataset, document why in the config’s comments so the manifest plus config tells the full story.
4. Consider freezing Python dependencies via `requirements.txt` or `environment.yaml` whenever you promote a dataset for long-term reuse.

With this setup you can drop in any new spatial dataset, point a Hydra config at it, run `make preprocess`, and immediately get reproducible outputs with a clear provenance trail.
