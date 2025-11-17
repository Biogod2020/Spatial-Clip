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
- Need cross-platform coverage without running the whole corpus? Use `configs/preprocess/hest_multitech_smoke.yaml`. It allowlists six slides chosen to cover every `st_technology` label (Visium, Visium HD, Spatial Transcriptomics, Xenium) with duplicate Visium/Xenium examples to reach six total.
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
python -m src.data.preprocessing \
  --config-name preprocess/hest_human_smoke.yaml \
  preprocess.run.stage=stage-3                        # Prefix overrides with 'preprocess.'

# Optional legacy compatibility shim (same flags as before):
python -m src.data.preprocessing.cli \
  --override preprocess.params.tiling.tile_size=256 \
  stage-3
```

- `run.stage` accepts `stage-1`, `stage-2`, `stage-3`, or `full` (the default). Comma-separated lists like `run.stage="stage-2,stage-3"` also work when prefixed as `preprocess.run.stage=...`.
- Hydra loads the config under the top-level key `preprocess`, so every CLI override must be scoped (for example `preprocess.params.tiling.tile_size=256`). Omitting the prefix triggers “Key 'run' is not in struct”.
- The Typer shim simply shells out to Hydra; pass overrides via repeated `--override <key>=<value>` flags *before* the subcommand so Typer recognizes them.
- All stages are idempotent: if an output already exists it will be skipped.
- Logs and artifacts land under `data/processed_intermediate/<dataset_key>` and `data/processed/<dataset_key>`.

### Quick-reference commands

| Goal | Command |
| --- | --- |
| Full pipeline with defaults | `python -m src.data.preprocessing preprocess.run.stage=full` |
| Stage 3 only, smoke dataset, 256 px tiles | `python -m src.data.preprocessing --config-name preprocess/hest_human_smoke.yaml preprocess.run.stage=stage-3 preprocess.params.tiling.tile_size=256` |
| Stage 1 only via Typer | `python -m src.data.preprocessing.cli --config-name preprocess/default.yaml stage-1` |

### Common pitfalls & prevention

1. **Missing `adata_final_for_sharding.h5ad`** – Stage 3 requires Stage 1 and 2 outputs under `data/processed_intermediate/<dataset_key>/`. If you see `Missing input for Stage 3`, rerun Stage 1→2 (or switch to a config such as `preprocess/hest_human_smoke.yaml` that already produced intermediates).
2. **Override errors (`Key 'run' is not in struct`)** – Always scope overrides with the config group name: `preprocess.run.stage=stage-3`, `preprocess.params.tiling.tile_size=256`, etc. Hydra validates the structure *before* our `_unwrap_cfg` helper runs, so unscoped overrides are rejected.
3. **Typer flag ordering** – Place `--override` flags before the subcommand: `python -m src.data.preprocessing.cli --override preprocess.run.stage=stage-3 stage-3`. Putting them afterward makes Typer treat them as unknown options.
4. **Partial reruns** – Tiling size changes only affect Stage 3. If you tweak tiling or sharding knobs, rerun Stage 3 (after confirming Stage 2 outputs exist) rather than regenerating the earlier stages unnecessarily.

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

## 8. Coordinate metadata reference

- `pxl_col_in_fullres` / `pxl_row_in_fullres` live in `adata.obs` and hold full-resolution pixel coordinates in microscope space. Stage 3 prefers them when present because they line up with the histology image without any additional transforms.
- `array_col` / `array_row` are the array grid indices reported by 10x for Visium/Visium HD/Xenium. They are useful for QA (spot ordering, neighborhood lookup) but are redundant for tiling whenever `pxl_*` exists.
- `obsm['spatial']` is a two-column matrix stored on every AnnData object. Its axes follow the same convention as 10x `tissue_positions_list.csv` (x = image column, y = image row) for every slide we checked, but the array is less explicit about its reference frame, so we validate it against `pxl_*` whenever both exist.

### Availability snapshot (HEST v1.1.0)

The aggregated counts that back the notes below are saved to `outputs/gap_report/coordinate_field_availability.csv` and can be regenerated with the script shown in this notebook session. Summary:

| Technology | Slides | `pxl_*` present | `pxl_*` missing | `array_*` present | `array_*` missing | `obsm['spatial']` present |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial Transcriptomics | 552 | 116 | 436 | 116 | 436 | 552 |
| Visium | 602 | 602 | 0 | 602 | 0 | 602 |
| Visium HD | 10 | 10 | 0 | 10 | 0 | 10 |
| Xenium | 65 | 48 | 17 | 48 | 17 | 65 |

- Overall counts: 1,229 slides total, 776 with `pxl_*`, 776 with `array_*`, and 1,229 with `obsm['spatial']`.
- All 453 slides missing `pxl_*` (Spatial Transcriptomics + 17 Xenium slides) still carry `obsm['spatial']`, so we can fall back to it without gaps.

### Orientation + interchangeability findings

- Comparing `obsm['spatial'][:, 0:2]` to `pxl_*` across the 776 slides that have both shows they are numerically identical for 774 slides when interpreted as `(x, y) = (column, row)`.
- Two Visium slides (`NCBI786`, `NCBI787`) are transposed: the first column matches `pxl_row_in_fullres` and the second matches `pxl_col_in_fullres`. These are the only observed swaps, but we guard against future surprises by auto-detecting orientation via correlation with `pxl_*` when both fields exist.
- For slides lacking `pxl_*`, we now treat `obsm['spatial']` as authoritative. When the pipeline loads a slide, it should:
  1. Prefer `pxl_*` if present.
  2. Otherwise use `obsm['spatial']`, but first check whether `pxl_*` exists on any matched sample and reuse that orientation hint (if not, assume `(x, y)` and log a warning).
- The fallback plan lets us tile every slide (including legacy Spatial Transcriptomics) while keeping traceability: any future discrepancy will surface through the orientation check and the logged coordinate source field.

## 9. Gap analysis report (Nov 2025)

### Goals & inputs

- Objective: quantify how well the raw HEST v1.1.0 corpus supports downstream tiling by checking (a) nearest-neighbor spacing versus the fixed 224 px patch size, (b) metadata/coordinate coverage, and (c) anomalous samples that need manual overrides.
- Script: `python scripts/compute_gap_statistics.py --raw data/raw/hest_v1 --out outputs/gap_report --patch-size 224`.
- Outputs:
  - `outputs/gap_report/per_sample_gap_stats.csv` with 1,229 per-sample rows (spot counts, NN stats, metadata columns listed in the script header).
  - `outputs/gap_report/gap_summary.json` summarizing corpus-wide statistics (excerpted below).
  - `outputs/gap_report/missing_spatial_coords.{csv,json}` enumerating the 453 slides that lack `pxl_*` columns.
  - `outputs/gap_report/coordinate_field_availability.csv` that tallies coordinate sources per technology (Section 8).

### Key findings

| Technology | Samples | Avg spots | NN mean (px) | Gap mean (px) |
| --- | ---: | ---: | ---: | ---: |
| Spatial Transcriptomics | 552 | 312 | 291.19 | 35.19 |
| Visium | 602 | 2,629 | 268.34 | 12.34 |
| Visium HD | 10 | 2,004 | 469.32 | 213.32 |
| Xenium | 65 | 5,319 | 425.26 | 169.26 |

- Total coverage: 1,229 slides (2.12e6 spots). Global nearest-neighbor mean is 288.54 px, leaving a comfortable +32.5 px buffer ahead of the 224 px patch size.
- Technology spread: classical Spatial Transcriptomics tiles are sparse (avg 312 spots) yet still keep only ~35 px between the 224 px patch and the next spot. Visium shows a narrow +12 px margin, while Visium HD and Xenium exhibit very large gaps because the fixed 224 px patch covers only a fraction of their native sampling density.
- Coordinate completeness: 453 slides (all Spatial Transcriptomics plus 17 Xenium) lack `pxl_*` columns. Every slide still exposes `obsm['spatial']`, and Section 8 documents the fallback strategy.
- Orientation anomalies: only `NCBI786` and `NCBI787` require `(y, x)` ordering. The correlation check baked into `get_spot_coordinates()` locks this down automatically, so the tiler receives consistent `(x, y)` arrays.
- Failure log: `outputs/gap_report/failures.json` is currently empty, which means every slide produced usable nearest-neighbor statistics. Any future rerun will surface problematic slides there for triage.

### Recommended actions

1. **Keep patch size at 224 px for Visium/ST.** The NN gaps stay positive, so tiles will not overlap excessively. For Visium HD and Xenium, consider experimenting with larger patches once we support per-tech overrides.
2. **Use `obsm['spatial']` as the universal fallback.** Section 8 already explains the auto-detection path; wiring that into the preprocessing stages removes the largest data completeness gap.
3. **Carry the per-sample CSV into dashboards.** Spot counts, NN medians, and metadata columns give fast QA hooks (e.g., flag slides with unusually low `spot_count` or large `gap_mean`).
4. **Schedule periodic reruns.** Re-running `scripts/compute_gap_statistics.py` after each dataset refresh adds a timestamped trail that can be diffed to catch regressions in metadata, patch sizing, or coordinate health.
