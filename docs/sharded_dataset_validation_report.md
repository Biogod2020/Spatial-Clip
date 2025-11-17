# Sharded Dataset Validation Report

## Overview
We ship a deterministic validator to guard CLIP pretraining against corrupted WebDataset shards. The utility replays coordinate and Top-50 gene sentence generation for each `.json`/`.txt` pair inside the final `tar` archives and cross-checks:

- Spot-level pixel coordinates against `adata_final_for_sharding.h5ad` rows (default tolerance: 1.5 px).
- Top-50 gene sentences against ranks recomputed via `get_top_k_genes`.
- Manifest coverage to confirm that every spot listed in the AnnData source is represented downstream.

Because the validator runs directly on production shards, any drift in tiling, captioning, or metadata serialization is caught before a training job begins.

## Validator CLI
- **Script**: `scripts/validate_sharded_dataset.py`
- **Inputs**: dataset directory containing `webdataset/` shards plus `adata_final_for_sharding.h5ad`
- **Outputs**: structured JSON reports emitted to `outputs/validation_reports/<dataset>.json`

### Running the validator
```bash
python scripts/validate_sharded_dataset.py data/processed/hest_v1_multitech_smoke \
    --max-spots-per-sample 0 \
    --output outputs/validation_reports/hest_v1_multitech_smoke.json

python scripts/validate_sharded_dataset.py data/processed/hest_v1_human_medium \
    --max-spots-per-sample 0 \
    --output outputs/validation_reports/hest_v1_human_medium.json
```

The CLI resolves the dataset manifest, loads the canonical AnnData file once, and streams shard samples with a small in-RAM spot cache. Statistics are printed to stdout and serialized to the chosen JSON artifact.

## Dataset Findings
### hest_v1_multitech_smoke ("Smoke")
- Coverage: `5,694 / 5,694` spots (100%) spanning Visium, Visium HD, Spatial Transcriptomics, and Xenium modalities.
- Coordinate mismatches: `0`; missing reference coordinates: `0`.
- Gene-ranking mismatches: `0`; generated sentences match recomputed Top-50 ranks for every spot.
- Report artifact: `outputs/validation_reports/hest_v1_multitech_smoke.json`.

### hest_v1_human_medium ("Medium")
- Coverage: `266,700 / 266,700` spots (100%) across the ~50-slide Homo sapiens allowlist.
- Coordinate mismatches: `0`; missing reference coordinates: `0` (slides lacking `pxl_*` automatically fall back to `obsm['spatial']`).
- Gene-ranking mismatches: `0`; generated sentences perfectly match recomputed Top-50 ranks.
- Report artifact: `outputs/validation_reports/hest_v1_human_medium.json`.

The slides listed below still ship without `pxl_*` metadata, so both Stage 3 and the validator source their coordinates from `obsm['spatial']`. The counts remain for awareness, but no samples are zeroed out anymore.

| Sample | Technology | Spots Using `obsm['spatial']` Fallback |
| --- | --- | --- |
| MEND17 | Spatial Transcriptomics | 261 |
| MEND18 | Spatial Transcriptomics | 209 |
| SPA33 | Spatial Transcriptomics | 243 |
| SPA34 | Spatial Transcriptomics | 242 |
| SPA35 | Spatial Transcriptomics | 238 |
| TENX111 | Xenium | 6,643 |
| TENX114 | Xenium | 6,006 |
| TENX121 | Xenium | 9,200 |
| TENX123 | Xenium | 10,810 |
| TENX132 | Xenium | 9,682 |
| TENX97 | Xenium | 11,845 |
| TENX98 | Xenium | 26,070 |

## Next Steps
1. Integrate the validator into CI or nightly preprocessing jobs so each pipeline run produces a fresh JSON report (and optionally appends to this document).
2. Re-run preprocessing + validation once TenX/Xenium vendors publish true pixel coordinates; at that point, update `get_spot_coordinates` to prefer the vendor-provided values over `obsm['spatial']`.
