# Medium Dataset Experiments

## Overview
This document summarizes the experiments conducted on the "medium" dataset (`hest_v1_human_medium`) to compare the performance of **Spatial Loss** versus **Normal CLIP Loss**.

## Dataset Details
- **Name**: `hest_v1_human_medium`
- **Format**: `shards_v1` (WebDataset-style shards)
- **Splits**:
  - Train: ~35 samples
  - Val: ~5 samples
  - Test: ~10 samples
- **Location**: `data/processed/hest_v1_human_medium/`

## Experiments

### 1. Spatial Loss (`medium_spatial`)
- **Config**: `configs/experiment/medium_spatial.yaml`
- **Loss Function**: `SpatialLoss` (includes local loss and neighbor consistency)
- **Key Hyperparameters**:
  - `local_loss`: True
  - `neighbor_alpha_scale`: 0.5
  - `temp_reg_weight`: 0.05

**Results**:
- **Training Loss**: ~2.45
- **Validation Loss**: ~4.64
- **Test Metrics**:
  - **R@1**: 0.0346
  - **R@5**: 0.1741
  - **R@10**: 0.3426
  - **Test Loss**: 4.626

### 2. Normal CLIP Loss (`medium_normal`)
- **Config**: `configs/experiment/medium_normal.yaml`
- **Loss Function**: `ClipLoss` (Standard Contrastive Loss)
- **Key Hyperparameters**:
  - `local_loss`: True (Note: `ClipLoss` implementation might ignore this or handle it differently than `SpatialLoss`)
  - `gather_with_grad`: True

**Results**:
- **Training Loss**: ~2.19
- **Validation Loss**: ~4.33
- **Test Metrics**:
  - **R@1**: 0.0361
  - **R@5**: 0.1717
  - **R@10**: 0.3347
  - **Test Loss**: 4.347

## Analysis & Observations
1. **Performance Parity**: The two loss functions perform comparably on this small dataset.
   - **Normal CLIP** has a slight edge in **R@1** (+0.0015) and **Test Loss** (-0.28).
   - **Spatial Loss** has a slight edge in **R@5** (+0.0024) and **R@10** (+0.0079).

2. **Training Dynamics**:
   - The **Normal CLIP** loss achieved a lower training loss (2.19 vs 2.45), suggesting it might be easier to optimize or less regularized than the Spatial Loss.
   - The **Spatial Loss** includes additional terms (neighbor consistency) which might act as a regularizer, potentially explaining the higher training loss but slightly better broad retrieval (R@5, R@10).

3. **Technical Fixes**:
   - During these experiments, a bug in `src/models/components/metrics.py` was identified and fixed. The `RecallAtK` metric was crashing when `batch_size < k` (e.g., validation set of 5 samples vs R@10). The fix involves dynamically clamping `k` to the batch size: `k_eff = min(self.k, logits.size(1))`.

## Reproduction Steps
To reproduce these results:

1. **Generate Splits** (if not already present):
   ```bash
   # Ensure train_samples.txt, val_samples.txt, test_samples.txt exist in data/processed/hest_v1_human_medium/
   ```

2. **Run Spatial Experiment**:
   ```bash
   python src/train.py experiment=medium_spatial
   ```

3. **Run Normal Experiment**:
   ```bash
   python src/train.py experiment=medium_normal
   ```
