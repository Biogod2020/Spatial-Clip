import torch
from torchmetrics import Metric
from typing import List
import os

class ZeroShotGeneExpressionMetric(Metric):
    def __init__(self, global_hvg_path: str = None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # State variables
        self.add_state("sum_pcc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")
        
        # Load global gene list (Gene Bank Index)
        self.gene_to_idx = {}
        self.num_global_genes = 0
        
        if global_hvg_path and os.path.exists(global_hvg_path):
            with open(global_hvg_path, 'r') as f:
                genes = [line.strip() for line in f if line.strip()]
            self.gene_to_idx = {gene: i for i, gene in enumerate(genes)}
            self.num_global_genes = len(genes)
        else:
            # If path is not provided or invalid, we can't compute metric correctly unless updated later
            pass

    def _compute_rank_weighted_vector(self, caption_list: List[str], device: torch.device) -> torch.Tensor:
        """
        Dynamically converts a list of captions into a rank-weighted gene expression vector.
        """
        batch_size = len(caption_list)
        target_vectors = torch.zeros((batch_size, self.num_global_genes), device=device, dtype=torch.float32)
        
        if self.num_global_genes == 0:
            return target_vectors

        for i, caption in enumerate(caption_list):
            # Assuming caption is "GeneA GeneB GeneC ..."
            genes_in_spot = caption.split()
            n_genes = len(genes_in_spot)
            
            if n_genes == 0:
                continue
                
            indices = []
            values = []
            
            for rank, gene in enumerate(genes_in_spot):
                if gene in self.gene_to_idx:
                    global_idx = self.gene_to_idx[gene]
                    # Rank decay logic: Rank 0 -> 1.0, Rank N -> 0.2
                    weight = 1.0 - (0.8 * rank / max(n_genes, 1))
                    indices.append(global_idx)
                    values.append(weight)
            
            if indices:
                # Use tensor indexing for assignment
                target_vectors[i, indices] = torch.tensor(values, device=device, dtype=torch.float32)
                
        return target_vectors

    def update(self, preds_logits: torch.Tensor, captions: List[str]):
        """
        preds_logits: (Batch, Num_Global_Genes) - Predicted Logits from Image Encoder
        captions: List[str] - Raw text captions from the batch
        """
        if self.num_global_genes == 0:
            return

        # 1. On-the-fly Ground Truth Generation
        targets = self._compute_rank_weighted_vector(captions, device=preds_logits.device)
        
        # 2. Calculate PCC (Sample-wise)
        # Center the vectors for Pearson Correlation
        preds_centered = preds_logits - preds_logits.mean(dim=1, keepdim=True)
        targets_centered = targets - targets.mean(dim=1, keepdim=True)
        
        # Numerator and Denominator
        numerator = (preds_centered * targets_centered).sum(dim=1)
        denominator = torch.sqrt((preds_centered ** 2).sum(dim=1)) * torch.sqrt((targets_centered ** 2).sum(dim=1))
        
        # Avoid division by zero
        valid_mask = denominator > 1e-6
        pcc = torch.zeros_like(numerator)
        pcc[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        self.sum_pcc += pcc.sum()
        self.total_count += pcc.numel()

    def compute(self):
        return self.sum_pcc / self.total_count if self.total_count > 0 else torch.tensor(0.0)
