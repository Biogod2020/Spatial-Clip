# src/models/components/metrics.py (New, Robust Version)

import torch
import torchmetrics
from torchmetrics import Metric

class RecallAtK(Metric):
    """
    Custom Metric to calculate Recall@k for in-batch retrieval.
    It is robust to varying batch sizes.
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        # Get top-k predictions
        _, top_k_preds = torch.topk(logits, self.k, dim=1)
        
        # Check if the target is in the top-k predictions
        # target.view(-1, 1) expands target to be comparable with each of the k predictions
        correct = torch.any(top_k_preds == target.view(-1, 1), dim=1)
        
        self.correct += torch.sum(correct)
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total


class ContrastiveMetrics(torchmetrics.MetricCollection):
    """
    A collection of robust metrics for contrastive learning tasks.
    """
    def __init__(self, prefix: str):
        # CodeGuardian: Notice `num_classes` is GONE. This component is now flexible.
        super().__init__(
            {
                "R@1": RecallAtK(k=1),
                "R@5": RecallAtK(k=5),
                "R@10": RecallAtK(k=10),
            },
            prefix=prefix,
        )