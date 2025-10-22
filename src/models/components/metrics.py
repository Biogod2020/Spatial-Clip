# src/models/components/metrics.py

import torchmetrics

class ContrastiveMetrics(torchmetrics.MetricCollection):
    """
    A collection of metrics for contrastive learning tasks.
    Calculates In-Batch Retrieval Accuracy (R@1) and Recall@k.
    """
    def __init__(self, num_classes: int, prefix: str):
        # `num_classes` here corresponds to the batch size.
        super().__init__(
            {
                "R@1": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1),
                "R@5": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5),
                "R@10": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=10),
            },
            prefix=prefix,
        )