# src/models/spatial_clip_module.py

from typing import Any, Dict

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.nn import Module as LossFunction
from torchmetrics import MetricCollection

from .components.spatial_clip_net import SpatialClipNet


class SpatialClipLitModule(LightningModule):
    """
    CodeGuardian: A compliant "Thin Orchestrator" LightningModule.
    Its `__init__` signature directly maps to the `spatial_clip.yaml` config structure.
    It is completely unaware of the specific loss function's implementation.
    """
    def __init__(
        self,
        net: SpatialClipNet,
        loss_fn: LossFunction, # Receives an instantiated nn.Module
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        train_metrics: MetricCollection,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
    ):
        super().__init__()
        # This saves all hyperparameters except the complex objects (net, loss_fn).
        self.save_hyperparameters(logger=True, ignore=["net", "loss_fn"])
        self.net = net
        self.loss_fn = loss_fn
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(images, texts)

    def model_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Centralized forward pass and loss calculation
        features = self.forward(batch["images"], batch["texts"])
        loss_input = {**features, **batch} # Combine features and batch data for the loss function
        loss_dict = self.loss_fn(**loss_input)
        
        # Prepare outputs for metrics and logging
        output = {"loss": loss_dict["contrastive_loss"]}
        
        # For metric calculation
        logits_per_image = features["image_features"] @ features["text_features"].T * features["logit_scale"]
        output["logits"] = logits_per_image
        
        return output

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        output = self.model_step(batch)
        self.log("train/loss", output["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_metrics(output["logits"], torch.arange(len(output["logits"]), device=self.device))
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, sync_dist=True)
        return output["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        output = self.model_step(batch)
        self.log("val/loss", output["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics(output["logits"], torch.arange(len(output["logits"]), device=self.device))
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        output = self.model_step(batch)
        self.log("test/loss", output["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics(output["logits"], torch.arange(len(output["logits"]), device=self.device))
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers."""
        optimizer = self.hparams.optimizer_cfg(params=self.parameters())
        
        if self.trainer is None:
            return {"optimizer": optimizer}

        # Dynamically calculate total steps for scheduler
        if self.trainer.max_steps == -1:
            total_steps = self.trainer.estimated_stepping_batches if self.trainer.max_epochs is not None else 1_000_000
        else:
            total_steps = self.trainer.max_steps
        
        if total_steps == float('inf') or total_steps == -1:
            total_steps = 1_000_000 
            self.print(f"Warning: Could not estimate total steps. Using default: {total_steps}.")

        scheduler = self.hparams.scheduler_cfg(optimizer=optimizer, num_training_steps=int(total_steps))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.get("optimized_metric", "val/loss"), # Use optimized metric from main config
                "interval": "step",
                "frequency": 1,
            },
        }