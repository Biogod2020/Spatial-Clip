# 文件路径: src/models/spatial_clip_module.py

from typing import Any, Dict

import hydra
import torch
from lightning import LightningModule
from omegaconf import DictConfig

from .components.spatial_clip_net import SpatialClipNet
from .components.losses import SpatialLoss

class SpatialClipLitModule(LightningModule):
    """
    一个“薄编排层” LightningModule。
    CodeGuardian: __init__ 直接接收实例化的对象，而不是配置。
    这与 Hydra 的默认递归实例化行为完全匹配。
    """
    def __init__(
        self,
        net: SpatialClipNet,
        loss_fn: SpatialLoss,
        optimizer_cfg: DictConfig,  # CodeGuardian: 这里接收的是一个 partial 对象
        scheduler_cfg: DictConfig,  # CodeGuardian: 这里接收的是一个 partial 对象
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "loss_fn"])
        self.net = net
        self.loss_fn = loss_fn

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(images, texts)

    def _step(self, batch: Dict[str, Any], batch_idx: int, step_name: str) -> torch.Tensor:
        features = self.forward(batch["images"], batch["texts"])
        
        loss_input = {
            "image_features": features["image_features"],
            "text_features": features["text_features"],
            "logit_scale": features["logit_scale"],
            "logit_bias": features.get("logit_bias"),
            "image_tile_ids": batch["image_tile_ids"],
            "text_tile_ids": batch["text_tile_ids"],
            "neighbor_tile_ids": batch["neighbor_tile_ids"],
            "neighbor_alphas": batch["neighbor_alphas"],
        }
        
        loss_dict = self.loss_fn(**loss_input, output_dict=True)
        loss = loss_dict["contrastive_loss"]
        
        self.log(f"{step_name}/loss", loss, on_step=(step_name=="train"), on_epoch=True, prog_bar=True, logger=True)
        if step_name == "train" and self.trainer and self.trainer.optimizers:
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self._step(batch, batch_idx, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self._step(batch, batch_idx, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        CodeGuardian's Architectural Fix:
        `self.hparams.optimizer_cfg` 和 `self.hparams.scheduler_cfg` 已经是 functools.partial 对象。
        我们不再使用 `hydra.utils.instantiate`，而是直接调用这些 partial 对象，并传入运行时参数。
        """
        # 调用 optimizer partial 对象，传入 `params`
        optimizer = self.hparams.optimizer_cfg(params=self.parameters())
        
        if self.trainer is None:
            return {"optimizer": optimizer}

        # 动态计算总步数
        if self.trainer.max_steps == -1:
            if self.trainer.max_epochs is None:
                total_steps = 1_000_000 
                self.print(f"Warning: Could not estimate total steps. Using a default of {total_steps}.")
            else:
                total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = self.trainer.max_steps
        
        if total_steps == float('inf') or total_steps == -1:
            total_steps = 1_000_000 
            self.print(f"Warning: Could not estimate total steps for scheduler. Using a default value of {total_steps}.")

        # 调用 scheduler partial 对象，传入 `optimizer` 和 `num_training_steps`
        scheduler = self.hparams.scheduler_cfg(optimizer=optimizer, num_training_steps=int(total_steps))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "step",
                "frequency": 1,
            },
        }