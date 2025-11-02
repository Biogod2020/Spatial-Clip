# src/models/spatial_clip_module.py
import inspect  # <-- 1. 导入 inspect 模块
from typing import Any, Dict

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.nn import Module as LossFunction
from torchmetrics import MetricCollection

from .components.spatial_clip_net import SpatialClipNet


class SpatialClipLitModule(LightningModule):
    def __init__(
        self,
        net: SpatialClipNet,
        loss_fn: LossFunction,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        train_metrics: MetricCollection,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True, ignore=["net", "loss_fn"])
        self.net = net
        self.loss_fn = loss_fn
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

        # 2. ✅ CodeGuardian: 在初始化时，一次性检查并缓存损失函数需要的参数名集合。
        self._loss_fn_arg_names = set(inspect.signature(self.loss_fn.forward).parameters.keys())
        # 这将得到一个类似 {'image_features', 'text_features', 'logit_scale', ...} 的集合

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(images, texts)

    def model_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        features = self.forward(batch["images"], batch["texts"])

        # 3. ✅ CodeGuardian: 实现“智能分发器”
        # 3a. 准备一个包含所有可用数据的“大字典” (The available data pool)
        available_data = {**features, **batch}

        # 3b. 根据之前缓存的参数名集合，智能筛选出当前 loss_fn 真正需要的参数。
        loss_input = {
            key: value for key, value in available_data.items() 
            if key in self._loss_fn_arg_names
        }
        
        # 3c. 安全地调用损失函数
        loss_dict = self.loss_fn(**loss_input)
        
        # ... 后续逻辑保持不变 ...
        output = {"loss": loss_dict["contrastive_loss"]}
        logits_per_image = features["image_features"] @ features["text_features"].T * features["logit_scale"]
        output["logits"] = logits_per_image
        return output

    # ... training_step, validation_step, test_step, 和 configure_optimizers 保持不变 ...
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
        optimizer = self.hparams.optimizer_cfg(params=self.parameters())
        if self.trainer is None:
            return {"optimizer": optimizer}
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
                "monitor": self.hparams.get("optimized_metric", "val/loss"),
                "interval": "step",
                "frequency": 1,
            },
        }