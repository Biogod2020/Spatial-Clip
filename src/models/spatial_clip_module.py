# src/models/spatial_clip_module.py
import inspect  # <-- 1. 导入 inspect 模块
from typing import Any, Dict, Optional
from pathlib import Path

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.nn import Module as LossFunction
from torchmetrics import MetricCollection

from .components.spatial_clip_net import SpatialClipNet
from src.metrics.zero_shot import ZeroShotGeneExpressionMetric


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
        global_hvg_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True, ignore=["net", "loss_fn"])
        self.net = net
        self.loss_fn = loss_fn
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        
        self.global_hvg_path = global_hvg_path
        self.zero_shot_metric = None
        self.gene_bank_embeddings = None
        
        if self.global_hvg_path:
            self.zero_shot_metric = ZeroShotGeneExpressionMetric(global_hvg_path=self.global_hvg_path)

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
        output["image_features"] = features["image_features"]
        return output

    def on_validation_start(self):
        if self.zero_shot_metric and self.gene_bank_embeddings is None:
            hvg_path = Path(self.global_hvg_path)
            if hvg_path.exists():
                with open(hvg_path, "r") as f:
                    gene_list = [line.strip() for line in f if line.strip()]
                
                if gene_list:
                    device = self.device
                    all_embeddings = []
                    batch_size = 256
                    
                    was_training = self.net.training
                    self.net.eval()
                    
                    with torch.no_grad():
                        for i in range(0, len(gene_list), batch_size):
                            batch_genes = gene_list[i:i+batch_size]
                            text_tokens = self.net.tokenizer(batch_genes).to(device)
                            embeddings = self.net.model.encode_text(text_tokens, normalize=True)
                            all_embeddings.append(embeddings)
                    
                    self.gene_bank_embeddings = torch.cat(all_embeddings, dim=0)
                    
                    if was_training:
                        self.net.train()
            else:
                print(f"Warning: Global HVG path {hvg_path} not found.")

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

        if self.zero_shot_metric and self.gene_bank_embeddings is not None:
            if "raw_text" in batch:
                raw_texts = batch["raw_text"]
                image_features = output["image_features"]
                logits = image_features @ self.gene_bank_embeddings.T
                self.zero_shot_metric.update(logits, raw_texts)
                self.log("val/zero_shot_pcc", self.zero_shot_metric, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        output = self.model_step(batch)
        self.log("test/loss", output["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics(output["logits"], torch.arange(len(output["logits"]), device=self.device))
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, sync_dist=True)

        if self.zero_shot_metric and self.gene_bank_embeddings is not None:
            if "raw_text" in batch:
                raw_texts = batch["raw_text"]
                image_features = output["image_features"]
                logits = image_features @ self.gene_bank_embeddings.T
                self.zero_shot_metric.update(logits, raw_texts)
                self.log("test/zero_shot_pcc", self.zero_shot_metric, on_step=False, on_epoch=True, sync_dist=True)
        
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