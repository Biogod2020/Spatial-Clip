# src/data/spatial_datamodule.py

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from lightning import LightningDataModule
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.data.datasets import create_spatial_dataset

log = logging.getLogger(__name__)


class SpatialClipDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the Spatial-CLIP project.
    """
    def __init__(
        self,
        data_dir: str,
        k_neighbors: int,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        dataset_format: str = "parquet_v1",
        dataset_format_kwargs: Optional[Dict[str, Any]] = None,
        splits: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "train"
        self.val_path = self.data_dir / "val"
        
        # 这些将在 setup() 中被赋值
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        
        # model-specific, will be set by LightningModule
        self.preprocess_fn: Optional[Callable] = None
        self.tokenizer: Optional[Callable] = None

        self.dataset_format = dataset_format
        self.dataset_format_kwargs = self._to_container(dataset_format_kwargs)
        default_splits = {"train": "train", "val": "val", "test": None}
        user_splits = self._to_container(splits) if splits else {}
        self.splits = {**default_splits, **user_splits}

    def prepare_data(self) -> None:
        """
        一次性数据准备。在我们的工作流中，这个步骤已经在 notebooks 中手动完成。
        这里我们只检查路径是否存在。
        """
        if self.dataset_format in {"parquet", "parquet_v1"}:
            missing: List[Path] = []
            for split_name in ("train", "val"):
                spec = self.splits.get(split_name)
                if isinstance(spec, str):
                    candidate = self.data_dir / spec
                    if not candidate.exists():
                        missing.append(candidate)
            if missing:
                raise FileNotFoundError(
                    "Missing parquet dataset splits: " + ", ".join(str(p) for p in missing)
                )
        else:
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Dataset directory '{self.data_dir}' not found.")
        log.info("Dataset paths verified for format %s", self.dataset_format)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        在每个 GPU 进程上加载数据。
        """
        # 确保模型模块已经设置了预处理器和分词器
        if self.preprocess_fn is None or self.tokenizer is None:
            raise ValueError("DataModule requires preprocess_fn and tokenizer to be set before setup().")
            
        if stage == "fit" or stage is None:
            if not self.data_train:
                log.info("Setting up train dataset (%s)", self.dataset_format)
                self.data_train = self._build_dataset("train")
            if not self.data_val:
                log.info("Setting up validation dataset (%s)", self.dataset_format)
                self.data_val = self._build_dataset("val")

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        # 如果有测试集，可以在这里实现
        return self._create_dataloader(self.data_val, shuffle=False)
        
    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        自定义的 collate 函数，将 list of dicts 转换为一个 dict of tensors。
        """
        images = torch.stack([item['image'] for item in batch])
        texts = torch.stack([item['text'] for item in batch])
        
        anchor_tile_ids = [item['anchor_tile_id'] for item in batch]
        neighbor_tile_ids = [item['neighbor_tile_ids'] for item in batch]
        neighbor_alphas = [item['neighbor_alphas'] for item in batch]

        batch_dict = {
            "images": images,
            "texts": texts,
            "image_tile_ids": torch.tensor(anchor_tile_ids, dtype=torch.long),
            "text_tile_ids": torch.tensor(anchor_tile_ids, dtype=torch.long), # 在我们的对称设置中，它们是相同的
            "neighbor_tile_ids": torch.tensor(neighbor_tile_ids, dtype=torch.long),
            "neighbor_alphas": torch.tensor(neighbor_alphas, dtype=torch.float32),
        }

        if 'raw_text' in batch[0]:
            batch_dict['raw_text'] = [item['raw_text'] for item in batch]

        if 'rank_weighted_vector' in batch[0] and batch[0]['rank_weighted_vector'].numel() > 0:
            batch_dict['rank_weighted_vector'] = torch.stack([item['rank_weighted_vector'] for item in batch])

        return batch_dict

    def _build_dataset(self, split_name: str) -> Dataset:
        split_spec = self.splits.get(split_name)
        if split_spec is None:
            raise ValueError(f"No split specification provided for '{split_name}'")
        return create_spatial_dataset(
            format_name=self.dataset_format,
            data_dir=self.data_dir,
            split_name=split_name,
            split_spec=split_spec,
            k_neighbors=self.hparams.k_neighbors,
            preprocess_fn=self.preprocess_fn,
            tokenizer=self.tokenizer,
            format_kwargs=self.dataset_format_kwargs,
        )

    @staticmethod
    def _to_container(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if data is None:
            return {}
        if isinstance(data, dict):
            return data
        return OmegaConf.to_container(data, resolve=True)