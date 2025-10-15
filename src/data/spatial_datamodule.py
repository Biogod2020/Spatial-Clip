# src/data/spatial_datamodule.py

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# 允许加载可能损坏的图像文件
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

log = logging.getLogger(__name__)


class SpatiallyAwareParquetDataset(Dataset):
    """
    一个专门为 SpaGLaM/Spatial-CLIP 设计的数据集，用于高效读取预处理后的 Parquet 文件。
    它在初始化时将节点和边信息加载到内存中，以实现快速的 __getitem__ 访问。
    """
    _nodes_map_cache: Dict[int, pd.Series] = {} # Worker间共享的缓存

    def __init__(
        self,
        data_path: str,
        k_neighbors: int,
        preprocess_fn: Callable,
        tokenizer: Callable,
    ):
        """
        Args:
            data_path (str): 指向包含 'nodes.parquet' 和 'edges.parquet' 的目录路径 (例如 '.../dataset_split/train').
            k_neighbors (int): 每个锚点要考虑的最大邻居数量。
            preprocess_fn (Callable): 应用于每个 PIL 图像的预处理函数。
            tokenizer (Callable): 用于将基因句子转换为 token ID 的分词器。
        """
        self.data_path = Path(data_path)
        self.k = k_neighbors
        self.preprocess = preprocess_fn
        self.tokenizer = tokenizer
        log.info(f"Initializing SpatiallyAwareParquetDataset for path: {self.data_path}")

        nodes_path = self.data_path / "nodes.parquet"
        edges_path = self.data_path / "edges.parquet"
        assert nodes_path.exists() and edges_path.exists(), f"Nodes or Edges file not found in {self.data_path}."

        self.nodes = pd.read_parquet(nodes_path)
        edges = pd.read_parquet(edges_path)
        
        # 优化：构建一个高效的 tile_id -> 邻居信息 的查找字典
        # 1. 确保每个锚点只保留最多 k 个邻居（按 alpha 排序）
        log.info(f"Performing top-k neighbor selection (k={self.k}) on {len(edges)} edges...")
        edges = edges.sort_values('alpha', ascending=False)
        top_k_edges = edges.groupby('src_tile_id').head(self.k)
        
        # 2. 将筛选后的边信息构建成字典
        self.edges_map = {
            tile_id: group.reset_index(drop=True)
            for tile_id, group in top_k_edges.groupby('src_tile_id')
        }
        log.info(f"Edges map created with {len(self.edges_map)} anchors.")

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取一个样本，包括锚点信息及其邻居信息。
        """
        anchor_info = self.nodes.iloc[idx]
        anchor_id = anchor_info['tile_id']
        
        try:
            # 加载和处理锚点图像
            anchor_image = self.preprocess(Image.open(anchor_info['image_path']).convert("RGB"))
            # 分词锚点文本
            anchor_text = self.tokenizer([anchor_info['gene_sentence']])[0]
        except Exception as e:
            log.error(f"Error loading data for anchor_id {anchor_id} at index {idx}: {e}")
            # 返回一个有效的 "空" 样本或重新尝试加载另一个样本
            # 这里我们简单地返回一个占位符，但在实际训练中可能需要更复杂的处理
            return self.__getitem__((idx + 1) % len(self))

        # 从预构建的 map 中获取邻居信息
        neighbors_df = self.edges_map.get(anchor_id, pd.DataFrame())
        
        nbr_ids = neighbors_df['nbr_tile_id'].tolist() if not neighbors_df.empty else []
        nbr_alphas = neighbors_df['alpha'].tolist() if not neighbors_df.empty else []
            
        # --- 关键步骤：填充到固定长度 k ---
        pad_count = self.k - len(nbr_ids)
        if pad_count > 0:
            nbr_ids.extend([-1] * pad_count)  # -1 作为无效邻居的标志
            nbr_alphas.extend([0.0] * pad_count)
            
        return {
            "image": anchor_image,
            "text": anchor_text,
            "anchor_tile_id": anchor_id,
            "neighbor_tile_ids": nbr_ids,
            "neighbor_alphas": nbr_alphas,
        }


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

    def prepare_data(self) -> None:
        """
        一次性数据准备。在我们的工作流中，这个步骤已经在 notebooks 中手动完成。
        这里我们只检查路径是否存在。
        """
        if not self.train_path.exists() or not self.val_path.exists():
            raise FileNotFoundError(
                f"Training path '{self.train_path}' or validation path '{self.val_path}' not found. "
                "Please run the preprocessing notebooks first to generate the split dataset."
            )
        log.info("Train and validation data paths verified.")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        在每个 GPU 进程上加载数据。
        """
        # 确保模型模块已经设置了预处理器和分词器
        if self.preprocess_fn is None or self.tokenizer is None:
            raise ValueError("DataModule requires preprocess_fn and tokenizer to be set before setup().")
            
        if stage == "fit" or stage is None:
            if not self.data_train:
                log.info("Setting up train dataset.")
                self.data_train = SpatiallyAwareParquetDataset(
                    data_path=str(self.train_path),
                    k_neighbors=self.hparams.k_neighbors,
                    preprocess_fn=self.preprocess_fn,
                    tokenizer=self.tokenizer,
                )
            if not self.data_val:
                log.info("Setting up validation dataset.")
                self.data_val = SpatiallyAwareParquetDataset(
                    data_path=str(self.val_path),
                    k_neighbors=self.hparams.k_neighbors,
                    preprocess_fn=self.preprocess_fn,
                    tokenizer=self.tokenizer,
                )

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

        return {
            "images": images,
            "texts": texts,
            "image_tile_ids": torch.tensor(anchor_tile_ids, dtype=torch.long),
            "text_tile_ids": torch.tensor(anchor_tile_ids, dtype=torch.long), # 在我们的对称设置中，它们是相同的
            "neighbor_tile_ids": torch.tensor(neighbor_tile_ids, dtype=torch.long),
            "neighbor_alphas": torch.tensor(neighbor_alphas, dtype=torch.float32),
        }