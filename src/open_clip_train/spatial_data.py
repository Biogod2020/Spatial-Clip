import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SpatiallyAwareDataset(Dataset):
    _nodes_map_cache = None

    # --- 这是新的、高效的 __init__ 方法 ---
    # --- 请用它替换掉 spatial_data.py 中的旧版本 ---
    def __init__(self, artifacts_dir: Path, k_neighbors: int, preprocess_fn: Callable, tokenizer: Callable):
        self.k = k_neighbors
        self.preprocess = preprocess_fn
        self.tokenizer = tokenizer
        logging.info(f"Initializing SpatiallyAwareDataset with k={k_neighbors}...")

        nodes_path = artifacts_dir / "nodes.parquet"
        edges_path = artifacts_dir / "edges.parquet"
        assert nodes_path.exists() and edges_path.exists(), f"Nodes or Edges file not found in {artifacts_dir}."

        self.nodes = pd.read_parquet(nodes_path)
        edges = pd.read_parquet(edges_path)
        
        # --- 优化开始 ---
        # 这是一个单一的、向量化的操作，速度极快
        logging.info(f"Performing vectorized top-k neighbor selection on {len(edges)} edges...")
        
        # 1. 首先按 alpha 降序对整个 edges DataFrame 进行排序
        edges = edges.sort_values('alpha', ascending=False)
        
        # 2. 然后对排序后的结果进行分组，并取每个组的头 k 个元素
        top_k_edges = edges.groupby('src_tile_id').head(self.k)
        
        # 3. 现在，从这个已经大幅缩小的 DataFrame 高效地构建字典
        self.edges_map = {
            tile_id: group.reset_index(drop=True)
            for tile_id, group in top_k_edges.groupby('src_tile_id')
        }
        logging.info("Vectorized selection complete. Dictionary created.")
        # --- 优化结束 ---
        
        if SpatiallyAwareDataset._nodes_map_cache is None:
            logging.info("Creating and caching nodes_map for all workers...")
            SpatiallyAwareDataset._nodes_map_cache = self.nodes.set_index('tile_id')
        
        logging.info(f"Dataset ready. Total anchors: {len(self.nodes)}.")

    def __len__(self):
        return len(self.nodes)

    # --- 修改 __getitem__ ---
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anchor_info = self.nodes.iloc[idx]
        anchor_id = anchor_info['tile_id']
        
        # 加载和处理图像
        anchor_image = self.preprocess(Image.open(anchor_info['image_path']).convert("RGB"))
        # 分词文本
        anchor_text = self.tokenizer([anchor_info['gene_sentence']])[0]

        neighbors = self.edges_map.get(anchor_id, pd.DataFrame())
        
        nbr_ids = neighbors['nbr_tile_id'].tolist() if not neighbors.empty else []
        nbr_alphas = neighbors['alpha'].tolist() if not neighbors.empty else []
            
        pad_count = self.k - len(nbr_ids)
        if pad_count > 0:
            nbr_ids.extend([-1] * pad_count)
            nbr_alphas.extend([0.0] * pad_count)
            
        return {
            "image": anchor_image,
            "text": anchor_text,
            "anchor_tile_id": anchor_id,
            "neighbor_tile_ids": nbr_ids,
            "neighbor_alphas": nbr_alphas,
        }


class SpatiallyAwareCollate:
    # --- 移除 __init__ 中的 preprocess 和 tokenizer ---
    def __init__(self, **kwargs):
        # 不再需要 preprocess_fn 和 tokenizer
        pass

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # batch 中的 'image' 和 'text' 已经是 tensors
        images = torch.stack([item['image'] for item in batch])
        texts = torch.stack([item['text'] for item in batch])
        
        anchor_tile_ids = [item['anchor_tile_id'] for item in batch]
        neighbor_tile_ids = [item['neighbor_tile_ids'] for item in batch]
        neighbor_alphas = [item['neighbor_alphas'] for item in batch]

        return {
            "images": images,
            "texts": texts,
            "image_tile_ids": torch.tensor(anchor_tile_ids, dtype=torch.long),
            "text_tile_ids": torch.tensor(anchor_tile_ids, dtype=torch.long),
            "neighbor_tile_ids": torch.tensor(neighbor_tile_ids, dtype=torch.long),
            "neighbor_alphas": torch.tensor(neighbor_alphas, dtype=torch.float32),
        }