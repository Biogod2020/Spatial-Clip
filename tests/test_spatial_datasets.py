from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.datasets import ParquetSpatialDataset, ShardedSpatialDataset, create_spatial_dataset


class _DummyTokenizer:
    def __call__(self, texts):
        return [torch.ones(8, dtype=torch.long) for _ in texts]


def _dummy_preprocess(image: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(np.array(image, dtype="float32"))
    return arr.permute(2, 0, 1)


def _write_png(path: Path, color: int) -> None:
    Image.new("RGB", (4, 4), color=(color, color, color)).save(path)


def _make_parquet_split(tmp_path: Path) -> Path:
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True)
    img0 = split_dir / "img0.png"
    img1 = split_dir / "img1.png"
    _write_png(img0, 10)
    _write_png(img1, 20)
    nodes = pd.DataFrame(
        {
            "tile_id": [1, 2],
            "image_path": [str(img0), str(img1)],
            "gene_sentence": ["gene A", "gene B"],
        }
    )
    edges = pd.DataFrame(
        {
            "src_tile_id": [1, 1, 2],
            "nbr_tile_id": [1, 2, 1],
            "alpha": [0.6, 0.4, 1.0],
        }
    )
    nodes.to_parquet(split_dir / "nodes.parquet")
    edges.to_parquet(split_dir / "edges.parquet")
    return split_dir


def _make_shard_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "processed"
    sample_dir = dataset_root / "SAMPLE_A"
    sample_dir.mkdir(parents=True, exist_ok=True)
    tar_path = sample_dir / "SAMPLE_A_000000.tar"
    with tarfile.open(tar_path, "w") as tar:
        for idx in range(3):
            base = f"SAMPLE_A_{idx:03d}"
            image = Image.new("RGB", (4, 4), color=(idx * 20, 0, 0))
            buf = BytesIO()
            image.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            txt_bytes = f"spot {idx}".encode("utf-8")
            meta = json.dumps({"sample_id": "SAMPLE_A", "x": idx * 5, "y": idx * 7}).encode("utf-8")
            for ext, payload in (("png", png_bytes), ("txt", txt_bytes), ("json", meta)):
                info = tarfile.TarInfo(name=f"{base}.{ext}")
                info.size = len(payload)
                tar.addfile(info, BytesIO(payload))
    return dataset_root


def test_parquet_dataset_roundtrip(tmp_path):
    split_dir = _make_parquet_split(tmp_path)
    dataset = ParquetSpatialDataset(
        data_path=split_dir,
        k_neighbors=2,
        preprocess_fn=_dummy_preprocess,
        tokenizer=_DummyTokenizer(),
    )
    sample = dataset[0]
    assert sample["image"].shape[0] == 3
    assert len(sample["neighbor_tile_ids"]) == 2


def test_sharded_dataset(tmp_path):
    dataset_root = _make_shard_dataset(tmp_path)
    dataset = ShardedSpatialDataset(
        dataset_root=dataset_root,
        split="train",
        sample_ids=["SAMPLE_A"],
        k_neighbors=2,
        preprocess_fn=_dummy_preprocess,
        tokenizer=_DummyTokenizer(),
        cache_dir=dataset_root / ".cache",
        rebuild_cache=True,
    )
    sample = dataset[0]
    assert sample["image"].shape[-1] == 4
    assert len(sample["neighbor_tile_ids"]) == 2


def test_factory_creates_correct_backend(tmp_path):
    split_dir = _make_parquet_split(tmp_path)
    data_dir = split_dir.parent
    dataset = create_spatial_dataset(
        format_name="parquet_v1",
        data_dir=data_dir,
        split_name="train",
        split_spec="train",
        k_neighbors=1,
        preprocess_fn=_dummy_preprocess,
        tokenizer=_DummyTokenizer(),
    )
    assert isinstance(dataset, ParquetSpatialDataset)


def test_factory_creates_sharded_backend(tmp_path):
    dataset_root = _make_shard_dataset(tmp_path)
    dataset = create_spatial_dataset(
        format_name="shards_v1",
        data_dir=dataset_root,
        split_name="train",
        split_spec=["SAMPLE_A"],
        k_neighbors=1,
        preprocess_fn=_dummy_preprocess,
        tokenizer=_DummyTokenizer(),
        format_kwargs={"cache_dir": dataset_root / ".cache", "rebuild_cache": True},
    )
    assert isinstance(dataset, ShardedSpatialDataset)