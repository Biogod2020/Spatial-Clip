"""Smoke tests for the preprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from anndata import AnnData
from omegaconf import OmegaConf

from src.data.preprocessing import pipeline


@pytest.fixture()
def smoke_cfg(tmp_path) -> OmegaConf:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    intermediate_dir = tmp_path / "processed_intermediate"
    output_dir = tmp_path / "processed"
    hvgs = tmp_path / "global_hvgs.txt"
    hvgs.write_text("g1\ng2\ng3\n", encoding="utf-8")
    hgnc = tmp_path / "hgnc.txt"
    hgnc.write_text("symbol\talias\n", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "dataset": {"key": "smoke_ds", "description": "Synthetic test dataset"},
            "source": {
                "raw_data_dir": str(raw_dir),
                "hgnc_path": str(hgnc),
                "global_hvg_path": str(hvgs),
            },
            "intermediate_dir": str(intermediate_dir),
            "output_dir": str(output_dir),
            "params": {
                "general": {"batch_key": "sample_id", "species_filter": "test"},
                "samples_to_exclude": [],
                "gene_alignment": {"keep_status": ["Approved"], "keep_locus_types": None},
                "sentence_generation": {"n_top_genes": 3},
                "sharding": {"max_samples_per_shard": 10},
                "tiling": {"tile_size": 32},
            },
            "performance": {"max_workers": 1, "limit_samples": -1},
        }
    )
    return cfg


def test_pipeline_emits_manifest(tmp_path, smoke_cfg, monkeypatch):
    class DummyDataset:
        def __init__(self, data_dir: str):
            self.data_dir = data_dir

        def get_samples(self, species: str | None = None):
            return [SimpleNamespace(sample_id="sample_a"), SimpleNamespace(sample_id="sample_b")]

    def fake_load_single_sample_adata(sample, batch_key: str):
        adata = AnnData(np.array([[1, 0, 0], [0, 1, 0]], dtype=float))
        adata.var_names = ["g1", "g2", "g3"]
        adata.obs_names = [f"{sample.sample_id}_0", f"{sample.sample_id}_1"]
        adata.obs[batch_key] = [sample.sample_id] * adata.n_obs
        return adata

    def fake_align_and_collapse_genes(adata, *_args, **_kwargs):
        return adata

    def fake_load_hgnc_resources(*_args, **_kwargs):
        return {"g1", "g2", "g3"}, {}

    def fake_normalize_adata(_adata):
        return None

    def fake_process_one_sample(sample_id, adata_sample, cfg):
        sample_dir = Path(cfg.output_dir) / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        shard_path = sample_dir / f"{sample_id}_000000.tar"
        shard_path.write_bytes(b"dummy")
        (sample_dir / f"{sample_id}.json").write_text(
            json.dumps({"spots": int(adata_sample.n_obs)}),
            encoding="utf-8",
        )
        return {"processed": int(adata_sample.n_obs), "failed": 0}

    monkeypatch.setattr(pipeline, "HESTDataset", DummyDataset)
    monkeypatch.setattr(pipeline, "load_single_sample_adata", fake_load_single_sample_adata)
    monkeypatch.setattr(pipeline, "align_and_collapse_genes", fake_align_and_collapse_genes)
    monkeypatch.setattr(pipeline, "load_hgnc_resources", fake_load_hgnc_resources)
    monkeypatch.setattr(pipeline, "normalize_adata", fake_normalize_adata)
    monkeypatch.setattr(pipeline, "_process_one_sample", fake_process_one_sample)

    pipeline.stage_01_merge_and_align(smoke_cfg)
    pipeline.stage_02_normalize_and_filter(smoke_cfg)
    stats = pipeline.stage_03_create_shards(smoke_cfg)

    manifest_path = Path(smoke_cfg.output_dir) / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stats"]["total_processed"] == stats["total_processed"]
    assert manifest["outputs"]["shard_count"] == 2
    assert set(manifest["outputs"]["sample_dirs"]) == {"sample_a", "sample_b"}
