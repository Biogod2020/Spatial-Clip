"""Unit tests for the Hydra-native preprocessing entrypoint."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.data.preprocessing import hydra_entry


def test_run_executes_all_stages_for_full_pipeline(monkeypatch):
    call_order: list[str] = []

    def _fake_stage(name: str, manifest_suffix: str):
        def _impl(_cfg):
            call_order.append(name)
            return {"manifest_path": f"/tmp/{manifest_suffix}"}

        return _impl

    fake_stage1 = _fake_stage("stage_1", "stage1.json")
    fake_stage2 = _fake_stage("stage_2", "stage2.json")
    fake_stage3 = _fake_stage("stage_3", "stage3.json")

    monkeypatch.setattr(hydra_entry, "stage_01_merge_and_align", fake_stage1)
    monkeypatch.setattr(hydra_entry, "stage_02_normalize_and_filter", fake_stage2)
    monkeypatch.setattr(hydra_entry, "stage_03_create_shards", fake_stage3)
    monkeypatch.setitem(hydra_entry._STAGE_FUNCS, "stage_1", fake_stage1)
    monkeypatch.setitem(hydra_entry._STAGE_FUNCS, "stage_2", fake_stage2)
    monkeypatch.setitem(hydra_entry._STAGE_FUNCS, "stage_3", fake_stage3)

    cfg = OmegaConf.create({"run": {"stage": "full-pipeline"}})
    result = hydra_entry.run(cfg)

    assert call_order == ["stage_1", "stage_2", "stage_3"]
    assert result == {"manifest_path": "/tmp/stage3.json"}


def test_run_accepts_comma_separated_stage_list(monkeypatch):
    call_order: list[str] = []

    def fake_stage2(_cfg):
        call_order.append("stage_2")

    def fake_stage3(_cfg):
        call_order.append("stage_3")

    monkeypatch.setitem(hydra_entry._STAGE_FUNCS, "stage_2", fake_stage2)
    monkeypatch.setitem(hydra_entry._STAGE_FUNCS, "stage_3", fake_stage3)

    cfg = OmegaConf.create({"run": {"stage": "stage-2, stage-3"}})
    hydra_entry.run(cfg)

    assert call_order == ["stage_2", "stage_3"]


def test_run_raises_for_unknown_stage():
    cfg = OmegaConf.create({"run": {"stage": "unknown"}})
    with pytest.raises(ValueError):
        hydra_entry.run(cfg)
