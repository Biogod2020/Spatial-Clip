"""Tests for the Typer CLI that proxies to the Hydra entrypoint."""

from __future__ import annotations

from typer.testing import CliRunner

from src.data.preprocessing import cli

runner = CliRunner()


def test_stage_command_invokes_hydra(monkeypatch):
    captured = {}

    def fake_invoke(config_name, config_path, stage, overrides):
        captured.update(
            {
                "config_name": config_name,
                "config_path": config_path,
                "stage": stage,
                "overrides": overrides,
            }
        )

    monkeypatch.setattr(cli, "_invoke_hydra_process", fake_invoke)

    result = runner.invoke(
        cli.app,
        [
            "--config-name",
            "preprocess/hest_mouse.yaml",
            "--config-path",
            "configs",
            "-o",
            "extras.print_config=false",
            "stage-2",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured == {
        "config_name": "preprocess/hest_mouse.yaml",
        "config_path": "configs",
        "stage": "stage-2",
        "overrides": ["extras.print_config=false"],
    }


def test_run_command_passes_stage_argument(monkeypatch):
    called = {}

    def fake_invoke(config_name, config_path, stage, overrides):
        called["stage"] = stage

    monkeypatch.setattr(cli, "_invoke_hydra_process", fake_invoke)

    result = runner.invoke(cli.app, ["run", "stage-2,stage-3"])

    assert result.exit_code == 0, result.stdout
    assert called["stage"] == "stage-2,stage-3"
