"""Tests for agentschool.config.paths."""

from __future__ import annotations

from pathlib import Path

from agentschool.config.paths import (
    get_config_dir,
    get_config_file_path,
    get_data_dir,
    get_logs_dir,
)


def test_get_config_dir_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AGENTSCHOOL_CONFIG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    config_dir = get_config_dir()
    assert config_dir == tmp_path / ".agentschool"
    assert config_dir.is_dir()


def test_get_config_dir_env_override(tmp_path: Path, monkeypatch):
    custom = tmp_path / "custom_config"
    monkeypatch.setenv("AGENTSCHOOL_CONFIG_DIR", str(custom))
    config_dir = get_config_dir()
    assert config_dir == custom
    assert config_dir.is_dir()


def test_get_config_file_path(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AGENTSCHOOL_CONFIG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = get_config_file_path()
    assert path == tmp_path / ".agentschool" / "settings.json"


def test_get_data_dir_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AGENTSCHOOL_CONFIG_DIR", raising=False)
    monkeypatch.delenv("AGENTSCHOOL_DATA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    data_dir = get_data_dir()
    assert data_dir == tmp_path / ".agentschool" / "data"
    assert data_dir.is_dir()


def test_get_data_dir_env_override(tmp_path: Path, monkeypatch):
    custom = tmp_path / "custom_data"
    monkeypatch.setenv("AGENTSCHOOL_DATA_DIR", str(custom))
    data_dir = get_data_dir()
    assert data_dir == custom
    assert data_dir.is_dir()


def test_get_logs_dir_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AGENTSCHOOL_CONFIG_DIR", raising=False)
    monkeypatch.delenv("AGENTSCHOOL_LOGS_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    logs_dir = get_logs_dir()
    assert logs_dir == tmp_path / ".agentschool" / "logs"
    assert logs_dir.is_dir()


def test_get_logs_dir_env_override(tmp_path: Path, monkeypatch):
    custom = tmp_path / "custom_logs"
    monkeypatch.setenv("AGENTSCHOOL_LOGS_DIR", str(custom))
    logs_dir = get_logs_dir()
    assert logs_dir == custom
    assert logs_dir.is_dir()
