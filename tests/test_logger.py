import importlib
from pathlib import Path

import pytest


def test_logger_setup_creates_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    import coreason_inference_engine.utils.logger

    # Ensure all file handles are closed before reloading
    coreason_inference_engine.utils.logger.logger.remove()

    importlib.reload(coreason_inference_engine.utils.logger)
    assert (tmp_path / "logs").exists()


def test_logger_setup_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    import coreason_inference_engine.utils.logger

    # Ensure all file handles are closed before reloading
    coreason_inference_engine.utils.logger.logger.remove()

    importlib.reload(coreason_inference_engine.utils.logger)
    assert log_path.exists()
