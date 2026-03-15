import importlib
import shutil
from pathlib import Path


def test_logger_setup_creates_dir() -> None:
    import coreason_inference_engine.utils.logger

    log_path = Path("logs")
    if log_path.exists():
        shutil.rmtree(log_path)

    importlib.reload(coreason_inference_engine.utils.logger)
    assert log_path.exists()


def test_logger_setup_exists() -> None:
    import coreason_inference_engine.utils.logger

    log_path = Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)

    importlib.reload(coreason_inference_engine.utils.logger)
    assert log_path.exists()
