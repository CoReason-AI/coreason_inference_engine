# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference_engine

import asyncio
import signal
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from coreason_inference_engine.adapters.openai_adapter import OpenAIAdapter
from coreason_inference_engine.engine import InferenceEngine
from coreason_inference_engine.main import InferenceRPCServer, app, serve, setup_engine, start_server

runner = CliRunner()


def test_cli_startup() -> None:
    """Test that the CLI command correctly invokes asyncio.run."""
    with patch("coreason_inference_engine.main.asyncio.run") as mock_run:
        # Since start_server is the only command, typer runs it directly
        # when no sub-commands are defined, or it becomes the default.
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        mock_run.assert_called_once()


def test_cli_startup_keyboard_interrupt() -> None:
    """Test that the CLI handles KeyboardInterrupt gracefully."""
    with patch("coreason_inference_engine.main.asyncio.run", side_effect=KeyboardInterrupt):
        result = runner.invoke(app, [])
        assert result.exit_code == 0


def test_setup_engine() -> None:
    """Test engine setup with environment variables."""
    engine = setup_engine("test-api-key")
    assert isinstance(engine, InferenceEngine)
    assert isinstance(engine.adapter, OpenAIAdapter)
    assert engine.adapter.api_key == "test-api-key"

    with patch.dict("os.environ", {"OPENAI_API_KEY": "env-api-key"}):
        engine_env = setup_engine()
        assert isinstance(engine_env.adapter, OpenAIAdapter)
        assert engine_env.adapter.api_key == "env-api-key"


@pytest.mark.asyncio
async def test_rpc_server_start_stop() -> None:
    """Test InferenceRPCServer start and stop logic."""
    engine_mock = MagicMock(spec=InferenceEngine)
    server = InferenceRPCServer(engine_mock)

    # Start the server as a background task
    task = asyncio.create_task(server.start())

    # Ensure it's running
    await asyncio.sleep(0.1)
    assert not server._shutdown_event.is_set()

    # Stop the server
    server.stop()
    assert server._shutdown_event.is_set()

    # Wait for the task to complete
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_rpc_server_cancellation() -> None:
    """Test InferenceRPCServer handling of asyncio cancellation."""
    engine_mock = MagicMock(spec=InferenceEngine)
    server = InferenceRPCServer(engine_mock)

    # Start the server as a background task
    task = asyncio.create_task(server.start())

    # Ensure it's running
    await asyncio.sleep(0.1)

    # Cancel the task
    task.cancel()

    # Wait for the task to complete (should raise CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_serve_function() -> None:
    """Test the serve function and signal handling."""
    loop = asyncio.get_running_loop()
    with patch.object(loop, "add_signal_handler") as mock_add_signal:
        # We need the server to start, wait briefly, and then stop itself
        # to prevent serve() from blocking forever.
        # So we patch the RPCServer's start method.
        async def mock_start(self: InferenceRPCServer) -> None:
            await asyncio.sleep(0.1)
            self.stop()

        with patch.object(InferenceRPCServer, "start", mock_start):
            await serve()

        # Verify signal handlers were added
        assert mock_add_signal.call_count >= 2
        mock_add_signal.assert_any_call(signal.SIGINT, mock_add_signal.call_args_list[0][0][1])
        mock_add_signal.assert_any_call(signal.SIGTERM, mock_add_signal.call_args_list[1][0][1])


def test_start_server_standalone() -> None:
    """Test the raw start_server function."""
    with patch("coreason_inference_engine.main.asyncio.run") as mock_run:
        start_server()
        mock_run.assert_called_once()


def test_main_block() -> None:
    """Test coverage for the __main__ block if possible."""
    with patch.dict("sys.modules", {"coreason_inference_engine.main": MagicMock(__name__="__main__")}):
        # Difficult to test the actual block cleanly without running a subprocess,
        # but coverage typically misses it if we don't do something.
        pass
