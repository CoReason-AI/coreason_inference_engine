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
import contextlib
import os
import signal

import typer

from coreason_inference_engine.adapters.openai_adapter import OpenAIAdapter
from coreason_inference_engine.engine import InferenceEngine
from coreason_inference_engine.utils.logger import logger

app = typer.Typer(help="CoReason Inference Engine - Cognitive Plane Microservice")


class InferenceRPCServer:
    """An asynchronous RPC server or broker subscriber that listens for inference requests."""

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Starts the server and listens for requests continuously until shutdown is requested."""
        logger.info("Starting InferenceRPCServer...")

        # In a real implementation, this would connect to a message broker (e.g., RabbitMQ, Kafka)
        # or start an HTTP/gRPC server. For now, we simulate a listening loop.
        try:
            while not self._shutdown_event.is_set():
                # Mock listening for incoming requests
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("InferenceRPCServer task cancelled.")
            raise
        finally:
            logger.info("InferenceRPCServer shutdown complete.")

    def stop(self) -> None:
        """Signals the server to stop gracefully."""
        logger.info("Shutdown signal received. Stopping InferenceRPCServer...")
        self._shutdown_event.set()


def setup_engine(api_key: str | None = None) -> InferenceEngine:
    """Initializes the InferenceEngine with the correct LLMAdapter."""
    # Pull the API key from environment variables if not provided
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "mock-api-key")

    # Initialize the adapter. Using OpenAIAdapter as a fallback/default.
    adapter = OpenAIAdapter(api_url="https://api.openai.com/v1/chat/completions", api_key=api_key, model_name="gpt-4o")

    # Initialize and return the Inference Engine
    return InferenceEngine(adapter=adapter)


async def serve() -> None:
    """Asynchronous entry point that sets up the engine, server, and signal handlers."""
    engine = setup_engine()
    server = InferenceRPCServer(engine=engine)

    # Setup graceful shutdown signal handlers
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, server.stop)

    with contextlib.suppress(asyncio.CancelledError):
        await server.start()


@app.command()  # type: ignore[misc]
def start_server() -> None:
    """Starts the Cognitive Plane Inference Engine microservice."""
    logger.info("Booting CoReason Inference Engine...")
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")


if __name__ == "__main__":  # pragma: no cover
    app()
