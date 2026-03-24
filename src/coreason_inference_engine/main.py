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
from typing import Any

import typer
import uvicorn
from coreason_manifest.spec.ontology import (
    ActionSpaceManifest,
    AgentNodeProfile,
    EpistemicLedgerState,
)
from fastapi import FastAPI, HTTPException, Request

from coreason_inference_engine.adapters.anthropic_adapter import AnthropicAdapter
from coreason_inference_engine.adapters.openai_adapter import OpenAIAdapter
from coreason_inference_engine.engine import InferenceEngine
from coreason_inference_engine.interfaces import InferenceConvergenceError, LLMAdapterProtocol
from coreason_inference_engine.utils.logger import logger

app = typer.Typer(help="CoReason Inference Engine - Cognitive Plane Microservice")


class InferenceRPCServer:
    """An asynchronous HTTP server that listens for inference requests."""

    def __init__(self, engine: InferenceEngine, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self.fastapi_app = FastAPI(title="CoReason Inference Engine")
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.fastapi_app.post("/v1/intent/generate")
        async def generate_intent(request: Request) -> Any:
            try:
                data = await request.json()
                node = AgentNodeProfile.model_validate(data.get("node"))
                ledger = EpistemicLedgerState.model_validate(data.get("ledger"))
                action_space = ActionSpaceManifest.model_validate(data.get("action_space"))
                node_id = data.get("node_id")

                if not node_id:
                    raise HTTPException(status_code=400, detail="node_id is required")

                intent, receipt, scratchpad, cognitive_receipt = await self.engine.generate_intent(
                    node=node.model_dump(),
                    ledger=ledger.model_dump(),
                    node_id=node_id,
                    action_space=action_space.model_dump(),
                )

                return {
                    "intent": intent if intent else None,
                    "receipt": receipt if receipt else None,
                    "scratchpad": scratchpad if scratchpad else None,
                    "cognitive_receipt": cognitive_receipt if cognitive_receipt else None,
                }
            except HTTPException:
                raise
            except InferenceConvergenceError as e:
                raise HTTPException(status_code=422, detail=str(e)) from e
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error") from e

    async def start(self) -> None:
        """Starts the FastAPI server continuously until shutdown is requested."""
        logger.info("Starting InferenceRPCServer...")

        config = uvicorn.Config(
            app=self.fastapi_app,
            host=self.host,
            port=self.port,
            log_level="info",
            loop="asyncio",
        )
        self.server = uvicorn.Server(config)

        try:
            await self.server.serve()
        except asyncio.CancelledError:
            logger.info("InferenceRPCServer task cancelled.")
            raise
        finally:
            logger.info("InferenceRPCServer shutdown complete.")

    def stop(self) -> None:
        """Signals the server to stop gracefully."""
        logger.info("Shutdown signal received. Stopping InferenceRPCServer...")
        if hasattr(self, "server"):
            self.server.should_exit = True


def setup_engine(api_key: str | None = None) -> InferenceEngine:
    """Initializes the InferenceEngine with the correct LLMAdapter."""
    provider = os.environ.get("COREASON_LLM_PROVIDER", "openai").lower()

    adapter: LLMAdapterProtocol
    if provider == "anthropic":
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "mock-api-key")
        model = os.environ.get("COREASON_LLM_MODEL", "claude-3-5-sonnet-20241022")
        adapter = AnthropicAdapter(
            api_url="https://api.anthropic.com/v1/messages",
            api_key=api_key,
            model_name=model,
        )
    else:
        # Default to OpenAI
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "mock-api-key")
        model = os.environ.get("COREASON_LLM_MODEL", "gpt-4o")
        adapter = OpenAIAdapter(
            api_url="https://api.openai.com/v1/chat/completions",
            api_key=api_key,
            model_name=model,
        )

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


@app.command()
def start_server() -> None:
    """Starts the Cognitive Plane Inference Engine microservice."""
    logger.info("Booting CoReason Inference Engine...")
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")


if __name__ == "__main__":  # pragma: no cover
    app()
