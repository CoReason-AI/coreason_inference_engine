# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

from collections.abc import AsyncGenerator
from typing import Any, Protocol

from coreason_manifest.spec.ontology import (
    ActionSpaceManifest,
    AgentNodeProfile,
    AnyIntent,
    EpistemicLedgerState,
    LatentScratchpadReceipt,
    PeftAdapterContract,
    TokenBurnReceipt,
)


class InferenceConvergenceError(Exception):
    """Raised when the LLM fails to converge on a valid topological structure after max retries."""


class InferenceEngineProtocol(Protocol):
    """Protocol for the core inference engine."""

    async def generate_intent(
        self,
        node: AgentNodeProfile,
        ledger: EpistemicLedgerState,
        node_id: str,
        action_space: ActionSpaceManifest,
    ) -> tuple[AnyIntent, TokenBurnReceipt, LatentScratchpadReceipt | None]:
        """
        Translates the passive ledger into active generation.
        Executes Context Hydration, the Forward Pass, and System 2 Remediation.

        Raises:
            InferenceConvergenceError: If max_loops are exceeded or upstream API fatally fails.
            asyncio.CancelledError: If preempted by the Orchestrator.
        """
        ...


class LLMAdapterProtocol(Protocol):
    rate_card: Any
    """Protocol for LLM provider adapters."""

    def count_tokens(self, text: str) -> int:
        """Synchronously calculates the exact token mass of a string."""
        ...

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translates JSON Schema dictionaries into the provider's native tool array format."""
        ...

    async def apply_peft_adapters(self, adapters: list[PeftAdapterContract]) -> None:
        """Hot-swaps declarative LoRA/PEFT weights into VRAM on compatible local backends."""
        ...

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        logit_biases: dict[int, float] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
        """
        Yields chunked string deltas and an optional usage dictionary.
        The final yield MUST contain the {"input_tokens": x, "output_tokens": y} mapping.
        """
        ...
