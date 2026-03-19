# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock

import pytest
from coreason_manifest.spec.ontology import (
    ActionSpaceManifest,
    AgentNodeProfile,
    AnyIntent,
    CognitiveRewardEvaluationReceipt,
    EpistemicLedgerState,
    LatentScratchpadReceipt,
    PeftAdapterContract,
    TokenBurnReceipt,
)

from coreason_inference_engine.interfaces import (
    InferenceConvergenceError,
    InferenceEngineProtocol,
    LLMAdapterProtocol,
)


def test_inference_convergence_error() -> None:
    """Test that the exception can be raised with a custom message."""
    error_msg = "LLM failed to converge after 3 attempts."
    with pytest.raises(InferenceConvergenceError, match=error_msg):
        raise InferenceConvergenceError(error_msg)


class DummyInferenceEngine:
    """A dummy implementation of the InferenceEngineProtocol for testing."""

    async def generate_intent(
        self,
        node: AgentNodeProfile,
        ledger: EpistemicLedgerState,
        node_id: str,
        action_space: ActionSpaceManifest,
    ) -> tuple[AnyIntent, TokenBurnReceipt, LatentScratchpadReceipt | None, CognitiveRewardEvaluationReceipt | None]:
        # Using typing.cast or similar is difficult because we have AnyIntent. We'll raise to prove invocation.
        raise NotImplementedError("Dummy implementation")


class DummyLLMAdapter:
    rate_card: Any = None
    """A dummy implementation of the LLMAdapterProtocol for testing."""

    def count_tokens(self, text: str) -> int:
        return len(text)

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return schemas

    async def apply_peft_adapters(self, adapters: list[PeftAdapterContract]) -> None:
        pass

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        logit_biases: dict[int, float] | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
        # Workaround unused args
        _ = messages
        _ = tools
        _ = temperature
        _ = logit_biases
        _ = max_tokens
        yield "dummy_chunk", {"input_tokens": 10, "output_tokens": 5}


@pytest.mark.asyncio
async def test_inference_engine_protocol() -> None:
    """Verify that a class correctly implementing the protocol works as expected."""
    engine: InferenceEngineProtocol = DummyInferenceEngine()

    node = MagicMock(spec=AgentNodeProfile)
    ledger = MagicMock(spec=EpistemicLedgerState)
    action_space = MagicMock(spec=ActionSpaceManifest)

    with pytest.raises(NotImplementedError, match="Dummy implementation"):
        await engine.generate_intent(
            node=node,
            ledger=ledger,
            node_id="did:example:123",
            action_space=action_space,
        )


@pytest.mark.asyncio
async def test_llm_adapter_protocol() -> None:
    """Verify that a class correctly implementing the protocol works as expected."""
    adapter: LLMAdapterProtocol = DummyLLMAdapter()

    assert adapter.count_tokens("hello") == 5
    assert adapter.project_tools([{"type": "function"}]) == [{"type": "function"}]

    await adapter.apply_peft_adapters([])

    chunks = [chunk async for chunk in adapter.generate_stream(messages=[], tools=[], temperature=0.0)]

    assert len(chunks) == 1
    assert chunks[0][0] == "dummy_chunk"
    assert chunks[0][1] == {"input_tokens": 10, "output_tokens": 5}
