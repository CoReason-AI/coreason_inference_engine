# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from coreason_manifest.spec.ontology import (
    ActionSpaceManifest,
    AgentNodeProfile,
    EpistemicLedgerState,
    PeftAdapterContract,
    SelfCorrectionPolicy,
)

from coreason_inference_engine.engine import InferenceEngine
from coreason_inference_engine.interfaces import InferenceConvergenceError, LLMAdapterProtocol


class DummyAdapter(LLMAdapterProtocol):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.applied_peft = False
        self.tools_projected = False

    def count_tokens(self, _text: str) -> int:
        return len(_text)

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.tools_projected = True
        return schemas

    async def apply_peft_adapters(self, _adapters: list[PeftAdapterContract]) -> None:
        self.applied_peft = True

    async def generate_stream(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
        _temperature: float,
        _logit_biases: dict[int, float] | None = None,
    ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
        if self.call_count >= len(self.responses):
            yield "", {"input_tokens": 0, "output_tokens": 0}
            return

        resp = self.responses[self.call_count]
        self.call_count += 1

        yield resp, {"input_tokens": 10, "output_tokens": 10}


@pytest.fixture
def mock_node() -> AgentNodeProfile:
    return AgentNodeProfile(
        description="Test node",
        type="agent",
        correction_policy=SelfCorrectionPolicy(max_loops=2, rollback_on_failure=True),
    )


@pytest.fixture
def mock_ledger() -> EpistemicLedgerState:
    return EpistemicLedgerState(history=[])


@pytest.fixture
def mock_action_space() -> ActionSpaceManifest:
    return ActionSpaceManifest(
        action_space_id="test_space",
        native_tools=[],
    )


@pytest.mark.asyncio
async def test_successful_generation(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    # We will simulate the LLM responding with a valid Intent
    # InformationalIntent requires a message
    valid_intent_json = '{"type": "informational", "message": "hello", "timeout_action": "proceed_default"}'

    adapter = DummyAdapter(responses=[valid_intent_json])
    engine = InferenceEngine(adapter)

    intent, receipt, _scratchpad = await engine.generate_intent(
        node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
    )

    assert intent.type == "informational"
    # Receipt accumulation test
    assert receipt.input_tokens == 10
    assert receipt.output_tokens == 10
    assert receipt.burn_magnitude == 20
    assert adapter.tools_projected is True


@pytest.mark.asyncio
async def test_remediation_loop_success(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    # 1st response: Invalid (missing 'message' field for InformationalIntent)
    invalid_intent_json = '{"type": "informational"}'
    # 2nd response: Valid
    valid_intent_json = '{"type": "informational", "message": "fixed", "timeout_action": "proceed_default"}'

    adapter = DummyAdapter(responses=[invalid_intent_json, valid_intent_json])
    engine = InferenceEngine(adapter)

    intent, receipt, _scratchpad = await engine.generate_intent(
        node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
    )

    assert intent.type == "informational"
    assert intent.message == "fixed"
    # 1st call: 10 in, 10 out. 2nd call: 10 in, 10 out. Total 20/20.
    assert receipt.input_tokens == 20
    assert receipt.output_tokens == 20
    assert receipt.burn_magnitude == 40


@pytest.mark.asyncio
async def test_remediation_loop_failure(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    # Continually failing responses
    invalid_intent_json = '{"type": "informational", "level": "info"}'

    adapter = DummyAdapter(responses=[invalid_intent_json, invalid_intent_json, invalid_intent_json])
    engine = InferenceEngine(adapter)

    with pytest.raises(InferenceConvergenceError, match="failed to converge after 2 attempts"):
        await engine.generate_intent(
            node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
        )


class CancellingAdapter(LLMAdapterProtocol):
    def count_tokens(self, _text: str) -> int:
        return 0

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return schemas

    async def apply_peft_adapters(self, _adapters: list[PeftAdapterContract]) -> None:
        pass

    async def generate_stream(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
        _temperature: float,
        _logit_biases: dict[int, float] | None = None,
    ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
        yield "start", {"input_tokens": 5, "output_tokens": 0}
        raise asyncio.CancelledError("Preempted")


@pytest.mark.asyncio
async def test_zero_leak_cancellation(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    adapter = CancellingAdapter()
    engine = InferenceEngine(adapter)

    with pytest.raises(asyncio.CancelledError):
        await engine.generate_intent(
            node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
        )


@pytest.mark.asyncio
async def test_peft_adapters_applied(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    # Use copy/update instead of mutate
    peft_adapters = [
        PeftAdapterContract(
            adapter_id="lora-123",
            safetensors_hash="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            base_model_hash="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            adapter_rank=8,
            target_modules=["q_proj"],
        )
    ]

    node_with_peft = mock_node.model_copy(update={"peft_adapters": peft_adapters})

    valid_intent_json = '{"type": "informational", "message": "hello", "timeout_action": "proceed_default"}'
    adapter = DummyAdapter(responses=[valid_intent_json])
    engine = InferenceEngine(adapter)

    await engine.generate_intent(
        node=node_with_peft, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
    )

    assert adapter.applied_peft is True
