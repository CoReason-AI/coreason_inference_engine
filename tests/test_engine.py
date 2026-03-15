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
    CognitiveFormatContract,
    EpistemicLedgerState,
    EpistemicRewardModelPolicy,
    JSONRPCErrorResponseState,
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
        self.max_tokens_received: list[int | None] = []

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
        temperature: float,
        logit_biases: dict[int, float] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
        _ = temperature
        _ = logit_biases
        self.max_tokens_received.append(max_tokens)
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
async def test_economic_dos_token_clamping(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    """Verifies that the remediation loop injects a strict max_tokens clamp (e.g., 500) to prevent Economic DoS."""
    # 1st response: Invalid (triggers remediation)
    invalid_intent_json = '{"type": "informational"}'
    # 2nd response: Valid
    valid_intent_json = '{"type": "informational", "message": "fixed", "timeout_action": "proceed_default"}'

    adapter = DummyAdapter(responses=[invalid_intent_json, valid_intent_json])
    engine = InferenceEngine(adapter)

    await engine.generate_intent(
        node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
    )

    assert len(adapter.max_tokens_received) == 2
    # First attempt: None
    assert adapter.max_tokens_received[0] is None
    # Second attempt (remediation): strictly clamped to 500
    assert adapter.max_tokens_received[1] == 500


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


class MockAsyncGenerator:
    def __init__(self) -> None:
        self.aclose_called = False

    def __aiter__(self) -> "MockAsyncGenerator":
        return self

    async def __anext__(self) -> tuple[str, dict[str, int]]:
        raise asyncio.CancelledError("Preempted")

    async def aclose(self) -> None:
        self.aclose_called = True


class CancellingAdapter(LLMAdapterProtocol):
    def __init__(self) -> None:
        self.mock_generator = MockAsyncGenerator()

    def count_tokens(self, _text: str) -> int:
        return 0

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return schemas

    async def apply_peft_adapters(self, _adapters: list[PeftAdapterContract]) -> None:
        pass

    def generate_stream(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
        temperature: float,
        logit_biases: dict[int, float] | None = None,
        max_tokens: int | None = None,
    ) -> Any:  # Returns the MockAsyncGenerator which has async aclose and async for protocol
        _ = temperature
        _ = logit_biases
        _ = max_tokens
        return self.mock_generator


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

    assert adapter.mock_generator.aclose_called is True


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


@pytest.fixture
def mock_node_with_think() -> AgentNodeProfile:
    return AgentNodeProfile(
        description="Test node with think tags",
        type="agent",
        correction_policy=SelfCorrectionPolicy(max_loops=2, rollback_on_failure=True),
        grpo_reward_policy=EpistemicRewardModelPolicy(
            policy_id="policy_1",
            reference_graph_id="graph_1",
            format_contract=CognitiveFormatContract(require_think_tags=True),
            beta_path_weight=0.5,
        ),
    )


@pytest.mark.asyncio
async def test_extract_latent_traces_with_tags(
    mock_node_with_think: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    raw_response = (
        "<think>\nThis is a reasoning trace.\n</think>\n"
        '```json\n{"type": "informational", "message": "hello", "timeout_action": "proceed_default"}\n```'
    )
    adapter = DummyAdapter(responses=[raw_response])
    engine = InferenceEngine(adapter)

    intent, _receipt, scratchpad = await engine.generate_intent(
        node=mock_node_with_think, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
    )

    assert intent.type == "informational"
    assert scratchpad is not None
    assert scratchpad.total_latent_tokens == len("This is a reasoning trace.")
    assert len(scratchpad.explored_branches) == 1
    branch = scratchpad.explored_branches[0]
    import hashlib

    expected_hash = hashlib.sha256(b"This is a reasoning trace.").hexdigest()
    assert branch.latent_content_hash == expected_hash


@pytest.mark.asyncio
async def test_extract_latent_traces_missing_tags_but_required(
    mock_node_with_think: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    # Tags are missing, should return raw_output. Since the payload is not valid JSON
    # (or maybe it is, let's just make it invalid to trigger validation error).
    # Actually, we should check that the raw_output is returned unaltered, and we can check that it succeeds.
    raw_response = '{"type": "informational", "message": "hello", "timeout_action": "proceed_default"}'
    adapter = DummyAdapter(responses=[raw_response])
    engine = InferenceEngine(adapter)

    intent, _receipt, scratchpad = await engine.generate_intent(
        node=mock_node_with_think, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
    )

    assert intent.type == "informational"
    assert scratchpad is None


@pytest.mark.asyncio
async def test_extract_latent_traces_no_tags_required(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    raw_response = (
        "<think>\nThis is a reasoning trace.\n</think>\n"
        '{"type": "informational", "message": "hello", "timeout_action": "proceed_default"}'
    )
    # mock_node does not have grpo_reward_policy, so require_think_tags is False
    # _extract_latent_traces will just return raw_response, None.
    # Since raw_response contains <think>, JSON validation will fail, triggering remediation loop.
    # We will give it a second response that is just valid JSON to pass.
    valid_response = '{"type": "informational", "message": "hello", "timeout_action": "proceed_default"}'
    adapter = DummyAdapter(responses=[raw_response, valid_response])
    engine = InferenceEngine(adapter)

    intent, _receipt, scratchpad = await engine.generate_intent(
        node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
    )

    assert intent.type == "informational"
    assert scratchpad is None
    # Ensure it went through remediation loop (used 2 responses)
    assert adapter.call_count == 2


@pytest.mark.asyncio
async def test_local_backpressure_fail_fast(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    """Verifies that exceeding the semaphore capacity immediately yields a 429 JSONRPCErrorResponseState."""
    valid_intent_json = '{"type": "informational", "message": "hello", "timeout_action": "proceed_default"}'
    adapter = DummyAdapter(responses=[valid_intent_json])
    # Set max_concurrent_tasks to 1 to easily test saturation
    engine = InferenceEngine(adapter, max_concurrent_tasks=1)

    # Acquire the semaphore to simulate a running task
    await engine._semaphore.acquire()

    try:
        # Since the semaphore is locked, generate_intent should fail-fast
        intent, receipt, scratchpad = await engine.generate_intent(
            node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
        )

        assert isinstance(intent, JSONRPCErrorResponseState)
        assert intent.error.code == 429
        assert "Local backpressure threshold exceeded" in intent.error.message
        assert receipt.input_tokens == 0
        assert receipt.output_tokens == 0
        assert receipt.burn_magnitude == 0
        assert scratchpad is None
    finally:
        # Release the semaphore
        engine._semaphore.release()
