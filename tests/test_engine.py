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

import httpx
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

from coreason_inference_engine.engine import InferenceEngine, _AnyIntentAdapter
from coreason_inference_engine.interfaces import InferenceConvergenceError, LLMAdapterProtocol


class DummyAdapter(LLMAdapterProtocol):
    rate_card: Any = None

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


class SeveredStreamAdapter(LLMAdapterProtocol):
    rate_card: Any = None

    def __init__(self, response: str) -> None:
        self.response = response
        self.tools_projected = False

    def count_tokens(self, text: str) -> int:
        return len(text)

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.tools_projected = True
        return schemas

    async def apply_peft_adapters(self, _adapters: list[PeftAdapterContract]) -> None:
        pass

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
        _ = max_tokens
        # Simulate returning a string response but omitting usage metrics entirely
        yield self.response, {}


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
async def test_ijson_early_termination(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    """Verifies that the engine severs the stream early using ijson if the structure violates expected bounds."""

    class StreamingAdapter(LLMAdapterProtocol):
        rate_card: Any = None

        def __init__(self) -> None:
            self.aclose_called = False
            self.stream_ended = False

        def count_tokens(self, text: str) -> int:
            return len(text)

        def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return schemas

        async def apply_peft_adapters(self, adapters: list[PeftAdapterContract]) -> None:
            pass

        def generate_stream(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            temperature: float,
            logit_biases: dict[int, float] | None = None,
            max_tokens: int | None = None,
        ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
            _ = messages
            _ = tools
            _ = temperature
            _ = logit_biases
            _ = max_tokens
            # Provide an invalid key mid-stream
            chunks = ['{"inva', 'lid_key": "val', 'ue"}']

            # Use a helper inner class to track aclose
            class AsyncGenWrapper:
                def __init__(self, parent: Any) -> None:
                    self.parent = parent
                    self.idx = 0

                def __aiter__(self) -> "AsyncGenWrapper":
                    return self

                async def __anext__(self) -> tuple[str, dict[str, int]]:
                    if self.idx < len(chunks):
                        c = chunks[self.idx]
                        self.idx += 1
                        return c, {"input_tokens": 10, "output_tokens": 0}
                    self.parent.stream_ended = True
                    raise StopAsyncIteration

                async def aclose(self) -> None:
                    self.parent.aclose_called = True

            return AsyncGenWrapper(self)  # type: ignore

    # First attempt: invalid key (triggers early termination). Second attempt: valid.
    adapter = StreamingAdapter()
    engine = InferenceEngine(adapter)

    # We need to monkey-patch or just test that it throws InferenceConvergenceError after max loops,
    # or give it a valid response on retry. Since StreamingAdapter always yields the invalid stream,
    # it will fail 2 times and raise InferenceConvergenceError.

    with pytest.raises(InferenceConvergenceError):
        await engine.generate_intent(
            node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
        )

    assert adapter.aclose_called is True
    assert adapter.stream_ended is False  # Meaning it broke out before finishing chunks


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
    assert receipt.burn_magnitude == 20000000
    assert adapter.tools_projected is True


@pytest.mark.asyncio
async def test_generate_intent_ttft_concurrency(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    """Verifies that ttft and span traces are tracked safely via local variables in a concurrent TaskGroup fan-out."""
    # We will simulate the LLM responding with a valid Intent, and block slightly to ensure overlapping execution.
    valid_intent_json = '{"type": "informational", "message": "concurrency", "timeout_action": "proceed_default"}'

    class DelayingAdapter(LLMAdapterProtocol):
        rate_card: Any = None

        def count_tokens(self, _text: str) -> int:
            return 10

        def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return schemas

        async def apply_peft_adapters(self, _adapters: list[PeftAdapterContract]) -> None:
            pass

        async def generate_stream(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            temperature: float,
            logit_biases: dict[int, float] | None = None,
            max_tokens: int | None = None,
        ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
            _ = messages
            _ = tools
            _ = temperature
            _ = logit_biases
            _ = max_tokens
            await asyncio.sleep(0.01)  # Yield to event loop, forcing concurrency
            yield valid_intent_json, {"input_tokens": 10, "output_tokens": 10}

    adapter = DelayingAdapter()

    # Create a custom queue telemetry emitter to capture the spans
    class SpanCollectorEmitter:
        def __init__(self) -> None:
            self.spans: list[Any] = []

        async def emit(self, event: Any) -> None:
            from coreason_manifest.spec.ontology import SpanTraceReceipt

            if isinstance(event, SpanTraceReceipt):
                self.spans.append(event)

        def redact_pii(self, payload: str, _policy: Any) -> str:
            return payload

    emitter = SpanCollectorEmitter()
    engine = InferenceEngine(adapter, telemetry=emitter)  # type: ignore

    # Fan out 10 concurrent requests
    tasks = [
        engine.generate_intent(
            node=mock_node, ledger=mock_ledger, node_id=f"did:test:{i}", action_space=mock_action_space
        )
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    assert len(emitter.spans) == 10

    # Ensure all span traces have valid ttft
    for span in emitter.spans:
        assert "ttft_nano" in span.context_profile
        ttft = span.context_profile["ttft_nano"]
        assert isinstance(ttft, int)
        assert ttft > 0  # Must have elapsed some time


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
    assert receipt.burn_magnitude == 40000000


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
    rate_card: Any = None

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


@pytest.mark.asyncio
async def test_severed_stream_token_fallback(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    """Verifies that if usage metrics are missing, the engine falls back to local token counting using safe decoding."""
    # Create an intent that includes potentially invalid UTF-8 bytes disguised in a valid structure
    valid_intent_json = '{"type": "informational", "message": "fixed\ud83d", "timeout_action": "proceed_default"}'

    adapter = SeveredStreamAdapter(response=valid_intent_json)
    engine = InferenceEngine(adapter)

    intent, receipt, _scratchpad = await engine.generate_intent(
        node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
    )

    # Safe decoding shouldn't explode and length of the safe output will be used
    safe_output = valid_intent_json.encode("utf-8", errors="replace").decode("utf-8")
    assert receipt.output_tokens == adapter.count_tokens(safe_output)
    assert receipt.input_tokens > 0  # Input tokens counted from messages
    assert intent.type == "informational"


def test_anyintent_adapter_includes_missing_intents() -> None:
    """Verifies that missing intent types can be successfully validated by _AnyIntentAdapter."""
    adapter = _AnyIntentAdapter()

    # ToolInvocationEvent
    tool_invocation_json = b"""{
        "type": "tool_invocation",
        "event_id": "test_id_1",
        "timestamp": 1234567890.0,
        "tool_name": "test_tool",
        "parameters": {},
        "authorized_budget_magnitude": 1,
        "agent_attestation": {
            "training_lineage_hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "developer_signature": "sig",
            "capability_merkle_root": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        },
        "zk_proof": {
            "proof_protocol": "zk-SNARK",
            "public_inputs_hash": "hash",
            "verifier_key_id": "key",
            "cryptographic_blob": "blob"
        }
    }"""
    tool_intent = adapter.model_validate_json(tool_invocation_json)
    assert tool_intent.type == "tool_invocation"
    assert tool_intent.tool_name == "test_tool"

    # StateMutationIntent
    state_mutation_json = b'{"op": "replace", "path": "/some/path", "value": "new_value"}'
    mutation_intent = adapter.model_validate_json(state_mutation_json)
    assert mutation_intent.op == "replace"
    assert mutation_intent.path == "/some/path"
    assert mutation_intent.value == "new_value"

    # System2RemediationIntent
    remediation_json = (
        b'{"fault_id": "fault_1", "target_node_id": "did:test:1", '
        b'"failing_pointers": ["/a"], "remediation_prompt": "fix it"}'
    )
    remed_intent = adapter.model_validate_json(remediation_json)
    assert remed_intent.fault_id == "fault_1"
    assert remed_intent.remediation_prompt == "fix it"


class HttpFaultAdapter(LLMAdapterProtocol):
    rate_card: Any = None

    def __init__(self, responses: list[str], status_codes: list[int]) -> None:
        self.responses = responses
        self.status_codes = status_codes
        self.call_count = 0

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
    ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
        _ = messages
        _ = tools
        _ = temperature
        _ = logit_biases
        _ = max_tokens
        status_code = self.status_codes[self.call_count]
        response = self.responses[self.call_count]
        self.call_count += 1

        if status_code != 200:
            req = httpx.Request("POST", "http://test")
            resp = httpx.Response(status_code, request=req)
            raise httpx.HTTPStatusError("Fault", request=req, response=resp)

        yield response, {"input_tokens": 10, "output_tokens": 20}


@pytest.mark.asyncio
async def test_transient_network_fault_backoff(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    # First call yields 429, second yields 200
    responses = ["", '{"type": "informational", "message": "hello", "timeout_action": "proceed_default"}']
    status_codes = [429, 200]

    adapter = HttpFaultAdapter(responses, status_codes)
    engine = InferenceEngine(adapter)

    intent, _receipt, _scratchpad = await engine.generate_intent(
        node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
    )

    assert intent.type == "informational"
    assert adapter.call_count == 2


@pytest.mark.asyncio
async def test_transient_network_fault_sla_exceeded(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    # First call yields 429
    responses = [""]
    status_codes = [429]

    adapter = HttpFaultAdapter(responses, status_codes)
    engine = InferenceEngine(adapter)

    _updated_policy = (
        mock_node.correction_policy.model_copy(update={"global_timeout_seconds": 0.0})
        if mock_node.correction_policy
        else None
    )
    correction_policy = (
        _updated_policy if _updated_policy is not None else SelfCorrectionPolicy(max_loops=2, rollback_on_failure=True)
    )
    object.__setattr__(correction_policy, "global_timeout_seconds", 0.0)
    _updated_policy = (
        mock_node.correction_policy.model_copy(update={"global_timeout_seconds": 0.0})
        if mock_node.correction_policy
        else None
    )
    correction_policy = (
        _updated_policy if _updated_policy is not None else SelfCorrectionPolicy(max_loops=2, rollback_on_failure=True)
    )
    object.__setattr__(correction_policy, "global_timeout_seconds", 0.0)

    node_with_small_timeout = mock_node.model_copy(update={"correction_policy": correction_policy})

    with pytest.raises(InferenceConvergenceError, match="SLA Contention: Required backoff delay"):
        await engine.generate_intent(
            node=node_with_small_timeout, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
        )


class HttpFaultMidStreamAdapter(LLMAdapterProtocol):
    rate_card: Any = None

    def __init__(self, responses: list[str], status_codes: list[int]) -> None:
        self.responses = responses
        self.status_codes = status_codes
        self.call_count = 0

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
    ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
        _ = messages
        _ = tools
        _ = temperature
        _ = logit_biases
        _ = max_tokens
        status_code = self.status_codes[self.call_count]
        response = self.responses[self.call_count]
        self.call_count += 1

        if status_code != 200:
            req = httpx.Request("POST", "http://test")
            resp = httpx.Response(status_code, request=req)
            raise httpx.HTTPStatusError("Fault", request=req, response=resp)

        yield response, {"input_tokens": 10, "output_tokens": 20}

        # Then fault mid-stream if we have another status code that is bad
        # Let's just create a mock generator that fails on the first yield
        # The first case already covers generator creation failure.
        # This will be similar, but let's test where it yields then fails.

    def generate_stream_faulty(self, *args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs

        async def mock_gen() -> AsyncGenerator[tuple[str, dict[str, int]]]:
            yield "part1", {"input_tokens": 10, "output_tokens": 0}
            req = httpx.Request("POST", "http://test")
            resp = httpx.Response(502, request=req)
            raise httpx.HTTPStatusError("Fault", request=req, response=resp)

        return mock_gen()


@pytest.mark.asyncio
async def test_transient_network_fault_mid_stream(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    class MidStreamFaultAdapter(LLMAdapterProtocol):
        rate_card: Any = None

        def __init__(self) -> None:
            self.call_count = 0

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
        ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
            _ = messages
            _ = tools
            _ = temperature
            _ = logit_biases
            _ = max_tokens
            self.call_count += 1
            if self.call_count == 1:
                yield "part1", {"input_tokens": 10, "output_tokens": 0}
                req = httpx.Request("POST", "http://test")
                resp = httpx.Response(502, request=req)
                raise httpx.HTTPStatusError("Fault", request=req, response=resp)
            else:
                yield (
                    '{"type": "informational", "message": "hello", "timeout_action": "proceed_default"}',
                    {"input_tokens": 10, "output_tokens": 20},
                )

    adapter = MidStreamFaultAdapter()
    engine = InferenceEngine(adapter)

    intent, _receipt, _scratchpad = await engine.generate_intent(
        node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
    )

    assert intent.type == "informational"
    assert adapter.call_count == 2


@pytest.mark.asyncio
async def test_transient_network_fault_mid_stream_sla_exceeded(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    class MidStreamFaultAdapter(LLMAdapterProtocol):
        rate_card: Any = None

        def __init__(self) -> None:
            self.call_count = 0

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
        ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
            _ = messages
            _ = tools
            _ = temperature
            _ = logit_biases
            _ = max_tokens
            self.call_count += 1
            yield "part1", {"input_tokens": 10, "output_tokens": 0}
            req = httpx.Request("POST", "http://test")
            resp = httpx.Response(502, request=req)
            raise httpx.HTTPStatusError("Fault", request=req, response=resp)

    adapter = MidStreamFaultAdapter()
    engine = InferenceEngine(adapter)

    correction_policy = (
        mock_node.correction_policy.model_copy()
        if mock_node.correction_policy
        else SelfCorrectionPolicy(max_loops=2, rollback_on_failure=True)
    )
    _updated_policy = (
        mock_node.correction_policy.model_copy(update={"global_timeout_seconds": 0.0})
        if mock_node.correction_policy
        else None
    )
    correction_policy = _updated_policy if _updated_policy is not None else correction_policy
    if correction_policy is None:
        correction_policy = SelfCorrectionPolicy(max_loops=2, rollback_on_failure=True)
        object.__setattr__(correction_policy, "global_timeout_seconds", 0.0)

    node_with_small_timeout = mock_node.model_copy(update={"correction_policy": correction_policy})

    with pytest.raises(InferenceConvergenceError, match="SLA Contention: Required backoff delay"):
        await engine.generate_intent(
            node=node_with_small_timeout, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
        )


@pytest.mark.asyncio
async def test_transient_network_fault_unhandled_status_code(
    mock_node: AgentNodeProfile, mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    # First call yields 404 which is not handled by transient fault logic
    responses = [""]
    status_codes = [404]

    adapter = HttpFaultAdapter(responses, status_codes)
    engine = InferenceEngine(adapter)

    with pytest.raises(httpx.HTTPStatusError):
        await engine.generate_intent(
            node=mock_node, ledger=mock_ledger, node_id="did:test:1", action_space=mock_action_space
        )
