from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import patch

import pytest
from coreason_manifest.spec.ontology import (
    ActionSpaceManifest,
    AgentNodeProfile,
    EpistemicLedgerState,
    PeftAdapterContract,
    PermissionBoundaryPolicy,
    SideEffectProfile,
    System1ReflexPolicy,
    ToolManifest,
)

from coreason_inference_engine.engine import InferenceEngine
from coreason_inference_engine.interfaces import InferenceConvergenceError, LLMAdapterProtocol


class DummyReflexAdapter(LLMAdapterProtocol):
    rate_card: Any = None

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.tools_projected: list[list[dict[str, Any]]] = []
        self.max_tokens_received: list[int | None] = []

    def count_tokens(self, text: str) -> int:
        return len(text)

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.tools_projected.append(schemas)
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
        **_kwargs: Any,
    ) -> AsyncGenerator[tuple[str, dict[str, int], dict[str, Any] | None]]:
        # use arguments to avoid unused variable warning while keeping function signature intact
        _ = messages
        _ = tools
        _ = temperature
        _ = logit_biases
        self.max_tokens_received.append(max_tokens)
        if self.call_count >= len(self.responses):
            yield "", {"input_tokens": 0, "output_tokens": 0}, None
            return

        resp = self.responses[self.call_count]
        self.call_count += 1
        yield resp, {"input_tokens": 10, "output_tokens": 10}, None


@pytest.fixture(autouse=True)
def mock_validate_payload() -> Any:
    """Mock validate_payload since the local coreason_manifest doesn't support AnyIntent/AnyStateEvent natively."""

    def _mocked_validate(schema_key: str, payload: bytes) -> Any:
        from coreason_manifest.spec.ontology import (
            AnyIntent,
            AnyStateEvent,
            CognitiveStateProfile,
            DocumentLayoutManifest,
            StateMutationIntent,
            System2RemediationIntent,
        )
        from pydantic import TypeAdapter

        schema_registry: dict[str, Any] = {
            "step8_vision": DocumentLayoutManifest,
            "state_differential": StateMutationIntent,
            "cognitive_sync": CognitiveStateProfile,
            "system2_remediation": System2RemediationIntent,
        }

        if schema_key in ("intent", "state_differential", "symbolic_handoff", "AnyIntent"):
            target_union = AnyIntent | AnyStateEvent | System2RemediationIntent | StateMutationIntent
            return TypeAdapter(target_union).validate_json(payload)

        target_schema = schema_registry.get(schema_key)
        if not target_schema:
            raise ValueError(f"FATAL: Unknown step '{schema_key}'. Valid steps: {list(schema_registry.keys())}")

        return target_schema.model_validate_json(payload)

    # Force the mock to be used globally within the module so `from ... import validate_payload` sees it.
    import coreason_manifest.utils.algebra

    original = coreason_manifest.utils.algebra.validate_payload
    coreason_manifest.utils.algebra.validate_payload = _mocked_validate

    with patch("coreason_inference_engine.engine.validate_payload", _mocked_validate, create=True):
        yield

    coreason_manifest.utils.algebra.validate_payload = original


@pytest.fixture
def mock_ledger() -> EpistemicLedgerState:
    return EpistemicLedgerState(history=[])


@pytest.fixture
def mock_action_space() -> ActionSpaceManifest:
    return ActionSpaceManifest(
        action_space_id="test_space",
        native_tools=[
            ToolManifest(
                tool_name="fast_tool",
                description="A passive tool.",
                input_schema={},
                side_effects=SideEffectProfile(mutates_state=False, is_idempotent=True),
                permissions=PermissionBoundaryPolicy(network_access=False, file_system_mutation_forbidden=True),
            ),
            ToolManifest(
                tool_name="slow_tool",
                description="A mutating tool.",
                input_schema={},
                side_effects=SideEffectProfile(mutates_state=True, is_idempotent=False),
                permissions=PermissionBoundaryPolicy(network_access=True, file_system_mutation_forbidden=False),
            ),
        ],
    )


@pytest.mark.asyncio
async def test_reflex_fast_path_success(
    mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    node = AgentNodeProfile(
        description="Test node",
        type="agent",
        reflex_policy=System1ReflexPolicy(confidence_threshold=0.9, allowed_passive_tools=["fast_tool"]),
    )

    fast_path_response = """{
        "type": "tool_invocation",
        "event_id": "fast_event_1",
        "timestamp": 1234567890.0,
        "tool_name": "fast_tool",
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

    adapter = DummyReflexAdapter(responses=[fast_path_response])
    engine = InferenceEngine(adapter)

    intent, receipt, _scratch, _ = await engine.generate_intent(
        node=node.model_dump(),
        ledger=mock_ledger.model_dump(),
        node_id="did:test:1",
        action_space=mock_action_space.model_dump(),
    )

    assert (intent.get("type") if isinstance(intent, dict) else getattr(intent, "type", None)) == "tool_invocation"
    assert (intent.get("tool_name") if isinstance(intent, dict) else getattr(intent, "tool_name", None)) == "fast_tool"
    assert adapter.call_count == 1
    assert adapter.max_tokens_received == [150]
    assert len(adapter.tools_projected[0]) == 1
    assert adapter.tools_projected[0][0]["tool_name"] == "fast_tool"
    assert receipt.get("input_tokens", 0) == 10
    assert receipt.get("output_tokens", 0) == 10


@pytest.mark.asyncio
async def test_reflex_fast_path_fallback_invalid_intent(
    mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    node = AgentNodeProfile(
        description="Test node",
        type="agent",
        reflex_policy=System1ReflexPolicy(confidence_threshold=0.9, allowed_passive_tools=["fast_tool"]),
    )

    fast_path_response = """{
        "type": "informational",
        "message": "I need more time",
        "timeout_action": "proceed_default"
    }"""

    deep_path_response = """{
        "type": "informational",
        "message": "Deep thought result",
        "timeout_action": "proceed_default"
    }"""

    adapter = DummyReflexAdapter(responses=[fast_path_response, deep_path_response])
    engine = InferenceEngine(adapter)

    intent, _receipt, _scratch, _ = await engine.generate_intent(
        node=node.model_dump(),
        ledger=mock_ledger.model_dump(),
        node_id="did:test:1",
        action_space=mock_action_space.model_dump(),
    )

    assert (intent.get("type") if isinstance(intent, dict) else getattr(intent, "type", None)) == "informational"
    assert (intent.get("message") if isinstance(intent, dict) else getattr(intent, "message", None)) == "Deep thought result"
    assert adapter.call_count == 2
    assert adapter.max_tokens_received == [150, None]


@pytest.mark.asyncio
async def test_reflex_fast_path_fallback_unallowed_tool(
    mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    node = AgentNodeProfile(
        description="Test node",
        type="agent",
        reflex_policy=System1ReflexPolicy(confidence_threshold=0.9, allowed_passive_tools=["fast_tool"]),
    )

    fast_path_response = """{
        "type": "tool_invocation",
        "event_id": "fast_event_1",
        "timestamp": 1234567890.0,
        "tool_name": "slow_tool",
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

    deep_path_response = """{
        "type": "informational",
        "message": "Deep thought result",
        "timeout_action": "proceed_default"
    }"""

    adapter = DummyReflexAdapter(responses=[fast_path_response, deep_path_response])
    engine = InferenceEngine(adapter)

    intent, _receipt, _scratch, _ = await engine.generate_intent(
        node=node.model_dump(),
        ledger=mock_ledger.model_dump(),
        node_id="did:test:1",
        action_space=mock_action_space.model_dump(),
    )

    assert (intent.get("type") if isinstance(intent, dict) else getattr(intent, "type", None)) == "informational"
    assert adapter.call_count == 2


@pytest.mark.asyncio
async def test_reflex_fast_path_empty_passivetools(
    mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    node = AgentNodeProfile(
        description="Test node",
        type="agent",
        reflex_policy=System1ReflexPolicy(confidence_threshold=0.9, allowed_passive_tools=["nonexistent_tool"]),
    )

    deep_path_response = """{
        "type": "informational",
        "message": "Deep thought result",
        "timeout_action": "proceed_default"
    }"""

    adapter = DummyReflexAdapter(responses=[deep_path_response])
    engine = InferenceEngine(adapter)

    intent, _receipt, _scratch, _ = await engine.generate_intent(
        node=node.model_dump(),
        ledger=mock_ledger.model_dump(),
        node_id="did:test:1",
        action_space=mock_action_space.model_dump(),
    )

    assert (intent.get("type") if isinstance(intent, dict) else getattr(intent, "type", None)) == "informational"
    assert adapter.call_count == 1
    assert adapter.max_tokens_received == [None]


@pytest.mark.asyncio
async def test_reflex_fast_path_missing_usage(
    mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    node = AgentNodeProfile(
        description="Test node",
        type="agent",
        reflex_policy=System1ReflexPolicy(confidence_threshold=0.9, allowed_passive_tools=["fast_tool"]),
    )

    fast_path_response = """{
        "type": "tool_invocation",
        "event_id": "fast_event_1",
        "timestamp": 1234567890.0,
        "tool_name": "fast_tool",
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

    class MissingUsageAdapter(DummyReflexAdapter):
        async def generate_stream(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            temperature: float,
            logit_biases: dict[int, float] | None = None,
            max_tokens: int | None = None,
            **_kwargs: Any,
        ) -> AsyncGenerator[tuple[str, dict[str, int], dict[str, Any] | None]]:
            _ = messages
            _ = tools
            _ = temperature
            _ = logit_biases
            self.max_tokens_received.append(max_tokens)
            yield fast_path_response, cast("dict[str, int]", {}), None

    adapter = MissingUsageAdapter(responses=[])
    engine = InferenceEngine(adapter)

    intent, receipt, _scratch, _ = await engine.generate_intent(
        node=node.model_dump(),
        ledger=mock_ledger.model_dump(),
        node_id="did:test:1",
        action_space=mock_action_space.model_dump(),
    )
    assert (intent.get("type") if isinstance(intent, dict) else getattr(intent, "type", None)) == "tool_invocation"
    assert (intent.get("type") if isinstance(intent, dict) else getattr(intent, "type", None)) == "tool_invocation"
    assert receipt.get("output_tokens", 0) > 0


@pytest.mark.asyncio
async def test_reflex_fast_path_exception(
    mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    node = AgentNodeProfile(
        description="Test node",
        type="agent",
        reflex_policy=System1ReflexPolicy(confidence_threshold=0.9, allowed_passive_tools=["fast_tool"]),
    )

    class ExceptionAdapter(DummyReflexAdapter):
        async def generate_stream(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            temperature: float,
            logit_biases: dict[int, float] | None = None,
            max_tokens: int | None = None,
            **_kwargs: Any,
        ) -> AsyncGenerator[tuple[str, dict[str, int], dict[str, Any] | None]]:
            _ = messages
            _ = tools
            _ = temperature
            _ = logit_biases
            if max_tokens == 150:
                raise ValueError("Some random error")
            yield "", {"input_tokens": 10, "output_tokens": 10}, None

    adapter = ExceptionAdapter(responses=[])
    engine = InferenceEngine(adapter)

    with pytest.raises(InferenceConvergenceError):
        await engine.generate_intent(
            node=node.model_dump(),
            ledger=mock_ledger.model_dump(),
            node_id="did:test:1",
            action_space=mock_action_space.model_dump(),
        )


@pytest.mark.asyncio
async def test_reflex_fast_path_system_message_exists(
    mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    node = AgentNodeProfile(
        description="Test node",
        type="agent",
        reflex_policy=System1ReflexPolicy(confidence_threshold=0.9, allowed_passive_tools=["fast_tool"]),
    )

    fast_path_response = """{
        "type": "tool_invocation",
        "event_id": "fast_event_1",
        "timestamp": 1234567890.0,
        "tool_name": "fast_tool",
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

    adapter = DummyReflexAdapter(responses=[fast_path_response])
    engine = InferenceEngine(adapter)

    class NoSystemHydrator:
        def compile(self, _node: Any, _ledger: Any) -> list[dict[str, Any]]:
            return [{"role": "user", "content": "hi"}]

    engine.hydrator = NoSystemHydrator()  # type: ignore

    intent, _receipt, _scratch, _ = await engine.generate_intent(
        node=node.model_dump(),
        ledger=mock_ledger.model_dump(),
        node_id="did:test:1",
        action_space=mock_action_space.model_dump(),
    )
    assert (intent.get("type") if isinstance(intent, dict) else getattr(intent, "type", None)) == "tool_invocation"


@pytest.mark.asyncio
async def test_reflex_fast_path_validation_error(
    mock_ledger: EpistemicLedgerState, mock_action_space: ActionSpaceManifest
) -> None:
    node = AgentNodeProfile(
        description="Test node",
        type="agent",
        reflex_policy=System1ReflexPolicy(confidence_threshold=0.9, allowed_passive_tools=["fast_tool"]),
    )

    fast_path_response = '{"invalid_json"'

    adapter = DummyReflexAdapter(
        responses=[
            fast_path_response,
            '{"type": "informational", "message": "hello", "timeout_action": "proceed_default"}',
        ]
    )
    engine = InferenceEngine(adapter)

    intent, _receipt, _scratch, _ = await engine.generate_intent(
        node=node.model_dump(),
        ledger=mock_ledger.model_dump(),
        node_id="did:test:1",
        action_space=mock_action_space.model_dump(),
    )
    assert (intent.get("type") if isinstance(intent, dict) else getattr(intent, "type", None)) == "informational"
    assert adapter.call_count == 2
