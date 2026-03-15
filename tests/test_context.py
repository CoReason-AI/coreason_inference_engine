# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import json

from coreason_manifest.spec.ontology import (
    AgentAttestationReceipt,
    AgentNodeProfile,
    CognitiveStateProfile,
    DefeasibleCascadeEvent,
    EpistemicLedgerState,
    ObservationEvent,
    RollbackIntent,
    System2RemediationIntent,
    ToolInvocationEvent,
    ZeroKnowledgeReceipt,
)

from coreason_inference_engine.context import ContextHydrator


def create_mock_attestation() -> AgentAttestationReceipt:
    return AgentAttestationReceipt(
        training_lineage_hash="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        developer_signature="mock_signature",
        capability_merkle_root="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    )


def create_mock_zk_proof() -> ZeroKnowledgeReceipt:
    return ZeroKnowledgeReceipt(
        proof_protocol="zk-SNARK",
        public_inputs_hash="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        verifier_key_id="mock_verifier",
        cryptographic_blob="mock_blob",
    )


def test_context_hydrator_system_prompt() -> None:
    hydrator = ContextHydrator()
    node = AgentNodeProfile(description="You are a helpful assistant.", type="agent")
    ledger = EpistemicLedgerState(history=[])

    messages = hydrator.compile(node, ledger)

    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."


def test_context_hydrator_cognitive_state() -> None:
    hydrator = ContextHydrator()
    node = AgentNodeProfile(
        description="Base instructions.",
        type="agent",
        baseline_cognitive_state=CognitiveStateProfile(
            urgency_index=0.8,
            caution_index=0.2,
            divergence_tolerance=0.5,
        ),
    )
    ledger = EpistemicLedgerState(history=[])

    messages = hydrator.compile(node, ledger)

    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert "Urgency Index: 0.8" in messages[0]["content"]
    assert "Caution Index: 0.2" in messages[0]["content"]


def test_context_hydrator_role_mapping() -> None:
    hydrator = ContextHydrator()
    node = AgentNodeProfile(description="System instructions.", type="agent")

    obs1 = ObservationEvent(
        event_id="obs_1",
        timestamp=100.0,
        payload={"data": "Initial observation"},
    )

    tool_inv = ToolInvocationEvent(
        event_id="tool_1",
        timestamp=110.0,
        tool_name="test_tool",
        parameters={"arg1": "value1"},
        agent_attestation=create_mock_attestation(),
        zk_proof=create_mock_zk_proof(),
    )

    obs2 = ObservationEvent(
        event_id="obs_2",
        timestamp=120.0,
        payload={"result": "Tool result"},
        triggering_invocation_id="tool_1",
    )

    # System2RemediationIntent isn't an AnyStateEvent so it can't be added to the ledger directly
    # In standard execution, it's an ephemeral message injected into the messages list
    # Let's mock the internal history for testing
    ledger = EpistemicLedgerState(history=[obs1, tool_inv, obs2])

    rem1 = System2RemediationIntent(
        fault_id="fault_1",
        target_node_id="did:coreason:agent:mock123",
        failing_pointers=["/foo/bar"],
        remediation_prompt="Fix the parameter.",
    )

    # We will hack the history list in the ledger just for this test,
    # because ContextHydrator currently relies on iterating ledger.history
    # and checks isinstance(System2RemediationIntent).
    from typing import Any, cast

    ledger.history.append(cast("Any", rem1))

    messages = hydrator.compile(node, ledger)

    assert len(messages) == 5  # 1 system + 4 history events

    assert messages[0]["role"] == "system"

    assert messages[1]["role"] == "user"
    assert json.loads(messages[1]["content"]) == {"data": "Initial observation"}

    assert messages[2]["role"] == "assistant"
    assert len(messages[2]["tool_calls"]) == 1
    assert messages[2]["tool_calls"][0]["id"] == "tool_1"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "test_tool"
    assert json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]) == {"arg1": "value1"}

    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "tool_1"
    assert json.loads(messages[3]["content"]) == {"result": "Tool result"}

    assert messages[4]["role"] == "system"
    assert "Fix the parameter" in messages[4]["content"]


def test_context_hydrator_o1_mapping() -> None:
    hydrator = ContextHydrator(provider_mode="o1")
    node = AgentNodeProfile(description="System instructions.", type="agent")

    # We will simulate a System2RemediationIntent inside the compile
    # but we can also just test the standard history
    ledger = EpistemicLedgerState(history=[])

    rem1 = System2RemediationIntent(
        fault_id="fault_1",
        target_node_id="did:coreason:agent:mock123",
        failing_pointers=["/foo/bar"],
        remediation_prompt="Fix the parameter.",
    )

    from typing import Any, cast

    ledger.history.append(cast("Any", rem1))

    messages = hydrator.compile(node, ledger)

    assert len(messages) == 2
    assert messages[0]["role"] == "developer"
    assert messages[0]["content"] == "System instructions."
    assert messages[1]["role"] == "developer"
    assert "Fix the parameter" in messages[1]["content"]


def test_context_hydrator_anthropic_mapping() -> None:
    hydrator = ContextHydrator(provider_mode="anthropic")
    node = AgentNodeProfile(description="System instructions.", type="agent")

    obs1 = ObservationEvent(
        event_id="obs_1",
        timestamp=100.0,
        payload={"data": "Initial observation"},
    )
    obs2 = ObservationEvent(
        event_id="obs_2",
        timestamp=101.0,
        payload={"data": "Second observation"},
    )
    tool_inv1 = ToolInvocationEvent(
        event_id="tool_1",
        timestamp=110.0,
        tool_name="test_tool",
        parameters={"arg1": "value1"},
        agent_attestation=create_mock_attestation(),
        zk_proof=create_mock_zk_proof(),
    )
    tool_inv2 = ToolInvocationEvent(
        event_id="tool_2",
        timestamp=111.0,
        tool_name="test_tool2",
        parameters={"arg2": "value2"},
        agent_attestation=create_mock_attestation(),
        zk_proof=create_mock_zk_proof(),
    )

    # Sequence: Obs1, Obs2 -> Collapsed User
    # ToolInv1, ToolInv2 -> Consecutive Assistants -> Inject dummy user in between
    # Trailing ToolInv2 -> Inject trailing dummy user
    # Add an empty tool response (observation) mapped to tool result

    obs_tool = ObservationEvent(
        event_id="obs_3", timestamp=110.5, payload={"data": "ToolResult"}, triggering_invocation_id="tool_1"
    )

    # Let's test consecutive assistants first without tool response
    ledger_consecutive = EpistemicLedgerState(history=[obs1, obs2, tool_inv1, tool_inv2])
    messages = hydrator.compile(node, ledger_consecutive)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "System instructions."

    assert messages[1]["role"] == "user"
    # Should contain XML tags since it's collapsed
    assert "<observation>" in messages[1]["content"]
    assert "Initial observation" in messages[1]["content"]
    assert "Second observation" in messages[1]["content"]

    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["id"] == "tool_1"

    assert messages[3]["role"] == "user"
    assert "Dummy acknowledgment" in messages[3]["content"]

    assert messages[4]["role"] == "assistant"
    assert messages[4]["tool_calls"][0]["id"] == "tool_2"

    assert messages[5]["role"] == "user"
    assert "Please continue or provide the next action." in messages[5]["content"]

    assert len(messages) == 6

    # Now let's test with the tool response interleaved
    ledger_with_tool = EpistemicLedgerState(history=[obs1, obs2, tool_inv1, obs_tool, tool_inv2])
    messages_tool = hydrator.compile(node, ledger_with_tool)

    assert messages_tool[3]["role"] == "user"
    assert "Tool tool_1 Result:" in messages_tool[3]["content"]

    assert messages_tool[4]["role"] == "assistant"
    assert messages_tool[4]["tool_calls"][0]["id"] == "tool_2"

    assert messages_tool[5]["role"] == "user"
    assert "Please continue or provide the next action." in messages_tool[5]["content"]

    assert len(messages_tool) == 6


def test_context_hydrator_anthropic_empty_messages() -> None:
    hydrator = ContextHydrator(provider_mode="anthropic")
    # if messages list is empty, it returns immediately
    # A compile with no description? AgentNodeProfile requires description.
    # To test the `if not messages:` check in `_apply_anthropic_grammar`,
    # we can call it directly.
    assert hydrator._apply_anthropic_grammar([]) == []


def test_context_hydrator_quarantine() -> None:
    hydrator = ContextHydrator()
    node = AgentNodeProfile(description="System instructions.", type="agent")

    obs_valid = ObservationEvent(
        event_id="obs_valid",
        timestamp=100.0,
        payload={"data": "Valid"},
    )

    obs_invalid1 = ObservationEvent(
        event_id="obs_invalid1",
        timestamp=110.0,
        payload={"data": "Invalid cascade"},
    )

    obs_invalid2 = ObservationEvent(
        event_id="obs_invalid2",
        timestamp=120.0,
        payload={"data": "Invalid rollback"},
    )

    ledger = EpistemicLedgerState(
        history=[obs_valid, obs_invalid1, obs_invalid2],
        active_cascades=[
            DefeasibleCascadeEvent(
                cascade_id="cascade_1",
                root_falsified_event_id="hypo_1",
                propagated_decay_factor=0.5,
                quarantined_event_ids=["obs_invalid1"],
            )
        ],
        active_rollbacks=[
            RollbackIntent(
                request_id="rb_1",
                target_event_id="obs_valid",
                invalidated_node_ids=["obs_invalid2"],
            )
        ],
    )

    messages = hydrator.compile(node, ledger)

    # 1 system + 1 valid event
    assert len(messages) == 2
    assert messages[0]["role"] == "system"

    assert messages[1]["role"] == "user"
    assert json.loads(messages[1]["content"]) == {"data": "Valid"}
