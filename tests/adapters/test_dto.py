# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

from coreason_inference_engine.adapters.dto import LocalActionSpace, LocalAgentNodeProfile, LocalLedgerState


def test_local_agent_node_profile_ignores_extra() -> None:
    data = {
        "description": "An agent",
        "baseline_cognitive_state": {
            "semantic_slicing": {"context_window_token_ceiling": 1000, "extra_field": "ignore me"},
            "extra_field": "ignore me",
        },
        "extra_top_level_field": "ignore me too",
    }

    node = LocalAgentNodeProfile.model_validate(data)
    assert node.baseline_cognitive_state is not None
    assert node.baseline_cognitive_state.semantic_slicing is not None
    assert node.baseline_cognitive_state.semantic_slicing.context_window_token_ceiling == 1000

    # Assert extra fields are dropped
    assert not hasattr(node, "extra_top_level_field")
    assert not hasattr(node.baseline_cognitive_state, "extra_field")
    assert not hasattr(node.baseline_cognitive_state.semantic_slicing, "extra_field")


def test_local_action_space_ignores_extra() -> None:
    data = {"native_tools": [{"name": "tool1"}], "mcp_servers": [{"name": "server1"}], "extra_field": "ignore me"}

    space = LocalActionSpace.model_validate(data)
    assert space.native_tools == [{"name": "tool1"}]
    assert not hasattr(space, "mcp_servers")
    assert not hasattr(space, "extra_field")


def test_local_ledger_state_ignores_extra() -> None:
    data = {"history": [{"event": "start"}], "checkpoints": ["cp1", "cp2"], "extra_field": "ignore me"}

    ledger = LocalLedgerState.model_validate(data)
    assert ledger.history == [{"event": "start"}]
    assert not hasattr(ledger, "checkpoints")
    assert not hasattr(ledger, "extra_field")
