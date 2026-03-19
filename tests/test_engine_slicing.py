# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import json
from collections.abc import AsyncGenerator
from typing import Any

from coreason_manifest.spec.ontology import (
    AgentNodeProfile,
    CognitiveStateProfile,
    EpistemicLedgerState,
    InformationClassificationProfile,
    LatentScratchpadReceipt,
    ObservationEvent,
    SemanticSlicingPolicy,
    System2RemediationIntent,
)

from coreason_inference_engine.engine import InferenceEngine
from coreason_inference_engine.interfaces import LLMAdapterProtocol


class TokenCountingAdapter(LLMAdapterProtocol):
    rate_card: Any = None
    """An adapter where each char is 1 token to easily test ceiling bounds."""

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return schemas

    def count_tokens(self, text: str) -> int:
        return len(text)

    async def apply_peft_adapters(self, adapters: list[Any]) -> None:
        pass

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        logit_biases: dict[int, float] | None = None,
        max_tokens: int | None = None,
        **_kwargs: Any,
    ) -> AsyncGenerator[tuple[str, dict[str, int], LatentScratchpadReceipt | None]]:
        _ = messages
        _ = tools
        _ = temperature
        _ = logit_biases
        _ = max_tokens
        yield "response", {"input_tokens": 0, "output_tokens": 0}, None


def test_null_policy_no_eviction() -> None:
    adapter = TokenCountingAdapter()
    engine = InferenceEngine(adapter)

    # Agent node without a slicing policy
    node = AgentNodeProfile(description="Test node", type="agent")

    obs1 = ObservationEvent(event_id="obs1", timestamp=100.0, payload={"data": "a" * 100})
    obs2 = ObservationEvent(event_id="obs2", timestamp=101.0, payload={"data": "b" * 100})
    ledger = EpistemicLedgerState(history=[obs1, obs2])

    messages = engine._apply_semantic_slicing(node, ledger)

    # Should retain both observations in the output
    assert len(messages) == 3  # 1 system + 2 observations
    assert "aaaaa" in messages[1]["content"]
    assert "bbbbb" in messages[2]["content"]


def test_positive_mass_below_ceiling() -> None:
    adapter = TokenCountingAdapter()
    engine = InferenceEngine(adapter)

    node = AgentNodeProfile(
        description="Test node",
        type="agent",
        baseline_cognitive_state=CognitiveStateProfile(
            urgency_index=0.5,
            caution_index=0.5,
            divergence_tolerance=0.1,
            semantic_slicing=SemanticSlicingPolicy(
                permitted_classification_tiers=[InformationClassificationProfile.PUBLIC],
                required_semantic_labels=["label1"],
                context_window_token_ceiling=1000,
            ),
        ),
    )

    obs1 = ObservationEvent(event_id="obs1", timestamp=100.0, payload={"data": "a" * 50})
    obs2 = ObservationEvent(event_id="obs2", timestamp=101.0, payload={"data": "b" * 50})
    ledger = EpistemicLedgerState(history=[obs1, obs2])

    messages = engine._apply_semantic_slicing(node, ledger)

    # The total JSON string len is definitely < 1000. No eviction should occur.
    assert len(messages) == 3
    assert "aaaaa" in messages[1]["content"]
    assert "bbbbb" in messages[2]["content"]


def test_boundary_edge_mass_above_ceiling_eviction() -> None:
    adapter = TokenCountingAdapter()
    engine = InferenceEngine(adapter)

    # Extremely tight ceiling to force eviction of the first observation
    # The ceiling of 120 is barely enough for the system prompt + obs2
    node = AgentNodeProfile(
        description="A",  # Minimal system prompt
        type="agent",
        baseline_cognitive_state=CognitiveStateProfile(
            urgency_index=0.5,
            caution_index=0.5,
            divergence_tolerance=0.1,
            semantic_slicing=SemanticSlicingPolicy(
                permitted_classification_tiers=[InformationClassificationProfile.PUBLIC],
                required_semantic_labels=["label1"],
                context_window_token_ceiling=150,
            ),
        ),
    )

    obs1 = ObservationEvent(event_id="obs1", timestamp=100.0, payload={"data": "X" * 100})  # Will push over 150
    obs2 = ObservationEvent(event_id="obs2", timestamp=101.0, payload={"data": "Y" * 10})  # Small
    ledger = EpistemicLedgerState(history=[obs1, obs2])

    messages = engine._apply_semantic_slicing(node, ledger)

    # Obs1 should be evicted because it makes the mass > 150
    # Obs2 should be retained
    assert len(messages) == 2  # 1 system + 1 obs
    content_combined = json.dumps(messages)
    assert "YYYYY" in content_combined
    assert "XXXXX" not in content_combined


def test_destructive_eviction_prevention() -> None:
    """Verifies that multiple System2RemediationIntent instances are capped to exactly 1 (the most recent)."""
    adapter = TokenCountingAdapter()
    engine = InferenceEngine(adapter)

    # Ceiling large enough not to evict observations due to size,
    # but we want to check the explicit cap on remediation intents.
    node = AgentNodeProfile(
        description="A",
        type="agent",
        baseline_cognitive_state=CognitiveStateProfile(
            urgency_index=0.5,
            caution_index=0.5,
            divergence_tolerance=0.1,
            semantic_slicing=SemanticSlicingPolicy(
                permitted_classification_tiers=[InformationClassificationProfile.PUBLIC],
                required_semantic_labels=["label1"],
                context_window_token_ceiling=10000,
            ),
        ),
    )

    rem1 = System2RemediationIntent(
        fault_id="fault1", target_node_id="did:test:1", failing_pointers=["/f1"], remediation_prompt="First failure"
    )
    rem2 = System2RemediationIntent(
        fault_id="fault2", target_node_id="did:test:1", failing_pointers=["/f2"], remediation_prompt="Second failure"
    )
    rem3 = System2RemediationIntent(
        fault_id="fault3", target_node_id="did:test:1", failing_pointers=["/f3"], remediation_prompt="Third failure"
    )

    obs1 = ObservationEvent(event_id="obs1", timestamp=103.0, payload={"data": "A"})

    # History with 3 remediation intents
    # Note: System2RemediationIntent is technically not an AnyStateEvent,
    # but the engine is designed to handle it if injected (e.g., during the retry loop).
    # Since Pydantic will reject it during instantiation of EpistemicLedgerState,
    # we bypass Pydantic validation by mutating `history` post-creation or avoiding the ledger.
    ledger = EpistemicLedgerState(history=[obs1])
    # Force injection for testing purposes
    history_with_remediations = [rem1, obs1, rem2, rem3]
    ledger = ledger.model_copy(update={"history": history_with_remediations})

    messages = engine._apply_semantic_slicing(node, ledger)

    content_combined = json.dumps(messages)
    # rem1 and rem2 should be removed. only rem3 should remain.
    assert "Third failure" in content_combined
    assert "First failure" not in content_combined
    assert "Second failure" not in content_combined

    # Check length: 1 system prompt + 1 observation + 1 remediation intent = 3
    assert len(messages) == 3


def test_eviction_exhaustion() -> None:
    """Verifies loop breaks when there are no more ObservationEvents to evict, even if over ceiling."""
    adapter = TokenCountingAdapter()
    engine = InferenceEngine(adapter)

    # Tight ceiling
    node = AgentNodeProfile(
        description="A super long system prompt that alone exceeds the ceiling...",
        type="agent",
        baseline_cognitive_state=CognitiveStateProfile(
            urgency_index=0.5,
            caution_index=0.5,
            divergence_tolerance=0.1,
            semantic_slicing=SemanticSlicingPolicy(
                permitted_classification_tiers=[InformationClassificationProfile.PUBLIC],
                required_semantic_labels=["label1"],
                context_window_token_ceiling=10,  # Ridiculously small
            ),
        ),
    )

    obs1 = ObservationEvent(event_id="obs1", timestamp=100.0, payload={"data": "A"})
    ledger = EpistemicLedgerState(history=[obs1])

    messages = engine._apply_semantic_slicing(node, ledger)

    # Obs1 should be evicted. But system prompt remains and mass is > 10.
    # The loop should break safely and return just the system prompt.
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
