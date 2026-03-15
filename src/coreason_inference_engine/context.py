# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import json
from typing import Any

from coreason_manifest.spec.ontology import (
    AgentNodeProfile,
    AnyStateEvent,
    EpistemicLedgerState,
    ObservationEvent,
    System2RemediationIntent,
    ToolInvocationEvent,
)

from coreason_inference_engine.utils.logger import logger


class ContextHydrator:
    """Translates the deterministic ledger into provider-agnostic conversational arrays."""

    def compile(self, node: AgentNodeProfile, ledger: EpistemicLedgerState) -> list[dict[str, Any]]:
        """
        Flattens the ledger into a conversational array.

        Args:
            node: The AgentNodeProfile defining the persona and constraints.
            ledger: The EpistemicLedgerState containing the causal history.

        Returns:
            A list of dictionary messages conforming to the standard LLM API format.
        """
        messages: list[dict[str, Any]] = []

        # Anchor System Prompt
        system_prompt = node.description

        # AGENT INSTRUCTION: Dynamically translate CognitiveStateProfile mathematical constraints
        if node.baseline_cognitive_state:
            state = node.baseline_cognitive_state
            system_prompt += f"\nUrgency Index: {state.urgency_index}"
            system_prompt += f"\nCaution Index: {state.caution_index}"

        messages.append({"role": "system", "content": system_prompt})

        # Generate Quarantined Event Set
        quarantined_event_ids: set[str] = set()

        # Extract defeasible cascades from the ledger
        if ledger.active_cascades:
            for cascade in ledger.active_cascades:
                quarantined_event_ids.update(cascade.quarantined_event_ids)

        # Extract active rollbacks from the ledger
        if ledger.active_rollbacks:
            for rollback in ledger.active_rollbacks:
                if rollback.invalidated_node_ids:
                    quarantined_event_ids.update(rollback.invalidated_node_ids)

        # Iterate Chronologically and Map Roles
        for event in ledger.history:
            # Type hinting the event
            typed_event: AnyStateEvent = event

            # Handle Event Exclusion
            event_id = getattr(typed_event, "event_id", None)
            if event_id in quarantined_event_ids:
                logger.debug("Quarantined event masked from context", event_id=event_id)
                continue

            # Standard Role Mapping
            if isinstance(typed_event, ObservationEvent):
                # We need to map ObservationEvent payloads into strings.
                content_dict = {
                    k: v.model_dump() if hasattr(v, "model_dump") and v is not None else v
                    for k, v in typed_event.payload.items()
                }
                content_str = json.dumps(content_dict)

                # If this observation is tied to a tool invocation, map as tool response
                if typed_event.triggering_invocation_id:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": typed_event.triggering_invocation_id,
                            "content": content_str,
                        }
                    )
                else:
                    messages.append({"role": "user", "content": content_str})

            elif isinstance(typed_event, ToolInvocationEvent):
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": typed_event.event_id,
                                "type": "function",
                                "function": {
                                    "name": typed_event.tool_name,
                                    "arguments": json.dumps(typed_event.parameters),
                                },
                            }
                        ],
                    }
                )

            elif isinstance(typed_event, System2RemediationIntent):
                # Map System2RemediationIntent to the system role as a mathematical reprimand
                # The target_node_id check is implicit as it's targeted for this execution branch
                messages.append(
                    {
                        "role": "system",
                        "content": typed_event.model_dump_json(),
                    }
                )

        return messages
