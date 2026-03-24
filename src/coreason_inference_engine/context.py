# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import json
from typing import Any, Literal

from coreason_manifest.spec.ontology import (
    AgentNodeProfile,
    AnyStateEvent,
    ContinuousObservationStream,
    EpistemicLedgerState,
    ObservationEvent,
    System2RemediationIntent,
    ToolInvocationEvent,
    StateMutationIntent
)

from coreason_inference_engine.utils.logger import logger

ProviderMode = Literal["standard", "anthropic", "o1"]


class ContextHydrator:
    """Translates the deterministic ledger into provider-agnostic conversational arrays."""

    def __init__(self, provider_mode: ProviderMode = "standard") -> None:
        self.provider_mode = provider_mode

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

        # FIX: Append domain extensions to mathematically guarantee constraint adherence
        if getattr(node, "domain_extensions", None):
            system_prompt += "\n\nCRITICAL SYSTEM EXTENSIONS:"
            for key, val in node.domain_extensions.items():
                system_prompt += f"\n- {key}: {val}"

        if self.provider_mode == "o1":
            # o1/o3 deprecate system role, use developer/user instead
            messages.append({"role": "developer", "content": system_prompt})
        else:
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
                if self.provider_mode == "o1":
                    messages.append(
                        {
                            "role": "developer",
                            "content": typed_event.model_dump_json(),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "system",
                            "content": typed_event.model_dump_json(),
                        }
                    )
            
            elif isinstance(typed_event, ContinuousObservationStream):
                buffer_content = "\n".join(str(token) for token in typed_event.token_buffer)
                messages.append({"role": "user", "content": buffer_content})

            elif isinstance(typed_event, StateMutationIntent):
                # Ensure the LLM remembers its own previously generated JSON objects
                # so it maintains context of its continuous intent projection pattern.
                messages.append({
                    "role": "assistant",
                    "content": typed_event.model_dump_json()
                })

        if self.provider_mode == "anthropic":
            messages = self._apply_anthropic_grammar(messages)

        print("\n\nDEBUG EXACT MESSAGES PAYLOAD:\n", json.dumps(messages, indent=2), "\n\n")
        return messages

    def _apply_anthropic_grammar(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Applies Anthropic Strict Alternation grammar.
        - Collapses consecutive user messages using explicit <observation> XML tags.
        - Injects dummy assistant acknowledgment if an asynchronous ToolInvocationEvent
          lacks a subsequent user observation.
        """
        if not messages:
            return messages

        # Anthropic allows system as the first message
        result: list[dict[str, Any]] = []
        user_buffer: list[str] = []

        def flush_user_buffer() -> None:
            if not user_buffer:
                return
            if len(user_buffer) == 1:
                result.append({"role": "user", "content": user_buffer[0]})
            else:
                combined_content = ""
                for obs in user_buffer:
                    combined_content += f"<observation>\n{obs}\n</observation>\n"
                result.append({"role": "user", "content": combined_content.strip()})
            user_buffer.clear()

        for msg in messages:
            role = msg["role"]
            if role == "system":
                flush_user_buffer()
                result.append(msg)
            elif role == "user":
                user_buffer.append(msg["content"])
            elif role == "tool":
                # Anthropic expects tool results as user messages containing tool_result blocks
                # We format it to match user role with a tool_result string for generic mapping
                # However, the user observation logic says "collapse consecutive user messages"
                # so we can buffer tool contents as user observations.
                # Actually, standard OpenAI has "tool" role. We must map "tool" to "user" for Anthropic
                # with an explicit structure, or just collapse it. The FRD says:
                # "collapse consecutive ObservationEvents into a single user turn using explicit <observation> XML tags"
                # For simplicity, we treat "tool" responses as user observations to be buffered.
                user_buffer.append(f"Tool {msg.get('tool_call_id', 'unknown')} Result: {msg['content']}")
            elif role == "assistant":
                # If the previous was also assistant, and we have NO user buffer to flush,
                # we need to inject a dummy user message to separate them.
                if result and result[-1]["role"] == "assistant" and not user_buffer:
                    result.append({"role": "user", "content": "<observation>\nDummy acknowledgment\n</observation>"})
                flush_user_buffer()
                result.append(msg)

        flush_user_buffer()

        # Handle trailing tool invocation: if the last message is assistant with tool_calls,
        # and no user observation follows, the API expects a user message with the tool results!
        # If there is no user observation, Anthropic will complain.
        # "If an asynchronous ToolInvocationEvent lacks a subsequent user observation,
        # the engine MUST inject a dummy assistant acknowledgment"
        # If the LAST event is ToolInvocationEvent (assistant), then the prompt ends with assistant.
        # But Anthropic prompt generation ALWAYS expects the assistant to speak next.
        # If it ends with assistant, the LLM cannot generate the next message.
        # So if it ends with assistant, we MUST append a user turn asking it to continue.
        if result and result[-1]["role"] == "assistant":
            result.append({"role": "user", "content": "Please continue or provide the next action."})

        return result
