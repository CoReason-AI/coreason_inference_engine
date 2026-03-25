# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import json
from typing import Any, Literal

from coreason_inference_engine.utils.logger import logger

ProviderMode = Literal["standard", "anthropic", "o1"]


class ContextHydrator:
    """Translates the deterministic ledger into provider-agnostic conversational arrays."""

    def __init__(self, provider_mode: ProviderMode = "standard") -> None:
        self.provider_mode = provider_mode

    def compile(self, node: Any, ledger: Any) -> list[dict[str, Any]]:
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
        system_prompt = node.get("description", "") if isinstance(node, dict) else getattr(node, "description", "")

        # AGENT INSTRUCTION: Dynamically translate CognitiveStateProfile mathematical constraints
        baseline_cognitive_state = (
            node.get("baseline_cognitive_state")
            if isinstance(node, dict)
            else getattr(node, "baseline_cognitive_state", None)
        )
        if baseline_cognitive_state:
            state = baseline_cognitive_state
            urgency = state.get("urgency_index") if isinstance(state, dict) else getattr(state, "urgency_index", None)
            system_prompt += f"\nUrgency Index: {urgency}"
            caution = state.get("caution_index") if isinstance(state, dict) else getattr(state, "caution_index", None)
            system_prompt += f"\nCaution Index: {caution}"

        if self.provider_mode == "o1":
            # o1/o3 deprecate system role, use developer/user instead
            messages.append({"role": "developer", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": system_prompt})

        # Generate Quarantined Event Set
        quarantined_event_ids: set[str] = set()

        # Extract defeasible cascades from the ledger
        active_cascades = (
            ledger.get("active_cascades", []) if isinstance(ledger, dict) else getattr(ledger, "active_cascades", [])
        )
        if active_cascades:
            for cascade in active_cascades:
                quarantined_event_ids.update(
                    cascade.get("quarantined_event_ids", [])
                    if isinstance(cascade, dict)
                    else getattr(cascade, "quarantined_event_ids", [])
                )

        # Extract active rollbacks from the ledger
        active_rollbacks = (
            ledger.get("active_rollbacks", []) if isinstance(ledger, dict) else getattr(ledger, "active_rollbacks", [])
        )
        if active_rollbacks:
            for rollback in active_rollbacks:
                invalidated_node_ids = (
                    rollback.get("invalidated_node_ids", [])
                    if isinstance(rollback, dict)
                    else getattr(rollback, "invalidated_node_ids", [])
                )
                if invalidated_node_ids:
                    quarantined_event_ids.update(invalidated_node_ids)

        # Iterate Chronologically and Map Roles
        history = ledger.get("history", []) if isinstance(ledger, dict) else getattr(ledger, "history", [])
        for event in history:
            # Type hinting the event
            typed_event: Any = event

            # Handle Event Exclusion
            event_id = (
                (
                    typed_event.get("event_id")
                    if isinstance(typed_event, dict)
                    else getattr(typed_event, "event_id", None)
                )
                if isinstance(typed_event, dict)
                else getattr(typed_event, "event_id", None)
            )
            if event_id in quarantined_event_ids:
                logger.debug("Quarantined event masked from context", event_id=event_id)
                continue

            # Standard Role Mapping
            if (isinstance(typed_event, dict) and typed_event.get("type") == "observation") or type(
                typed_event
            ).__name__ == "ObservationEvent":
                # We need to map ObservationEvent payloads into strings.
                content_dict = {
                    k: v.model_dump() if hasattr(v, "model_dump") and v is not None else v
                    for k, v in (
                        typed_event.get("payload", {})
                        if isinstance(typed_event, dict)
                        else getattr(typed_event, "payload", {})
                    ).items()
                }
                content_str = json.dumps(content_dict)

                # If this observation is tied to a tool invocation, map as tool response
                if (
                    typed_event.get("triggering_invocation_id")
                    if isinstance(typed_event, dict)
                    else getattr(typed_event, "triggering_invocation_id", None)
                ):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": (
                                typed_event.get("triggering_invocation_id")
                                if isinstance(typed_event, dict)
                                else getattr(typed_event, "triggering_invocation_id", None)
                            ),
                            "content": content_str,
                        }
                    )
                else:
                    messages.append({"role": "user", "content": content_str})

            elif (isinstance(typed_event, dict) and typed_event.get("type") == "tool_invocation") or type(
                typed_event
            ).__name__ == "ToolInvocationEvent":
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": (
                                    typed_event.get("event_id")
                                    if isinstance(typed_event, dict)
                                    else getattr(typed_event, "event_id", None)
                                ),
                                "type": "function",
                                "function": {
                                    "name": (
                                        typed_event.get("tool_name")
                                        if isinstance(typed_event, dict)
                                        else getattr(typed_event, "tool_name", None)
                                    ),
                                    "arguments": json.dumps(
                                        typed_event.get("parameters", {})
                                        if isinstance(typed_event, dict)
                                        else getattr(typed_event, "parameters", {})
                                    ),
                                },
                            }
                        ],
                    }
                )

            elif (isinstance(typed_event, dict) and typed_event.get("type") == "system2_remediation") or type(
                typed_event
            ).__name__ == "System2RemediationIntent":
                # Map System2RemediationIntent to the system role as a mathematical reprimand
                # The target_node_id check is implicit as it's targeted for this execution branch
                if self.provider_mode == "o1":
                    messages.append(
                        {
                            "role": "developer",
                            "content": json.dumps(
                                typed_event if isinstance(typed_event, dict) else typed_event.model_dump()
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "system",
                            "content": json.dumps(
                                typed_event if isinstance(typed_event, dict) else typed_event.model_dump()
                            ),
                        }
                    )

            elif (isinstance(typed_event, dict) and typed_event.get("type") == "continuous_observation_stream") or type(
                typed_event
            ).__name__ == "ContinuousObservationStream":  # pragma: no cover
                buffer_content = "\n".join(
                    str(token)
                    for token in (
                        typed_event.get("token_buffer", [])
                        if isinstance(typed_event, dict)
                        else getattr(typed_event, "token_buffer", [])
                    )
                )
                messages.append({"role": "user", "content": buffer_content})

            elif (isinstance(typed_event, dict) and typed_event.get("type") == "state_mutation") or type(
                typed_event
            ).__name__ == "StateMutationIntent":  # pragma: no cover
                # Ensure the LLM remembers its own previously generated JSON objects
                # so it maintains context of its continuous intent projection pattern.
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            typed_event if isinstance(typed_event, dict) else typed_event.model_dump()
                        ),
                    }
                )

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
