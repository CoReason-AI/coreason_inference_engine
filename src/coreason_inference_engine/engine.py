# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import asyncio
import hashlib
import json
import re
import time
import uuid
from typing import Any, cast

import httpx
from coreason_manifest.spec.ontology import (
    ActionSpaceManifest,
    AgentNodeProfile,
    AnyIntent,
    EpistemicLedgerState,
    JSONRPCErrorResponseState,
    JSONRPCErrorState,
    LatentScratchpadReceipt,
    ObservationEvent,
    StateMutationIntent,
    System2RemediationIntent,
    ThoughtBranchState,
    TokenBurnReceipt,
    ToolInvocationEvent,
)
from coreason_manifest.utils.algebra import SCHEMA_REGISTRY, generate_correction_prompt, validate_payload
from pydantic import TypeAdapter, ValidationError

from coreason_inference_engine.context import ContextHydrator
from coreason_inference_engine.interfaces import (
    InferenceConvergenceError,
    InferenceEngineProtocol,
    LLMAdapterProtocol,
)
from coreason_inference_engine.utils.logger import logger


class _AnyIntentAdapter:
    def model_validate_json(self, b: bytes) -> Any:
        # StateMutationIntent does not have a "type" field natively, so we cannot use a simple string discriminator.
        # We'll just rely on a non-discriminated Union for the patched types.
        patched_intent = AnyIntent | ToolInvocationEvent | StateMutationIntent | System2RemediationIntent
        return TypeAdapter(patched_intent).validate_json(b)


if "AnyIntent" not in SCHEMA_REGISTRY:
    SCHEMA_REGISTRY["AnyIntent"] = cast("Any", _AnyIntentAdapter())


class InferenceEngine(InferenceEngineProtocol):
    """The stateless cognitive bridge connecting deterministic rules to LLMs."""

    def __init__(
        self, adapter: LLMAdapterProtocol, hydrator: ContextHydrator | None = None, max_concurrent_tasks: int = 1000
    ) -> None:
        self.adapter = adapter
        self.hydrator = hydrator or ContextHydrator()
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)

    def _compile_watermark_biases(self, _watermark: Any | None) -> dict[int, float] | None:
        # Placeholder for steganography contract
        return None

    def _extract_latent_traces(
        self, raw_output: str, node: AgentNodeProfile
    ) -> tuple[str, LatentScratchpadReceipt | None]:
        # FR-3.1: Structural extraction of <think> tags
        require_think_tags = False
        if node.grpo_reward_policy and node.grpo_reward_policy.format_contract:
            require_think_tags = node.grpo_reward_policy.format_contract.require_think_tags

        if not require_think_tags:
            return raw_output, None

        start_tag = "<think>"
        end_tag = "</think>"
        start_idx = raw_output.find(start_tag)
        end_idx = raw_output.find(end_tag)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            think_content = raw_output[start_idx + len(start_tag) : end_idx].strip()
            clean_json_str = raw_output[:start_idx] + raw_output[end_idx + len(end_tag) :]
            clean_json_str = clean_json_str.strip()

            # Remove optional ```json ... ``` wrapper
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_json_str, re.DOTALL)
            if json_match:
                clean_json_str = json_match.group(1).strip()

            content_hash = hashlib.sha256(think_content.encode("utf-8")).hexdigest()
            branch_id = f"branch_{uuid.uuid4().hex[:8]}"

            branch = ThoughtBranchState(
                branch_id=branch_id,
                parent_branch_id=None,
                latent_content_hash=content_hash,
                prm_score=None,
            )

            receipt = LatentScratchpadReceipt(
                trace_id=f"trace_{uuid.uuid4().hex[:8]}",
                explored_branches=[branch],
                discarded_branches=[],
                resolution_branch_id=branch_id,
                total_latent_tokens=self.adapter.count_tokens(think_content),
            )

            return clean_json_str, receipt

        return raw_output, None

    def _determine_target_schema(self, _node: AgentNodeProfile) -> str:
        # TODO: Return dynamic key based on schema constraints
        return "AnyIntent"

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> int:
        # Placeholder
        return input_tokens + output_tokens

    def _apply_semantic_slicing(self, node: AgentNodeProfile, ledger: EpistemicLedgerState) -> list[dict[str, Any]]:
        """
        Applies semantic slicing policy to prevent context window overflow.
        Recursively evicts the oldest ObservationEvent payloads if token ceiling is exceeded.
        Caps historical System2RemediationIntent payloads to only the most recent one.
        """
        ceiling = None
        if node.baseline_cognitive_state and node.baseline_cognitive_state.semantic_slicing:
            ceiling = node.baseline_cognitive_state.semantic_slicing.context_window_token_ceiling

        history = list(ledger.history)

        # Destructive Eviction Prevention: cap System2RemediationIntent to the most recent one
        remediation_indices = [i for i, event in enumerate(history) if isinstance(event, System2RemediationIntent)]
        if len(remediation_indices) > 1:
            indices_to_remove = set(remediation_indices[:-1])
            history = [event for i, event in enumerate(history) if i not in indices_to_remove]

        # EpistemicLedgerState is frozen, so we must use model_copy(update={"history": history})
        sliced_ledger = ledger.model_copy(update={"history": history})

        messages = self.hydrator.compile(node, sliced_ledger)

        if not ceiling:
            return messages

        messages_str = json.dumps(messages)
        token_mass = self.adapter.count_tokens(messages_str)

        while token_mass > ceiling:
            # Find the oldest ObservationEvent to evict
            obs_indices = [i for i, event in enumerate(sliced_ledger.history) if isinstance(event, ObservationEvent)]
            if not obs_indices:
                break  # Cannot evict any more observations

            # Remove the oldest ObservationEvent
            oldest_obs_idx = obs_indices[0]
            new_history = list(sliced_ledger.history)
            new_history.pop(oldest_obs_idx)
            sliced_ledger = sliced_ledger.model_copy(update={"history": new_history})

            messages = self.hydrator.compile(node, sliced_ledger)
            messages_str = json.dumps(messages)
            token_mass = self.adapter.count_tokens(messages_str)

        return messages

    async def generate_intent(
        self,
        node: AgentNodeProfile,
        ledger: EpistemicLedgerState,
        node_id: str,
        action_space: ActionSpaceManifest,
    ) -> tuple[AnyIntent, TokenBurnReceipt, LatentScratchpadReceipt | None]:
        """
        Translates the passive ledger into active generation.
        Executes Context Hydration, the Forward Pass, and System 2 Remediation.

        Raises:
            InferenceConvergenceError: If max_loops are exceeded or upstream API fatally fails.
            asyncio.CancelledError: If preempted by the Orchestrator.
        """
        # CRITICAL FIX: Deadlock Prevention via Local Backpressure
        if self._semaphore.locked():
            logger.warning("Semaphore saturated, yielding 429 JSONRPCErrorResponseState", node_id=node_id)
            error_intent = cast(
                "AnyIntent",
                JSONRPCErrorResponseState(
                    jsonrpc="2.0",
                    error=JSONRPCErrorState(
                        code=429,
                        message="Too Many Requests: Local backpressure threshold exceeded.",
                    ),
                ),
            )
            receipt = TokenBurnReceipt(
                event_id=f"burn_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                tool_invocation_id="",
                input_tokens=0,
                output_tokens=0,
                burn_magnitude=0,
            )
            return error_intent, receipt, None

        async with self._semaphore:
            # FR-1.2.5: System 1 Fast-Path Evaluation (TODO: implementation logic omitted for this skeleton)
            # if node.reflex_policy: ...

            # FR-1.3 & FR-2.5: Context Compilation & Semantic Slicing
            messages = self._apply_semantic_slicing(node, ledger)

            # Explicitly map tools from injected action space (air-gapped resolution)
            # Assuming action_space.native_tools holds the list of standard tool schemas
            tools = self.adapter.project_tools([t.model_dump() for t in action_space.native_tools])

            max_loops = node.correction_policy.max_loops if node.correction_policy else 3
            total_input_tokens = 0
            total_output_tokens = 0

            # Apply PEFT adapters if requested
            if node.peft_adapters:
                await self.adapter.apply_peft_adapters(node.peft_adapters)

            logit_biases = self._compile_watermark_biases(node.logit_steganography)

            import random

            # Initialize backoff state outside the loop
            current_backoff = 1.0
            max_backoff = 60.0
            global_start_time = time.time()

            # Need to get global timeout, default to a sensible value if not provided
            global_timeout = 300.0
            if node.correction_policy and getattr(node.correction_policy, "global_timeout_seconds", None) is not None:
                global_timeout = float(getattr(node.correction_policy, "global_timeout_seconds", 300.0))

            attempt = 0
            while attempt < max_loops:
                raw_output = ""
                usage_metrics = {"input_tokens": 0, "output_tokens": 0}

                current_max_tokens = 500 if attempt > 0 else None

                stream = self.adapter.generate_stream(
                    messages, tools, temperature=0.0, logit_biases=logit_biases, max_tokens=current_max_tokens
                )
                try:
                    import ijson  # type: ignore

                    events = ijson.sendable_list()
                    parser = ijson.parse_coro(events)
                    structural_violation = False

                    try:
                        # FR-1.6 / FR-2.6: Active Preemption Check & Stream Consumption
                        async for chunk, usage in stream:
                            raw_output += chunk
                            if usage:
                                usage_metrics = usage

                            # FR-3.2: Fail-fast incremental JSON parsing
                            try:
                                # Encode using errors="replace" to prevent panic on severed/invalid UTF-8 mid-stream
                                parser.send(chunk.encode("utf-8", errors="replace"))
                                for prefix, event, value in events:
                                    if prefix == "" and event == "map_key":
                                        # Only specific top-level keys are expected in AnyIntent structures
                                        allowed_keys = {
                                            "tool_name",
                                            "parameters",
                                            "agent_attestation",
                                            "zk_proof",  # ToolInvocationEvent
                                            "mutations",
                                            "ledger_hash_pre",
                                            "ledger_hash_post",  # StateMutationIntent
                                            "fault_id",
                                            "target_node_id",
                                            "failing_pointers",
                                            "remediation_prompt",  # System2RemediationIntent
                                        }
                                        if (
                                            value not in allowed_keys
                                            and value != "type"
                                            and value != "event_id"
                                            and value != "timestamp"
                                        ):
                                            structural_violation = True
                                            break
                                events.clear()
                            except ijson.JSONError, UnicodeEncodeError:
                                # We ignore standard parse errors during streaming since it's incomplete
                                events.clear()

                            if structural_violation:
                                # Sever connection immediately to save output tokens
                                await stream.aclose()
                                break

                    except httpx.HTTPStatusError as e:
                        if e.response.status_code in (429, 502, 503):
                            # Calculate full jitter exponential backoff
                            jitter = random.uniform(0, current_backoff)  # noqa: S311

                            elapsed_time = time.time() - global_start_time
                            if elapsed_time + jitter > global_timeout:
                                raise InferenceConvergenceError(
                                    f"SLA Contention: Required backoff delay {jitter}s exceeds remaining global SLA "
                                    f"({global_timeout - elapsed_time}s)"
                                ) from e

                            await asyncio.sleep(jitter)
                            current_backoff = min(current_backoff * 2, max_backoff)
                            continue  # Retry without incrementing attempt counter since it's a transient fault
                        raise

                    # CRITICAL FIX: Severed Stream Token Tracking & Panic Prevention
                    # Fallback to local count_tokens if usage metrics are missing
                    in_tokens = usage_metrics.get("input_tokens", 0)
                    if not in_tokens:
                        safe_input = json.dumps(messages).encode("utf-8", errors="replace").decode("utf-8")
                        in_tokens = self.adapter.count_tokens(safe_input)

                    out_tokens = usage_metrics.get("output_tokens", 0)
                    if not out_tokens:
                        safe_output = raw_output.encode("utf-8", errors="replace").decode("utf-8")
                        out_tokens = self.adapter.count_tokens(safe_output)

                    total_input_tokens += in_tokens
                    total_output_tokens += out_tokens

                    # Optional: Fail-fast JSON stream parsing could happen inside the loop above

                    clean_json_str, scratchpad = self._extract_latent_traces(raw_output, node)

                    target_schema_key = self._determine_target_schema(node)

                    # Zero-Trust Egress: Pass byte string to validation functor
                    # validate_payload raises ValidationError on failure
                    valid_intent = validate_payload(target_schema_key, clean_json_str.encode("utf-8", errors="replace"))

                    # FR-4.5: Hallucinated Tool Escalation
                    if isinstance(valid_intent, ToolInvocationEvent):
                        allowed_tools = {t.tool_name for t in action_space.native_tools}
                        if valid_intent.tool_name not in allowed_tools:
                            from pydantic_core import ValidationError as PydanticCoreValidationError

                            raise PydanticCoreValidationError.from_exception_data(
                                title=target_schema_key,
                                line_errors=[
                                    {
                                        "type": "value_error",
                                        "loc": ("tool_name",),
                                        "input": valid_intent.tool_name,
                                        "ctx": {
                                            "error": ValueError(
                                                f"Tool '{valid_intent.tool_name}' not found in ActionSpaceManifest."
                                            )
                                        },
                                    }
                                ],
                            )

                    # Check for tool invocation ID
                    invocation_cid = valid_intent.event_id if isinstance(valid_intent, ToolInvocationEvent) else None

                    # Build tracking receipt
                    burn_receipt = TokenBurnReceipt(
                        event_id=f"burn_{uuid.uuid4().hex[:8]}",
                        timestamp=time.time(),
                        tool_invocation_id=invocation_cid or "",
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        burn_magnitude=self._calculate_cost(total_input_tokens, total_output_tokens),
                    )

                    return cast("AnyIntent", valid_intent), burn_receipt, scratchpad

                except ValidationError as e:
                    # FR-3.3, FR-3.4: Trap validation failure and generate mathematical reprimand
                    fault_id = f"fault_{uuid.uuid4().hex[:8]}"

                    # CRITICAL: Pass exact DID string (node_id) to prevent remediation crash
                    remediation = generate_correction_prompt(error=e, target_node_id=node_id, fault_id=fault_id)

                    # Inject prompt and retry
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": remediation.model_dump_json()})

                    logger.warning(
                        "Validation error during generation; entering remediation loop",
                        attempt=attempt,
                        max_loops=max_loops,
                        error=str(e),
                    )
                    attempt += 1  # Increment attempt on ValidationError
                except asyncio.CancelledError:
                    # FR-1.6: TCP Teardown Shielding
                    # Await aclose shielded to guarantee the TCP FIN/RST packet is successfully dispatched
                    # despite the cancellation context, ensuring zero-leak termination.
                    await asyncio.shield(stream.aclose())
                    raise

            # FR-4.3: Convergence Failure (Loop Bounding)
            raise InferenceConvergenceError(f"LLM failed to converge after {max_loops} attempts.")
