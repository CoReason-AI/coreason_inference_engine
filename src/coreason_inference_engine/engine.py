# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import asyncio
import time
import uuid
from typing import Any, cast

from coreason_manifest.spec.ontology import (
    ActionSpaceManifest,
    AgentNodeProfile,
    AnyIntent,
    EpistemicLedgerState,
    LatentScratchpadReceipt,
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
        return TypeAdapter(AnyIntent).validate_json(b)


if "AnyIntent" not in SCHEMA_REGISTRY:
    SCHEMA_REGISTRY["AnyIntent"] = _AnyIntentAdapter()


class InferenceEngine(InferenceEngineProtocol):
    """The stateless cognitive bridge connecting deterministic rules to LLMs."""

    def __init__(self, adapter: LLMAdapterProtocol, hydrator: ContextHydrator | None = None) -> None:
        self.adapter = adapter
        self.hydrator = hydrator or ContextHydrator()

    def _compile_watermark_biases(self, _watermark: Any | None) -> dict[int, float] | None:
        # Placeholder for steganography contract
        return None

    def _extract_latent_traces(
        self, raw_output: str, _node: AgentNodeProfile
    ) -> tuple[str, LatentScratchpadReceipt | None]:
        # TODO: Implement structured extraction of <think> tags (FR-3.1).
        # For now, just return the raw payload.
        return raw_output, None

    def _determine_target_schema(self, _node: AgentNodeProfile) -> str:
        # TODO: Return dynamic key based on schema constraints
        return "AnyIntent"

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> int:
        # Placeholder
        return input_tokens + output_tokens

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
        # FR-1.2.5: System 1 Fast-Path Evaluation (TODO: implementation logic omitted for this skeleton)
        # if node.reflex_policy: ...

        # FR-1.3: Context Compilation (Hydration & Projection)
        messages = self.hydrator.compile(node, ledger)

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

        for attempt in range(max_loops):
            raw_output = ""
            usage_metrics = {"input_tokens": 0, "output_tokens": 0}

            try:
                # FR-1.6 / FR-2.6: Active Preemption Check & Stream Consumption
                async for chunk, usage in self.adapter.generate_stream(
                    messages, tools, temperature=0.0, logit_biases=logit_biases
                ):
                    raw_output += chunk
                    if usage:
                        usage_metrics = usage

                total_input_tokens += usage_metrics.get("input_tokens", 0)
                total_output_tokens += usage_metrics.get("output_tokens", 0)

                # Optional: Fail-fast JSON stream parsing could happen inside the loop above

                clean_json_str, scratchpad = self._extract_latent_traces(raw_output, node)

                target_schema_key = self._determine_target_schema(node)

                # Zero-Trust Egress: Pass byte string to validation functor
                # validate_payload raises ValidationError on failure
                valid_intent = validate_payload(target_schema_key, clean_json_str.encode("utf-8"))

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
            except asyncio.CancelledError:
                # Provide strict teardown (In standard httpx client usage, canceling task cleans socket)
                # Ensure the generator cleanly drops the stream.
                raise

        # FR-4.3: Convergence Failure (Loop Bounding)
        raise InferenceConvergenceError(f"LLM failed to converge after {max_loops} attempts.")
