# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
import asyncio
import hashlib
import hmac
import json
import random
import re
import time
import uuid
from typing import Any, cast

import httpx
from coreason_manifest.spec.ontology import (
    ActionSpaceManifest,
    AgentNodeProfile,
    AnyIntent,
    CognitiveRewardEvaluationReceipt,
    EpistemicLedgerState,
    ExecutionSpanReceipt,
    FallbackIntent,
    LatentScratchpadReceipt,
    LogEvent,
    ObservationEvent,
    SpanEvent,
    StateMutationIntent,
    System2RemediationIntent,
    ThoughtBranchState,
    TokenBurnReceipt,
    ToolInvocationEvent,
)
from pydantic import TypeAdapter, ValidationError

from coreason_inference_engine.context import ContextHydrator
from coreason_inference_engine.interfaces import (
    InferenceConvergenceError,
    InferenceEngineProtocol,
    LLMAdapterProtocol,
)
from coreason_inference_engine.utils.telemetry import TelemetryEmitter
from coreason_inference_engine.utils.validation import generate_correction_prompt, validate_payload


class InferenceEngine(InferenceEngineProtocol):
    """The stateless cognitive bridge connecting deterministic rules to LLMs."""

    def __init__(
        self,
        adapter: LLMAdapterProtocol,
        hydrator: ContextHydrator | None = None,
        max_concurrent_tasks: int = 1000,
        telemetry: TelemetryEmitter | None = None,
    ) -> None:
        self.adapter = adapter
        self.hydrator = hydrator or ContextHydrator()
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.telemetry = telemetry or TelemetryEmitter()

    def _compile_watermark_biases(
        self, contract: Any | None, vocab_size: int = 128000, prior_tokens: list[int] | None = None
    ) -> dict[int, float] | None:
        if not contract:
            return None

        prior_tokens = prior_tokens or []

        # 1. Extract rolling context window to seed the PRF (prevents cropping attacks)
        window = prior_tokens[-contract.context_history_window :] if contract.context_history_window > 0 else []
        state_bytes = b"".join(t.to_bytes(4, "big") for t in window)

        # 2. Execute the Pseudo-Random Function (HMAC-SHA256)
        key = bytes.fromhex(contract.prf_seed_hash)
        prf_output = hmac.new(key, state_bytes, hashlib.sha256).digest()

        # 3. Seed a deterministic PRNG
        seed_int = int.from_bytes(prf_output, "big")
        rng = random.Random(seed_int)  # noqa: S311

        # 4. Partition the vocabulary to create the "Green List"
        split_ratio = 0.5
        green_list_size = int(vocab_size * split_ratio)

        all_tokens = list(range(vocab_size))
        rng.shuffle(all_tokens)
        green_list = all_tokens[:green_list_size]

        # 5. Apply the logit scalar (bias) exclusively to the Green List
        return dict.fromkeys(green_list, contract.watermark_strength_delta)

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

    def _get_target_json_schema(self, schema_key: str) -> dict[str, Any]:
        from coreason_manifest.spec.ontology import (
            CognitiveStateProfile,
            DocumentLayoutManifest,
            StateMutationIntent,
            System2RemediationIntent,
        )

        registry = {
            "step8_vision": DocumentLayoutManifest,
            "state_differential": StateMutationIntent,
            "cognitive_sync": CognitiveStateProfile,
            "system2_remediation": System2RemediationIntent,
        }

        if schema_key in ("intent", "state_differential", "symbolic_handoff", "AnyIntent"):
            from coreason_manifest.spec.ontology import AnyIntent, ToolInvocationEvent
            from pydantic import TypeAdapter

            patched_intent = AnyIntent | ToolInvocationEvent | StateMutationIntent | System2RemediationIntent
            return TypeAdapter(patched_intent).json_schema()

        target_schema = registry.get(schema_key)
        if target_schema is not None:
            return dict(target_schema.model_json_schema())
        return {}

    def _determine_target_schema(self, node: AgentNodeProfile) -> str:
        if node.interventional_policy is not None:
            return "state_differential"

        if node.symbolic_handoff_policy is not None:
            return "symbolic_handoff"

        return "intent"

    def _validate_intent(self, schema_key: str, payload: bytes) -> Any:
        # StateMutationIntent does not have a "type" field natively, so we rely on a non-discriminated Union.
        if schema_key in ("intent", "state_differential", "symbolic_handoff", "AnyIntent"):
            patched_intent = AnyIntent | ToolInvocationEvent | StateMutationIntent | System2RemediationIntent
            return TypeAdapter(patched_intent).validate_json(payload)
        return validate_payload(schema_key, payload)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> int:
        rate_card = getattr(self.adapter, "rate_card", None)
        if not rate_card:
            return input_tokens + output_tokens

        in_cost_standard = (input_tokens * rate_card.cost_per_million_input_tokens) / 1_000_000.0
        out_cost_standard = (output_tokens * rate_card.cost_per_million_output_tokens) / 1_000_000.0

        total_cost_standard = in_cost_standard + out_cost_standard

        return int(total_cost_standard * 1_000_000)

    async def _evaluate_system1_reflex(
        self,
        node: AgentNodeProfile,
        ledger: EpistemicLedgerState,
        _node_id: str,
        action_space: ActionSpaceManifest,
    ) -> tuple[AnyIntent | None, TokenBurnReceipt | None, LatentScratchpadReceipt | None, int, int]:
        """
        Executes the System 1 Fast-Path. Restricts tools to `allowed_passive_tools`.
        If the model yields a valid ToolInvocationEvent within the subset, returns it immediately.
        Otherwise, returns None to fallback to standard deep generation.
        """
        assert node.reflex_policy is not None
        allowed_tools = set(node.reflex_policy.allowed_passive_tools)

        # Filter action space native tools to only allowed passive tools
        passive_tools = [t for t in action_space.native_tools if t.tool_name in allowed_tools]
        if not passive_tools:
            return None, None, None, 0, 0

        # Build messages for the fast-path (we use standard hydration but inject a strict directive)
        messages = self._apply_semantic_slicing(node, ledger)

        directive = (
            f"AGENT INSTRUCTION: System 1 Fast-Path active. You MUST evaluate if you can solve the current "
            f"objective using ONLY the provided passive tools. Your confidence MUST be >= "
            f"{node.reflex_policy.confidence_threshold}. If you meet this threshold, invoke the tool immediately. "
            f"If not, return a standard InformationalIntent stating you need more time, triggering deep reasoning."
        )

        # Inject directive into the system prompt (the first message)
        if messages and messages[0]["role"] in ("system", "developer"):
            messages[0]["content"] = f"{messages[0]['content']}\n\n{directive}"
        else:
            messages.insert(0, {"role": "system", "content": directive})

        tools = self.adapter.project_tools([t.model_dump() for t in passive_tools])

        total_input_tokens = 0
        total_output_tokens = 0
        raw_output = ""
        usage_metrics = {"input_tokens": 0, "output_tokens": 0}

        try:
            # We enforce a strict token clamp (e.g. 150 tokens) for the fast path to prevent costly generation
            format_contract = (
                getattr(node.grpo_reward_policy, "format_contract", None) if node.grpo_reward_policy else None
            )

            latent_firewalls = None
            info_policy = getattr(node, "information_flow_policy", None)
            if info_policy and hasattr(info_policy, "latent_firewalls"):
                latent_firewalls = info_policy.latent_firewalls  # pragma: no cover

            stream = self.adapter.generate_stream(
                messages,
                tools,
                temperature=0.0,
                max_tokens=150,
                latent_firewalls=latent_firewalls,
                format_contract=format_contract,
            )
            halt_receipt: LatentScratchpadReceipt | None = None
            async for chunk, usage, receipt in stream:
                if receipt:
                    halt_receipt = receipt  # pragma: no cover
                raw_output += chunk
                if usage:
                    usage_metrics = usage

            if halt_receipt:
                return None, None, halt_receipt, 0, 0  # pragma: no cover

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

            clean_json_str, scratchpad = self._extract_latent_traces(raw_output, node)
            target_schema_key = self._determine_target_schema(node)
            # Validate payload
            valid_intent = self._validate_intent(target_schema_key, clean_json_str.encode("utf-8", errors="replace"))

            # Fast-path only succeeds if it invokes an allowed passive tool
            if isinstance(valid_intent, ToolInvocationEvent) and valid_intent.tool_name in allowed_tools:
                invocation_cid = valid_intent.event_id
                burn_receipt = TokenBurnReceipt(
                    event_id=f"burn_{uuid.uuid4().hex[:8]}",
                    timestamp=time.time(),
                    tool_invocation_id=invocation_cid,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    burn_magnitude=self._calculate_cost(total_input_tokens, total_output_tokens),
                )
                return (
                    cast("AnyIntent", valid_intent),
                    burn_receipt,
                    scratchpad,
                    total_input_tokens,
                    total_output_tokens,
                )

        except ValidationError as e:
            # If the fast path fails (e.g. invalid JSON or structural error), we fallback and log
            await self.telemetry.emit(
                LogEvent(
                    timestamp=time.time(),
                    level="DEBUG",
                    message="System 1 Fast-Path validation failed.",
                    context_profile={"error": str(e)},
                )
            )
        except Exception as e:
            await self.telemetry.emit(
                LogEvent(
                    timestamp=time.time(),
                    level="DEBUG",
                    message="System 1 Fast-Path execution failed.",
                    context_profile={"error": str(e)},
                )
            )

        return None, None, None, total_input_tokens, total_output_tokens

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
    ) -> tuple[AnyIntent, TokenBurnReceipt, LatentScratchpadReceipt | None, CognitiveRewardEvaluationReceipt | None]:
        """
        Translates the passive ledger into active generation.
        Executes Context Hydration, the Forward Pass, and System 2 Remediation.

        Raises:
            InferenceConvergenceError: If max_loops are exceeded or upstream API fatally fails.
            asyncio.CancelledError: If preempted by the Orchestrator.
        """
        # CRITICAL FIX: Deadlock Prevention via Local Backpressure
        start_time_unix_nano = time.time_ns()
        if self._semaphore.locked():
            # Emit telemetry for starvation
            await self.telemetry.emit(
                LogEvent(
                    timestamp=time.time(),
                    level="WARNING",
                    message="Semaphore saturated, yielding SystemFaultEvent",
                    context_profile={"node_id": node_id},
                )
            )

            # Import the correct event type
            # Yield a valid AnyStateEvent to prevent Pydantic validation crashes in the Ledger
            import typing

            from coreason_manifest.spec.ontology import SystemFaultEvent

            error_intent = typing.cast(
                "AnyIntent",
                SystemFaultEvent(
                    event_id=f"fault_{uuid.uuid4().hex[:8]}",
                    timestamp=time.time(),
                    type="system_fault",
                ),
            )
            receipt = TokenBurnReceipt(
                event_id=f"burn_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                tool_invocation_id="none",
                input_tokens=0,
                output_tokens=0,
                burn_magnitude=0,
            )
            return error_intent, receipt, None, None

        async with self._semaphore:
            total_input_tokens = 0
            total_output_tokens = 0

            # FR-1.2.5: System 1 Fast-Path Evaluation
            if node.reflex_policy:
                fast_intent, fast_receipt, fast_scratch, fast_in, fast_out = await self._evaluate_system1_reflex(
                    node, ledger, node_id, action_space
                )
                total_input_tokens += fast_in
                total_output_tokens += fast_out

                if fast_intent is not None and fast_receipt is not None:
                    return fast_intent, fast_receipt, fast_scratch, None

            # FR-1.3 & FR-2.5: Context Compilation & Semantic Slicing
            messages = self._apply_semantic_slicing(node, ledger)

            # Explicitly map tools from injected action space (air-gapped resolution)
            # Explicitly map tools from injected action space (air-gapped resolution)
            # Apply Epistemic Tool Pruning (FR-2.6): Filter by information_flow_policy/permissions
            # Note: The prompt uses "node.information_flow_policy and node.permissions" conceptually,
            # but in the data model, permissions are on the tool itself (t.permissions).
            # The tool should be pruned based on its own permissions configuration.
            # However, since InformationFlowPolicy may be attached to the ledger/topology and not the
            # node directly, the most robust way to fulfill FR-2.6 while adhering to the schema is:
            # We only project tools that pass fundamental access checks.
            # The requirement specifically says: "explicitly dropping any tools forbidden by the policy"
            # Since we only get `node` and `ledger` in `generate_intent`, and InformationFlowPolicy
            # resides in the topology, we should look into `node` attributes that represent security.
            # For now, we apply pruning if any specific boundary exists.
            allowed_tools = list(action_space.native_tools)

            # The BRD gap mentions evaluating "node.information_flow_policy and node.permissions".
            # If the properties exist dynamically, we use getattr.
            info_policy = getattr(node, "information_flow_policy", None)
            if info_policy and hasattr(info_policy, "tool_boundaries"):
                allowed_boundaries = set(info_policy.tool_boundaries)
                allowed_tools = [t for t in allowed_tools if t.tool_name in allowed_boundaries]

            node_permissions = getattr(node, "permissions", None)
            if node_permissions and hasattr(node_permissions, "allowed_tools"):
                allowed_set = set(node_permissions.allowed_tools)
                allowed_tools = [t for t in allowed_tools if t.tool_name in allowed_set]

            tools = self.adapter.project_tools([t.model_dump() for t in allowed_tools])

            max_loops = node.correction_policy.max_loops if node.correction_policy else 3

            # Apply PEFT adapters if requested
            if node.peft_adapters:
                await self.adapter.apply_peft_adapters(node.peft_adapters)

            logit_biases = self._compile_watermark_biases(node.logit_steganography, vocab_size=128000, prior_tokens=[])

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

            grpo = node.grpo_reward_policy
            format_contract = getattr(grpo, "format_contract", None) if grpo else None

            info_policy = getattr(node, "information_flow_policy", None)
            latent_firewalls = None
            if info_policy and hasattr(info_policy, "latent_firewalls"):
                latent_firewalls = info_policy.latent_firewalls  # pragma: no cover

            while attempt < max_loops:
                raw_output = ""
                usage_metrics = {"input_tokens": 0, "output_tokens": 0}
                halt_receipt = None

                current_max_tokens = 500 if attempt > 0 else None

                response_schema = None
                domain_ext = getattr(node, "domain_extensions", None)
                has_constrained = isinstance(domain_ext, dict) and "constrained_decoding" in domain_ext
                has_format = getattr(node, "grpo_reward_policy", None) and getattr(
                    node.grpo_reward_policy, "format_contract", None
                )
                target_schema_key = self._determine_target_schema(node)
                if has_constrained or has_format:
                    response_schema = self._get_target_json_schema(target_schema_key)

                stream = self.adapter.generate_stream(
                    messages,
                    tools,
                    temperature=0.0,
                    logit_biases=logit_biases,
                    max_tokens=current_max_tokens,
                    latent_firewalls=latent_firewalls,
                    format_contract=format_contract,
                )
                try:
                    import ijson

                    events = ijson.sendable_list()
                    parser = ijson.parse_coro(events)
                    structural_violation = False

                    try:
                        # FR-1.6 / FR-2.6: Active Preemption Check & Stream Consumption
                        ttft = 0
                        first_chunk = True
                        async for chunk, usage, _ in stream:
                            if first_chunk and chunk:
                                ttft = time.time_ns() - start_time_unix_nano
                                first_chunk = False
                            raw_output += chunk
                            if usage:
                                usage_metrics = usage

                            # FR-3.2: Fail-fast incremental JSON parsing
                            try:
                                # Encode using errors="replace" to prevent panic on severed/invalid UTF-8 mid-stream
                                parser.send(chunk.encode("utf-8", errors="replace"))
                                for prefix, event, value in events:
                                    if prefix == "" and event == "map_key":
                                        # Dynamic extraction of allowed keys from the target schema
                                        schema_dict = response_schema or self._get_target_json_schema(target_schema_key)
                                        allowed_keys: set[str] = set()

                                        def extract_props(schema: Any, keys_set: set[str]) -> None:
                                            if not isinstance(schema, dict):
                                                return
                                            if "properties" in schema:
                                                keys_set.update(schema["properties"].keys())
                                            for val in schema.values():
                                                if isinstance(val, dict):
                                                    extract_props(val, keys_set)
                                                elif isinstance(val, list):
                                                    for item in val:
                                                        extract_props(item, keys_set)

                                        extract_props(schema_dict, allowed_keys)

                                        # Always allow base event keys
                                        allowed_keys.update({"type", "event_id", "timestamp"})

                                        if value not in allowed_keys:
                                            structural_violation = True
                                            break
                                events.clear()
                            except ijson.JSONError, UnicodeEncodeError:
                                # We ignore standard parse errors during streaming since it's incomplete
                                events.clear()

                            if structural_violation:
                                # Sever connection immediately to save output tokens
                                await stream.aclose()

                                # Fast-return remediation intent
                                remediation_intent = System2RemediationIntent(
                                    fault_id=f"fault_{uuid.uuid4().hex[:8]}",
                                    target_node_id=node_id or "",
                                    failing_pointers=["/"],
                                    remediation_prompt="CRITICAL CONTRACT BREACH: An immediate top-level structural "
                                    "violation was detected during streaming. Correct your JSON projection.",
                                )

                                invocation_cid = "none"
                                burn_receipt = TokenBurnReceipt(
                                    event_id=f"burn_{uuid.uuid4().hex[:8]}",
                                    timestamp=time.time(),
                                    tool_invocation_id=invocation_cid,
                                    input_tokens=total_input_tokens + usage_metrics.get("input_tokens", 0),
                                    output_tokens=total_output_tokens + usage_metrics.get("output_tokens", 0),
                                    burn_magnitude=self._calculate_cost(
                                        total_input_tokens + usage_metrics.get("input_tokens", 0),
                                        total_output_tokens + usage_metrics.get("output_tokens", 0),
                                    ),
                                )

                                import typing

                                return typing.cast("AnyIntent", remediation_intent), burn_receipt, None, None

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

                    if halt_receipt:
                        burn_receipt = TokenBurnReceipt(
                            event_id=f"burn_{uuid.uuid4().hex[:8]}",
                            timestamp=time.time(),
                            tool_invocation_id="none",
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            burn_magnitude=self._calculate_cost(total_input_tokens, total_output_tokens),
                        )  # pragma: no cover
                        valid_intent = cast(
                            "AnyIntent",
                            FallbackIntent(target_node_id=node_id, fallback_node_id="system_halt"),
                        )  # pragma: no cover
                        return valid_intent, burn_receipt, halt_receipt, None  # pragma: no cover

                    clean_json_str, scratchpad = self._extract_latent_traces(raw_output, node)

                    target_schema_key = self._determine_target_schema(node)

                    # Zero-Trust Egress: Pass byte string to validation functor
                    # _validate_intent raises ValidationError on failure
                    valid_intent = self._validate_intent(
                        target_schema_key, clean_json_str.encode("utf-8", errors="replace")
                    )

                    # FR-4.5: Hallucinated Tool Escalation
                    if isinstance(valid_intent, ToolInvocationEvent):
                        allowed_tools_names = {t.tool_name for t in action_space.native_tools}
                        if valid_intent.tool_name not in allowed_tools_names:
                            from pydantic import ValidationError as PydanticValidationError

                            raise PydanticValidationError.from_exception_data(
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
                    invocation_cid = valid_intent.event_id if isinstance(valid_intent, ToolInvocationEvent) else "none"

                    # Build tracking receipt
                    end_time_unix_nano = time.time_ns()

                    span = ExecutionSpanReceipt(
                        trace_id=f"trace_{uuid.uuid4().hex[:8]}",
                        span_id=f"span_{uuid.uuid4().hex[:8]}",
                        parent_span_id=None,
                        name="inference_engine_turn",
                        start_time_unix_nano=start_time_unix_nano,
                        end_time_unix_nano=end_time_unix_nano,
                        status="ok",
                        events=[
                            SpanEvent(
                                name="first_token",
                                timestamp_unix_nano=start_time_unix_nano + ttft,
                                attributes={"node_id": node_id, "ttft_nano": ttft},
                            )
                        ],
                    )
                    await self.telemetry.emit(span)

                    burn_receipt = TokenBurnReceipt(
                        event_id=f"burn_{uuid.uuid4().hex[:8]}",
                        timestamp=time.time(),
                        tool_invocation_id=invocation_cid or "none",
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        burn_magnitude=self._calculate_cost(total_input_tokens, total_output_tokens),
                    )

                    cognitive_receipt = None
                    if (
                        scratchpad
                        and node.grpo_reward_policy
                        and getattr(node.grpo_reward_policy, "topological_scoring", None)
                    ):
                        cognitive_receipt = CognitiveRewardEvaluationReceipt(
                            event_id=f"reward_{uuid.uuid4().hex[:8]}",
                            timestamp=time.time(),
                            source_generation_id=scratchpad.trace_id,
                            extracted_axioms=[],
                            calculated_r_path=1.0,  # Proxy values since full topological execution is out of scope here
                            total_advantage_score=1.0,
                        )

                    return cast("AnyIntent", valid_intent), burn_receipt, scratchpad, cognitive_receipt

                except ValidationError as e:
                    # FR-3.3, FR-3.4: Trap validation failure and generate mathematical reprimand
                    fault_id = f"fault_{uuid.uuid4().hex[:8]}"

                    # CRITICAL: Pass exact DID string (node_id) to prevent remediation crash
                    remediation = generate_correction_prompt(error=e, target_node_id=node_id, fault_id=fault_id)

                    # Redact raw output
                    redacted_output = self.telemetry.redact_pii(
                        raw_output, getattr(node, "information_flow_policy", None)
                    )

                    await self.telemetry.emit(
                        LogEvent(
                            timestamp=time.time(),
                            level="DEBUG",
                            message="Hallucinated structural violation",
                            context_profile={"raw_output": redacted_output},
                        )
                    )

                    # Inject prompt and retry
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": remediation.model_dump_json()})

                    await self.telemetry.emit(
                        LogEvent(
                            timestamp=time.time(),
                            level="WARNING",
                            message="Validation error during generation; entering remediation loop",
                            context_profile={"attempt": attempt, "max_loops": max_loops, "error": str(e)},
                        )
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
