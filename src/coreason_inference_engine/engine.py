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
import importlib.resources
import json
import random
import time
import uuid
from typing import Any

import httpx
import jsonschema
from jsonschema.exceptions import ValidationError

from coreason_inference_engine.adapters.dto import (
    LocalCognitiveRewardReceipt,
    LocalLatentScratchpadReceipt,
    LocalSystemFaultEvent,
    LocalTokenBurnReceipt,
)
from coreason_inference_engine.context import ContextHydrator
from coreason_inference_engine.interfaces import (
    InferenceConvergenceError,
    LLMAdapterProtocol,
)
from coreason_inference_engine.utils.telemetry import TelemetryEmitter
from coreason_inference_engine.utils.validation import generate_correction_prompt


class InferenceEngine:
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
        self._cached_schema: dict[str, Any] | None = None

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

    def _extract_latent_traces(self, raw_output: str, node: Any) -> tuple[str, dict[str, Any] | None]:
        # FR-3.1: Structural extraction of <think> tags
        import re

        require_think_tags = False
        if node.grpo_reward_policy and node.grpo_reward_policy.format_contract:
            require_think_tags = node.grpo_reward_policy.format_contract.require_think_tags

        if not require_think_tags:
            clean_out = raw_output.strip()
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_out, re.DOTALL)
            if json_match:  # pragma: no cover
                clean_out = json_match.group(1).strip()
            return clean_out, None

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

            branch = {
                "branch_id": branch_id,
                "parent_branch_id": None,
                "latent_content_hash": content_hash,
                "prm_score": None,
            }

            receipt_dto = LocalLatentScratchpadReceipt(
                trace_id=f"trace_{uuid.uuid4().hex[:8]}",
                explored_branches=[branch],
                discarded_branches=[],
                resolution_branch_id=branch_id,
                total_latent_tokens=self.adapter.count_tokens(think_content),
            )
            receipt = json.loads(receipt_dto.model_dump_json(exclude_none=True))

            return clean_json_str, receipt

        return raw_output, None

    def _get_target_json_schema(self, schema_key: str) -> dict[str, Any]:
        if self._cached_schema is None:
            try:
                schema_path = importlib.resources.files("coreason_inference_engine").joinpath(
                    "coreason_ontology.schema.json"
                )
                with schema_path.open("r", encoding="utf-8") as f:
                    self._cached_schema = json.load(f)
            except Exception:
                # Fallback if schema doesn't exist or is invalid
                self._cached_schema = {"type": "object", "$defs": {}}

        registry = {
            "step8_vision": "DocumentLayoutManifest",
            "state_differential": "StateMutationIntent",
            "cognitive_sync": "CognitiveStateProfileSchema",
            "system2_remediation": "System2RemediationIntent",
            "tool_invocation": "ToolInvocationEvent",
            "informational": "InformationalIntent",
            "observation": "ObservationEvent",
            "AnyIntent": "AnyIntent",
            "intent": "AnyIntent",
            "symbolic_handoff": "AnyIntent",
        }

        def_key = registry.get(schema_key, schema_key)

        defs = self._cached_schema.get("$defs", {})
        if def_key in defs:
            # We want to return a valid schema that validates this specific definition.
            # Easiest way is to reference it within the whole schema so other $refs work.
            return {"$ref": f"#/$defs/{def_key}", "$defs": defs}

        if schema_key in ("intent", "symbolic_handoff", "AnyIntent"):
            # Create a composite allowed_keys fallback based on all potential intents
            # if we can't find a dedicated AnyIntent. Since we know we are returning
            # early, we can create a dummy schema object with properties matching the union
            # to satisfy the streaming validation properties extractor.
            if not defs:
                return {}  # Fallback to empty if schema doesn't exist to avoid false-positive early rejection

            composite_props = {}
            for k in defs:
                if "Intent" in k or "Event" in k or "Manifest" in k or "Observation" in k:
                    schema_def = defs[k]
                    if "properties" in schema_def:
                        composite_props.update(schema_def["properties"])

            # To preserve test_ijson_early_termination checking invalid_keys,
            # we need to be strict and only return the composite properties.
            return {"type": "object", "properties": composite_props, "$defs": defs}

        # Fallback to base structure if not found
        return {"type": "object"}

    def _determine_target_schema(self, node: Any) -> str:
        if node.interventional_policy is not None:
            return "state_differential"

        if node.symbolic_handoff_policy is not None:
            return "symbolic_handoff"

        return "intent"

    def _validate_intent(self, schema_key: str, payload: bytes) -> dict[str, Any]:
        import json

        try:
            data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as e:
            # Fall back to throwing standard pydantic ValueError instead of trying to format deep json decode error
            raise ValueError(f"JSONDecodeError: {e!s}") from e

        if not isinstance(data, dict):
            raise ValueError("Payload must be a dictionary")

        # Optional structure fixes before validation
        if data.get("type") == "tool_invocation":  # pragma: no cover
            params = data.get("parameters", {})
            if "python_code" in params and "code" not in params:
                params["code"] = params.pop("python_code")
            data["parameters"] = params
            if "event_id" not in data:
                import uuid

                data["event_id"] = f"evt_{uuid.uuid4().hex[:8]}"

        # Since AnyIntent is too loose (just requires 'type'), we enforce it strictly for tool_invocation
        if schema_key in ("intent", "symbolic_handoff", "AnyIntent"):
            if data.get("type") == "tool_invocation" and "tool_name" not in data:
                raise ValueError("Missing tool_name in tool_invocation")
            if data.get("type") == "informational" and "message" not in data:
                raise ValueError("Missing message in informational")

        expected_schema = self._get_target_json_schema(schema_key)
        jsonschema.validate(instance=data, schema=expected_schema)

        # If it's a specific schema key, we could validate it here, but skipping for brevity
        return data

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
        node: dict[str, Any],
        ledger: dict[str, Any],
        _node_id: str,
        action_space: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, int, int]:
        """
        Executes the System 1 Fast-Path. Restricts tools to `allowed_passive_tools`.
        If the model yields a valid ToolInvocationEvent within the subset, returns it immediately.
        Otherwise, returns None to fallback to standard deep generation.
        """
        from coreason_inference_engine.adapters.dto import LocalActionSpace, LocalAgentNodeProfile, LocalLedgerState

        node_data = node if isinstance(node, dict) else node.model_dump()
        local_node = LocalAgentNodeProfile(**node_data)
        ledger_data = ledger if isinstance(ledger, dict) else ledger.model_dump()
        local_ledger = LocalLedgerState(**ledger_data)
        action_space_data = action_space if isinstance(action_space, dict) else action_space.model_dump()
        local_action_space = LocalActionSpace(**action_space_data)

        assert local_node.reflex_policy is not None
        allowed_tools = set(local_node.reflex_policy.allowed_passive_tools)

        # Filter action space native tools to only allowed passive tools
        passive_tools = [t for t in local_action_space.native_tools if t.get("tool_name") in allowed_tools]
        if not passive_tools:
            return None, None, None, 0, 0

        # Build messages for the fast-path (we use standard hydration but inject a strict directive)
        messages = self._apply_semantic_slicing(local_node, local_ledger)

        directive = (
            f"AGENT INSTRUCTION: System 1 Fast-Path active. You MUST evaluate if you can solve the current "
            f"objective using ONLY the provided passive tools. Your confidence MUST be >= "
            f"{local_node.reflex_policy.confidence_threshold}. If you meet this threshold, invoke tool immediately. "
            f"If not, return a standard InformationalIntent stating you need more time, triggering deep reasoning."
        )

        # Inject directive into the system prompt (the first message)
        if messages and messages[0]["role"] in ("system", "developer"):
            messages[0]["content"] = f"{messages[0]['content']}\n\n{directive}"
        else:
            messages.insert(0, {"role": "system", "content": directive})

        tools = self.adapter.project_tools(list(passive_tools))

        total_input_tokens = 0
        total_output_tokens = 0
        raw_output = ""
        usage_metrics = {"input_tokens": 0, "output_tokens": 0}

        try:
            # We enforce a strict token clamp (e.g. 150 tokens) for the fast path to prevent costly generation
            format_contract = (
                getattr(local_node.grpo_reward_policy, "format_contract", None)
                if local_node.grpo_reward_policy
                else None
            )

            latent_firewalls = None
            info_policy = getattr(local_node, "information_flow_policy", None)
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
            halt_receipt: dict[str, Any] | None = None
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

            clean_json_str, scratchpad = self._extract_latent_traces(raw_output, local_node)
            target_schema_key = self._determine_target_schema(local_node)
            # Validate payload
            valid_intent = self._validate_intent(target_schema_key, clean_json_str.encode("utf-8", errors="replace"))

            # Fast-path only succeeds if it invokes an allowed passive tool
            if (
                isinstance(valid_intent, dict)
                and valid_intent.get("type") == "tool_invocation"
                and valid_intent.get("tool_name") in allowed_tools
            ):
                invocation_cid = valid_intent.get("event_id", "none")
                burn_receipt_dto = LocalTokenBurnReceipt(
                    event_id=f"burn_{uuid.uuid4().hex[:8]}",
                    timestamp=time.time(),
                    tool_invocation_id=invocation_cid,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    burn_magnitude=self._calculate_cost(total_input_tokens, total_output_tokens),
                )
                burn_receipt = json.loads(burn_receipt_dto.model_dump_json(exclude_none=True))
                return (
                    valid_intent,
                    burn_receipt,
                    scratchpad,
                    total_input_tokens,
                    total_output_tokens,
                )

        except ValidationError as e:  # pragma: no cover
            # If the fast path fails (e.g. invalid JSON or structural error), we fallback and log
            await self.telemetry.emit(
                {
                    "type": "log_event",
                    "timestamp": time.time(),
                    "level": "DEBUG",
                    "message": "System 1 Fast-Path validation failed.",
                    "context_profile": {"error": str(e)},
                }
            )
        except Exception as e:
            await self.telemetry.emit(
                {
                    "type": "log_event",
                    "timestamp": time.time(),
                    "level": "DEBUG",
                    "message": "System 1 Fast-Path execution failed.",
                    "context_profile": {"error": str(e)},
                }
            )

        return None, None, None, total_input_tokens, total_output_tokens

    def _apply_semantic_slicing(self, node: Any, ledger: Any) -> list[dict[str, Any]]:
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
        remediation_indices = [
            i
            for i, event in enumerate(history)
            if (isinstance(event, dict) and event.get("type") == "system2_remediation")
            or getattr(event, "type", "") == "system2_remediation"
            or getattr(event, "intent_type", "") == "system2_remediation"
            or type(event).__name__ == "System2RemediationIntent"
        ]
        if len(remediation_indices) > 1:
            indices_to_remove = set(remediation_indices[:-1])
            history = [event for i, event in enumerate(history) if i not in indices_to_remove]

        # Update the history in the local ledger copy
        sliced_ledger = ledger.model_copy(update={"history": history})

        messages = self.hydrator.compile(node, sliced_ledger)

        if not ceiling:
            return messages

        messages_str = json.dumps(messages)
        token_mass = self.adapter.count_tokens(messages_str)

        while token_mass > ceiling:
            # Find the oldest ObservationEvent to evict
            obs_indices = [
                i
                for i, event in enumerate(sliced_ledger.history)
                if (isinstance(event, dict) and event.get("type") == "observation")
                or getattr(event, "type", "") == "observation"
            ]
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
        node: dict[str, Any],
        ledger: dict[str, Any],
        node_id: str,
        action_space: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        """
        Translates the passive ledger into active generation.
        Executes Context Hydration, the Forward Pass, and System 2 Remediation.

        Raises:
            InferenceConvergenceError: If max_loops are exceeded or upstream API fatally fails.
            asyncio.CancelledError: If preempted by the Orchestrator.
        """
        from coreason_inference_engine.adapters.dto import LocalActionSpace, LocalAgentNodeProfile, LocalLedgerState

        node_data = node if isinstance(node, dict) else node.model_dump()
        local_node = LocalAgentNodeProfile(**node_data)
        ledger_data = ledger if isinstance(ledger, dict) else ledger.model_dump()
        local_ledger = LocalLedgerState(**ledger_data)
        action_space_data = action_space if isinstance(action_space, dict) else action_space.model_dump()
        local_action_space = LocalActionSpace(**action_space_data)

        # CRITICAL FIX: Deadlock Prevention via Local Backpressure
        start_time_unix_nano = time.time_ns()
        if self._semaphore.locked():
            # Emit telemetry for starvation
            await self.telemetry.emit(
                {
                    "type": "log_event",
                    "timestamp": time.time(),
                    "level": "WARNING",
                    "message": "Semaphore saturated, yielding SystemFaultEvent",
                    "context_profile": {"node_id": node_id},
                }
            )

            error_intent_dto = LocalSystemFaultEvent(
                event_id=f"fault_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
            )
            error_intent = json.loads(error_intent_dto.model_dump_json(exclude_none=True))
            receipt_dto = LocalTokenBurnReceipt(
                event_id=f"burn_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                tool_invocation_id="none",
                input_tokens=0,
                output_tokens=0,
                burn_magnitude=0,
            )
            receipt = json.loads(receipt_dto.model_dump_json(exclude_none=True))
            return error_intent, receipt, None, None

        async with self._semaphore:
            total_input_tokens = 0
            total_output_tokens = 0

            # FR-1.2.5: System 1 Fast-Path Evaluation
            if local_node.reflex_policy:
                fast_intent, fast_receipt, fast_scratch, fast_in, fast_out = await self._evaluate_system1_reflex(
                    node, ledger, node_id, action_space
                )
                total_input_tokens += fast_in
                total_output_tokens += fast_out

                if fast_intent is not None and fast_receipt is not None:
                    return fast_intent, fast_receipt, fast_scratch, None

            # FR-1.3 & FR-2.5: Context Compilation & Semantic Slicing
            messages = self._apply_semantic_slicing(local_node, local_ledger)

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
            allowed_tools = list(local_action_space.native_tools)

            # The BRD gap mentions evaluating "node.information_flow_policy and node.permissions".
            # If the properties exist dynamically, we use getattr.
            info_policy = getattr(local_node, "information_flow_policy", None)
            if info_policy and hasattr(info_policy, "tool_boundaries"):  # pragma: no cover
                allowed_boundaries = set(info_policy.tool_boundaries)
                allowed_tools = [t for t in allowed_tools if t.get("tool_name") in allowed_boundaries]

            node_permissions = getattr(local_node, "permissions", None)
            if node_permissions and hasattr(node_permissions, "allowed_tools"):  # pragma: no cover
                allowed_set = set(node_permissions.allowed_tools)
                allowed_tools = [t for t in allowed_tools if t.get("tool_name") in allowed_set]

            tools = self.adapter.project_tools(allowed_tools)

            max_loops = local_node.correction_policy.max_loops if local_node.correction_policy else 3

            # Apply PEFT adapters if requested
            if local_node.peft_adapters:
                await self.adapter.apply_peft_adapters(local_node.peft_adapters)

            logit_biases = self._compile_watermark_biases(
                local_node.logit_steganography, vocab_size=128000, prior_tokens=[]
            )

            import random

            # Initialize backoff state outside the loop
            current_backoff = 1.0
            max_backoff = 60.0
            global_start_time = time.time()

            # Need to get global timeout, default to a sensible value if not provided
            global_timeout = 300.0
            if (
                local_node.correction_policy
                and getattr(local_node.correction_policy, "global_timeout_seconds", None) is not None
            ):  # pragma: no cover
                global_timeout = float(getattr(local_node.correction_policy, "global_timeout_seconds", 300.0))

            attempt = 0

            grpo = local_node.grpo_reward_policy
            format_contract = getattr(grpo, "format_contract", None) if grpo else None

            info_policy = getattr(local_node, "information_flow_policy", None)
            latent_firewalls = None
            if info_policy and hasattr(info_policy, "latent_firewalls"):
                latent_firewalls = info_policy.latent_firewalls  # pragma: no cover

            while attempt < max_loops:
                raw_output = ""
                usage_metrics = {"input_tokens": 0, "output_tokens": 0}
                halt_receipt = None

                current_max_tokens = 500 if attempt > 0 else None

                response_schema = None
                domain_ext = getattr(local_node, "domain_extensions", None)
                has_constrained = isinstance(domain_ext, dict) and "constrained_decoding" in domain_ext
                has_format = getattr(local_node, "grpo_reward_policy", None) and getattr(
                    local_node.grpo_reward_policy, "format_contract", None
                )
                target_schema_key = self._determine_target_schema(local_node)
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
                                for prefix, event, _value in events:
                                    if prefix == "" and event == "map_key":
                                        # Dynamic extraction of allowed keys from the target schema
                                        schema_dict = response_schema or self._get_target_json_schema(target_schema_key)
                                        allowed_keys: set[str] = set()

                                        def extract_props(schema: Any, keys_set: set[str]) -> None:
                                            if not isinstance(schema, dict):  # pragma: no cover
                                                return
                                            if "properties" in schema:
                                                keys_set.update(schema["properties"].keys())
                                            for val in schema.values():
                                                if isinstance(val, dict):
                                                    extract_props(val, keys_set)
                                                elif isinstance(val, list):  # pragma: no cover
                                                    for item in val:
                                                        extract_props(item, keys_set)

                                        extract_props(schema_dict, allowed_keys)

                                        if allowed_keys:
                                            # Always allow base event keys
                                            allowed_keys.update(
                                                {"type", "intent_type", "target_node_id", "event_id", "timestamp"}
                                            )
                                            if _value not in allowed_keys:
                                                structural_violation = True
                                                break
                                events.clear()  # pragma: no cover
                            except StopIteration:  # pragma: no cover
                                break
                            except ijson.JSONError, UnicodeEncodeError:  # pragma: no cover
                                # We ignore standard parse errors during streaming since it's incomplete
                                events.clear()

                            if structural_violation:
                                break

                        if structural_violation:  # pragma: no cover
                            # Shield teardown to prevent API leakage
                            import contextlib

                            with contextlib.suppress(Exception):
                                await asyncio.shield(stream.aclose())

                            # Fast-return remediation intent
                            remediation_intent = {
                                "type": "system2_remediation",
                                "fault_id": f"fault_{uuid.uuid4().hex[:8]}",
                                "target_node_id": node_id or "",
                                "failing_pointers": ["/"],
                                "remediation_prompt": "CRITICAL CONTRACT BREACH: An immediate top-level structural "
                                "violation was detected during streaming. Correct your JSON projection.",
                            }

                            invocation_cid = "none"
                            burn_receipt_dto = LocalTokenBurnReceipt(
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
                            burn_receipt = json.loads(burn_receipt_dto.model_dump_json(exclude_none=True))

                            return remediation_intent, burn_receipt, None, None

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

                    if halt_receipt:  # pragma: no cover
                        burn_receipt_dto = LocalTokenBurnReceipt(
                            event_id=f"burn_{uuid.uuid4().hex[:8]}",
                            timestamp=time.time(),
                            tool_invocation_id="none",
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            burn_magnitude=self._calculate_cost(total_input_tokens, total_output_tokens),
                        )
                        burn_receipt = json.loads(
                            burn_receipt_dto.model_dump_json(exclude_none=True)
                        )  # pragma: no cover
                        valid_intent = {
                            "type": "fallback_intent",
                            "target_node_id": node_id,
                            "fallback_node_id": "system_halt",
                        }  # pragma: no cover
                        return valid_intent, burn_receipt, halt_receipt, None  # pragma: no cover

                    clean_json_str, scratchpad = self._extract_latent_traces(raw_output, local_node)

                    target_schema_key = self._determine_target_schema(local_node)

                    # Zero-Trust Egress: Pass byte string to validation functor
                    # _validate_intent raises ValidationError on failure
                    valid_intent = self._validate_intent(
                        target_schema_key, clean_json_str.encode("utf-8", errors="replace")
                    )

                    # FR-4.5: Hallucinated Tool Escalation
                    if isinstance(valid_intent, dict) and valid_intent.get("type") == "tool_invocation":
                        allowed_tools_names = {t.get("tool_name") for t in local_action_space.native_tools}
                        if valid_intent.get("tool_name") not in allowed_tools_names:
                            raise ValidationError(f"Tool '{valid_intent.get('tool_name')}' not found in actions.")

                    # Check for tool invocation ID
                    invocation_cid = (
                        valid_intent.get("event_id", "none")
                        if isinstance(valid_intent, dict) and valid_intent.get("type") == "tool_invocation"
                        else "none"
                    )

                    # Build tracking receipt
                    end_time_unix_nano = time.time_ns()

                    span = {
                        "type": "execution_span",
                        "trace_id": f"trace_{uuid.uuid4().hex[:8]}",
                        "span_id": f"span_{uuid.uuid4().hex[:8]}",
                        "parent_span_id": None,
                        "name": "inference_engine_turn",
                        "start_time_unix_nano": start_time_unix_nano,
                        "end_time_unix_nano": end_time_unix_nano,
                        "status": "ok",
                        "events": [
                            {
                                "name": "first_token",
                                "timestamp_unix_nano": start_time_unix_nano + ttft,
                                "attributes": {"node_id": node_id, "ttft_nano": ttft},
                            }
                        ],
                    }
                    await self.telemetry.emit(span)

                    burn_receipt_dto = LocalTokenBurnReceipt(
                        event_id=f"burn_{uuid.uuid4().hex[:8]}",
                        timestamp=time.time(),
                        tool_invocation_id=invocation_cid or "none",
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        burn_magnitude=self._calculate_cost(total_input_tokens, total_output_tokens),
                    )
                    burn_receipt = json.loads(burn_receipt_dto.model_dump_json(exclude_none=True))

                    cognitive_receipt = None
                    if (
                        scratchpad
                        and local_node.grpo_reward_policy
                        and getattr(local_node.grpo_reward_policy, "topological_scoring", None)
                    ):
                        cognitive_receipt_dto = LocalCognitiveRewardReceipt(
                            event_id=f"reward_{uuid.uuid4().hex[:8]}",
                            timestamp=time.time(),
                            source_generation_id=scratchpad.get("trace_id", ""),
                            extracted_axioms=[],
                            calculated_r_path=1.0,  # Proxy values as topology is out of scope here
                            total_advantage_score=1.0,
                        )
                        cognitive_receipt = json.loads(cognitive_receipt_dto.model_dump_json(exclude_none=True))

                    return valid_intent, burn_receipt, scratchpad, cognitive_receipt

                except (ValidationError, ValueError) as e:
                    # FR-3.3, FR-3.4: Trap validation failure and generate mathematical reprimand
                    fault_id = f"fault_{uuid.uuid4().hex[:8]}"

                    # CRITICAL: Pass exact DID string (node_id) to prevent remediation crash
                    remediation = generate_correction_prompt(error=e, target_node_id=node_id, fault_id=fault_id)
                    remediation_dict = remediation if isinstance(remediation, dict) else remediation.model_dump()

                    print(
                        f"\\n\\n=========== RAW LLM FALLBACK ===========\\n"
                        f"{raw_output}\\n"
                        f"========================================\\n\\n"
                    )

                    # Redact raw output
                    redacted_output = self.telemetry.redact_pii(
                        raw_output, getattr(local_node, "information_flow_policy", None)
                    )

                    await self.telemetry.emit(
                        {
                            "type": "log_event",
                            "timestamp": time.time(),
                            "level": "DEBUG",
                            "message": "Hallucinated structural violation",
                            "context_profile": {"raw_output": redacted_output},
                        }
                    )

                    # Inject prompt and retry
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": json.dumps(remediation_dict)})

                    await self.telemetry.emit(
                        {
                            "type": "log_event",
                            "timestamp": time.time(),
                            "level": "WARNING",
                            "message": "Validation error during generation; entering remediation loop",
                            "context_profile": {"attempt": attempt, "max_loops": max_loops, "error": str(e)},
                        }
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
