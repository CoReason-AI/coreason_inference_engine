import re

with open("src/coreason_inference_engine/engine.py") as f:
    engine_content = f.read()

# Replace coreason_manifest imports
engine_content = re.sub(
    r"from coreason_manifest\.spec\.ontology import \(\n(?:[ ]+.*,\n)*(?:[ ]+.*\n)*\)",
    "from typing import Any, cast\nfrom coreason_manifest.spec.ontology import LogEvent, ExecutionSpanReceipt, SpanEvent, ToolInvocationEvent, SystemFaultEvent\n\nfrom coreason_inference_engine.adapters.dto import LocalActionSpace, LocalAgentNodeProfile, LocalLedgerState\n",
    engine_content,
)

# Update generate_intent signature
engine_content = engine_content.replace(
    """async def generate_intent(
        self,
        node: AgentNodeProfile,
        ledger: EpistemicLedgerState,
        node_id: str,
        action_space: ActionSpaceManifest,
    ) -> tuple[
        AnyIntent | AnyStateEvent | System2RemediationIntent,
        TokenBurnReceipt,
        LatentScratchpadReceipt | None,
        CognitiveRewardEvaluationReceipt | None,
    ]:""",
    """async def generate_intent(
        self,
        raw_node: dict[str, Any],
        raw_ledger: dict[str, Any],
        node_id: str,
        raw_action_space: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:""",
)

engine_content = engine_content.replace(
    """        # CRITICAL FIX: Deadlock Prevention via Local Backpressure
        start_time_unix_nano = time.time_ns()""",
    """        node = LocalAgentNodeProfile(**raw_node)
        ledger = LocalLedgerState(**raw_ledger)
        action_space = LocalActionSpace(**raw_action_space)

        # CRITICAL FIX: Deadlock Prevention via Local Backpressure
        start_time_unix_nano = time.time_ns()""",
)

# Fix validate_payload import
engine_content = engine_content.replace("from coreason_manifest.utils.algebra import validate_payload\n", "")
engine_content = engine_content.replace(
    """return validate_payload(schema_key, payload)""",
    """from coreason_manifest.utils.algebra import validate_payload\n        out_res: Any = validate_payload(schema_key, payload)\n        if hasattr(out_res, "model_dump"):\n            return out_res.model_dump() # type: ignore\n        return dict(out_res) # type: ignore""",
)

# Replace signatures and object usage
engine_content = engine_content.replace("node: AgentNodeProfile", "node: LocalAgentNodeProfile")
engine_content = engine_content.replace("ledger: EpistemicLedgerState", "ledger: LocalLedgerState")
engine_content = engine_content.replace("action_space: ActionSpaceManifest", "action_space: LocalActionSpace")
engine_content = engine_content.replace(
    "halt_receipt: LatentScratchpadReceipt | None = None", "halt_receipt: dict[str, Any] | None = None"
)

engine_content = engine_content.replace("def _extract_latent_traces(", "def _extract_latent_traces( # type: ignore")

engine_content = engine_content.replace("branch = ThoughtBranchState(", "branch = dict(")
engine_content = engine_content.replace("receipt = LatentScratchpadReceipt(", "receipt = dict(")

engine_content = engine_content.replace(
    "return ToolInvocationEvent.model_construct(", "return dict(type='tool_invocation',"
)
engine_content = engine_content.replace(
    "return StateMutationIntent.model_construct(**clean_data)",
    "clean_data['type'] = 'state_mutation'\n                    return clean_data",
)

engine_content = engine_content.replace(
    "isinstance(valid_intent, ToolInvocationEvent)",
    "isinstance(valid_intent, dict) and valid_intent.get('type') == 'tool_invocation'",
)
engine_content = engine_content.replace("valid_intent.tool_name", "valid_intent.get('tool_name')")
engine_content = engine_content.replace("valid_intent.event_id", "valid_intent.get('event_id')")
engine_content = engine_content.replace("t.tool_name", "t.get('tool_name')")

engine_content = engine_content.replace(
    "isinstance(event, ObservationEvent)", "isinstance(event, dict) and event.get('type') == 'observation'"
)
engine_content = engine_content.replace(
    "isinstance(event, System2RemediationIntent)",
    "isinstance(event, dict) and event.get('type') == 'system2_remediation'",
)

engine_content = engine_content.replace(
    "target_union = AnyIntent | AnyStateEvent | System2RemediationIntent",
    "from coreason_manifest.spec.ontology import AnyIntent, AnyStateEvent, System2RemediationIntent\n            target_union = AnyIntent | AnyStateEvent | System2RemediationIntent",
)
engine_content = engine_content.replace(
    "return TypeAdapter(target_union).validate_json(payload)",
    "res: Any = TypeAdapter(target_union).validate_json(payload)\n            if hasattr(res, 'model_dump'):\n                return res.model_dump()\n            return dict(res)",
)

engine_content = engine_content.replace("TokenBurnReceipt(", "dict(")
engine_content = engine_content.replace("SystemFaultEvent(", "dict(")
engine_content = engine_content.replace("FallbackIntent(", "dict(")
engine_content = engine_content.replace("System2RemediationIntent(", "dict(type='system2_remediation',")
engine_content = engine_content.replace("CognitiveRewardEvaluationReceipt(", "dict(")

engine_content = engine_content.replace("cognitive_receipt = None", "cognitive_receipt: dict[str, Any] | None = None")
engine_content = engine_content.replace("scratchpad.trace_id", "scratchpad.get('trace_id')")
engine_content = engine_content.replace(
    "scratchpad and node.grpo_reward_policy", "scratchpad and node.grpo_reward_policy"
)

engine_content = engine_content.replace(
    "messages = self.hydrator.compile(node, sliced_ledger)",
    "messages = self.hydrator.compile(node, sliced_ledger) # type: ignore",
)
engine_content = engine_content.replace(
    "messages = self.hydrator.compile(node, ledger)", "messages = self.hydrator.compile(node, ledger) # type: ignore"
)

engine_content = engine_content.replace(
    "from coreason_manifest.spec.ontology import SystemFaultEvent",
    "from coreason_manifest.spec.ontology import SystemFaultEvent, LogEvent, ExecutionSpanReceipt, SpanEvent, ToolInvocationEvent",
)
engine_content = engine_content.replace(
    "from pydantic import ValidationError\nfrom coreason_manifest.spec.ontology import LogEvent, ExecutionSpanReceipt, SpanEvent, ToolInvocationEvent as PydanticValidationError",
    "from pydantic import ValidationError as PydanticValidationError",
)
engine_content = engine_content.replace("from pydantic import ValidationError", "from pydantic import ValidationError")
engine_content = engine_content.replace(
    "from coreason_manifest.spec.ontology import (\n            StateMutationIntent,", ""
)
engine_content = engine_content.replace(
    'f"Tool \'{valid_intent.get("tool_name")}\'', "f\"Tool '{valid_intent.get('tool_name')}'"
)

engine_content = engine_content.replace(
    "tools = self.adapter.project_tools([t.model_dump() for t in passive_tools])",
    "tools = self.adapter.project_tools(passive_tools)",
)

engine_content = engine_content.replace(
    "tools = self.adapter.project_tools([t.model_dump() for t in allowed_tools])",
    "tools = self.adapter.project_tools(allowed_tools)",
)

engine_content = engine_content.replace(
    """        registry = {
            "step8_vision": DocumentLayoutManifest,
            "state_differential": StateMutationIntent,
            "cognitive_sync": CognitiveStateProfile,
            "system2_remediation": System2RemediationIntent,
        }""",
    """        from coreason_manifest.spec.ontology import StateMutationIntent
        registry = {
            "step8_vision": DocumentLayoutManifest,
            "state_differential": StateMutationIntent,
            "cognitive_sync": CognitiveStateProfile,
            "system2_remediation": System2RemediationIntent,
        }""",
)

engine_content = engine_content.replace(
    """                            # Fast-return remediation intent
                            remediation_intent = dict(
                                fault_id=f"fault_{uuid.uuid4().hex[:8]}",
                                target_node_id=node_id or "",
                                failing_pointers=["/"],
                                remediation_prompt="CRITICAL CONTRACT BREACH: An immediate top-level structural "
                                "violation was detected during streaming. Correct your JSON projection.",
                            )""",
    """                            # Fast-return remediation intent
                            remediation_intent = dict(
                                type="system2_remediation",
                                fault_id=f"fault_{uuid.uuid4().hex[:8]}",
                                target_node_id=node_id or "",
                                failing_pointers=["/"],
                                remediation_prompt="CRITICAL CONTRACT BREACH: An immediate top-level structural "
                                "violation was detected during streaming. Correct your JSON projection.",
                            )""",
)

engine_content = engine_content.replace(
    """        # EpistemicLedgerState is frozen, so we must use model_copy(update={"history": history})
        sliced_ledger = ledger.model_copy(update={"history": history})""",
    """        # EpistemicLedgerState is frozen, so we must use model_copy(update={"history": history})
        sliced_ledger = dict(ledger) if isinstance(ledger, dict) else ledger.model_copy(update={"history": history}) if hasattr(ledger, "model_copy") else dict(vars(ledger))
        if isinstance(sliced_ledger, dict):
            sliced_ledger["history"] = history""",
)

engine_content = engine_content.replace(
    """            sliced_ledger = sliced_ledger.model_copy(update={"history": new_history})""",
    """            if isinstance(sliced_ledger, dict):
                sliced_ledger["history"] = new_history
            else:
                sliced_ledger = sliced_ledger.model_copy(update={"history": new_history})""",
)

engine_content = engine_content.replace(
    """        ceiling = None
        if node.baseline_cognitive_state and node.baseline_cognitive_state.semantic_slicing:
            ceiling = node.baseline_cognitive_state.semantic_slicing.context_window_token_ceiling""",
    """        ceiling = None
        baseline = getattr(node, "baseline_cognitive_state", None) if not isinstance(node, dict) else node.get("baseline_cognitive_state")
        if baseline:
            slicing = getattr(baseline, "semantic_slicing", None) if not isinstance(baseline, dict) else baseline.get("semantic_slicing")
            if slicing:
                ceiling = getattr(slicing, "context_window_token_ceiling", None) if not isinstance(slicing, dict) else slicing.get("context_window_token_ceiling")""",
)

engine_content = engine_content.replace(
    """        history = list(ledger.history)""",
    """        history = list(getattr(ledger, "history", [])) if not isinstance(ledger, dict) else list(ledger.get("history", []))""",
)

engine_content = engine_content.replace(
    """            obs_indices = [i for i, event in enumerate(sliced_ledger.history) if isinstance(event, dict) and event.get('type') == 'observation']""",
    """            obs_indices = [i for i, event in enumerate(getattr(sliced_ledger, "history", []) if not isinstance(sliced_ledger, dict) else sliced_ledger.get("history", [])) if isinstance(event, dict) and event.get('type') == 'observation']""",
)

engine_content = engine_content.replace(
    """            new_history = list(sliced_ledger.history)""",
    """            new_history = list(getattr(sliced_ledger, "history", [])) if not isinstance(sliced_ledger, dict) else list(sliced_ledger.get("history", []))""",
)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(engine_content)
