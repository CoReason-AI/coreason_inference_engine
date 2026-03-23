with open("tests/test_engine.py", "r") as f:
    content = f.read()

content = content.replace(
    "response = self.responses[self.call_count]",
    "response = self.responses[self.call_count] if self.call_count < len(self.responses) else ''",
)
content = content.replace(
    'assert getattr(intent, "fault_id", None) is not None', 'assert intent and intent.get("fault_id") is not None'
)
content = content.replace(
    'assert intent.get("type") == "informational"', 'assert intent and intent.get("type") == "informational"'
)
content = content.replace("assert scratchpad is not None", "assert scratchpad is not None")
content = content.replace(
    "assert scratchpad.get('total_latent_tokens') == len(\"This is a reasoning trace.\")",
    "assert scratchpad and scratchpad.get('total_latent_tokens') == len(\"This is a reasoning trace.\")",
)
content = content.replace(
    "assert len(scratchpad.get('explored_branches')) == 1",
    "assert scratchpad and len(scratchpad.get('explored_branches', [])) == 1",
)
content = content.replace("assert intent.get('fault_id')", "assert intent and intent.get('fault_id')")
content = content.replace(
    "assert intent.get('remediation_prompt')", "assert intent and intent.get('remediation_prompt')"
)
content = content.replace("assert intent.get('failing_pointers')", "assert intent and intent.get('failing_pointers')")
content = content.replace("mutation_intent.get('value')", "mutation_intent.get('value')")

# For test_epistemic_tool_pruning, fixing "unexpected keyword argument 'ledger'"
content = content.replace(
    """        await engine.generate_intent(
            raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries,
            ledger=EpistemicLedgerState(history=[]),""",
    """        await engine.generate_intent(
            raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries,
            raw_ledger=EpistemicLedgerState(history=[]).model_dump(),""",
)

# For test_local_backpressure_fail_fast
# `UnboundLocalError: cannot access local variable 'ExecutionSpanReceipt'`
# Wait, I already added `from coreason_manifest.spec.ontology import ExecutionSpanReceipt, SpanEvent` there, but test_local_backpressure_fail_fast doesn't execute that block. It returns `error_intent` directly. So it doesn't fail there.

with open("tests/test_engine.py", "w") as f:
    f.write(content)
