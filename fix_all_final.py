with open("tests/test_engine.py", "r") as f:
    content = f.read()

content = content.replace(
    "raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries",
    "raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries",
)

# And in test_epistemic_tool_pruning:
content = content.replace(
    "raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries,\n            ledger=EpistemicLedgerState(history=[]),",
    "raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries,\n            raw_ledger=EpistemicLedgerState(history=[]).model_dump(),",
)

# Fix SLA Exceeded:
content = content.replace("status_codes = [429]", "status_codes = [429, 429]")
content = content.replace(
    "status_code = self.status_codes[self.call_count] if self.call_count < len(self.status_codes) else 200",
    "status_code = self.status_codes[self.call_count] if self.call_count < len(self.status_codes) else 429",
)

with open("tests/test_engine.py", "w") as f:
    f.write(content)

with open("src/coreason_inference_engine/engine.py", "r") as f:
    content = f.read()

content = content.replace(
    "from coreason_manifest.spec.ontology import ExecutionSpanReceipt, SpanEvent",
    "from coreason_manifest.spec.ontology import ExecutionSpanReceipt, SpanEvent, LogEvent, SystemFaultEvent",
)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(content)
