
with open("tests/test_engine.py") as f:
    content = f.read()

content = content.replace(
    "raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries,\n            ledger=EpistemicLedgerState(history=[]),",
    "raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries,\n            raw_ledger=EpistemicLedgerState(history=[]).model_dump(),",
)

with open("tests/test_engine.py", "w") as f:
    f.write(content)
