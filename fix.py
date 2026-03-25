with open("tests/test_engine.py", "r") as f:
    code = f.read()

# Fix ijson early termination test checking system_halt fallback
code = code.replace(
    'assert intent.get("type") in ("system2_remediation", "fallback_intent", "system_fault") or getattr(intent, "fallback_node_id", None) == "system_halt" or getattr(intent, "target_node_id", None) == "system_halt" or intent.get("target_node_id") == "did:test:1"',
    'assert isinstance(intent, dict) and intent.get("type") == "system2_remediation"'
)

with open("tests/test_engine.py", "w") as f:
    f.write(code)
