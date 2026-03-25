with open("tests/test_engine.py", "r") as f:
    code = f.read()

# Fix ijson early termination test checking system_halt fallback
code = code.replace(
    'assert intent.get("target_node_id") == "did:test:1" or intent.get("target_node_id") == "system_halt" or intent.get("type") == "system2_remediation"',
    'assert intent.get("type") == "system2_remediation" or intent.get("fallback_node_id") == "system_halt" or intent.get("target_node_id") == "system_halt"'
)

with open("tests/test_engine.py", "w") as f:
    f.write(code)

with open("src/coreason_inference_engine/utils/validation.py", "r") as f:
    code = f.read()

code = code.replace(
    'if not hasattr(error, "errors"):',
    'if not hasattr(error, "errors") or not callable(getattr(error, "errors")):'
)

with open("src/coreason_inference_engine/utils/validation.py", "w") as f:
    f.write(code)
