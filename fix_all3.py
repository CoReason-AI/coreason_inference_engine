import os

test_files = [
    "tests/test_engine.py",
    "tests/test_engine_reflex.py",
    "tests/test_engine_slicing.py",
    "tests/test_main.py",
    "tests/test_hallucinated_tool.py",
    "tests/test_interfaces.py",
    "tests/test_cost.py",
]

for file in test_files:
    if not os.path.exists(file):
        continue
    with open(file) as f:
        content = f.read()

    # In tests/test_engine.py, test_epistemic_tool_pruning:
    content = content.replace(
        "raw_raw_node=node.model_dump() if hasattr(node, 'model_dump') else node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries",
        "raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries",
    )

    # In test_engine.py, test_peft_adapters_applied:
    content = content.replace(
        "node=node.model_dump() if hasattr(node, 'model_dump') else node_with_peft.model_dump() if hasattr(node_with_peft, 'model_dump') else node_with_peft",
        "raw_node=node_with_peft.model_dump() if hasattr(node_with_peft, 'model_dump') else node_with_peft",
    )

    # Some variables like `node` are not defined because it should have been `mock_node`. The regex `node.model_dump()` matched `mock_node.model_dump()`.
    # Let's fix the specific test failures in `test_engine.py`.
    # For example, "AttributeError: 'dict' object has no attribute 'model_dump'" because they are already converted.

    # test_generate_intent_ttft_concurrency: "AttributeError: 'dict' object has no attribute 'model_dump'"
    # If the fixture was updated to return dicts?
    # No, we did `node.model_dump() if hasattr(node, 'model_dump') else node`. This shouldn't raise AttributeError!
    # Ah, the error is `AttributeError: 'LocalAgentNodeProfile' object has no attribute 'model_dump'`?
    # No, LocalAgentNodeProfile inherits BaseModel and has model_dump!

    # Let's just fix test_engine.py manually where needed.

    with open(file, "w") as f:
        f.write(content)
