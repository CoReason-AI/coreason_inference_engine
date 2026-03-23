with open("tests/test_engine.py", "r") as f:
    content = f.read()

content = content.replace("raw_node=node.model_dump() if hasattr(node, 'model_dump') else node_with_peft.model_dump() if hasattr(node_with_peft, \"model_dump\") else node_with_peft", "raw_node=node_with_peft.model_dump() if hasattr(node_with_peft, \"model_dump\") else node_with_peft")
content = content.replace("raw_node=node.model_dump() if hasattr(node, 'model_dump') else node_with_small_timeout.model_dump()\n                if hasattr(node_with_small_timeout, \"model_dump\")\n                else node_with_small_timeout", "raw_node=node_with_small_timeout.model_dump()\n                if hasattr(node_with_small_timeout, \"model_dump\")\n                else node_with_small_timeout")
content = content.replace("raw_node=node.model_dump() if hasattr(node, 'model_dump') else node_with_boundaries.model_dump()\n            if hasattr(node_with_boundaries, \"model_dump\")\n            else node_with_boundaries", "raw_node=node_with_boundaries.model_dump()\n            if hasattr(node_with_boundaries, \"model_dump\")\n            else node_with_boundaries")

# Also `AttributeError: 'dict' object has no attribute 'urgency_index'`
content = content.replace("cog_intent.urgency_index", "cog_intent.get('urgency_index')")

# Fix `test_remediation_loop_success`
content = content.replace("assert getattr(intent, \"message\", None)", "assert intent and intent.get('message')")

with open("tests/test_engine.py", "w") as f:
    f.write(content)
