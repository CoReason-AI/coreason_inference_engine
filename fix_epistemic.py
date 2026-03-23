with open("tests/test_engine.py", "r") as f:
    content = f.read()
content = content.replace("raw_node=node.model_dump() if hasattr(node, 'model_dump') else node_with_boundaries.model_dump()", "raw_node=node_with_boundaries.model_dump()")
with open("tests/test_engine.py", "w") as f:
    f.write(content)
