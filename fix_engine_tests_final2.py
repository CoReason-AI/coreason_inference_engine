import re

with open("tests/test_engine.py", "r") as f:
    content = f.read()

content = content.replace(
    "raw_node=node.model_dump() if hasattr(node, 'model_dump') else node_with_small_timeout",
    "raw_node=node_with_small_timeout.model_dump() if hasattr(node_with_small_timeout, 'model_dump') else node_with_small_timeout",
)
content = content.replace(
    "raw_node=node.model_dump() if hasattr(node, 'model_dump') else node_with_peft",
    "raw_node=node_with_peft.model_dump() if hasattr(node_with_peft, 'model_dump') else node_with_peft",
)

with open("tests/test_engine.py", "w") as f:
    f.write(content)
