import re

with open("tests/test_engine.py", "r") as f:
    content = f.read()

content = content.replace("raw_raw_ledger=", "raw_ledger=")
content = content.replace("raw_raw_action_space=", "raw_action_space=")

content = content.replace(
    "raw_node=mock_node.model_dump() if hasattr(mock_node, \"model_dump\") else mock_node",
    "raw_node=mock_node.model_dump() if hasattr(mock_node, \"model_dump\") else mock_node"
)

# And fix TypeError: got an unexpected keyword argument 'raw_raw_ledger'
content = content.replace("raw_raw_ledger", "raw_ledger")
content = content.replace("raw_raw_action_space", "raw_action_space")
content = content.replace("raw_raw_node", "raw_node")

with open("tests/test_engine.py", "w") as f:
    f.write(content)
