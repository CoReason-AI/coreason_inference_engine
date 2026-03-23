with open("tests/test_engine.py", "r") as f:
    content = f.read()

content = content.replace("raw_node=node.model_dump() if hasattr(node, 'model_dump') else node_with_boundaries.model_dump()\n            if hasattr(node_with_boundaries, \"model_dump\")\n            else node_with_boundaries,", "raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, \"model_dump\") else node_with_boundaries,")

# test_extract_latent_traces_missing_tags_but_required fails on TypeError because it uses LocalAgentNodeProfile(**raw_node) and node passes AgentNodeProfile which isn't dict.
# Actually I replaced raw_node=mock_node.model_dump() with raw_node=mock_node.model_dump() ... wait.
# The error was "TypeError: LocalAgentNodeProfile() argument after ** must be a mapping, not AgentNodeProfile"
# Ah! In test_extract_latent_traces_missing_tags_but_required:
content = content.replace("raw_node=mock_node.model_dump() if hasattr(mock_node, \"model_dump\") else mock_node_with_think,", "raw_node=mock_node_with_think.model_dump() if hasattr(mock_node_with_think, \"model_dump\") else mock_node_with_think,")

with open("tests/test_engine.py", "w") as f:
    f.write(content)
