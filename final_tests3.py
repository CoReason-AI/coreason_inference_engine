with open("tests/test_engine.py", "r") as f:
    content = f.read()

content = content.replace("raw_node=node.model_dump() if hasattr(node, 'model_dump') else node_with_boundaries.model_dump()\n            if hasattr(node_with_boundaries, \"model_dump\")\n            else node_with_boundaries", "raw_node=node_with_boundaries.model_dump()\n            if hasattr(node_with_boundaries, \"model_dump\")\n            else node_with_boundaries")

# test_extract_latent_traces_with_tags fails on "AttributeError: 'dict' object has no attribute 'latent_content_hash'"
content = content.replace("branch.latent_content_hash", "branch.get('latent_content_hash')")

# For missing tags required: AttributeError: 'dict' object has no attribute 'latent_content_hash' or similar? Let's fix that too.

with open("tests/test_engine.py", "w") as f:
    f.write(content)
