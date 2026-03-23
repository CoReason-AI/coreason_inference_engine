with open("tests/test_engine.py", "r") as f:
    content = f.read()

content = content.replace("raw_node=node.model_dump() if hasattr(node, 'model_dump') else node_with_small_timeout.model_dump()\n                if hasattr(node_with_small_timeout, \"model_dump\")\n                else node_with_small_timeout,", "raw_node=node_with_small_timeout.model_dump()\n                if hasattr(node_with_small_timeout, \"model_dump\")\n                else node_with_small_timeout,")
content = content.replace("raw_node=node.model_dump() if hasattr(node, 'model_dump') else node_with_boundaries.model_dump()\n            if hasattr(node_with_boundaries, \"model_dump\")\n            else node_with_boundaries,", "raw_node=node_with_boundaries.model_dump()\n            if hasattr(node_with_boundaries, \"model_dump\")\n            else node_with_boundaries,")

# extract_latent_traces_missing_tags_but_required failure:
content = content.replace("assert scratchpad.get('total_latent_tokens') == len(\"This is a reasoning trace.\")", "assert scratchpad and scratchpad.get('total_latent_tokens') == len(\"This is a reasoning trace.\")")
content = content.replace("assert len(scratchpad.get('explored_branches', [])) == 1", "assert scratchpad and len(scratchpad.get('explored_branches', [])) == 1")
content = content.replace("branch = scratchpad.get('explored_branches')[0]", "branch = scratchpad.get('explored_branches')[0] if scratchpad else {}")

with open("tests/test_engine.py", "w") as f:
    f.write(content)
