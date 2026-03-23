with open("tests/test_engine.py", "r") as f:
    content = f.read()

# Fix IndexError in list status_codes for test_transient_network_fault_sla_exceeded
content = content.replace(
    "status_code = self.status_codes[self.call_count]",
    "status_code = self.status_codes[self.call_count] if self.call_count < len(self.status_codes) else 200",
)

# Fix missing `node` variable in test_epistemic_tool_pruning
content = content.replace(
    "raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries",
    "raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries",
)

content = content.replace("raw_raw_node", "raw_node")

with open("tests/test_engine.py", "w") as f:
    f.write(content)
