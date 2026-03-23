with open("src/coreason_inference_engine/engine.py", "r") as f:
    content = f.read()

content = content.replace(
    "span = ExecutionSpanReceipt(",
    "from coreason_manifest.spec.ontology import ExecutionSpanReceipt, SpanEvent\n                    span = ExecutionSpanReceipt(",
)
content = content.replace(
    "UnboundLocalError: cannot access local variable 'ExecutionSpanReceipt' where it is not associated with a value", ""
)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(content)
