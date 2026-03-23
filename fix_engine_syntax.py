import re

with open("src/coreason_inference_engine/engine.py", "r") as f:
    engine_content = f.read()

# Fix syntax error at line 754 "expected 'except' or 'finally' block"
# Look for "from pydantic import ValidationError as PydanticValidationError"
engine_content = engine_content.replace(
    "from pydantic import ValidationError\nfrom coreason_manifest.spec.ontology import LogEvent, ExecutionSpanReceipt, SpanEvent, ToolInvocationEvent, StateMutationIntent",
    "from pydantic import ValidationError",
)
engine_content = engine_content.replace(
    "from pydantic import ValidationError\nfrom pydantic import ValidationError as PydanticValidationError",
    "from pydantic import ValidationError as PydanticValidationError",
)

# I'll just check line 754 directly
lines = engine_content.split("\n")
for i, line in enumerate(lines):
    if "from pydantic import ValidationError" in line and "as PydanticValidationError" not in line:
        if i > 700:
            lines[i] = ""

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write("\n".join(lines))
