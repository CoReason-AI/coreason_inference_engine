import re

with open("tests/test_engine.py", "r") as f:
    code = f.read()

# Fix generate_intent calls to use model_dump()
code = re.sub(
    r"node=([a-zA-Z_0-9]+)(\s*,)",
    r"node=\1.model_dump()\2",
    code
)

code = re.sub(
    r"ledger=([a-zA-Z_0-9]+)(\s*,)",
    r"ledger=\1.model_dump()\2",
    code
)

code = re.sub(
    r"action_space=([a-zA-Z_0-9]+)(\s*[,)])",
    r"action_space=\1.model_dump()\2",
    code
)

# Fix LatentScratchpadReceipt references in stream signatures
code = code.replace("LatentScratchpadReceipt | None", "dict[str, Any] | None")

with open("tests/test_engine.py", "w") as f:
    f.write(code)

with open("tests/test_engine_reflex.py", "r") as f:
    code = f.read()

code = re.sub(
    r"node=([a-zA-Z_0-9]+)(\s*,)",
    r"node=\1.model_dump()\2",
    code
)

code = re.sub(
    r"ledger=([a-zA-Z_0-9]+)(\s*,)",
    r"ledger=\1.model_dump()\2",
    code
)

code = re.sub(
    r"action_space=([a-zA-Z_0-9]+)(\s*[,)])",
    r"action_space=\1.model_dump()\2",
    code
)

code = code.replace("LatentScratchpadReceipt | None", "dict[str, Any] | None")

with open("tests/test_engine_reflex.py", "w") as f:
    f.write(code)

with open("tests/test_engine_slicing.py", "r") as f:
    code = f.read()

code = re.sub(
    r"node=([a-zA-Z_0-9]+)(\s*,)",
    r"node=\1.model_dump()\2",
    code
)

code = re.sub(
    r"ledger=([a-zA-Z_0-9]+)(\s*,)",
    r"ledger=\1.model_dump()\2",
    code
)

with open("tests/test_engine_slicing.py", "w") as f:
    f.write(code)
