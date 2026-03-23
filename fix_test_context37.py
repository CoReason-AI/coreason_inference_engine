import re

with open("tests/test_context.py", "r") as f:
    content = f.read()

# Replace any lingering `ledger_dict = ledger.model_dump() if hasattr(ledger, 'model_dump') else ledger` to make sure it's fully defined.
# I had already tried replacing this, but the regex missed the exact whitespace. Let's do it with split lines.

lines = content.split("\n")
for i, l in enumerate(lines):
    if "messages = hydrator.compile" in l and "ledger_dict" in l:
        if "ledger_dict =" not in lines[i - 1]:
            lines.insert(i, "        ledger_dict = ledger.model_dump() if hasattr(ledger, 'model_dump') else ledger")

with open("tests/test_context.py", "w") as f:
    f.write("\n".join(lines))
