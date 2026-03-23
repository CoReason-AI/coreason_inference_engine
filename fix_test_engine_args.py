import re
import os

test_files = [
    "tests/test_engine.py",
    "tests/test_engine_reflex.py",
    "tests/test_engine_slicing.py",
    "tests/test_main.py",
    "tests/test_hallucinated_tool.py",
]

for file in test_files:
    if not os.path.exists(file):
        continue
    with open(file, "r") as f:
        content = f.read()

    # The issue: intents are returning as dicts. `assert getattr(intent, "type", None) == "informational"`
    content = content.replace('getattr(intent, "type", None)', 'intent.get("type")')

    with open(file, "w") as f:
        f.write(content)
