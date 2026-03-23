import re

with open("src/coreason_inference_engine/engine.py", "r") as f:
    content = f.read()

content = content.replace(
    'type=\'tool_invocation\',\n                        event_id=data.get("event_id", f"evt_{uuid.uuid4().hex[:8]}"),\n                        timestamp=data.get("timestamp", time.time()),\n                        type="tool_invocation"',
    'event_id=data.get("event_id", f"evt_{uuid.uuid4().hex[:8]}"),\n                        timestamp=data.get("timestamp", time.time()),\n                        type="tool_invocation"',
)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(content)
