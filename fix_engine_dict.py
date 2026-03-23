
with open("src/coreason_inference_engine/engine.py") as f:
    content = f.read()

content = content.replace(
    """return dict(
                        event_id=data.get("event_id", f"evt_{uuid.uuid4().hex[:8]}"),
                        timestamp=data.get("timestamp", time.time()),
                        type="tool_invocation",""",
    """return {
                        "event_id": data.get("event_id", f"evt_{uuid.uuid4().hex[:8]}"),
                        "timestamp": data.get("timestamp", time.time()),
                        "type": "tool_invocation",""",
)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(content)
