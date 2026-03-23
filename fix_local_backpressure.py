import re

with open("src/coreason_inference_engine/engine.py", "r") as f:
    content = f.read()

content = content.replace(
    "            await self.telemetry.emit(\n                LogEvent(",
    "            from coreason_manifest.spec.ontology import LogEvent\n            await self.telemetry.emit(\n                LogEvent(",
)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(content)
