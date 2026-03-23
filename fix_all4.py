import re

with open("src/coreason_inference_engine/adapters/dto.py", "r") as f:
    content = f.read()

content = content.replace(
    'class LocalAgentNodeProfile(BaseModel):\n    """\n    Internal Pydantic model for the inference engine to map Node payloads.\n    Silently drops unknown fields.\n    """\n\n    model_config = ConfigDict(extra="ignore")\n\n    baseline_cognitive_state: LocalCognitiveStateProfile | None = None',
    'class LocalAgentNodeProfile(BaseModel):\n    """\n    Internal Pydantic model for the inference engine to map Node payloads.\n    Silently drops unknown fields.\n    """\n\n    model_config = ConfigDict(extra="ignore")\n    description: str = ""\n\n    baseline_cognitive_state: LocalCognitiveStateProfile | None = None',
)

with open("src/coreason_inference_engine/adapters/dto.py", "w") as f:
    f.write(content)

with open("src/coreason_inference_engine/context.py", "r") as f:
    content = f.read()
content = content.replace(
    "node.description",
    "getattr(node, 'description', '') if not isinstance(node, dict) else node.get('description', '')",
)
with open("src/coreason_inference_engine/context.py", "w") as f:
    f.write(content)
