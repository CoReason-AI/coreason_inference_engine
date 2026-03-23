with open("src/coreason_inference_engine/engine.py", "r") as f:
    content = f.read()

content = content.replace("tool_name=data.get(\"tool_name\", \"unknown\"),", "\"tool_name\": data.get(\"tool_name\", \"unknown\"),")
content = content.replace("parameters=params,", "\"parameters\": params,")
content = content.replace("authorized_budget_magnitude=1,", "\"authorized_budget_magnitude\": 1,")
content = content.replace("agent_attestation=dict(", "\"agent_attestation\": dict(")
content = content.replace("zk_proof=dict(", "\"zk_proof\": dict(")

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(content)
