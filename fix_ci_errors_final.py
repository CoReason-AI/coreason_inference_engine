with open("src/coreason_inference_engine/engine.py", "r") as f:
    content = f.read()

# Fix mypy error: dict no attribute model_dump
content = content.replace("return res.model_dump()", "return res.model_dump() if hasattr(res, 'model_dump') else res")
content = content.replace("return out_res.model_dump()", "return out_res.model_dump() if hasattr(out_res, 'model_dump') else out_res")
content = content.replace("import time\n                    import uuid\n\n                    return {\n                        \"event_id\": data.get(\"event_id\", f\"evt_{uuid.uuid4().hex[:8]}\"),\n                        \"timestamp\": data.get(\"timestamp\", time.time()),\n                        \"type\": \"tool_invocation\",\n                        tool_name=data.get(\"tool_name\", \"unknown\"),\n                        parameters=params,\n                        # Provide explicit Nones/empty mocks for strict schema requirements\n                        authorized_budget_magnitude=1,\n                        agent_attestation={\n                            training_lineage_hash=\"0\" * 64,\n                            developer_signature=\"mock\",\n                            capability_merkle_root=\"0\" * 64,\n                            credential_presentations=[],\n                        },\n                        zk_proof={\n                            proof_protocol=\"zk-SNARK\",\n                            public_inputs_hash=\"0\" * 64,\n                            verifier_key_id=\"mock-key\",\n                            cryptographic_blob=\"mock-blob\",\n                        },\n                    }", "import time\n                    import uuid\n\n                    return {\n                        \"event_id\": data.get(\"event_id\", f\"evt_{uuid.uuid4().hex[:8]}\"),\n                        \"timestamp\": data.get(\"timestamp\", time.time()),\n                        \"type\": \"tool_invocation\",\n                        \"tool_name\": data.get(\"tool_name\", \"unknown\"),\n                        \"parameters\": params,\n                        \"authorized_budget_magnitude\": 1,\n                        \"agent_attestation\": {\n                            \"training_lineage_hash\": \"0\" * 64,\n                            \"developer_signature\": \"mock\",\n                            \"capability_merkle_root\": \"0\" * 64,\n                            \"credential_presentations\": [],\n                        },\n                        \"zk_proof\": {\n                            \"proof_protocol\": \"zk-SNARK\",\n                            \"public_inputs_hash\": \"0\" * 64,\n                            \"verifier_key_id\": \"mock-key\",\n                            \"cryptographic_blob\": \"mock-blob\",\n                        },\n                    }")

# Fix test_interfaces.py missing raw_node, etc.
with open("tests/test_interfaces.py", "r") as f:
    test_content = f.read()

test_content = test_content.replace(
    "await engine.generate_intent(\n            node=node_with_logit.model_dump() if hasattr(node_with_logit, 'model_dump') else node_with_logit, ledger=mock_ledger.model_dump() if hasattr(mock_ledger, 'model_dump') else mock_ledger, node_id=\"did:test:1\", action_space=mock_action_space.model_dump() if hasattr(mock_action_space, 'model_dump') else mock_action_space\n        )",
    "await engine.generate_intent(\n            raw_node=node_with_logit.model_dump() if hasattr(node_with_logit, 'model_dump') else node_with_logit, raw_ledger=mock_ledger.model_dump() if hasattr(mock_ledger, 'model_dump') else mock_ledger, node_id=\"did:test:1\", raw_action_space=mock_action_space.model_dump() if hasattr(mock_action_space, 'model_dump') else mock_action_space\n        )"
)
test_content = test_content.replace("node=node.model_dump()", "raw_node=node.model_dump()")
test_content = test_content.replace("node=node", "raw_node=node.model_dump() if hasattr(node, 'model_dump') else node")
test_content = test_content.replace("ledger=ledger", "raw_ledger=ledger.model_dump() if hasattr(ledger, 'model_dump') else ledger")
test_content = test_content.replace("action_space=action_space", "raw_action_space=action_space.model_dump() if hasattr(action_space, 'model_dump') else action_space")

with open("tests/test_interfaces.py", "w") as f:
    f.write(test_content)

# Fix test_engine.py missing ledger etc.
with open("tests/test_engine.py", "r") as f:
    test_engine_content = f.read()

test_engine_content = test_engine_content.replace("node=node.model_dump()", "raw_node=node.model_dump()")
test_engine_content = test_engine_content.replace("node=node", "raw_node=node.model_dump() if hasattr(node, 'model_dump') else node")
test_engine_content = test_engine_content.replace("ledger=mock_ledger", "raw_ledger=mock_ledger.model_dump() if hasattr(mock_ledger, 'model_dump') else mock_ledger")
test_engine_content = test_engine_content.replace("action_space=mock_action_space", "raw_action_space=mock_action_space.model_dump() if hasattr(mock_action_space, 'model_dump') else mock_action_space")
test_engine_content = test_engine_content.replace("assert receipt.get(\"input_tokens\") < receipt.get(\"output_tokens\")", "assert receipt.get(\"input_tokens\", 0) < receipt.get(\"output_tokens\", 0)")
test_engine_content = test_engine_content.replace("assert receipt.get(\"input_tokens\") > 0", "assert receipt.get(\"input_tokens\", 0) > 0")

with open("tests/test_engine.py", "w") as f:
    f.write(test_engine_content)

# Fix test_engine_reflex.py missing dict bounds checks
with open("tests/test_engine_reflex.py", "r") as f:
    test_reflex_content = f.read()
test_reflex_content = test_reflex_content.replace("assert receipt.get(\"input_tokens\") < receipt.get(\"output_tokens\")", "assert receipt.get(\"input_tokens\", 0) < receipt.get(\"output_tokens\", 0)")
with open("tests/test_engine_reflex.py", "w") as f:
    f.write(test_reflex_content)
