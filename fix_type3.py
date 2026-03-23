with open("src/coreason_inference_engine/engine.py", "r") as f:
    content = f.read()

content = content.replace(
    """                        "zk_proof": dict(
                            proof_protocol="zk-SNARK",
                            public_inputs_hash="0" * 64,
                            verifier_key_id="mock-key",
                            cryptographic_blob="mock-blob",
                        ),
                    )""",
    """                        "zk_proof": dict(
                            proof_protocol="zk-SNARK",
                            public_inputs_hash="0" * 64,
                            verifier_key_id="mock-key",
                            cryptographic_blob="mock-blob",
                        ),
                    }"""
)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(content)
