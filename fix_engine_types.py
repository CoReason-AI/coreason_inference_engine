
with open("src/coreason_inference_engine/engine.py") as f:
    content = f.read()

# Fix types in _evaluate_system1_reflex
content = content.replace(
    """    async def _evaluate_system1_reflex(
        self,
        node: LocalAgentNodeProfile,
        ledger: LocalLedgerState,
        _node_id: str,
        action_space: LocalActionSpace,
    ) -> tuple[AnyIntent | AnyStateEvent | None, TokenBurnReceipt | None, LatentScratchpadReceipt | None, int, int]:""",
    """    async def _evaluate_system1_reflex(
        self,
        node: LocalAgentNodeProfile,
        ledger: LocalLedgerState,
        _node_id: str,
        action_space: LocalActionSpace,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, int, int]:""",
)

# Fix Unused ignores
content = content.replace("return res.model_dump() # type: ignore", "return res.model_dump()")
content = content.replace("return dict(res) # type: ignore", "return dict(res)")
content = content.replace("return out_res.model_dump() # type: ignore", "return out_res.model_dump()")
content = content.replace("return dict(out_res) # type: ignore", "return dict(out_res)")
content = content.replace("def _extract_latent_traces( # type: ignore", "def _extract_latent_traces(")

# Remove Pydantic models from _validate_intent
content = content.replace(
    """                        agent_attestation=AgentAttestationReceipt(
                            training_lineage_hash="0" * 64,
                            developer_signature="mock",
                            capability_merkle_root="0" * 64,
                            credential_presentations=[],
                        ),
                        zk_proof=ZeroKnowledgeReceipt(
                            proof_protocol="zk-SNARK",
                            public_inputs_hash="0" * 64,
                            verifier_key_id="mock-key",
                            cryptographic_blob="mock-blob",
                        ),""",
    """                        agent_attestation=dict(
                            training_lineage_hash="0" * 64,
                            developer_signature="mock",
                            capability_merkle_root="0" * 64,
                            credential_presentations=[],
                        ),
                        zk_proof=dict(
                            proof_protocol="zk-SNARK",
                            public_inputs_hash="0" * 64,
                            verifier_key_id="mock-key",
                            cryptographic_blob="mock-blob",
                        ),""",
)

content = content.replace(
    """invocation_cid = valid_intent.get('event_id') if isinstance(valid_intent, dict) and valid_intent.get('type') == 'tool_invocation' else "none\"""",
    """invocation_cid = valid_intent.get('event_id', "none") if isinstance(valid_intent, dict) and valid_intent.get('type') == 'tool_invocation' else "none\"""",
)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(content)
