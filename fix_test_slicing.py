with open("tests/test_engine_slicing.py", "r") as f:
    content = f.read()

content = content.replace(
    "messages = engine._apply_semantic_slicing(node, ledger)",
    "messages = engine._apply_semantic_slicing(LocalAgentNodeProfile(**node.model_dump()) if hasattr(node, 'model_dump') else LocalAgentNodeProfile(**node), LocalLedgerState(**ledger.model_dump()) if hasattr(ledger, 'model_dump') else LocalLedgerState(**ledger))",
)
content = content.replace(
    'messages = engine._apply_semantic_slicing(node, ledger.model_copy(update={"history": history_with_remediations}))',
    "ledger_dict = ledger.model_copy(update={'history': history_with_remediations}).model_dump()\n        messages = engine._apply_semantic_slicing(LocalAgentNodeProfile(**node.model_dump()) if hasattr(node, 'model_dump') else LocalAgentNodeProfile(**node), LocalLedgerState(**ledger_dict))",
)
content = content.replace(
    "history_with_remediations = [rem1, obs1, rem2, rem3]",
    "history_with_remediations = [{'type': 'system2_remediation', 'fault_id': rem1.fault_id, 'target_node_id': rem1.target_node_id, 'failing_pointers': rem1.failing_pointers, 'remediation_prompt': rem1.remediation_prompt}, obs1, {'type': 'system2_remediation', 'fault_id': rem2.fault_id, 'target_node_id': rem2.target_node_id, 'failing_pointers': rem2.failing_pointers, 'remediation_prompt': rem2.remediation_prompt}, {'type': 'system2_remediation', 'fault_id': rem3.fault_id, 'target_node_id': rem3.target_node_id, 'failing_pointers': rem3.failing_pointers, 'remediation_prompt': rem3.remediation_prompt}]",
)
content = content.replace(
    "from coreason_inference_engine.adapters.dto import LocalAgentNodeProfile, LocalLedgerState\n        ",
    "from coreason_inference_engine.adapters.dto import LocalAgentNodeProfile, LocalLedgerState\n        ",
)

with open("tests/test_engine_slicing.py", "w") as f:
    f.write(content)
