1. **Change the method signature in `engine.py`**:
   - Update `generate_intent` to take `node: dict[str, Any]`, `ledger: dict[str, Any]`, `node_id: str`, and `action_space: dict[str, Any]`.
   - Update `_evaluate_system1_reflex` and `_apply_semantic_slicing` similarly, mapping into Local DTOs if necessary.
   - Return raw dicts for intents, receipts, etc., instead of `coreason_manifest` objects.

2. **Strip global coupling**:
   - Remove imports of `coreason_manifest.spec.ontology` inside `engine.py`.
   - Update any `BaseModel` instantiations to return dictionaries.

3. **Convert to use Local DTOs**:
   - In `generate_intent`, parse the incoming dicts using the internal Local DTOs defined in `src/coreason_inference_engine/adapters/dto.py`. E.g., `local_node = LocalAgentNodeProfile(**node)`, `local_ledger = LocalLedgerState(**ledger)`, etc.
   - Use `local_node`, `local_ledger`, and `local_action_space` throughout the logic.

4. **Return dictionaries**:
   - Ensure `TokenBurnReceipt`, `LatentScratchpadReceipt`, `CognitiveRewardEvaluationReceipt`, and intents (like `System2RemediationIntent`, `FallbackIntent`, etc.) are returned as standard Python dictionaries or lists of dictionaries.

5. **Fix Type Definitions**:
   - Change type hinting in `engine.py` to `dict[str, Any]` everywhere `coreason_manifest` objects were previously used.

6. **Fix downstream usages in tests**:
   - Although the task explicitly targets `engine.py`, we may need to touch tests to ensure `generate_intent` callers pass dicts instead of models, and tests expect dicts as returns to maintain 100% code coverage. Wait, maybe that's out of scope unless tests fail. We'll fix tests if necessary.

7. **Pre-commit steps**: Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
