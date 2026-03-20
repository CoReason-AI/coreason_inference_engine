# Deep-Dive Critical Omissions & Risk Analysis
**Target:** `coreason_inference_engine` (`engine.py`, `context.py`, adapters)

## 1. The "Ghosted Feature" Audit (Mechanistic & Constrained Decoding)
**Analysis & Findings:**
- In `InferenceEngine.generate_intent()`, the variables `latent_firewalls` and `format_contract` are correctly extracted from the `node` (via `information_flow_policy` and `grpo_reward_policy`).
- However, when passed to `self.adapter.generate_stream(..., latent_firewalls=latent_firewalls, format_contract=format_contract)`, these features are completely dropped by the HTTP adapters.
- **Dead Code:** `OpenAIAdapter` and `AnthropicAdapter` ignore `**kwargs` in `generate_stream`. They do not map `latent_firewalls` or `format_contract` to the API payloads, meaning `SaeLatentPolicy` and `ConstrainedDecodingPolicy` (FSM logit masking) are effectively "ghosted" on remote providers. Only the local vLLM adapter makes an attempt to use them, but the cognitive plane assumes they are universally enforced.

## 2. Logic Leakage & Boundary Violations
**Analysis & Findings:**
- The `coreason_manifest` (Shared Kernel) is mandated to be strictly passive, holding data contracts and schemas.
- **Boundary Violation in `engine.py`:** `_determine_target_schema()` and `_get_target_json_schema()` contain hardcoded string mappings (e.g., `"state_differential"`, `"symbolic_handoff"`, `"intent"`, `"step8_vision"`) and explicitly map them to Pydantic models imported directly from the ontology.
- `_validate_intent()` also hallucinates schema resolution logic by checking `if schema_key in ("intent", "state_differential", "symbolic_handoff", "AnyIntent"):`.
- **Location:** `src/coreason_inference_engine/engine.py` lines ~100-140. This creates brittle coupling where the inference engine implements business logic instead of relying on the Shared Kernel's algebraic functors or a registry.

## 3. Thermodynamic & Token Economy Risk Assessment
**Analysis & Findings:**
- **Token Leakage on Severed Streams:** In `engine.py`, if a stream is severed early due to a `structural_violation` (FR-3.2 Fail-fast JSON parsing), the engine calculates the burn as `total_output_tokens + usage_metrics.get("output_tokens", 0)`. Because APIs (like OpenAI/Anthropic) only emit usage stats in the *final* chunk, an early severance means `usage_metrics` is empty. The engine fails to fallback to `self.adapter.count_tokens(raw_output)` in this early-exit path, leaking tokens and failing to charge for the burned output.
- **VRAM OOM / JSON Bombing in Semantic Slicing:** The `_apply_semantic_slicing` method only evicts the oldest `ObservationEvent` payloads when the token ceiling is exceeded. If an adversary or faulty tool injects massive `ToolInvocationEvent` payloads or large `System2RemediationIntent` payloads, the slicing policy will infinitely loop or fail to reduce the token mass, leading to VRAM explosion and OOM crashes.

## 4. Required Autonomous Remediation Plan
### Prioritized Punch-List:
1. **Fix Token Leakage:** Update the early-exit structural violation block in `engine.py` to calculate raw token mass using `self.adapter.count_tokens` if `usage_metrics` is empty.
2. **Fix Semantic Slicing:** Update `_apply_semantic_slicing` to evict the oldest events regardless of type (excluding critical system boundaries or the single retained remediation intent).
3. **Rip Hardcoded Schemas:** Refactor `_determine_target_schema` and `_validate_intent` to dispatch to a generic `SchemaRegistry` exposed by the `coreason_manifest` instead of using inline string mapping.
4. **Adapter Protocol Update:** Implement `format_contract` and `latent_firewalls` integration in `LLMAdapterProtocol` and push support or strict explicit rejections in `OpenAIAdapter` and `AnthropicAdapter`.

### Pseudo-code for Remediation:
**Schema Resolution (Kernel Side):**
```python
# coreason_manifest/registry.py
def resolve_schema_for_node(node: AgentNodeProfile) -> Type[BaseModel]:
    if node.interventional_policy: return StateMutationIntent
    # ...
```
**Engine Side:**
```python
# engine.py
def _validate_intent(self, node: AgentNodeProfile, payload: bytes) -> Any:
    target_schema = SchemaRegistry.resolve_schema_for_node(node)
    return target_schema.model_validate_json(payload)
```
**Token Leakage Fix:**
```python
# engine.py (inside structural_violation check)
out_tokens = usage_metrics.get("output_tokens", 0) or self.adapter.count_tokens(raw_output)
in_tokens = usage_metrics.get("input_tokens", 0) or self.adapter.count_tokens(json.dumps(messages))
burn_receipt = TokenBurnReceipt(..., input_tokens=total_input_tokens + in_tokens, output_tokens=total_output_tokens + out_tokens, ...)
```
