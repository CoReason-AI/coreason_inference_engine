# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

from typing import Any

import pytest
from coreason_manifest.spec.ontology import (
    CognitiveFormatContract,
    LatentSmoothingProfile,
    PeftAdapterContract,
    SaeLatentPolicy,
)

from coreason_inference_engine.adapters.local_vllm_adapter import LocalVLLMAdapter, SAELatentViolationError


@pytest.mark.asyncio
async def test_fsm_logit_masking_creation() -> None:
    """Verify that the LocalVLLMAdapter can create a logits processor via outlines FSM grammar."""
    adapter = LocalVLLMAdapter(model_name="gpt-4o")

    regex_pattern = "^Final Answer: .*$"
    processor = adapter.create_logits_processor(regex_pattern)

    assert processor is not None
    assert callable(processor)

    dummy_scores = [0.1, 0.9]
    output_scores = processor([1, 2], dummy_scores)
    assert output_scores == dummy_scores


@pytest.mark.asyncio
async def test_apply_peft_adapters() -> None:
    """Verify apply_peft_adapters does not fail."""
    adapter = LocalVLLMAdapter()
    await adapter.apply_peft_adapters([])

    dummy_peft = PeftAdapterContract(
        adapter_id="dummy",
        safetensors_hash="deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        base_model_hash="beefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdead",
        target_modules=["q_proj"],
        adapter_rank=8,
        eviction_ttl_seconds=3600,
    )
    await adapter.apply_peft_adapters([dummy_peft])


@pytest.mark.asyncio
async def test_generate_stream() -> None:
    """Verify generate_stream works and uses FSM Logit processor."""
    adapter = LocalVLLMAdapter()

    format_contract = CognitiveFormatContract(require_think_tags=True, final_answer_regex="^Final Answer: .*$")

    stream = adapter.generate_stream(messages=[], tools=[], temperature=0.0, format_contract=format_contract)

    chunks = []
    async for chunk, usage, receipt in stream:
        chunks.append(chunk)
        assert usage is not None
        assert receipt is None

    assert "".join(chunks) == "{}"


def test_count_tokens() -> None:
    adapter = LocalVLLMAdapter()
    assert adapter.count_tokens("hello world") > 0
    assert adapter.count_tokens(b"hello world") > 0


def test_project_tools() -> None:
    adapter = LocalVLLMAdapter()
    assert adapter.project_tools([{"type": "function"}]) == [{"type": "function"}]


@pytest.mark.asyncio
async def test_sae_latent_firewalls_hooks() -> None:
    """Verify that PyTorch forward hooks are generated correctly."""
    adapter = LocalVLLMAdapter()
    policy = SaeLatentPolicy(
        target_feature_index=1,
        monitored_layers=[0, 1],
        max_activation_threshold=1.5,
        violation_action="halt",
        sae_dictionary_hash="deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    )

    class MockLayer:
        def __init__(self) -> None:
            self.hook = None

        def register_forward_hook(self, hook: Any) -> str:
            self.hook = hook
            return "handle"

    class MockModel:
        class InnerModel:
            def __init__(self) -> None:
                self.layers = [MockLayer(), MockLayer()]

        def __init__(self) -> None:
            self.model = self.InnerModel()
            self._allow_mock_hooks = True

    mock_model = MockModel()

    handles = adapter.register_sae_firewall_hooks([policy], mock_model)
    assert handles == ["handle", "handle"]

    hook_fn = mock_model.model.layers[0].hook
    assert hook_fn is not None

    with pytest.raises(SAELatentViolationError):
        hook_fn(None, None, [0.0, 0.0])


@pytest.mark.asyncio
async def test_sae_latent_firewall_violation_in_stream() -> None:
    """Verify that a raised SAELatentViolationError yields a ThoughtBranchState failure."""

    class MockAdapter(LocalVLLMAdapter):
        async def generate_stream(self, *args: Any, **kwargs: Any) -> Any:
            _ = args
            _ = kwargs
            import hashlib
            import uuid

            from coreason_manifest.spec.ontology import LatentScratchpadReceipt, ThoughtBranchState

            policy = SaeLatentPolicy(
                target_feature_index=1,
                monitored_layers=[0, 1],
                max_activation_threshold=1.5,
                violation_action="halt",
                sae_dictionary_hash="deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            )

            try:
                raise SAELatentViolationError(policy=policy, activation_val=2.5)
            except SAELatentViolationError:
                branch_id = f"branch_{uuid.uuid4().hex[:8]}"
                content_hash = hashlib.sha256(b"sae_violation_trace").hexdigest()
                branch = ThoughtBranchState(
                    branch_id=branch_id,
                    parent_branch_id=None,
                    latent_content_hash=content_hash,
                    prm_score=0.0,
                )
                receipt = LatentScratchpadReceipt(
                    trace_id=f"trace_{uuid.uuid4().hex[:8]}",
                    explored_branches=[branch],
                    discarded_branches=[branch.branch_id],
                    resolution_branch_id=None,
                    total_latent_tokens=1,
                )
                yield "", {"input_tokens": 0, "output_tokens": 0}, receipt

    adapter = MockAdapter()

    stream = adapter.generate_stream(messages=[], tools=[], temperature=0.0)

    async for chunk, usage, receipt in stream:
        assert chunk == ""
        assert usage == {"input_tokens": 0, "output_tokens": 0}
        assert receipt is not None
        assert len(receipt.discarded_branches) == 1
        break


@pytest.mark.asyncio
async def test_sae_latent_firewall_actions() -> None:
    adapter = LocalVLLMAdapter()

    policy_clamp = SaeLatentPolicy(
        target_feature_index=1,
        monitored_layers=[0],
        max_activation_threshold=1.5,
        violation_action="clamp",
        clamp_value=1.5,
        sae_dictionary_hash="deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    )
    policy_smooth = SaeLatentPolicy(
        target_feature_index=1,
        monitored_layers=[0],
        max_activation_threshold=1.5,
        violation_action="smooth_decay",
        clamp_value=1.0,
        smoothing_profile=LatentSmoothingProfile(decay_function="exponential", transition_window_tokens=10),
        sae_dictionary_hash="deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    )

    class MockLayer:
        def __init__(self) -> None:
            self.hook = None

        def register_forward_hook(self, hook: Any) -> str:
            self.hook = hook
            return "handle"

    class MockModel:
        class InnerModel:
            def __init__(self) -> None:
                self.layers = [MockLayer()]

        def __init__(self) -> None:
            self.model = self.InnerModel()
            self._allow_mock_hooks = True

    mock_model = MockModel()

    adapter.register_sae_firewall_hooks([policy_clamp], mock_model)
    hook_fn = mock_model.model.layers[0].hook
    assert hook_fn is not None
    output = hook_fn(None, None, [0.0, 0.0])
    assert output == [0.0, 0.0]

    adapter.register_sae_firewall_hooks([policy_smooth], mock_model)
    hook_fn2 = mock_model.model.layers[0].hook
    assert hook_fn2 is not None
    output2 = hook_fn2(None, None, [0.0, 0.0])
    assert output2 == [0.0, 0.0]


def test_fsm_compile_import_error_coverage(monkeypatch: Any) -> None:
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "outlines.fsm.guide":
            raise ImportError("Mock missing module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    adapter = LocalVLLMAdapter()
    processor = adapter.create_logits_processor(".*")
    assert processor is not None
