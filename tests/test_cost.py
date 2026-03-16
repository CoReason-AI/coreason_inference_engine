from collections.abc import AsyncGenerator
from typing import Any

from coreason_manifest.spec.ontology import (
    AgentNodeProfile,
    InterventionalCausalTask,
    LogitSteganographyContract,
)

from coreason_inference_engine.engine import InferenceEngine
from coreason_inference_engine.interfaces import LLMAdapterProtocol


class MockAdapterWithCost(LLMAdapterProtocol):
    def __init__(self) -> None:
        self.rate_card = type(
            "DummyCard",
            (),
            {
                "cost_per_million_input_tokens": 10.0,
                "cost_per_million_output_tokens": 20.0,
                "magnitude_unit": "micro_usd",
            },
        )()

    def count_tokens(self, text: str) -> int:  # noqa: ARG002
        return 0

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:  # noqa: ARG002
        return []

    async def apply_peft_adapters(self, adapters: list[Any]) -> None:
        pass

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],  # noqa: ARG002
        tools: list[dict[str, Any]],  # noqa: ARG002
        temperature: float,  # noqa: ARG002
        logit_biases: dict[int, float] | None = None,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> AsyncGenerator[tuple[str, dict[str, int]]]:
        yield "", {}


def test_calculate_cost_with_rate_card() -> None:
    adapter = MockAdapterWithCost()
    engine = InferenceEngine(adapter)

    cost = engine._calculate_cost(1000, 2000)
    assert cost == 0.05

    # test None
    adapter.rate_card = None
    cost2 = engine._calculate_cost(1000, 2000)
    assert cost2 == 3000


def test_compile_watermark_biases() -> None:
    adapter = MockAdapterWithCost()
    engine = InferenceEngine(adapter)

    contract = LogitSteganographyContract(
        verification_public_key_id="pub",
        prf_seed_hash="0" * 64,
        watermark_strength_delta=5.0,
        target_bits_per_token=2.0,
        context_history_window=10,
    )

    res = engine._compile_watermark_biases(contract, vocab_size=1000, prior_tokens=[1, 2, 3])
    assert res is not None
    assert len(res) == 500
    assert next(iter(res.values())) == 5.0

    res2 = engine._compile_watermark_biases(None)
    assert res2 is None

    res3 = engine._compile_watermark_biases(contract, vocab_size=1000, prior_tokens=None)
    assert res3 is not None
    assert len(res3) == 500

    contract2 = LogitSteganographyContract(
        verification_public_key_id="pub",
        prf_seed_hash="0" * 64,
        watermark_strength_delta=5.0,
        target_bits_per_token=2.0,
        context_history_window=0,
    )
    res4 = engine._compile_watermark_biases(contract2, vocab_size=1000, prior_tokens=[1, 2, 3])
    assert res4 is not None
    assert len(res4) == 500


def test_determine_target_schema() -> None:
    adapter = MockAdapterWithCost()
    engine = InferenceEngine(adapter)

    node = AgentNodeProfile(description="desc", type="agent")
    assert engine._determine_target_schema(node) == "intent"

    node2 = AgentNodeProfile(
        description="desc",
        type="agent",
        interventional_policy=InterventionalCausalTask(
            task_id="1",
            target_hypothesis_id="2",
            intervention_variable="x",
            do_operator_state="y",
            expected_causal_information_gain=0.5,
            execution_cost_budget_magnitude=10,
        ),
    )
    assert engine._determine_target_schema(node2) == "state_differential"

    # Use a simple class that quacks like NeuroSymbolicHandoffContract
    class DummyNeuro:
        pass

    node3 = AgentNodeProfile(
        description="desc",
        type="agent",
    )
    # Pydantic is too strict, just monkeypatch it
    object.__setattr__(node3, "symbolic_handoff_policy", DummyNeuro())
    assert engine._determine_target_schema(node3) == "symbolic_handoff"
