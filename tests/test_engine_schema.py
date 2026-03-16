from coreason_manifest.spec.ontology import AgentNodeProfile

from coreason_inference_engine.engine import InferenceEngine


def test_determine_target_schema_action_space_id() -> None:
    # Test action_space_id fallback branch
    engine = InferenceEngine(adapter=None)  # type: ignore
    node = AgentNodeProfile(description="test", type="agent", action_space_id="my_custom_schema")
    res = engine._determine_target_schema(node)
    assert res == "my_custom_schema"
