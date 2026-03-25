import json
from unittest.mock import MagicMock, mock_open, patch

from coreason_inference_engine.engine import InferenceEngine
from tests.test_engine import DummyAdapter


@patch("importlib.resources.files")
def test_engine_schema_caching_success(mock_files: MagicMock) -> None:
    # Setup mock to return a valid schema
    schema_data = {"type": "object", "$defs": {"AnyIntent": {"type": "object"}}}
    mock_path = mock_files.return_value.joinpath.return_value
    mock_path.open = mock_open(read_data=json.dumps(schema_data))

    adapter = DummyAdapter(responses=[])
    engine = InferenceEngine(adapter)

    # First call should hit the disk/mock
    schema = engine._get_target_json_schema("intent")
    assert engine._cached_schema is not None
    assert "$defs" in engine._cached_schema
    assert schema["$defs"] == schema_data["$defs"]

    # Second call should return cached
    mock_path.open.reset_mock()
    schema2 = engine._get_target_json_schema("intent")
    assert schema2["$defs"] == schema_data["$defs"]
    mock_path.open.assert_not_called()


@patch("importlib.resources.files")
def test_engine_schema_caching_failure(mock_files: MagicMock) -> None:
    mock_files.side_effect = Exception("disk error")

    adapter = DummyAdapter(responses=[])
    engine = InferenceEngine(adapter)

    _schema = engine._get_target_json_schema("intent")
    assert engine._cached_schema is not None
    assert engine._cached_schema == {"type": "object", "$defs": {}}
