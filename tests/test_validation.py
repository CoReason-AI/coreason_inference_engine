from typing import Any

import jsonschema

from coreason_inference_engine.utils.validation import generate_correction_prompt


def test_generate_correction_prompt_jsonschema_required() -> None:
    error = jsonschema.exceptions.ValidationError("msg", validator="required", path=["a", "b"])
    res = generate_correction_prompt(error, "did:1", "fault1")
    assert "required semantic boundary is completely missing" in res["remediation_prompt"]
    assert "/a/b" in res["failing_pointers"]


def test_generate_correction_prompt_jsonschema_other() -> None:
    error = jsonschema.exceptions.ValidationError("invalid type", validator="type", path=["x"])
    res = generate_correction_prompt(error, "did:1", "fault1")
    assert "invalid type" in res["remediation_prompt"]
    assert "/x" in res["failing_pointers"]


def test_generate_correction_prompt_jsonschema_no_path() -> None:
    error = jsonschema.exceptions.ValidationError("invalid type", validator="type", path=[])
    res = generate_correction_prompt(error, "did:1", "fault1")
    assert "invalid type" in res["remediation_prompt"]
    assert "/" in res["failing_pointers"]


class MockError(Exception):
    def __init__(self, errs: list[dict[str, Any]]) -> None:
        self._errs = errs

    def errors(self) -> list[dict[str, Any]]:
        return self._errs


def test_generate_correction_prompt_pydantic_missing() -> None:
    error = MockError([{"type": "missing", "loc": ("a",)}])
    res = generate_correction_prompt(error, "did:1", "fault1")
    assert "required semantic boundary is completely missing" in res["remediation_prompt"]
    assert "/a" in res["failing_pointers"]


def test_generate_correction_prompt_pydantic_other() -> None:
    error = MockError([{"type": "value_error", "loc": ("b",), "msg": "bad val"}])
    res = generate_correction_prompt(error, "did:1", "fault1")
    assert "bad val" in res["remediation_prompt"]
    assert "/b" in res["failing_pointers"]


def test_generate_correction_prompt_fallback() -> None:
    error = ValueError("generic error")
    res = generate_correction_prompt(error, "did:1", "fault1")
    assert "generic error" in res["remediation_prompt"]
    assert "/" in res["failing_pointers"]
