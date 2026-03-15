# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from coreason_inference_engine.adapters.http_adapter import BaseHttpAdapter
from coreason_inference_engine.utils.network import SSRFValidationError


@pytest.fixture
def mock_adapter() -> BaseHttpAdapter:
    return BaseHttpAdapter(api_url="https://api.openai.com/v1/chat/completions", api_key="test_key")


@pytest.mark.asyncio
async def test_base_http_adapter_ssrf_validation_failure() -> None:
    adapter = BaseHttpAdapter(api_url="http://127.0.0.1/v1/completions", api_key="test_key")
    gen = adapter.generate_stream(messages=[], tools=[], temperature=0.0)
    with pytest.raises(SSRFValidationError):
        # We need to manually consume the generator
        await gen.__anext__()


@pytest.mark.asyncio
async def test_base_http_adapter_success(mock_adapter: BaseHttpAdapter) -> None:
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock(return_value=None)

    # Mock stream response lines
    async def mock_aiter_lines() -> Any:
        lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"usage": {"prompt_tokens": 5, "completion_tokens": 10}}',
            "data: [DONE]",
        ]
        for line in lines:
            yield line

    mock_response.aiter_lines = mock_aiter_lines

    # We need to mock the context manager
    class MockStreamContext:
        async def __aenter__(self) -> Any:
            return mock_response

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    with (
        patch(
            "coreason_inference_engine.adapters.http_adapter.validate_url_for_ssrf", new_callable=AsyncMock
        ) as mock_validate,
        patch.object(mock_adapter.client, "stream", return_value=MockStreamContext()),
    ):
        chunks = []
        usages = []
        async for chunk, usage in mock_adapter.generate_stream(messages=[], tools=[], temperature=0.0):
            chunks.append(chunk)
            usages.append(usage)

        assert "Hello" in chunks
        assert usages[-1] == {"input_tokens": 5, "output_tokens": 10}
        mock_validate.assert_called_once_with("https://api.openai.com/v1/chat/completions")


@pytest.mark.asyncio
async def test_base_http_adapter_network_failure(mock_adapter: BaseHttpAdapter) -> None:
    # Simulate httpx.HTTPStatusError
    class MockStreamContext:
        async def __aenter__(self) -> Any:
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            resp = httpx.Response(502, request=req)
            raise httpx.HTTPStatusError("Bad Gateway", request=req, response=resp)

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    with (
        patch("coreason_inference_engine.adapters.http_adapter.validate_url_for_ssrf", new_callable=AsyncMock),
        patch.object(mock_adapter.client, "stream", return_value=MockStreamContext()),
        pytest.raises(httpx.HTTPStatusError),
    ):
        async for _chunk, _usage in mock_adapter.generate_stream(messages=[], tools=[], temperature=0.0):
            pass


def test_base_http_adapter_count_tokens(mock_adapter: BaseHttpAdapter) -> None:
    assert mock_adapter.count_tokens("hello") == 5


@pytest.mark.asyncio
async def test_base_http_adapter_close(mock_adapter: BaseHttpAdapter) -> None:
    with patch.object(mock_adapter.client, "aclose", new_callable=AsyncMock) as mock_aclose:
        await mock_adapter.close()
        mock_aclose.assert_called_once()


def test_base_http_adapter_project_tools(mock_adapter: BaseHttpAdapter) -> None:
    schemas: list[dict[str, Any]] = [{"type": "function", "function": {"name": "test"}}]
    assert mock_adapter.project_tools(schemas) == schemas


@pytest.mark.asyncio
async def test_base_http_adapter_apply_peft_adapters(mock_adapter: BaseHttpAdapter) -> None:
    # Just verify it does not raise
    await mock_adapter.apply_peft_adapters([])


@pytest.mark.asyncio
async def test_base_http_adapter_prepare_request_payload(mock_adapter: BaseHttpAdapter) -> None:
    payload = mock_adapter._prepare_request_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "test"}}],
        temperature=0.5,
        logit_biases={123: 10.0},
        max_tokens=100,
    )
    assert payload["messages"][0]["content"] == "hi"
    assert payload["tools"][0]["function"]["name"] == "test"
    assert payload["temperature"] == 0.5
    assert payload["logit_bias"] == {123: 10.0}
    assert payload["max_tokens"] == 100


@pytest.mark.asyncio
async def test_base_http_adapter_stream_response_decode_error(mock_adapter: BaseHttpAdapter) -> None:
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock(return_value=None)

    async def mock_aiter_lines() -> Any:
        lines = [
            'data: {"invalid_json',
            "data: ",  # empty data string test (continue)
            "data: [DONE]",
        ]
        for line in lines:
            yield line

    mock_response.aiter_lines = mock_aiter_lines

    class MockStreamContext:
        async def __aenter__(self) -> Any:
            return mock_response

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    with (
        patch("coreason_inference_engine.adapters.http_adapter.validate_url_for_ssrf", new_callable=AsyncMock),
        patch.object(mock_adapter.client, "stream", return_value=MockStreamContext()),
    ):
        chunks = []
        async for chunk, _usage in mock_adapter.generate_stream(messages=[], tools=[], temperature=0.0):
            chunks.append(chunk)

        # no valid json, chunks should be empty (though we might yield empty strings based on current naive parsing)
        # actually, json decode error does a pass, so it won't yield
        assert chunks == []
