# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coreason_manifest.spec.ontology import PeftAdapterContract

from coreason_inference_engine.adapters.openai_adapter import OpenAIAdapter


@pytest.fixture
def adapter() -> OpenAIAdapter:
    return OpenAIAdapter(api_url="https://api.openai.com/v1/chat/completions", api_key="test_key")


@pytest.fixture
def fallback_adapter() -> OpenAIAdapter:
    # Test fallback to cl100k_base
    return OpenAIAdapter(
        api_url="https://api.openai.com/v1/chat/completions", api_key="test_key", model_name="unknown-model-xyz"
    )


def test_openai_adapter_count_tokens(adapter: OpenAIAdapter) -> None:
    # "hello world" in cl100k_base is typically 2 tokens
    assert adapter.count_tokens("hello world") == 2
    assert adapter.count_tokens(b"hello world") == 2


def test_openai_adapter_count_tokens_fallback(fallback_adapter: OpenAIAdapter) -> None:
    assert fallback_adapter.count_tokens("hello world") == 2


def test_openai_adapter_project_tools(adapter: OpenAIAdapter) -> None:
    schemas: list[dict[str, Any]] = [
        # Already correct
        {"type": "function", "function": {"name": "test", "description": "desc", "parameters": {"type": "object"}}},
        # Anthropic style or missing type
        {"name": "test2", "input_schema": {"type": "object", "properties": {"a": "b"}}},
        # Missing parameters completely
        {"name": "test3", "description": "desc3"},
    ]
    tools = adapter.project_tools(schemas)
    assert len(tools) == 3

    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "test"

    assert tools[1]["type"] == "function"
    assert tools[1]["function"]["name"] == "test2"
    assert tools[1]["function"]["parameters"] == {"type": "object", "properties": {"a": "b"}}

    assert tools[2]["type"] == "function"
    assert tools[2]["function"]["name"] == "test3"
    assert tools[2]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_openai_adapter_prepare_request_payload(adapter: OpenAIAdapter) -> None:
    messages = [
        {"role": "system", "content": "sys1"},
        {"role": "user", "content": "user"},
    ]
    tools = [{"type": "function", "function": {"name": "test"}}]
    payload = adapter._prepare_request_payload(messages, tools, 0.5, logit_biases={123: 10.0}, max_tokens=100)

    assert payload["model"] == "gpt-4o"
    assert payload["messages"][0]["role"] == "system"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100
    assert payload["tools"] == tools
    assert payload["logit_bias"] == {123: 10.0}
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_openai_adapter_stream_response(adapter: OpenAIAdapter) -> None:
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock(return_value=None)

    async def mock_aiter_lines() -> Any:
        lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " World"}}]}',
            'data: {"choices": [], "usage": {"prompt_tokens": 15, "completion_tokens": 10}}',
            "data: [DONE]",
            "data: ",  # should not be reached due to break
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
        patch.object(adapter.client, "stream", return_value=MockStreamContext()),
    ):
        chunks = []
        usages = []
        async for chunk, usage, _ in adapter.generate_stream(messages=[], tools=[], temperature=0.0):
            if chunk:
                chunks.append(chunk)
            if usage:
                usages.append(usage)

        assert chunks == ["Hello", " World"]
        assert usages == [{"input_tokens": 15, "output_tokens": 10}]


@pytest.mark.asyncio
async def test_openai_adapter_stream_response_decode_error_and_empty(adapter: OpenAIAdapter) -> None:
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock(return_value=None)

    async def mock_aiter_lines() -> Any:
        lines = [
            "data: {invalid json",
            "data: ",
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
        patch.object(adapter.client, "stream", return_value=MockStreamContext()),
    ):
        chunks = []
        async for chunk, _usage, _ in adapter.generate_stream(messages=[], tools=[], temperature=0.0):
            chunks.append(chunk)

        assert chunks == []


@pytest.mark.asyncio
async def test_openai_adapter_apply_peft_adapters(adapter: OpenAIAdapter) -> None:
    # Test early return
    await adapter.apply_peft_adapters([])

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock(return_value=None)

    contract = PeftAdapterContract(
        adapter_id="test_adapter",
        safetensors_hash="0" * 64,
        base_model_hash="1" * 64,
        adapter_rank=16,
        target_modules=["q_proj", "v_proj"],
        eviction_ttl_seconds=300,
    )

    with patch.object(adapter.client, "post", return_value=mock_response) as mock_post:
        await adapter.apply_peft_adapters([contract])
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.openai.com/v1/adapters/mount"
        assert kwargs["json"]["adapter_id"] == "test_adapter"
        assert kwargs["json"]["safetensors_uri"] == f"s3://coreason-cold-storage/{'0' * 64}.safetensors"
        assert kwargs["json"]["eviction_ttl_seconds"] == 300


@pytest.mark.asyncio
async def test_openai_adapter_apply_peft_adapters_fallback_ttl(adapter: OpenAIAdapter) -> None:
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock(return_value=None)

    contract = PeftAdapterContract(
        adapter_id="test_adapter2",
        safetensors_hash="2" * 64,
        base_model_hash="3" * 64,
        adapter_rank=8,
        target_modules=["q_proj"],
        eviction_ttl_seconds=None,
    )

    with patch.object(adapter.client, "post", return_value=mock_response) as mock_post:
        await adapter.apply_peft_adapters([contract])
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["eviction_ttl_seconds"] == 3600


@pytest.mark.asyncio
async def test_openai_adapter_structured_output() -> None:
    adapter = OpenAIAdapter("https://test.openai.com", "test-key")
    payload = adapter._prepare_request_payload(
        messages=[{"role": "user", "content": "test"}],
        tools=[],
        temperature=0.0,
        response_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
