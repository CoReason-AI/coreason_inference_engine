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

from coreason_inference_engine.adapters.anthropic_adapter import AnthropicAdapter


@pytest.fixture
def adapter() -> AnthropicAdapter:
    return AnthropicAdapter(api_url="https://api.anthropic.com/v1/messages", api_key="test_key")


def test_anthropic_adapter_count_tokens(adapter: AnthropicAdapter) -> None:
    assert adapter.count_tokens("hello world") == 2
    assert adapter.count_tokens(b"hello world") == 2


def test_anthropic_adapter_project_tools(adapter: AnthropicAdapter) -> None:
    schemas: list[dict[str, Any]] = [
        {"type": "function", "function": {"name": "test", "description": "desc", "parameters": {"type": "object"}}},
        {"name": "test2", "input_schema": {"type": "object"}},
        {"name": "test3", "description": "desc3"},
    ]
    tools = adapter.project_tools(schemas)
    assert len(tools) == 3
    assert tools[0]["name"] == "test"
    assert tools[0]["input_schema"] == {"type": "object"}
    assert tools[1]["name"] == "test2"
    assert tools[2]["name"] == "test3"


def test_anthropic_adapter_strict_alternation(adapter: AnthropicAdapter) -> None:
    messages = [
        {"role": "user", "content": "msg1"},
        {"role": "user", "content": "msg2"},
        {"role": "assistant", "content": "ans1"},
        {"role": "assistant", "content": "ans2"},
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "msg3"},
    ]
    strict = adapter._apply_strict_alternation(messages)
    assert len(strict) == 3
    assert strict[0]["role"] == "user"
    assert strict[0]["content"] == "<observation>\nmsg1\n</observation>\n<observation>\nmsg2\n</observation>"
    assert strict[1]["role"] == "assistant"
    assert strict[1]["content"] == "ans1\nans2"
    assert strict[2]["role"] == "user"
    assert strict[2]["content"] == "msg3"


def test_anthropic_adapter_strict_alternation_empty(adapter: AnthropicAdapter) -> None:
    assert adapter._apply_strict_alternation([]) == []


def test_anthropic_adapter_prepare_request_payload(adapter: AnthropicAdapter) -> None:
    messages = [
        {"role": "system", "content": "sys1"},
        {"role": "system", "content": "sys2"},
        {"role": "user", "content": "user"},
    ]
    tools = [{"name": "test"}]
    payload = adapter._prepare_request_payload(messages, tools, 0.5)

    assert payload["system"] == "sys1\nsys2"
    assert payload["messages"][0]["role"] == "user"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 8192
    assert payload["tools"] == tools


@pytest.mark.asyncio
async def test_anthropic_adapter_stream_response(adapter: AnthropicAdapter) -> None:
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock(return_value=None)

    async def mock_aiter_lines() -> Any:
        lines = [
            "event: message_start",
            'data: {"type": "message_start", "message": {"usage": {"input_tokens": 15}}}',
            "event: content_block_delta",
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}',
            "event: message_delta",
            'data: {"type": "message_delta", "usage": {"output_tokens": 10}}',
            "event: message_stop",
            "data: {}",
            "data: ",  # empty data
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
            chunks.append(chunk)
            usages.append(usage)

        assert "Hello" in chunks
        assert usages[-1] == {"input_tokens": 15, "output_tokens": 10}


@pytest.mark.asyncio
async def test_anthropic_adapter_stream_response_decode_error(adapter: AnthropicAdapter) -> None:
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock(return_value=None)

    async def mock_aiter_lines() -> Any:
        lines = [
            "event: message_start",
            "data: {invalid json",
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
