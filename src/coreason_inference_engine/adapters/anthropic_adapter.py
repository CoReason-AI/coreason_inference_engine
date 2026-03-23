# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import json
from collections.abc import AsyncGenerator
from typing import Any

import tiktoken
from coreason_manifest.spec.ontology import ComputeRateContract

from coreason_inference_engine.adapters.http_adapter import BaseHttpAdapter


class AnthropicAdapter(BaseHttpAdapter):
    """
    Anthropic-specific adapter extending BaseHttpAdapter.
    Enforces strict user/assistant alternation and parses Anthropic's custom SSE format.
    """

    def __init__(
        self, api_url: str, api_key: str | None, model_name: str | None = None, max_connections: int = 1000
    ) -> None:
        super().__init__(api_url, api_key or "", max_connections)
        self.model_name = model_name or "claude-3-5-sonnet-20241022"
        self.rate_card = ComputeRateContract(
            cost_per_million_input_tokens=3,
            cost_per_million_output_tokens=15,
            magnitude_unit="USD",
        )
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str | bytes) -> int:
        """Synchronously calculates token mass. Uses tiktoken cl100k_base for fallback accuracy."""
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return len(self._encoding.encode(text, disallowed_special=()))

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translates JSON Schema dictionaries into Anthropic's native tool array format."""
        # Anthropic tool format expects name, description, and input_schema
        anthropic_tools = []
        for schema in schemas:
            if "function" in schema:
                # Convert OpenAI format to Anthropic
                func = schema["function"]
                anthropic_tools.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            elif "name" in schema and "input_schema" in schema:
                anthropic_tools.append(schema)
            else:
                anthropic_tools.append(
                    {
                        "name": schema.get("name", "unknown_tool"),
                        "description": schema.get("description", ""),
                        "input_schema": schema.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
        return anthropic_tools

    def _apply_strict_alternation(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Anthropic strictly rejects adjacent user messages.
        Collapses consecutive user messages into a single user turn wrapped in <observation> tags.
        """
        if not messages:
            return []

        collapsed_messages: list[dict[str, Any]] = []
        current_role = None
        current_content: list[str] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                if current_role == "user":
                    if len(current_content) == 1 and not current_content[0].startswith("<observation>"):
                        # Wrap the first one as well if it's being collapsed
                        current_content[0] = f"<observation>\n{current_content[0]}\n</observation>"
                    current_content.append(f"<observation>\n{content}\n</observation>")
                else:
                    if current_role is not None:
                        collapsed_messages.append({"role": current_role, "content": "\n".join(current_content)})
                    current_role = "user"
                    current_content = [content]
            elif role == "assistant":
                if current_role == "assistant":
                    current_content.append(content)
                else:
                    if current_role is not None:
                        collapsed_messages.append({"role": current_role, "content": "\n".join(current_content)})
                    current_role = "assistant"
                    current_content = [content]
            else:
                pass

        if current_role is not None:
            collapsed_messages.append({"role": current_role, "content": "\n".join(current_content)})

        return collapsed_messages

    def _prepare_request_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        logit_biases: dict[int, float] | None = None,  # noqa: ARG002
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepares the JSON payload for Anthropic."""
        system_prompt = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                if system_prompt:
                    system_prompt += "\n" + msg.get("content", "")
                else:
                    system_prompt = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        strict_messages = self._apply_strict_alternation(filtered_messages)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": strict_messages,
            "temperature": temperature,
            "stream": True,
            "max_tokens": max_tokens if max_tokens is not None else 8192,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = tools

        if response_schema:
            # Enforce output by appending as a tool and forcing selection
            forced_tool = {
                "name": "yield_target_schema",
                "description": "Yield the final structured output",
                "input_schema": response_schema,
            }
            if "tools" not in payload:
                payload["tools"] = []
            payload["tools"].append(forced_tool)
            payload["tool_choice"] = {"type": "tool", "name": "yield_target_schema"}

        return payload

    async def _stream_response(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],  # noqa: ARG002
    ) -> AsyncGenerator[tuple[str, dict[str, int], dict[str, Any] | None]]:
        anthropic_headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

        async with self.client.stream("POST", self.api_url, json=payload, headers=anthropic_headers) as response:
            response.raise_for_status()

            current_event_type = None
            input_tokens = 0
            output_tokens = 0

            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    current_event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_str = line[6:].strip()
                    if not data_str:
                        continue

                    try:
                        data = json.loads(data_str)
                        delta = ""
                        usage = {}

                        if current_event_type == "message_start":
                            if "message" in data and "usage" in data["message"]:
                                input_tokens = data["message"]["usage"].get("input_tokens", 0)
                                usage["input_tokens"] = input_tokens

                        elif current_event_type == "content_block_delta":
                            if "delta" in data and data["delta"].get("type") == "text_delta":
                                delta = data["delta"].get("text", "")

                        elif current_event_type == "message_delta" and "usage" in data:
                            output_tokens = data["usage"].get("output_tokens", 0)
                            usage["output_tokens"] = output_tokens

                        if delta or usage:
                            if not usage:
                                usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
                            else:
                                if "input_tokens" not in usage:
                                    usage["input_tokens"] = input_tokens
                                if "output_tokens" not in usage:
                                    usage["output_tokens"] = output_tokens

                            yield delta, usage, None

                    except json.JSONDecodeError:
                        pass
