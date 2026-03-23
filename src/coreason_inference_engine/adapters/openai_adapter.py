# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import json
import urllib.parse
from collections.abc import AsyncGenerator
from typing import Any

import tiktoken
from coreason_manifest.spec.ontology import ComputeRateContract, LatentScratchpadReceipt, PeftAdapterContract

from coreason_inference_engine.adapters.http_adapter import BaseHttpAdapter


class OpenAIAdapter(BaseHttpAdapter):
    """
    OpenAI-specific adapter extending BaseHttpAdapter.
    Utilizes tiktoken for exact token counting and parses OpenAI's SSE format,
    including capturing stream usage options.
    """

    def __init__(
        self, api_url: str, api_key: str | None, model_name: str | None = None, max_connections: int = 1000
    ) -> None:
        super().__init__(api_url, api_key or "", max_connections)
        self.model_name = model_name or "gpt-4o"
        self.rate_card = ComputeRateContract(
            cost_per_million_input_tokens=5,
            cost_per_million_output_tokens=15,
            magnitude_unit="USD",
        )
        # Pre-load the encoding for the model
        try:
            self._encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback to a standard encoding if model is unknown
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str | bytes) -> int:
        """Synchronously calculates token mass using tiktoken. Fallbacks to safe decoding if bytes."""
        if isinstance(text, bytes):
            # Safe decoding using replace to prevent tokenizer panic on severed chunks
            text = text.decode("utf-8", errors="replace")
        return len(self._encoding.encode(text, disallowed_special=()))

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translates JSON Schema dictionaries into OpenAI's native tool array format."""
        openai_tools = []
        for schema in schemas:
            # OpenAI expects {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
            if "type" in schema and schema["type"] == "function" and "function" in schema:
                # Already in OpenAI format
                openai_tools.append(schema)
            elif ("name" in schema or "tool_name" in schema) and ("parameters" in schema or "input_schema" in schema):
                # We need to wrap it
                parameters = schema.get("parameters") or schema.get(
                    "input_schema", {"type": "object", "properties": {}}
                )
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": schema.get("name") or schema.get("tool_name", "unknown_tool"),
                            "description": schema.get("description", ""),
                            "parameters": parameters,
                        },
                    }
                )
            else:
                # Fallback, wrap the whole schema in parameters if not recognizable, or try to extract
                name = schema.get("name") or schema.get("tool_name", "unknown_tool")
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": schema.get("description", ""),
                            "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                        },
                    }
                )
        return openai_tools

    async def apply_peft_adapters(self, adapters: list[PeftAdapterContract]) -> None:
        if not adapters:
            return

        parsed = urllib.parse.urlparse(self.api_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        mount_url = f"{base_url}/v1/adapters/mount"

        for adapter in adapters:
            payload = {
                "adapter_id": adapter.adapter_id,
                "safetensors_uri": f"s3://coreason-cold-storage/{adapter.safetensors_hash}.safetensors",
                "base_model_hash": adapter.base_model_hash,
                "target_modules": adapter.target_modules,
                "adapter_rank": adapter.adapter_rank,
                "eviction_ttl_seconds": adapter.eviction_ttl_seconds or 3600,
            }

            response = await self.client.post(mount_url, json=payload)
            response.raise_for_status()

    def _prepare_request_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        logit_biases: dict[int, float] | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepares the JSON payload for OpenAI."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            payload["tools"] = tools
        if logit_biases:
            payload["logit_bias"] = logit_biases

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "target_schema", "schema": response_schema, "strict": True},
            }

        return payload

    async def _stream_response(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncGenerator[tuple[str, dict[str, int], LatentScratchpadReceipt | None]]:
        # Accumulators for native tool calls
        tool_name = ""
        tool_args_str = ""

        async with self.client.stream("POST", self.api_url, json=payload, headers=headers) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    if not data_str:
                        continue

                    try:
                        data = json.loads(data_str)
                        delta = ""
                        usage = {}

                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            if "delta" in choice:
                                d = choice["delta"]

                                # Extract standard text content
                                if d.get("content"):
                                    delta = d["content"]

                                # Extract and accumulate native tool calls
                                if d.get("tool_calls"):
                                    for tc in d["tool_calls"]:
                                        if "function" in tc:
                                            func = tc["function"]
                                            if func.get("name"):
                                                tool_name += func["name"]
                                            if func.get("arguments"):
                                                tool_args_str += func["arguments"]

                        if data.get("usage"):
                            usage = {
                                "input_tokens": data["usage"].get("prompt_tokens", 0),
                                "output_tokens": data["usage"].get("completion_tokens", 0),
                            }

                        if delta or usage:
                            yield delta, usage, None

                    except json.JSONDecodeError:
                        pass

        # After the stream completes, check if a tool was called natively.
        # If so, project it back as a strictly valid CoReason JSON ontology event.
        if tool_name:
            import time
            import uuid

            try:
                args = json.loads(tool_args_str) if tool_args_str else {}
            except json.JSONDecodeError:
                args = {}

            tool_json = {
                "type": "tool_invocation",
                "tool_name": tool_name,
                "parameters": args,  # Properly mapped to 'parameters' to pass validation
                "event_id": f"evt_{uuid.uuid4().hex[:8]}",
                "timestamp": time.time(),
            }
            yield json.dumps(tool_json), {}, None
