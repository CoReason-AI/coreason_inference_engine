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

import httpx
from coreason_manifest.spec.ontology import ComputeRateContract, LatentScratchpadReceipt, PeftAdapterContract

from coreason_inference_engine.interfaces import LLMAdapterProtocol
from coreason_inference_engine.utils.network import validate_url_for_ssrf


class BaseHttpAdapter(LLMAdapterProtocol):
    """
    Base HTTP Adapter implementing LLMAdapterProtocol with a singleton httpx.AsyncClient.
    Enforces SSRF validation before dispatching requests.
    """

    def __init__(self, api_url: str, api_key: str, max_connections: int = 1000) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.rate_card: ComputeRateContract | None = None
        # Singleton Transport & SSRF Firewall: explicitly bounded limits
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        self.client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(60.0))

    async def close(self) -> None:
        """Closes the underlying httpx.AsyncClient."""
        await self.client.aclose()

    def count_tokens(self, text: str) -> int:
        """Synchronously calculates the exact token mass of a string. To be overridden by subclasses."""
        # By default, a naive fallback if not overridden
        return len(text)

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translates JSON Schema dictionaries into the provider's native tool array format."""
        return schemas

    async def apply_peft_adapters(self, adapters: list[PeftAdapterContract]) -> None:
        """Hot-swaps declarative LoRA/PEFT weights into VRAM on compatible local backends."""
        if not adapters:
            return

        management_url = self.api_url.replace("/chat/completions", "/adapters/load")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for adapter in adapters:
            await validate_url_for_ssrf(management_url)

            payload = {
                "adapter_name": adapter.adapter_id,
                "remote_path": getattr(adapter, "huggingface_repo_id", adapter.adapter_id),
            }

            try:
                response = await self.client.post(management_url, json=payload, headers=headers)
                response.raise_for_status()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                raise RuntimeError(f"Hardware PEFT swap failed: {e}") from e

    def _prepare_request_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        logit_biases: dict[int, float] | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Prepares the JSON payload for the provider. To be overridden by subclasses if needed."""
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if logit_biases:
            payload["logit_bias"] = logit_biases
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        logit_biases: dict[int, float] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> AsyncGenerator[tuple[str, dict[str, int], LatentScratchpadReceipt | None]]:
        """
        Yields chunked string deltas and an optional usage dictionary.
        The final yield MUST contain the {"input_tokens": x, "output_tokens": y} mapping.
        """
        # SSRF Firewall: run target URI through ipaddress bogon filter
        await validate_url_for_ssrf(self.api_url)

        payload = self._prepare_request_payload(messages, tools, temperature, logit_biases, max_tokens, response_schema)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        # Send POST request, yield chunks
        # Need to return async generator
        async for chunk, usage, receipt in self._stream_response(payload, headers):
            yield chunk, usage, receipt

    async def _stream_response(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncGenerator[tuple[str, dict[str, int], LatentScratchpadReceipt | None]]:
        # We must use client.stream
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
                        # Naive parsing (assuming OpenAI-like format)
                        # Subclasses should override generate_stream or _stream_response for specific parsing
                        delta = ""
                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if content:
                                    delta = content

                        usage = {}
                        if data.get("usage"):
                            usage = {
                                "input_tokens": data["usage"].get("prompt_tokens", 0),
                                "output_tokens": data["usage"].get("completion_tokens", 0),
                            }

                        yield delta, usage, None
                    except json.JSONDecodeError:
                        pass
