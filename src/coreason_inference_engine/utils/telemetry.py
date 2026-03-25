# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import asyncio
import contextlib
import re
from typing import Any

type TelemetryEvent = Any


class TelemetryEmitter:
    """Handles out-of-band emission of telemetry models to maintain the Hollow Data Plane."""

    def __init__(self, queue: asyncio.Queue[TelemetryEvent] | None = None) -> None:
        self.queue = queue or asyncio.Queue()

    async def emit(self, event: TelemetryEvent) -> None:
        """Asynchronously queues the telemetry event for external collection."""
        await self.queue.put(event)

    def redact_pii(self, payload: str, policy: Any | None) -> str:
        """
        Passes the payload string through the Any rules
        to redact sensitive geometric coordinates or secure identifiers.
        """
        if not policy or not policy.active or not policy.rules:
            return payload

        redacted = payload
        for rule in policy.rules:
            if rule.action != "redact":
                # We only handle string replacement redaction here
                continue

            replacement = rule.replacement_token or "[REDACTED]"

            if rule.target_regex_pattern:
                with contextlib.suppress(re.error):
                    redacted = re.sub(rule.target_regex_pattern, replacement, redacted)

            if rule.target_pattern:
                redacted = redacted.replace(rule.target_pattern, replacement)

        return redacted
