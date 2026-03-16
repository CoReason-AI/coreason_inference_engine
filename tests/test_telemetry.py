# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import asyncio

import pytest
from coreason_manifest.spec.ontology import (
    InformationClassificationProfile,
    InformationFlowPolicy,
    LogEvent,
    RedactionPolicy,
)

from coreason_inference_engine.utils.telemetry import TelemetryEmitter, TelemetryEvent


@pytest.mark.asyncio
async def test_telemetry_emitter_emit() -> None:
    queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue()
    emitter = TelemetryEmitter(queue)
    event = LogEvent(timestamp=0, level="INFO", message="test", context_profile={})
    await emitter.emit(event)
    out_event = await queue.get()
    assert out_event == event


def test_telemetry_emitter_redact_pii_no_policy() -> None:
    emitter = TelemetryEmitter()
    assert emitter.redact_pii("test email@example.com", None) == "test email@example.com"


def test_telemetry_emitter_redact_pii_inactive_policy() -> None:
    emitter = TelemetryEmitter()
    policy = InformationFlowPolicy(policy_id="p1", active=False, rules=[])
    assert emitter.redact_pii("test email@example.com", policy) == "test email@example.com"


def test_telemetry_emitter_redact_pii_regex() -> None:
    emitter = TelemetryEmitter()
    rule = RedactionPolicy(
        rule_id="r1",
        classification=InformationClassificationProfile.CONFIDENTIAL,
        target_regex_pattern=r"[\w\.-]+@[\w\.-]+",
        target_pattern="",
        action="redact",
        replacement_token="[EMAIL]",  # noqa: S106,
    )
    policy = InformationFlowPolicy(policy_id="p1", active=True, rules=[rule])
    assert emitter.redact_pii("test email@example.com", policy) == "test [EMAIL]"


def test_telemetry_emitter_redact_pii_string() -> None:
    emitter = TelemetryEmitter()
    rule = RedactionPolicy(
        rule_id="r1",
        classification=InformationClassificationProfile.RESTRICTED,
        target_pattern="my_secret_key",
        target_regex_pattern="",
        action="redact",
        replacement_token="***",  # noqa: S106,
    )
    policy = InformationFlowPolicy(policy_id="p1", active=True, rules=[rule])
    assert emitter.redact_pii("Here is my_secret_key data", policy) == "Here is *** data"


def test_telemetry_emitter_redact_pii_invalid_regex() -> None:
    emitter = TelemetryEmitter()
    rule = RedactionPolicy(
        rule_id="r1",
        classification=InformationClassificationProfile.CONFIDENTIAL,
        target_regex_pattern=r"[(bad regex",
        target_pattern="",
        action="redact",
        replacement_token="***",  # noqa: S106,
    )
    policy = InformationFlowPolicy(policy_id="p1", active=True, rules=[rule])
    # Should ignore invalid regex and not crash
    assert emitter.redact_pii("test", policy) == "test"


def test_telemetry_emitter_redact_pii_ignore_non_redact_action() -> None:
    emitter = TelemetryEmitter()
    rule = RedactionPolicy(
        rule_id="r1",
        classification=InformationClassificationProfile.RESTRICTED,
        target_pattern="secret",
        target_regex_pattern="",
        action="hash",
    )
    policy = InformationFlowPolicy(policy_id="p1", active=True, rules=[rule])
    # Since action != redact, it should not replace string
    assert emitter.redact_pii("this is a secret", policy) == "this is a secret"


def test_telemetry_emitter_redact_pii_missing_replacement() -> None:
    emitter = TelemetryEmitter()
    rule = RedactionPolicy(
        rule_id="r1",
        classification=InformationClassificationProfile.CONFIDENTIAL,
        target_pattern="secret",
        target_regex_pattern="",
        action="redact",
        replacement_token="",
    )
    # The requirement is to replace with [REDACTED] if token is None or empty. Let's fix TelemetryEmitter
    # or just rely on what it does.
    policy = InformationFlowPolicy(policy_id="p1", active=True, rules=[rule])
    assert emitter.redact_pii("this is a secret", policy) == "this is a [REDACTED]"
