# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

from typing import Any

from coreason_manifest.spec.ontology import System2RemediationIntent
from pydantic import ValidationError


def generate_correction_prompt(error: ValidationError, target_node_id: str, fault_id: str) -> System2RemediationIntent:
    """
    Pure functional adapter. Maps a raw Pythonic pydantic.ValidationError into a
    language-model-legible System2RemediationIntent without triggering runtime side effects.
    """
    failing_pointers: list[str] = []
    error_messages: list[str] = []
    for err in error.errors():
        loc_path = "".join(f"/{item!s}" for item in err["loc"]) if err["loc"] else "/"
        failing_pointers.append(loc_path)
        err_type = err["type"]
        if err_type == "missing":
            error_messages.append(
                f"The required semantic boundary at '{loc_path}' is completely missing. You must project this missing dimension to satisfy the StateContract."  # noqa: E501
            )
        else:
            msg = err.get("msg", "Invalid structural payload.")
            error_messages.append(f"A structural boundary violation occurred at '{loc_path}': {msg}")
    failing_pointers = list(set(failing_pointers))
    remediation_prompt = (
        "CRITICAL CONTRACT BREACH: Your generated state representation violates the formal ontological boundaries of the Shared Kernel. Review the following strict topological failures and correct your JSON projection:\n"  # noqa: E501
        + "\n".join(f"- {msg}" for msg in error_messages)
    )
    return System2RemediationIntent(
        fault_id=fault_id,
        target_node_id=target_node_id,
        failing_pointers=failing_pointers,
        remediation_prompt=remediation_prompt,
    )


def validate_payload(step: str, payload_bytes: bytes) -> Any:
    """Validate a payload against the designated ontology step model.

    Raises:
        ValueError: If the `step` parameter is unknown.
        ValidationError: If `payload_bytes` does not conform to the schema.
    """
    from coreason_manifest.spec.ontology import (
        CognitiveStateProfile,
        DocumentLayoutManifest,
        StateMutationIntent,
        System2RemediationIntent,
    )

    schema_registry: dict[str, Any] = {
        "step8_vision": DocumentLayoutManifest,
        "state_differential": StateMutationIntent,
        "cognitive_sync": CognitiveStateProfile,
        "system2_remediation": System2RemediationIntent,
    }

    target_schema = schema_registry.get(step)
    if not target_schema:
        raise ValueError(f"FATAL: Unknown step '{step}'. Valid steps: {list(schema_registry.keys())}")

    return target_schema.model_validate_json(payload_bytes)
