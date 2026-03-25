# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.


from coreason_manifest.spec.ontology import System2RemediationIntent
from pydantic import BaseModel


class ConstrainedDecodingPolicy(BaseModel):
    pass


def generate_correction_prompt(error: Exception, target_node_id: str, fault_id: str) -> System2RemediationIntent:
    """
    Pure functional adapter. Maps a raw Pythonic pydantic.ValidationError into a
    language-model-legible System2RemediationIntent without triggering runtime side effects.
    """
    failing_pointers = []
    remediation_prompts = []

    if not hasattr(error, "errors"):
        return System2RemediationIntent(
            fault_id=fault_id, target_node_id=target_node_id, failing_pointers=["/"], remediation_prompt=str(error)
        )
    for err in error.errors():
        loc_path = "".join(f"/{item!s}" for item in err["loc"]) if err["loc"] else "/"
        err_type = str(err.get("type", "unknown"))

        if err_type == "missing":
            msg = (
                "The required semantic boundary is completely missing. "
                "You must project this missing dimension to satisfy the StateContract."
            )
        else:
            msg = str(err.get("msg", "Invalid structural payload."))

        failing_pointers.append(loc_path)
        remediation_prompts.append(msg)

    combined_prompt = " ".join(remediation_prompts) if remediation_prompts else "Unknown schema validation error."

    return System2RemediationIntent(
        fault_id=fault_id,
        target_node_id=target_node_id,
        failing_pointers=failing_pointers or ["/"],
        remediation_prompt=combined_prompt,
    )
