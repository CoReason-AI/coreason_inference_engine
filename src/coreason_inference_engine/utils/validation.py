# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.


from coreason_manifest.spec.ontology import System2RemediationIntent, ManifestViolationReceipt
from pydantic import BaseModel, ValidationError


class ConstrainedDecodingPolicy(BaseModel):
    pass


def generate_correction_prompt(error: ValidationError, target_node_id: str, fault_id: str) -> System2RemediationIntent:
    """
    Pure functional adapter. Maps a raw Pythonic pydantic.ValidationError into a
    language-model-legible System2RemediationIntent without triggering runtime side effects.
    """
    violation_receipts: list[ManifestViolationReceipt] = []

    for err in error.errors():
        loc_path = "".join(f"/{item!s}" for item in err["loc"]) if err["loc"] else "/"
        err_type = str(err.get("type", "unknown"))
        
        if err_type == "missing":
            msg = "The required semantic boundary is completely missing. You must project this missing dimension to satisfy the StateContract."
        else:
            msg = str(err.get("msg", "Invalid structural payload."))
            
        violation_receipts.append(ManifestViolationReceipt(
            failing_pointer=loc_path,
            violation_type=err_type,
            diagnostic_message=msg,
        ))

    return System2RemediationIntent(
        fault_id=fault_id,
        target_node_id=target_node_id,
        violation_receipts=violation_receipts,
    )
