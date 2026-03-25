# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LocalFormatContract(BaseModel):
    model_config = ConfigDict(extra="ignore")
    require_think_tags: bool = False


class LocalEpistemicRewardModelPolicy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    format_contract: LocalFormatContract | None = None
    topological_scoring: Any | None = None


class LocalSemanticSlicingPolicy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    context_window_token_ceiling: int


class LocalCognitiveStateProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    semantic_slicing: LocalSemanticSlicingPolicy | None = None


class LocalSelfCorrectionPolicy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    max_loops: int = 3
    global_timeout_seconds: float | None = None


class LocalSystem1ReflexPolicy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    confidence_threshold: float
    allowed_passive_tools: list[str]


class LocalLogitSteganographyContract(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prf_seed_hash: str
    watermark_strength_delta: float


class LocalPeftAdapterContract(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # Using generic dict parsing as the engine offloads hot-swapping to LLM adapters
    # We will just pass dicts, or define fields if needed.


class LocalPermissions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    allowed_tools: list[str] = Field(default_factory=list)


class LocalAgentNodeProfile(BaseModel):
    description: str | None = None
    behavioral_directives: str | None = None
    constraints: list[str] | None = None
    active_inference_policy: Any | None = None
    architectural_intent: Any | None = None
    """
    Internal Pydantic model for the inference engine to map Node payloads.
    Silently drops unknown fields.
    """

    model_config = ConfigDict(extra="ignore")

    baseline_cognitive_state: LocalCognitiveStateProfile | None = None
    correction_policy: LocalSelfCorrectionPolicy | None = None
    grpo_reward_policy: LocalEpistemicRewardModelPolicy | None = None
    information_flow_policy: Any | None = None
    interventional_policy: Any | None = None
    logit_steganography: LocalLogitSteganographyContract | None = None
    peft_adapters: list[dict[str, Any]] | None = None
    permissions: LocalPermissions | None = None
    reflex_policy: LocalSystem1ReflexPolicy | None = None
    symbolic_handoff_policy: Any | None = None
    domain_extensions: Any | None = None


class LocalActionSpace(BaseModel):
    """
    Internal Pydantic model for the inference engine to map Action Space payloads.
    Silently drops unknown fields.
    """

    model_config = ConfigDict(extra="ignore")

    native_tools: list[dict[str, Any]] = Field(default_factory=list)


class LocalLedgerState(BaseModel):
    """
    Internal Pydantic model for the inference engine to map Ledger payloads.
    Silently drops unknown fields.
    """

    model_config = ConfigDict(extra="ignore")

    history: list[dict[str, Any]] = Field(default_factory=list)
    active_cascades: list[Any] | None = None
    active_rollbacks: list[Any] | None = None


class LocalStateMutationIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "state_mutation"
    op: str
    path: str
    value: Any | None = None
    from_: str | None = Field(None, alias="from")
    from_path: str | None = None


class LocalCognitiveStateProfileSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "cognitive_sync"


class LocalSystem2RemediationIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "system2_remediation"
    fault_id: str
    target_node_id: str
    failing_pointers: list[str]
    remediation_prompt: str


class LocalDocumentLayoutManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "document_layout"


class LocalInformationalIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "informational"
    message: str
    timeout_action: str | None = None


class LocalLogEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "log_event"
    level: str
    message: str
    context_profile: dict[str, Any] | None = None


LocalAnyIntent = LocalInformationalIntent | LocalLogEvent


class LocalToolInvocationEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "tool_invocation"
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    event_id: str | None = None
    timestamp: float | None = None


class LocalTokenBurnReceipt(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["token_burn_receipt"] = "token_burn_receipt"
    event_id: str
    timestamp: float
    tool_invocation_id: str
    input_tokens: int
    output_tokens: int
    burn_magnitude: int


class LocalLatentScratchpadReceipt(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["latent_scratchpad_receipt"] = "latent_scratchpad_receipt"
    trace_id: str
    explored_branches: list[dict[str, Any]]
    discarded_branches: list[dict[str, Any]]
    resolution_branch_id: str
    total_latent_tokens: int


class LocalSystemFaultEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["system_fault"] = "system_fault"
    event_id: str
    timestamp: float


class LocalCognitiveRewardReceipt(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["cognitive_reward_receipt"] = "cognitive_reward_receipt"
    event_id: str
    timestamp: float
    source_generation_id: str
    extracted_axioms: list[Any]
    calculated_r_path: float
    total_advantage_score: float
