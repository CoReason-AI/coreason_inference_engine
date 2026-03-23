# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

from typing import Any

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
    """
    Internal Pydantic model for the inference engine to map Node payloads.
    Silently drops unknown fields.
    """

    model_config = ConfigDict(extra="ignore")
    description: str = ""

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
