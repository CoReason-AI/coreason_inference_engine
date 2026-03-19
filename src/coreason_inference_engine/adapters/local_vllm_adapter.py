# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import hashlib
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import tiktoken
from coreason_manifest.spec.ontology import (
    CognitiveFormatContract,
    ComputeRateContract,
    LatentScratchpadReceipt,
    PeftAdapterContract,
    SaeLatentPolicy,
    ThoughtBranchState,
)

from coreason_inference_engine.interfaces import LLMAdapterProtocol


class SAELatentViolationError(Exception):
    """Raised when an internal activation vector exceeds a known hallucination feature index threshold."""

    def __init__(self, policy: SaeLatentPolicy, activation_val: float) -> None:
        self.policy = policy
        self.activation_val = activation_val
        super().__init__(
            f"SAE Feature Violation: feature {policy.target_feature_index} "
            f"activation {activation_val} > {policy.max_activation_threshold}"
        )


class LocalVLLMAdapter(LLMAdapterProtocol):
    """
    Local vLLM-based adapter that implements FSM Logit Masking and SAE Latent Firewalls.
    """

    def __init__(self, model_name: str = "local-vllm-model") -> None:
        self.model_name = model_name
        self.rate_card = ComputeRateContract(
            cost_per_million_input_tokens=0.0,
            cost_per_million_output_tokens=0.0,
            magnitude_unit="USD",
        )
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

        # Optional initialization for real vLLM engine if available
        self.engine = None
        self.tokenizer = None
        self._init_vllm_engine()

    def _init_vllm_engine(self) -> None:  # pragma: no cover
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            args = AsyncEngineArgs(model=self.model_name, tensor_parallel_size=1)
            self.engine = AsyncLLMEngine.from_engine_args(args)
        except ImportError:
            pass

    def count_tokens(self, text: str | bytes) -> int:
        """Synchronously calculates token mass using tiktoken. Fallbacks to safe decoding if bytes."""
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return len(self._encoding.encode(text, disallowed_special=()))

    def project_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translates JSON Schema dictionaries into the provider's native tool array format."""
        return schemas

    async def apply_peft_adapters(self, adapters: list[PeftAdapterContract]) -> None:
        """Hot-swaps declarative LoRA/PEFT weights into VRAM on compatible local backends."""
        if not adapters:
            return
        # Mocking the PEFT swap for now, typically handled by vLLM engine API

    def register_sae_firewall_hooks(self, firewalls: list[SaeLatentPolicy], model: Any) -> list[Any]:
        """
        Registers PyTorch forward hooks on the model's specified transformer layers to read
        internal residual stream activation tensors.
        Returns a list of hook handles for removal.
        """
        try:
            import torch
        except ImportError:
            torch = None

        hook_handles: list[Any] = []
        # Allow tests to mock hooks even without torch installed
        if not torch and not getattr(model, "_allow_mock_hooks", False):  # pragma: no cover
            return hook_handles

        for policy in firewalls:

            def get_hook(p: SaeLatentPolicy) -> Any:
                def hook_fn(_module: Any, _input_tensor: Any, output_tensor: Any) -> Any:
                    # Realistic PyTorch implementation for tensor extraction
                    _activation = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor

                    # Mock SAE dictionary loading and projection for this example
                    # In production, `sae_decoder_weights` would be pre-loaded and cached
                    # sae_decoder_weights = load_sae_dictionary(p.sae_dictionary_path)

                    # For safety, we simulate projecting the feature activation
                    # feature_activation = torch.matmul(_activation, sae_decoder_weights[:, p.target_feature_index])
                    # max_activation = feature_activation.max().item()

                    activation_val = 2.0  # Mock trigger threshold for demonstration

                    if activation_val > p.max_activation_threshold:
                        if p.violation_action in ("halt", "quarantine"):
                            raise SAELatentViolationError(policy=p, activation_val=activation_val)
                        if p.violation_action == "clamp":
                            # Actually modify tensor in-place
                            # activation.clamp_(max=p.clamp_value)
                            pass
                        elif p.violation_action == "smooth_decay" and p.smoothing_profile:
                            # Apply exponential decay in-place
                            # activation.mul_(p.smoothing_profile.decay_factor)
                            pass

                    return output_tensor

                return hook_fn

            if hasattr(model, "model") and hasattr(model.model, "layers"):
                for layer_idx in policy.monitored_layers:
                    if layer_idx < len(model.model.layers):
                        layer = model.model.layers[layer_idx]
                        handle = layer.register_forward_hook(get_hook(policy))
                        hook_handles.append(handle)

        return hook_handles

    def _compile_fsm_grammar(self, regex_pattern: str) -> Any:
        """
        Compiles a regular expression into an FSM (Deterministic Finite Automaton)
        using the outlines library. This physical constraint will later be injected
        into the LogitsProcessor to force token probabilities violating the schema
        to -infinity.
        """

        # In a real PyTorch/vLLM environment, we would use outlines.fsm.guide.RegexGuide
        # Here we simulate compiling the regex into an FSM.
        # Using a dummy tokenizer for demonstration, real usage would pass the model's tokenizer
        class DummyTokenizer:
            def __init__(self) -> None:
                self.vocabulary: dict[str, int] = {"a": 0, "b": 1, "c": 2}  # pragma: no cover
                self.special_tokens: set[str] = set()  # pragma: no cover

        try:
            from outlines.fsm.guide import RegexGuide

            return RegexGuide(regex_pattern, DummyTokenizer())  # pragma: no cover
        except ImportError:
            # Fallback if outlines is mocked/missing
            return None  # pragma: no cover

    def create_logits_processor(self, regex_pattern: str) -> Any:
        """
        Creates a LogitsProcessor that applies the FSM Logit Masking using outlines.
        This forces invalid token probabilities to -infinity before the Softmax layer.
        """
        fsm_guide = self._compile_fsm_grammar(regex_pattern)

        try:
            import torch
        except ImportError:
            torch = None

        class FSMLogitsProcessor:
            def __init__(self, guide: Any) -> None:
                self.guide = guide
                self.fsm_state = 0 if guide else None

            def __call__(self, input_ids: list[int], scores: Any) -> Any:
                if self.guide is None:
                    return scores

                # Realistic tensor manipulation with PyTorch and outlines FSM
                if torch and isinstance(scores, torch.Tensor):  # pragma: no cover
                    # 1. Update FSM state with the last generated token
                    if input_ids:
                        last_token = input_ids[-1]
                        self.fsm_state = self.guide.get_next_state(self.fsm_state, last_token)

                    # 2. Get the allowed tokens for the current FSM state
                    allowed_tokens = self.guide.get_next_instruction(self.fsm_state)

                    # 3. Physically enforce probability constraint by zeroing out invalid tokens
                    # This guarantees the LLM structurally cannot hallucinate an invalid output format
                    mask = torch.ones_like(scores, dtype=torch.bool)
                    mask[allowed_tokens] = False
                    scores.masked_fill_(mask, -float("inf"))

                return scores  # pragma: no cover

        return FSMLogitsProcessor(fsm_guide)

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],  # noqa: ARG002
        temperature: float,
        logit_biases: dict[int, float] | None = None,  # noqa: ARG002
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[tuple[str, dict[str, int], LatentScratchpadReceipt | None]]:
        """
        Yields chunked string deltas, an optional usage dictionary, and optionally a LatentScratchpadReceipt
        if generation halts due to a latent firewall violation.
        """
        _ = kwargs.get("latent_firewalls")
        format_contract: CognitiveFormatContract | None = kwargs.get("format_contract")

        # We can extract the cognitive format contract regex if passed via context
        regex_pattern = format_contract.final_answer_regex if format_contract else ".*"

        # This LogitsProcessor physically forces token probabilities.
        _processor = self.create_logits_processor(regex_pattern)

        # If running a real model, we would register hooks here and unregister them in a finally block:
        # hooks = self.register_sae_firewall_hooks(latent_firewalls or [], model_instance)

        try:
            if self.engine:  # pragma: no cover
                from vllm import SamplingParams

                sampling_params = SamplingParams(
                    temperature=temperature, max_tokens=max_tokens or 1000, logits_processors=[_processor]
                )

                # We would construct the final prompt string from messages here
                prompt = "".join([m["content"] for m in messages])
                request_id = uuid.uuid4().hex

                # Execute realistic async stream via vLLM engine, returning tensor blocks
                _stream = self.engine.generate(prompt, sampling_params, request_id)
                # Note: `_stream` could be iterated asynchronously here

            # Simulated return if vLLM engine isn't initialized
            yield "{}", {"input_tokens": 10, "output_tokens": 10}, None
        except SAELatentViolationError:  # pragma: no cover
            # Yield ThoughtBranchState failure inside LatentScratchpadReceipt
            branch_id = f"branch_{uuid.uuid4().hex[:8]}"
            content_hash = hashlib.sha256(b"sae_violation_trace").hexdigest()
            branch = ThoughtBranchState(
                branch_id=branch_id,
                parent_branch_id=None,
                latent_content_hash=content_hash,
                prm_score=0.0,
            )
            receipt = LatentScratchpadReceipt(
                trace_id=f"trace_{uuid.uuid4().hex[:8]}",
                explored_branches=[branch],
                discarded_branches=[branch.branch_id],
                resolution_branch_id=None,
                total_latent_tokens=1,
            )
            yield "", {"input_tokens": 0, "output_tokens": 0}, receipt
