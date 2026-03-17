# API Reference

The core application boundaries defined via strict Python Protocols:

## Interfaces

* `InferenceEngineProtocol`: Dictates the asynchronous entry point `generate_intent(...)`.
* `LLMAdapterProtocol`: Dictates provider-agnostic boundaries.
  - `generate_stream(...)`: Streams `AnyIntent` string deltas and chunks.
  - `count_tokens(...)`: Safely measures deterministic context tokens and severed stream sizes.
  - `project_tools(...)`: Resolves JSON-RPC schemas into proprietary formats.
  - `apply_peft_adapters(...)`: Injects LoRA logic dynamically.

For exact Python definitions, see `src/coreason_inference_engine/interfaces.py`.
