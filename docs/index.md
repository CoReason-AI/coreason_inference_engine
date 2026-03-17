# Welcome to CoReason Inference Engine

The **Cognitive Plane** microservice of the CoReason ecosystem.

The stateless cognitive bridge connecting deterministic rules to Large Language Models (LLMs). This service isolates probabilistic compute, enforces strict egress validation via `coreason_manifest`, and implements robust self-healing workflows including System 2 Remediation logic.

## Project Vision
By operating within a strict Zero-Trust architecture, the engine remains physically air-gapped from the execution environment. It ingests an immutable `EpistemicLedgerState`, hydrates it into a provider-specific context window, negotiates with external or local APIs, and mathematically forces the output into rigidly typed JSON structures (`AnyIntent`).

## Guides

- [Architecture & Protocol Overview](architecture.md)
- [API Reference](api.md)

## Development Requirements
- **Python 3.14+**
- Developed with `uv` package manager and `pytest` for test-driven design.
