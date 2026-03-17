# coreason_inference_engine

The stateless cognitive bridge connecting deterministic rules to LLMs

[![CI/CD](https://github.com/CoReason-AI/coreason_inference_engine/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/CoReason-AI/coreason_inference_engine/actions/workflows/ci-cd.yml)
[![PyPI](https://img.shields.io/pypi/v/coreason_inference_engine.svg)](https://pypi.org/project/coreason_inference_engine/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/coreason_inference_engine.svg)](https://pypi.org/project/coreason_inference_engine/)
[![License](https://img.shields.io/github/license/CoReason-AI/coreason_inference_engine)](https://github.com/CoReason-AI/coreason_inference_engine/blob/main/LICENSE)
[![Codecov](https://codecov.io/gh/CoReason-AI/coreason_inference_engine/branch/main/graph/badge.svg)](https://codecov.io/gh/CoReason-AI/coreason_inference_engine)
[![Downloads](https://static.pepy.tech/badge/coreason_inference_engine)](https://pepy.tech/project/coreason_inference_engine)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Getting Started

### Prerequisites

- Python 3.14+
- uv

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/CoReason-AI/coreason_inference_engine.git
    cd coreason_inference_engine
    ```
2.  Install dependencies:
    ```sh
    uv sync --all-extras --dev
    ```

## Architecture

The Inference Engine operates exclusively within the **Cognitive Plane** of the CoReason ecosystem. It acts as a mathematically bounded, stateless, asynchronous transformation engine. It receives rigid topological state (`EpistemicLedgerState`, `AgentNodeProfile`), translates it into a probabilistic vector space via LLM APIs, and forces the output back into a strict geometric boundary (`AnyIntent`).

### Core Components

*   **Inference Engine (`src/coreason_inference_engine/engine.py`)**: The primary class responsible for the generation loop, fail-fast parsing, semantic slicing, and invoking the System 2 Remediation Loop upon validation failure.
*   **Context Hydrator (`src/coreason_inference_engine/context.py`)**: Responsible for deterministically flattening the causal DAG (`EpistemicLedgerState`) into provider-compliant conversational arrays, supporting standard, reasoning, and strictly-alternating (Anthropic) prompt grammars.
*   **LLM Adapters (`src/coreason_inference_engine/adapters/`)**: Classes conforming to `LLMAdapterProtocol` that abstract provider-specific APIs (OpenAI, Anthropic). These include SSRF firewalls and connection pooling logic.
*   **RPC Server (`src/coreason_inference_engine/main.py`)**: A Typer-backed FastAPI server exposing the async generation entry point over HTTP.

### Usage

### Running the Server

Start the Cognitive Plane microservice with the default Typer command:

```sh
uv run coreason_inference_engine
```

Configure the LLM adapter using environment variables:

```sh
export COREASON_LLM_PROVIDER="openai" # or "anthropic"
export COREASON_LLM_MODEL="gpt-4o"
export OPENAI_API_KEY="your-api-key"
```

### Development

-   Run the linter and formatters:
    ```sh
    uv run ruff check . --fix
    uv run ruff format .
    uv run pre-commit run --all-files
    ```
-   Run the tests:
    ```sh
    uv run pytest
    ```
