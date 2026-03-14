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

### Usage

-   Run the linter:
    ```sh
    uv run pre-commit run --all-files
    ```
-   Run the tests:
    ```sh
    uv run pytest
    ```
