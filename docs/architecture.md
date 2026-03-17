# Architecture

The Inference Engine implements several key architectural strategies:

### Context Hydration & Semantic Slicing
* Iterates through the topological state array (`EpistemicLedgerState`).
* Determines provider-specific mappings based on OpenAI, Anthropic, or reasoning mode parameters.
* Implements Destructive Eviction Prevention (caps historical remediation traces, preserving observations).

### Provider Agnostic Streaming & Token Panic Prevention
* Employs asynchronous SSE (Server Sent Events) parsing (e.g. `ijson`).
* Uses safe fallback strategies (`errors="replace"`) to stop strict BPE tokenizers from panicking when streams are preempted mid-byte.

### SSRF Protection & Deadlocks
* Uses `ipaddress` bogon checking prior to HTTP connections.
* Local backpressure constraints (`asyncio.Semaphore`) with rapid `429 Too Many Requests` failure signals back to the orchestration layer.

### System 2 Remediation
* Native Python `try...except ValidationError` wrapping to generate structural reprimands (`System2RemediationIntent`).
* Dynamically re-injects prompts and forces JSON structures in a mathematically precise loop.

### Token Clamping
* Employs token budgets (e.g. `max_tokens=500`) specifically during System 2 retry loops to prevent extreme SLA burn rates due to large halluciations.
