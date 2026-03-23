import re

with open("src/coreason_inference_engine/adapters/dto.py", "r") as f:
    content = f.read()

content = content.replace(
    'class LocalLedgerState(BaseModel):\n    """\n    Internal Pydantic model for the inference engine to map Ledger payloads.\n    Silently drops unknown fields.\n    """\n\n    model_config = ConfigDict(extra="ignore")\n\n    history: list[dict[str, Any]] = Field(default_factory=list)',
    'class LocalLedgerState(BaseModel):\n    """\n    Internal Pydantic model for the inference engine to map Ledger payloads.\n    Silently drops unknown fields.\n    """\n\n    model_config = ConfigDict(extra="ignore")\n\n    history: list[dict[str, Any]] = Field(default_factory=list)\n    active_cascades: list[Any] | None = None\n    active_rollbacks: list[Any] | None = None',
)

with open("src/coreason_inference_engine/adapters/dto.py", "w") as f:
    f.write(content)


with open("src/coreason_inference_engine/context.py", "r") as f:
    content = f.read()

content = content.replace(
    "if ledger.active_cascades:",
    "active_cascades = getattr(ledger, 'active_cascades', None) if not isinstance(ledger, dict) else ledger.get('active_cascades')\n        if active_cascades:\n            for cascade in active_cascades:\n                if isinstance(cascade, dict):\n                    quarantined_event_ids.update(cascade.get('quarantined_event_ids', []))\n                else:\n                    quarantined_event_ids.update(getattr(cascade, 'quarantined_event_ids', []))",
)

content = content.replace(
    "for cascade in ledger.active_cascades:\n                quarantined_event_ids.update(cascade.quarantined_event_ids)",
    "",
)

content = content.replace(
    "if ledger.active_rollbacks:",
    "active_rollbacks = getattr(ledger, 'active_rollbacks', None) if not isinstance(ledger, dict) else ledger.get('active_rollbacks')\n        if active_rollbacks:\n            for rollback in active_rollbacks:\n                if isinstance(rollback, dict):\n                    invalid_ids = rollback.get('invalidated_node_ids', [])\n                else:\n                    invalid_ids = getattr(rollback, 'invalidated_node_ids', [])\n                if invalid_ids:\n                    quarantined_event_ids.update(invalid_ids)",
)

content = content.replace(
    "for rollback in ledger.active_rollbacks:\n                if rollback.invalidated_node_ids:\n                    quarantined_event_ids.update(rollback.invalidated_node_ids)",
    "",
)

with open("src/coreason_inference_engine/context.py", "w") as f:
    f.write(content)
