import re
import os

test_files = [
    "tests/test_engine.py",
    "tests/test_engine_reflex.py",
    "tests/test_engine_slicing.py",
    "tests/test_main.py",
    "tests/test_hallucinated_tool.py",
    "tests/test_interfaces.py",
    "tests/test_cost.py",
]

for file in test_files:
    if not os.path.exists(file):
        continue
    with open(file, "r") as f:
        content = f.read()

    content = content.replace(
        "assert isinstance(intent, ToolInvocationEvent)",
        "assert isinstance(intent, dict) and intent.get('type') == 'tool_invocation'",
    )
    content = content.replace(
        "assert isinstance(intent, System2RemediationIntent)",
        "assert isinstance(intent, dict) and intent.get('type') == 'system2_remediation'",
    )
    content = content.replace(
        "assert isinstance(intent, StateMutationIntent)",
        "assert isinstance(intent, dict) and intent.get('type') == 'state_mutation'",
    )
    content = content.replace(
        "assert isinstance(intent, SystemFaultEvent)",
        "assert isinstance(intent, dict) and intent.get('type') == 'system_fault'",
    )

    content = content.replace("intent.type", "intent.get('type')")
    content = content.replace("intent.tool_name", "intent.get('tool_name')")
    content = content.replace("intent.parameters", "intent.get('parameters')")
    content = content.replace("intent.failing_pointers", "intent.get('failing_pointers')")
    content = content.replace("intent.remediation_prompt", "intent.get('remediation_prompt')")
    content = content.replace("intent.event_id", "intent.get('event_id')")
    content = content.replace("intent.message", 'intent.get("message")')
    content = content.replace("intent.fault_id", "intent.get('fault_id')")
    content = content.replace("intent.op", "intent.get('op')")
    content = content.replace("intent.path", "intent.get('path')")

    content = content.replace("receipt.burn_magnitude", "receipt.get('burn_magnitude')")
    content = content.replace("receipt.input_tokens", "receipt.get('input_tokens')")
    content = content.replace("receipt.output_tokens", "receipt.get('output_tokens')")
    content = content.replace("receipt.tool_invocation_id", "receipt.get('tool_invocation_id')")

    content = content.replace("scratchpad.trace_id", "scratchpad.get('trace_id')")
    content = content.replace("scratchpad.total_latent_tokens", "scratchpad.get('total_latent_tokens')")
    content = content.replace("scratchpad.explored_branches", "scratchpad.get('explored_branches')")

    content = content.replace(
        "cognitive_receipt.total_advantage_score", "cognitive_receipt.get('total_advantage_score')"
    )
    content = content.replace("mutation_intent.value", "mutation_intent.get('value')")

    content = content.replace("assert getattr(intent, 'message', None)", "assert intent and intent.get('message')")
    content = content.replace("assert getattr(intent, 'tool_name', None)", "assert intent and intent.get('tool_name')")
    content = content.replace("assert getattr(intent, 'type', None)", "assert intent and intent.get('type')")

    content = content.replace(
        '            intent=InformationalIntent(message="test", timeout_action="proceed_default"),',
        "            intent={'type': 'informational', 'message': 'test', 'timeout_action': 'proceed_default'},",
    )
    content = content.replace(
        '            receipt=TokenBurnReceipt(\n                event_id="1",\n                timestamp=0.0,\n                tool_invocation_id="none",\n                input_tokens=10,\n                output_tokens=20,\n                burn_magnitude=100,\n            ),\n            scratchpad=None,\n            cognitive_receipt=None,\n        )',
        "            receipt={'event_id': '1', 'timestamp': 0.0, 'tool_invocation_id': 'none', 'input_tokens': 10, 'output_tokens': 20, 'burn_magnitude': 100},\n            scratchpad=None,\n            cognitive_receipt=None,\n        )",
    )

    with open(file, "w") as f:
        f.write(content)
