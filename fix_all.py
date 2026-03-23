import os
import re

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
    with open(file) as f:
        content = f.read()

    # We changed keyword arguments from node= to raw_node=, ledger= to raw_ledger=, action_space= to raw_action_space=
    content = content.replace(
        "node=node_with_boundaries",
        "raw_node=node_with_boundaries.model_dump() if hasattr(node_with_boundaries, 'model_dump') else node_with_boundaries",
    )
    content = content.replace("node=node", "raw_node=node.model_dump() if hasattr(node, 'model_dump') else node")
    content = content.replace(
        "ledger=ledger", "raw_ledger=ledger.model_dump() if hasattr(ledger, 'model_dump') else ledger"
    )
    content = content.replace(
        "action_space=action_space",
        "raw_action_space=action_space.model_dump() if hasattr(action_space, 'model_dump') else action_space",
    )
    content = content.replace(
        "node=mock_node", "raw_node=mock_node.model_dump() if hasattr(mock_node, 'model_dump') else mock_node"
    )
    content = content.replace(
        "ledger=mock_ledger",
        "raw_ledger=mock_ledger.model_dump() if hasattr(mock_ledger, 'model_dump') else mock_ledger",
    )
    content = content.replace(
        "action_space=mock_action_space",
        "raw_action_space=mock_action_space.model_dump() if hasattr(mock_action_space, 'model_dump') else mock_action_space",
    )

    content = content.replace(
        "node_with_logit.model_dump() if hasattr(node_with_logit, 'model_dump') else node_with_logit",
        "node_with_logit.model_dump() if hasattr(node_with_logit, 'model_dump') else node_with_logit",
    )
    content = content.replace(
        "node=node_with_logit",
        "raw_node=node_with_logit.model_dump() if hasattr(node_with_logit, 'model_dump') else node_with_logit",
    )
    content = content.replace(
        "node=node_with_peft",
        "raw_node=node_with_peft.model_dump() if hasattr(node_with_peft, 'model_dump') else node_with_peft",
    )
    content = content.replace(
        "node=mock_node_with_think",
        "raw_node=mock_node_with_think.model_dump() if hasattr(mock_node_with_think, 'model_dump') else mock_node_with_think",
    )
    content = content.replace(
        "node=node_with_small_timeout",
        "raw_node=node_with_small_timeout.model_dump() if hasattr(node_with_small_timeout, 'model_dump') else node_with_small_timeout",
    )
    content = content.replace(
        "node=new_node", "raw_node=new_node.model_dump() if hasattr(new_node, 'model_dump') else new_node"
    )
    content = content.replace(
        "node=dummy_node", "raw_node=dummy_node.model_dump() if hasattr(dummy_node, 'model_dump') else dummy_node"
    )
    content = content.replace(
        "ledger=dummy_ledger",
        "raw_ledger=dummy_ledger.model_dump() if hasattr(dummy_ledger, 'model_dump') else dummy_ledger",
    )
    content = content.replace(
        "action_space=dummy_action_space",
        "raw_action_space=dummy_action_space.model_dump() if hasattr(dummy_action_space, 'model_dump') else dummy_action_space",
    )

    # Positional generate_intent args in test_engine.py
    content = re.sub(
        r"""await engine\.generate_intent\(\s*node,\s*ledger,\s*"node_1",\s*action_space\s*\)""",
        r"""await engine.generate_intent(raw_node=node.model_dump() if hasattr(node, 'model_dump') else node, raw_ledger=ledger.model_dump() if hasattr(ledger, 'model_dump') else ledger, node_id="node_1", raw_action_space=action_space.model_dump() if hasattr(action_space, 'model_dump') else action_space)""",
        content,
    )
    content = re.sub(
        r"""engine\.generate_intent\(\s*node,\s*ledger,\s*"node_1",\s*action_space\s*\)""",
        r"""engine.generate_intent(raw_node=node.model_dump() if hasattr(node, 'model_dump') else node, raw_ledger=ledger.model_dump() if hasattr(ledger, 'model_dump') else ledger, node_id="node_1", raw_action_space=action_space.model_dump() if hasattr(action_space, 'model_dump') else action_space)""",
        content,
    )

    with open(file, "w") as f:
        f.write(content)
