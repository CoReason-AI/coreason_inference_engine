with open("src/coreason_inference_engine/engine.py") as f:
    content = f.read()

content = content.replace(
    'burn_receipt = dict(\n                    event_id=f"burn_{uuid.uuid4().hex[:8]}",\n                    timestamp=time.time(),\n                    tool_invocation_id=invocation_cid,\n                    input_tokens=total_input_tokens,\n                    output_tokens=total_output_tokens,\n                    burn_magnitude=self._calculate_cost(total_input_tokens, total_output_tokens),\n                )',
    'burn_receipt = {\n                    "event_id": f"burn_{uuid.uuid4().hex[:8]}",\n                    "timestamp": time.time(),\n                    "tool_invocation_id": invocation_cid,\n                    "input_tokens": total_input_tokens,\n                    "output_tokens": total_output_tokens,\n                    "burn_magnitude": self._calculate_cost(total_input_tokens, total_output_tokens),\n                }',
)
content = content.replace(
    'error_intent = dict(\n                event_id=f"fault_{uuid.uuid4().hex[:8]}",\n                timestamp=time.time(),\n                type="system_fault",\n            )',
    'error_intent = {\n                "event_id": f"fault_{uuid.uuid4().hex[:8]}",\n                "timestamp": time.time(),\n                "type": "system_fault",\n            }',
)
content = content.replace(
    'receipt = dict(\n                event_id=f"burn_{uuid.uuid4().hex[:8]}",\n                timestamp=time.time(),\n                tool_invocation_id="none",\n                input_tokens=0,\n                output_tokens=0,\n                burn_magnitude=0,\n            )',
    'receipt = {\n                "event_id": f"burn_{uuid.uuid4().hex[:8]}",\n                "timestamp": time.time(),\n                "tool_invocation_id": "none",\n                "input_tokens": 0,\n                "output_tokens": 0,\n                "burn_magnitude": 0,\n            }',
)
content = content.replace(
    'remediation_intent = dict(\n                                type="system2_remediation",\n                                fault_id=f"fault_{uuid.uuid4().hex[:8]}",\n                                target_node_id=node_id or "",\n                                failing_pointers=["/"],\n                                remediation_prompt="CRITICAL CONTRACT BREACH: An immediate top-level structural "\n                                "violation was detected during streaming. Correct your JSON projection.",\n                            )',
    'remediation_intent = {\n                                "type": "system2_remediation",\n                                "fault_id": f"fault_{uuid.uuid4().hex[:8]}",\n                                "target_node_id": node_id or "",\n                                "failing_pointers": ["/"],\n                                "remediation_prompt": "CRITICAL CONTRACT BREACH: An immediate top-level structural "\n                                "violation was detected during streaming. Correct your JSON projection.",\n                            }',
)
content = content.replace(
    'burn_receipt = dict(\n                                event_id=f"burn_{uuid.uuid4().hex[:8]}",\n                                timestamp=time.time(),\n                                tool_invocation_id=invocation_cid,\n                                input_tokens=total_input_tokens + usage_metrics.get("input_tokens", 0),\n                                output_tokens=total_output_tokens + usage_metrics.get("output_tokens", 0),\n                                burn_magnitude=self._calculate_cost(\n                                    total_input_tokens + usage_metrics.get("input_tokens", 0),\n                                    total_output_tokens + usage_metrics.get("output_tokens", 0),\n                                ),\n                            )',
    'burn_receipt = {\n                                "event_id": f"burn_{uuid.uuid4().hex[:8]}",\n                                "timestamp": time.time(),\n                                "tool_invocation_id": invocation_cid,\n                                "input_tokens": total_input_tokens + usage_metrics.get("input_tokens", 0),\n                                "output_tokens": total_output_tokens + usage_metrics.get("output_tokens", 0),\n                                "burn_magnitude": self._calculate_cost(\n                                    total_input_tokens + usage_metrics.get("input_tokens", 0),\n                                    total_output_tokens + usage_metrics.get("output_tokens", 0),\n                                ),\n                            }',
)
content = content.replace(
    'burn_receipt = dict(\n                            event_id=f"burn_{uuid.uuid4().hex[:8]}",\n                            timestamp=time.time(),\n                            tool_invocation_id="none",\n                            input_tokens=total_input_tokens,\n                            output_tokens=total_output_tokens,\n                            burn_magnitude=self._calculate_cost(total_input_tokens, total_output_tokens),\n                        )',
    'burn_receipt = {\n                            "event_id": f"burn_{uuid.uuid4().hex[:8]}",\n                            "timestamp": time.time(),\n                            "tool_invocation_id": "none",\n                            "input_tokens": total_input_tokens,\n                            "output_tokens": total_output_tokens,\n                            "burn_magnitude": self._calculate_cost(total_input_tokens, total_output_tokens),\n                        }',
)
content = content.replace(
    'valid_intent = dict(target_node_id=node_id, fallback_node_id="system_halt")',
    'valid_intent = {"target_node_id": node_id, "fallback_node_id": "system_halt"}',
)
content = content.replace(
    'burn_receipt = dict(\n                        event_id=f"burn_{uuid.uuid4().hex[:8]}",\n                        timestamp=time.time(),\n                        tool_invocation_id=invocation_cid or "none",\n                        input_tokens=total_input_tokens,\n                        output_tokens=total_output_tokens,\n                        burn_magnitude=self._calculate_cost(total_input_tokens, total_output_tokens),\n                    )',
    'burn_receipt = {\n                        "event_id": f"burn_{uuid.uuid4().hex[:8]}",\n                        "timestamp": time.time(),\n                        "tool_invocation_id": invocation_cid or "none",\n                        "input_tokens": total_input_tokens,\n                        "output_tokens": total_output_tokens,\n                        "burn_magnitude": self._calculate_cost(total_input_tokens, total_output_tokens),\n                    }',
)
content = content.replace(
    'cognitive_receipt = dict(\n                            event_id=f"reward_{uuid.uuid4().hex[:8]}",\n                            timestamp=time.time(),\n                            source_generation_id=scratchpad.get("trace_id"),\n                            extracted_axioms=[],\n                            calculated_r_path=1.0,\n                            total_advantage_score=1.0,\n                        )',
    'cognitive_receipt = {\n                            "event_id": f"reward_{uuid.uuid4().hex[:8]}",\n                            "timestamp": time.time(),\n                            "source_generation_id": scratchpad.get("trace_id"),\n                            "extracted_axioms": [],\n                            "calculated_r_path": 1.0,\n                            "total_advantage_score": 1.0,\n                        }',
)
content = content.replace(
    "branch = dict(\n                branch_id=branch_id,\n                parent_branch_id=None,\n                latent_content_hash=content_hash,\n                prm_score=None,\n            )",
    'branch = {\n                "branch_id": branch_id,\n                "parent_branch_id": None,\n                "latent_content_hash": content_hash,\n                "prm_score": None,\n            }',
)
content = content.replace(
    'receipt = dict(\n                trace_id=f"trace_{uuid.uuid4().hex[:8]}",\n                explored_branches=[branch],\n                discarded_branches=[],\n                resolution_branch_id=branch_id,\n                total_latent_tokens=self.adapter.count_tokens(think_content),\n            )',
    'receipt = {\n                "trace_id": f"trace_{uuid.uuid4().hex[:8]}",\n                "explored_branches": [branch],\n                "discarded_branches": [],\n                "resolution_branch_id": branch_id,\n                "total_latent_tokens": self.adapter.count_tokens(think_content),\n            }',
)

# Fix main.py usage:
with open("src/coreason_inference_engine/main.py") as f:
    main_content = f.read()

main_content = main_content.replace(
    "intent, receipt, latent_scratchpad, reward = await engine.generate_intent(",
    "intent, receipt, latent_scratchpad, reward = await engine.generate_intent(",
)

# wait main.py has:
# intent, receipt, latent_scratchpad, reward = await engine.generate_intent(
#     node=node, ledger=ledger, node_id=request.target_node_id, action_space=action_space
# )
main_content = main_content.replace(
    """    intent, receipt, latent_scratchpad, reward = await engine.generate_intent(
        node=node, ledger=ledger, node_id=request.target_node_id, action_space=action_space
    )""",
    """    intent, receipt, latent_scratchpad, reward = await engine.generate_intent(
        raw_node=node.model_dump(), raw_ledger=ledger.model_dump(), node_id=request.target_node_id, raw_action_space=action_space.model_dump()
    )""",
)

# And in main.py: `intent.model_dump()` etc. must be removed.
main_content = main_content.replace(
    """    return {
        "intent": intent.model_dump(),
        "receipt": receipt.model_dump(),
        "latent_scratchpad": latent_scratchpad.model_dump() if latent_scratchpad else None,
        "cognitive_reward": reward.model_dump() if reward else None,
    }""",
    """    return {
        "intent": intent,
        "receipt": receipt,
        "latent_scratchpad": latent_scratchpad,
        "cognitive_reward": reward,
    }""",
)

with open("src/coreason_inference_engine/main.py", "w") as f:
    f.write(main_content)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(content)
