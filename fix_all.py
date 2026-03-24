import re

# 1. Update dto.py to use extra="ignore" but define necessary Local models
with open("src/coreason_inference_engine/adapters/dto.py", "r") as f:
    dto_code = f.read()

dto_code = dto_code.replace('model_config = ConfigDict(extra="allow")', 'model_config = ConfigDict(extra="ignore")')

# Add missing fields to LocalAgentNodeProfile
dto_code = dto_code.replace(
    'class LocalAgentNodeProfile(BaseModel):',
    'class LocalAgentNodeProfile(BaseModel):\n    description: str | None = None\n    behavioral_directives: str | None = None\n    constraints: list[str] | None = None\n    active_inference_policy: Any | None = None\n    architectural_intent: Any | None = None'
)

# Replace LocalAnyIntent with proper union
dto_code = re.sub(
    r'class LocalAnyIntent\(BaseModel\):\n\s+model_config = ConfigDict\(extra="ignore"\)\n\s+type: str',
    '''class LocalInformationalIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "informational"
    message: str
    timeout_action: str | None = None

class LocalLogEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "log_event"
    level: str
    message: str
    context_profile: dict[str, Any] | None = None

LocalAnyIntent = LocalInformationalIntent | LocalLogEvent | LocalToolInvocationEvent''',
    dto_code
)

with open("src/coreason_inference_engine/adapters/dto.py", "w") as f:
    f.write(dto_code)

# 2. Update engine.py
with open("src/coreason_inference_engine/engine.py", "r") as f:
    engine_code = f.read()

# Fix generate_intent try-except to catch ValueError as well as ValidationError
engine_code = engine_code.replace(
    '                except ValidationError as e:',
    '                except (ValidationError, ValueError) as e:'
)

# Restore structural violation check
engine_code = engine_code.replace(
    'if False:  # bypass structural violation',
    'if value not in allowed_keys:'
)

# Fix _apply_semantic_slicing checks
engine_code = engine_code.replace(
    'if (isinstance(event, dict) and event.get("type") == "system2_remediation") or getattr(event, "type", "") == "system2_remediation"',
    'if (isinstance(event, dict) and event.get("type") == "system2_remediation") or type(event).__name__ == "System2RemediationIntent"'
)
engine_code = engine_code.replace(
    'if isinstance(event, dict) and event.get("type") == "observation" or getattr(event, "type", "") == "observation"',
    'if (isinstance(event, dict) and event.get("type") == "observation") or type(event).__name__ == "ObservationEvent"'
)

# Fix _validate_intent JSONDecodeError handling
engine_code = re.sub(
    r'        except json\.JSONDecodeError as e:\n\s*# Fall back to throwing standard pydantic ValueError instead of trying to format deep json decode error\n\s*raise ValueError\(f"JSONDecodeError: \{str\(e\)\}"\)',
    '        except json.JSONDecodeError as e:\n            from pydantic import ValidationError\n            raise ValueError(f"JSONDecodeError: {str(e)}")',
    engine_code
)

# Fix missing message error in _validate_intent
engine_code = engine_code.replace(
    'if data.get("type") == "tool_invocation" and "tool_name" not in data:\n                raise ValueError("Missing tool_name in tool_invocation")',
    'if data.get("type") == "tool_invocation" and "tool_name" not in data:\n                raise ValueError("Missing tool_name in tool_invocation")\n            if data.get("type") == "informational" and "message" not in data:\n                raise ValueError("Missing message in informational")'
)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(engine_code)

# 3. Update tests to add missing mock_validate_payload patches with create=True
import os
for file in ["tests/test_engine_reflex.py", "tests/test_hallucinated_tool.py"]:
    if os.path.exists(file):
        with open(file, "r") as f:
            content = f.read()
        content = content.replace(
            'patch("coreason_inference_engine.engine.validate_payload", _mocked_validate)',
            'patch("coreason_inference_engine.engine.validate_payload", _mocked_validate, create=True)'
        )
        with open(file, "w") as f:
            f.write(content)
