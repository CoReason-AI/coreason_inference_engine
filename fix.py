with open("src/coreason_inference_engine/engine.py", "r") as f:
    code = f.read()

code = code.replace(
    'raise ValueError(f"JSONDecodeError: {e!s}")',
    'raise ValueError(f"JSONDecodeError: {e!s}") from e'
)

code = code.replace(
    'if not isinstance(data, dict):\n            # Fall back to throwing standard pydantic ValueError instead of trying to format deep json decode error\n            raise ValueError(f"JSONDecodeError: {e!s}") from e',
    'if not isinstance(data, dict):\n            raise ValueError("Payload must be a dictionary")'
)

with open("src/coreason_inference_engine/engine.py", "w") as f:
    f.write(code)
