with open("tests/test_engine.py") as f:
    content = f.read()

content = content.replace(
    'patch("coreason_inference_engine.engine.validate_payload", side_effect=_mocked_validate),', ""
)

with open("tests/test_engine.py", "w") as f:
    f.write(content)
