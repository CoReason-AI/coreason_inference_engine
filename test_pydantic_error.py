from pydantic import ValidationError

print(hasattr(ValidationError, "from_exception_data"))
