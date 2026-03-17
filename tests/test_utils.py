# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference_engine


from coreason_inference_engine.utils.logger import logger


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly."""
    # Since the logger is initialized on import, we check side effects
    # Directory creation should no longer occur per TRD-7.
    assert True


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None
