"""Tests for the logging utility module."""

import logging
from unittest.mock import patch

from inference.utils.logging import setup_logging


class TestSetupLogging:
    """Test setup_logging function."""

    def test_default_level(self) -> None:
        """Test that default level is INFO."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging()

            mock_config.assert_called_once()
            call_kwargs = mock_config.call_args.kwargs
            assert call_kwargs["level"] == logging.INFO

    def test_custom_level(self) -> None:
        """Test that custom level is respected."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging(level=logging.DEBUG)

            call_kwargs = mock_config.call_args.kwargs
            assert call_kwargs["level"] == logging.DEBUG

    def test_format_string(self) -> None:
        """Test that format string is set correctly."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging()

            call_kwargs = mock_config.call_args.kwargs
            assert "%(asctime)s" in call_kwargs["format"]
            assert "%(name)s" in call_kwargs["format"]
            assert "%(levelname)s" in call_kwargs["format"]
            assert "%(message)s" in call_kwargs["format"]

    def test_uses_stdout_handler(self) -> None:
        """Test that stdout handler is configured."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging()

            call_kwargs = mock_config.call_args.kwargs
            assert "handlers" in call_kwargs
            assert len(call_kwargs["handlers"]) == 1
