"""Tests for the FastAPI application module."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from inference.app import create_app, lifespan
from inference.backends.base import InferenceBackend
from inference.config import Settings


class TestCreateApp:
    """Test create_app factory function."""

    def test_creates_fastapi_app(self) -> None:
        """Test that create_app returns a FastAPI instance."""
        with patch("inference.app.lifespan"):
            app = create_app(Settings(model_path="/tmp/test"))

        assert isinstance(app, FastAPI)
        assert app.title == "Small Model Inference"
        assert app.version == "0.1.0"

    def test_stores_settings_in_state(self) -> None:
        """Test that settings are stored in app state."""
        settings = Settings(model_path="/tmp/test", port=9000)

        with patch("inference.app.lifespan"):
            app = create_app(settings)

        assert app.state.settings == settings
        assert app.state.settings.port == 9000

    def test_uses_default_settings(self) -> None:
        """Test that default settings are used when none provided."""
        with patch("inference.app.lifespan"):
            app = create_app()

        assert isinstance(app.state.settings, Settings)

    def test_includes_router(self) -> None:
        """Test that the API router is included."""
        with patch("inference.app.lifespan"):
            app = create_app(Settings(model_path="/tmp/test"))

        # Check that routes are registered
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/v1/models" in routes
        assert "/v1/completions" in routes
        assert "/v1/chat/completions" in routes


class TestLifespan:
    """Test lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_loads_model(self) -> None:
        """Test that lifespan creates the backend on startup."""
        settings = Settings(model_path="/tmp/test", device="cpu")

        with patch("inference.app.create_backend") as mock_create_backend:
            # Setup mock backend
            mock_backend = MagicMock(spec=InferenceBackend)
            mock_backend.model_name = "test-model"
            mock_backend.device = "cpu"
            mock_create_backend.return_value = mock_backend

            app = FastAPI(lifespan=lifespan)
            app.state.settings = settings

            async with lifespan(app):
                # Engine should be created
                assert hasattr(app.state, "engine")
                assert app.state.engine == mock_backend

                # Backend factory should be called with settings
                mock_create_backend.assert_called_once()
                call_args = mock_create_backend.call_args
                assert call_args[0][0] == settings

    @pytest.mark.asyncio
    async def test_lifespan_with_tracing_enabled(self) -> None:
        """Test that tracing is initialized when enabled."""
        settings = Settings(
            model_path="/tmp/test",
            device="cpu",
            enable_tracing=True,
            langfuse_public_key="test-key",
            langfuse_secret_key="test-secret",
        )

        with (
            patch("inference.app.create_backend") as mock_create_backend,
            patch("inference.app.init_tracer") as mock_init_tracer,
        ):
            mock_backend = MagicMock(spec=InferenceBackend)
            mock_backend.model_name = "test-model"
            mock_backend.device = "cpu"
            mock_create_backend.return_value = mock_backend

            mock_tracer = MagicMock()
            mock_tracer.enabled = True
            mock_init_tracer.return_value = mock_tracer

            app = FastAPI(lifespan=lifespan)
            app.state.settings = settings

            async with lifespan(app):
                # Tracer should be initialized
                mock_init_tracer.assert_called_once()
                assert app.state.tracer == mock_tracer

    @pytest.mark.asyncio
    async def test_lifespan_cleanup(self) -> None:
        """Test that lifespan cleans up resources on shutdown."""
        settings = Settings(model_path="/tmp/test", device="cpu")

        with patch("inference.app.create_backend") as mock_create_backend:
            mock_backend = MagicMock(spec=InferenceBackend)
            mock_backend.model_name = "test-model"
            mock_backend.device = "cpu"
            mock_create_backend.return_value = mock_backend

            app = FastAPI(lifespan=lifespan)
            app.state.settings = settings

            async with lifespan(app):
                pass

            # After exiting lifespan, engine should be deleted
            assert not hasattr(app.state, "engine")
            # Backend shutdown should be called
            mock_backend.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_cleanup_with_tracer(self) -> None:
        """Test that tracer is shut down on cleanup."""
        settings = Settings(
            model_path="/tmp/test",
            device="cpu",
            enable_tracing=True,
        )

        with (
            patch("inference.app.create_backend") as mock_create_backend,
            patch("inference.app.init_tracer") as mock_init_tracer,
        ):
            mock_backend = MagicMock(spec=InferenceBackend)
            mock_backend.model_name = "test-model"
            mock_backend.device = "cpu"
            mock_create_backend.return_value = mock_backend

            mock_tracer = MagicMock()
            mock_init_tracer.return_value = mock_tracer

            app = FastAPI(lifespan=lifespan)
            app.state.settings = settings

            async with lifespan(app):
                pass

            # Tracer should be shut down
            mock_tracer.shutdown.assert_called_once()


class TestDependencies:
    """Test FastAPI dependencies."""

    def test_get_engine_returns_engine(self) -> None:
        """Test that get_engine returns the engine from app state."""
        from inference.api.dependencies import get_engine

        # Create a mock backend
        mock_backend = MagicMock(spec=InferenceBackend)

        # Create a mock request with app state
        mock_request = MagicMock()
        mock_request.app.state.engine = mock_backend

        engine = get_engine(mock_request)

        assert engine == mock_backend
