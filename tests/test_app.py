"""Tests for the FastAPI application module."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from inference.app import create_app, lifespan
from inference.config import Settings
from inference.engine.inference import InferenceEngine


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
    async def test_lifespan_loads_model(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model_config: MagicMock,
    ) -> None:
        """Test that lifespan loads the model on startup."""
        settings = Settings(model_path="/tmp/test", device="cpu")

        with (
            patch("inference.app.ModelLoader") as mock_loader_class,
            patch("inference.app.CPUOptimizer") as mock_optimizer_class,
            patch("inference.app.resolve_device", return_value="cpu"),
        ):
            # Setup mocks
            mock_loader = MagicMock()
            mock_loader.load.return_value = (mock_model, mock_tokenizer)
            mock_loader_class.return_value = mock_loader

            mock_optimizer = MagicMock()
            mock_optimizer.optimize_model.return_value = mock_model
            mock_optimizer_class.return_value = mock_optimizer

            app = FastAPI(lifespan=lifespan)
            app.state.settings = settings

            async with lifespan(app):
                # Engine should be created
                assert hasattr(app.state, "engine")
                assert isinstance(app.state.engine, InferenceEngine)

                # CPU optimizer should be called
                mock_optimizer.setup_environment.assert_called_once()
                mock_optimizer.optimize_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_skips_cpu_optimizer_for_gpu(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Test that CPU optimizer is skipped for GPU devices."""
        settings = Settings(model_path="/tmp/test", device="cuda")

        with (
            patch("inference.app.ModelLoader") as mock_loader_class,
            patch("inference.app.CPUOptimizer") as mock_optimizer_class,
            patch("inference.app.resolve_device", return_value="cuda"),
        ):
            mock_loader = MagicMock()
            mock_loader.load.return_value = (mock_model, mock_tokenizer)
            mock_loader_class.return_value = mock_loader

            app = FastAPI(lifespan=lifespan)
            app.state.settings = settings

            async with lifespan(app):
                # CPU optimizer should not be instantiated
                mock_optimizer_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_lifespan_cleanup(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Test that lifespan cleans up resources on shutdown."""
        settings = Settings(model_path="/tmp/test", device="cpu")

        with (
            patch("inference.app.ModelLoader") as mock_loader_class,
            patch("inference.app.CPUOptimizer") as mock_optimizer_class,
            patch("inference.app.resolve_device", return_value="cpu"),
        ):
            mock_loader = MagicMock()
            mock_loader.load.return_value = (mock_model, mock_tokenizer)
            mock_loader_class.return_value = mock_loader

            mock_optimizer = MagicMock()
            mock_optimizer.optimize_model.return_value = mock_model
            mock_optimizer_class.return_value = mock_optimizer

            app = FastAPI(lifespan=lifespan)
            app.state.settings = settings

            async with lifespan(app):
                pass

            # After exiting lifespan, engine should be deleted
            assert not hasattr(app.state, "engine")


class TestDependencies:
    """Test FastAPI dependencies."""

    def test_get_engine_returns_engine(
        self,
        mock_engine: InferenceEngine,
    ) -> None:
        """Test that get_engine returns the engine from app state."""
        from inference.api.dependencies import get_engine

        # Create a mock request with app state
        mock_request = MagicMock()
        mock_request.app.state.engine = mock_engine

        engine = get_engine(mock_request)

        assert engine == mock_engine
