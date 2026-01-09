"""Tests for the API dependencies module."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from inference.api.dependencies import get_engine


class TestGetEngine:
    """Tests for get_engine dependency."""

    def test_get_engine_returns_engine(self) -> None:
        """Test that get_engine returns the engine from app state."""
        app = FastAPI()
        mock_engine = MagicMock()
        app.state.engine = mock_engine

        @app.get("/test")
        def test_endpoint(request: Request):
            engine = get_engine(request)
            return {"has_engine": engine is not None}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert response.json()["has_engine"] is True

    def test_get_engine_returns_correct_engine(self) -> None:
        """Test that get_engine returns the correct engine instance."""
        app = FastAPI()
        mock_engine = MagicMock()
        mock_engine.model_name = "test-model"
        app.state.engine = mock_engine

        @app.get("/model-name")
        def test_endpoint(request: Request):
            engine = get_engine(request)
            return {"model_name": engine.model_name}

        client = TestClient(app)
        response = client.get("/model-name")

        assert response.status_code == 200
        assert response.json()["model_name"] == "test-model"

    def test_get_engine_raises_when_no_engine(self) -> None:
        """Test that accessing engine raises error when not set."""
        app = FastAPI()
        # Don't set app.state.engine

        @app.get("/test")
        def test_endpoint(request: Request):
            engine = get_engine(request)
            return {"engine": str(engine)}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")

        # Should raise AttributeError since engine isn't set
        assert response.status_code == 500

    def test_get_engine_with_dependency_injection(self) -> None:
        """Test get_engine used as FastAPI dependency."""
        from fastapi import Depends

        app = FastAPI()
        mock_engine = MagicMock()
        mock_engine.device = "cpu"
        app.state.engine = mock_engine

        @app.get("/device")
        def test_endpoint(engine=Depends(get_engine)):
            return {"device": engine.device}

        client = TestClient(app)
        response = client.get("/device")

        assert response.status_code == 200
        assert response.json()["device"] == "cpu"

    def test_get_engine_works_with_different_states(self) -> None:
        """Test that get_engine works with different app states."""
        # First app with engine A
        app1 = FastAPI()
        engine_a = MagicMock()
        engine_a.name = "engine_a"
        app1.state.engine = engine_a

        @app1.get("/name")
        def endpoint_a(request: Request):
            return {"name": get_engine(request).name}

        # Second app with engine B
        app2 = FastAPI()
        engine_b = MagicMock()
        engine_b.name = "engine_b"
        app2.state.engine = engine_b

        @app2.get("/name")
        def endpoint_b(request: Request):
            return {"name": get_engine(request).name}

        client1 = TestClient(app1)
        client2 = TestClient(app2)

        assert client1.get("/name").json()["name"] == "engine_a"
        assert client2.get("/name").json()["name"] == "engine_b"


class TestDependencyIntegration:
    """Integration tests for dependencies with routes."""

    def test_dependency_with_backend_protocol(self) -> None:
        """Test that dependency works with InferenceBackend protocol."""
        from inference.backends.base import InferenceBackend

        app = FastAPI()

        # Create a mock that implements the protocol
        mock_backend = MagicMock(spec=InferenceBackend)
        mock_backend.model_name = "test-model"
        mock_backend.device = "cpu"
        mock_backend.generate.return_value = ("Hello", {"total_tokens": 1})
        mock_backend.get_model_info.return_value = {"name": "test"}

        app.state.engine = mock_backend

        @app.get("/info")
        def get_info(request: Request):
            engine = get_engine(request)
            return engine.get_model_info()

        client = TestClient(app)
        response = client.get("/info")

        assert response.status_code == 200
        assert response.json()["name"] == "test"

    def test_dependency_injection_type_hints(self) -> None:
        """Test that type hints work with dependency injection."""
        from typing import TYPE_CHECKING

        from fastapi import Depends

        if TYPE_CHECKING:
            from inference.backends.base import InferenceBackend

        app = FastAPI()
        mock_backend = MagicMock()
        mock_backend.model_name = "typed-model"
        app.state.engine = mock_backend

        @app.get("/typed")
        def typed_endpoint(engine: "InferenceBackend" = Depends(get_engine)):
            return {"model": engine.model_name}

        client = TestClient(app)
        response = client.get("/typed")

        assert response.status_code == 200
        assert response.json()["model"] == "typed-model"
