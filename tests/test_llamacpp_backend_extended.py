"""Extended tests for LlamaCppBackend - state persistence, caching, and shutdown."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inference.config import Settings


class TestLlamaCppBackendStatePersistence:
    """Tests for LlamaCppBackend state persistence features."""

    @pytest.fixture
    def mock_llama(self) -> MagicMock:
        """Create a mock Llama instance."""
        mock = MagicMock()
        mock.save_state.return_value = b"state_data"
        mock.load_state.return_value = None
        mock.tokenize.return_value = [1, 2, 3, 4, 5]
        return mock

    @pytest.fixture
    def settings_with_state_cache(self, tmp_path: Path) -> Settings:
        """Create settings with state cache enabled."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")
        cache_dir = tmp_path / "cache"

        return Settings(
            model_path=str(model_path),
            backend="llama-cpp",
            enable_persistent_cache=True,
            persistent_cache_dir=str(cache_dir),
        )

    def test_check_state_support(self, tmp_path: Path) -> None:
        """Test _check_state_support method."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")

        settings = Settings(model_path=str(model_path), backend="llama-cpp")

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)

            # Mock model has save_state and load_state
            assert backend._supports_state_persistence is True

    def test_check_state_support_missing_methods(self, tmp_path: Path) -> None:
        """Test _check_state_support when methods are missing."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")

        settings = Settings(model_path=str(model_path), backend="llama-cpp")

        mock_llama = MagicMock(spec=[])  # No save_state/load_state
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)
            assert backend._supports_state_persistence is False

    def test_state_cache_initialization(
        self, settings_with_state_cache: Settings, mock_llama: MagicMock
    ) -> None:
        """Test that state cache is initialized when enabled."""
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings_with_state_cache, None)

            assert backend._state_cache is not None
            info = backend.get_model_info()
            assert info["state_cache_enabled"] is True

    def test_state_cache_disabled_by_default(self, tmp_path: Path) -> None:
        """Test that state cache is disabled by default."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")

        settings = Settings(
            model_path=str(model_path),
            backend="llama-cpp",
            enable_persistent_cache=False,
        )

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)
            assert backend._state_cache is None


class TestLlamaCppBackendCaching:
    """Tests for LlamaCppBackend caching functionality."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> MagicMock:
        """Create a backend with response cache enabled."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")

        settings = Settings(
            model_path=str(model_path),
            backend="llama-cpp",
            enable_response_cache=True,
            response_cache_size=100,
        )

        mock_llama = MagicMock()
        mock_llama.return_value = {
            "choices": [{"text": "response"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            return LlamaCppBackend(settings, None)

    def test_get_cache_stats(self, backend) -> None:
        """Test get_cache_stats returns correct structure."""
        stats = backend.get_cache_stats()

        assert "backend" in stats
        assert stats["backend"] == "llama-cpp"
        assert "response_cache" in stats

    def test_get_cache_stats_with_state_cache(self, tmp_path: Path) -> None:
        """Test get_cache_stats includes state cache stats."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")
        cache_dir = tmp_path / "cache"

        settings = Settings(
            model_path=str(model_path),
            backend="llama-cpp",
            enable_response_cache=True,
            enable_persistent_cache=True,
            persistent_cache_dir=str(cache_dir),
        )

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)
            stats = backend.get_cache_stats()

            assert "state_cache" in stats

    def test_clear_caches(self, backend) -> None:
        """Test clear_caches clears response cache."""
        # Generate to populate cache
        from inference.backends.base import GenerationConfig

        config = GenerationConfig(temperature=0.0)
        backend.generate("test prompt", config)

        # Verify cache has entry
        assert backend._response_cache.stats()["size"] > 0

        # Clear
        backend.clear_caches()

        # Verify cache is empty
        assert backend._response_cache.stats()["size"] == 0

    def test_clear_caches_no_caches(self, tmp_path: Path) -> None:
        """Test clear_caches when no caches are enabled."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")

        settings = Settings(
            model_path=str(model_path),
            backend="llama-cpp",
            enable_response_cache=False,
            enable_persistent_cache=False,
        )

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)
            # Should not raise
            backend.clear_caches()


class TestLlamaCppBackendShutdown:
    """Tests for LlamaCppBackend shutdown behavior."""

    def test_shutdown_cleans_up(self, tmp_path: Path) -> None:
        """Test that shutdown properly cleans up resources."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")

        settings = Settings(model_path=str(model_path), backend="llama-cpp")

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)
            # Should not raise
            backend.shutdown()

    def test_shutdown_with_draft_model(self, tmp_path: Path) -> None:
        """Test shutdown cleans up draft model."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")
        draft_path = tmp_path / "draft.gguf"
        draft_path.write_bytes(b"mock draft")

        settings = Settings(
            model_path=str(model_path),
            backend="llama-cpp",
            enable_speculative_decoding=True,
            draft_model_path=str(draft_path),
        )

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)
            # Should not raise
            backend.shutdown()


class TestLlamaCppBackendTracing:
    """Tests for LlamaCppBackend tracing integration."""

    def test_create_trace_disabled(self, tmp_path: Path) -> None:
        """Test _create_trace when tracing is disabled."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")

        settings = Settings(model_path=str(model_path), backend="llama-cpp")

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)
            trace = backend._create_trace("test")
            assert trace is None

    def test_create_trace_enabled(self, tmp_path: Path) -> None:
        """Test _create_trace when tracing is enabled."""
        from inference.observability.tracing import InferenceTracer

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")

        settings = Settings(model_path=str(model_path), backend="llama-cpp")

        # Create a mock tracer that's enabled
        mock_tracer = MagicMock(spec=InferenceTracer)
        mock_tracer.enabled = True
        mock_trace = MagicMock()
        mock_tracer.trace_generation.return_value = mock_trace

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, mock_tracer)
            trace = backend._create_trace("test", user_id="user1", session_id="sess1")

            assert trace == mock_trace
            mock_tracer.trace_generation.assert_called_once_with(
                name="test",
                user_id="user1",
                session_id="sess1",
            )


class TestLlamaCppBackendModelLoading:
    """Tests for model loading edge cases."""

    def test_load_model_from_directory(self, tmp_path: Path) -> None:
        """Test loading model from directory containing .gguf file."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model_file = model_dir / "model.gguf"
        model_file.write_bytes(b"mock")

        settings = Settings(model_path=str(model_dir), backend="llama-cpp")

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)
            assert backend.model_name == "model"  # Stem of gguf file

    def test_load_model_multiple_gguf_files(self, tmp_path: Path) -> None:
        """Test loading when directory has multiple .gguf files."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "model_a.gguf").write_bytes(b"mock")
        (model_dir / "model_b.gguf").write_bytes(b"mock")

        settings = Settings(model_path=str(model_dir), backend="llama-cpp")

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            # Should use first file found
            backend = LlamaCppBackend(settings, None)
            assert backend.model_name in ["model_a", "model_b"]

    def test_load_model_empty_directory(self, tmp_path: Path) -> None:
        """Test loading from directory with no .gguf files."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        settings = Settings(model_path=str(model_dir), backend="llama-cpp")

        mock_llama_class = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            with pytest.raises(FileNotFoundError, match="No .gguf files found"):
                LlamaCppBackend(settings, None)

    def test_load_model_invalid_path(self, tmp_path: Path) -> None:
        """Test loading from path that is neither file nor directory."""
        settings = Settings(model_path="/nonexistent/path", backend="llama-cpp")

        mock_llama_class = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            with pytest.raises(FileNotFoundError, match="Model path does not exist"):
                LlamaCppBackend(settings, None)

    def test_custom_model_name(self, tmp_path: Path) -> None:
        """Test custom model name override."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"mock")

        settings = Settings(
            model_path=str(model_path),
            backend="llama-cpp",
            model_name="my-custom-model",
        )

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)
            assert backend.model_name == "my-custom-model"
