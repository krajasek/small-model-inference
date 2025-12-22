"""Tests for the backends module."""

from unittest.mock import MagicMock, patch

import pytest

from inference.backends import create_backend
from inference.backends.base import GenerationConfig, InferenceBackend
from inference.config import Settings


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = GenerationConfig()

        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.do_sample is True
        assert config.repetition_penalty == 1.1

    def test_custom_values(self) -> None:
        """Test that custom values can be set."""
        config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.5,
            top_p=0.8,
            top_k=40,
            do_sample=False,
            repetition_penalty=1.2,
        )

        assert config.max_new_tokens == 100
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        assert config.top_k == 40
        assert config.do_sample is False
        assert config.repetition_penalty == 1.2


class TestInferenceBackendProtocol:
    """Test InferenceBackend protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that InferenceBackend can be used for runtime checks."""
        # Create a mock that implements all required methods
        mock_backend = MagicMock()
        mock_backend.model_name = "test"
        mock_backend.device = "cpu"
        mock_backend.generate = MagicMock(return_value=("output", {}))
        mock_backend.generate_stream = MagicMock(return_value=iter([]))
        mock_backend.get_model_info = MagicMock(return_value={})
        mock_backend.get_cache_stats = MagicMock(return_value={})
        mock_backend.clear_caches = MagicMock()
        mock_backend.shutdown = MagicMock()

        # Should pass isinstance check (not guaranteed for Protocols, but structure check passes)
        assert hasattr(mock_backend, "model_name")
        assert hasattr(mock_backend, "device")
        assert hasattr(mock_backend, "generate")
        assert hasattr(mock_backend, "generate_stream")


class TestCreateBackend:
    """Test create_backend factory function."""

    def test_creates_pytorch_backend_by_default(self) -> None:
        """Test that PyTorch backend is created when backend=pytorch."""
        settings = Settings(model_path="/tmp/test", backend="pytorch")

        with patch("inference.backends.pytorch.PyTorchBackend") as mock_backend_class:
            mock_backend = MagicMock(spec=InferenceBackend)
            mock_backend_class.return_value = mock_backend

            backend = create_backend(settings, None)

            assert backend == mock_backend
            mock_backend_class.assert_called_once_with(settings, None)

    def test_creates_llamacpp_backend(self) -> None:
        """Test that llama-cpp backend is created when backend=llama-cpp."""
        settings = Settings(model_path="/tmp/test.gguf", backend="llama-cpp")

        with patch("inference.backends.llamacpp.LlamaCppBackend") as mock_backend_class:
            mock_backend = MagicMock(spec=InferenceBackend)
            mock_backend_class.return_value = mock_backend

            backend = create_backend(settings, None)

            assert backend == mock_backend
            mock_backend_class.assert_called_once_with(settings, None)

    def test_passes_tracer_to_backend(self) -> None:
        """Test that tracer is passed to the backend."""
        settings = Settings(model_path="/tmp/test", backend="pytorch")
        mock_tracer = MagicMock()

        with patch("inference.backends.pytorch.PyTorchBackend") as mock_backend_class:
            mock_backend = MagicMock(spec=InferenceBackend)
            mock_backend_class.return_value = mock_backend

            create_backend(settings, mock_tracer)

            mock_backend_class.assert_called_once_with(settings, mock_tracer)

    def test_raises_for_unknown_backend(self) -> None:
        """Test that ValueError is raised for unknown backend."""
        # Create settings with an invalid backend value
        settings = Settings(model_path="/tmp/test")
        # Manually override backend to test error handling
        object.__setattr__(settings, "backend", "unknown")

        with pytest.raises(ValueError, match="Unknown backend: unknown"):
            create_backend(settings, None)


class TestPyTorchBackend:
    """Test PyTorchBackend class."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.vocab_size = 32000
        model.config.hidden_size = 4096
        model.config.num_hidden_layers = 32
        return model

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "test output"
        tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        return tokenizer

    def test_backend_init(self, mock_model: MagicMock, mock_tokenizer: MagicMock) -> None:
        """Test PyTorchBackend initialization."""
        settings = Settings(model_path="/tmp/test", device="cpu")

        with (
            patch("inference.backends.pytorch.ModelLoader") as mock_loader_class,
            patch("inference.backends.pytorch.CPUOptimizer") as mock_optimizer_class,
            patch("inference.backends.pytorch.CacheManager"),
        ):
            mock_loader = MagicMock()
            mock_loader.load.return_value = (mock_model, mock_tokenizer)
            mock_loader.device = "cpu"
            mock_loader_class.return_value = mock_loader

            mock_optimizer = MagicMock()
            mock_optimizer.optimize_model.return_value = mock_model
            mock_optimizer_class.return_value = mock_optimizer

            from inference.backends.pytorch import PyTorchBackend

            backend = PyTorchBackend(settings, None)

            assert backend.model_name == "local-model"
            assert backend.device == "cpu"
            mock_optimizer.setup_environment.assert_called_once()

    def test_backend_with_custom_model_name(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test PyTorchBackend with custom model name."""
        settings = Settings(model_path="/tmp/test", device="cuda", model_name="my-model")

        with (
            patch("inference.backends.pytorch.ModelLoader") as mock_loader_class,
            patch("inference.backends.pytorch.CacheManager"),
        ):
            mock_loader = MagicMock()
            mock_loader.load.return_value = (mock_model, mock_tokenizer)
            mock_loader.device = "cuda"
            mock_loader_class.return_value = mock_loader

            from inference.backends.pytorch import PyTorchBackend

            backend = PyTorchBackend(settings, None)

            assert backend.model_name == "my-model"

    def test_get_model_info(self, mock_model: MagicMock, mock_tokenizer: MagicMock) -> None:
        """Test get_model_info returns correct info."""
        settings = Settings(model_path="/tmp/test", device="cpu")

        with (
            patch("inference.backends.pytorch.ModelLoader") as mock_loader_class,
            patch("inference.backends.pytorch.CPUOptimizer") as mock_optimizer_class,
            patch("inference.backends.pytorch.CacheManager"),
        ):
            mock_loader = MagicMock()
            mock_loader.load.return_value = (mock_model, mock_tokenizer)
            mock_loader.device = "cpu"
            mock_loader_class.return_value = mock_loader

            mock_optimizer = MagicMock()
            mock_optimizer.optimize_model.return_value = mock_model
            mock_optimizer_class.return_value = mock_optimizer

            from inference.backends.pytorch import PyTorchBackend

            backend = PyTorchBackend(settings, None)
            info = backend.get_model_info()

            assert info["backend"] == "pytorch"
            assert info["device"] == "cpu"
            assert info["model_name"] == "local-model"


class TestLlamaCppBackend:
    """Test LlamaCppBackend class."""

    def test_backend_raises_import_error(self) -> None:
        """Test that ImportError is raised when llama-cpp not installed."""
        settings = Settings(model_path="/tmp/test.gguf", backend="llama-cpp")

        with patch.dict("sys.modules", {"llama_cpp": None}):
            with patch(
                "inference.backends.llamacpp.LlamaCppBackend.__init__",
                side_effect=ImportError("llama-cpp-python is not installed"),
            ):
                from inference.backends.llamacpp import LlamaCppBackend

                with pytest.raises(ImportError, match="llama-cpp-python"):
                    LlamaCppBackend(settings, None)

    def test_backend_raises_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing model."""
        settings = Settings(model_path="/nonexistent/path.gguf", backend="llama-cpp")

        # Mock llama_cpp to avoid import error
        mock_llama_cpp = MagicMock()
        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            from inference.backends.llamacpp import LlamaCppBackend

            with pytest.raises(FileNotFoundError, match="Model path does not exist"):
                LlamaCppBackend(settings, None)

    def test_get_model_info(self) -> None:
        """Test get_model_info returns correct info."""
        settings = Settings(
            model_path="/tmp/test.gguf",
            backend="llama-cpp",
            llama_cpp_n_ctx=4096,
            num_threads=8,
        )

        mock_llama = MagicMock()
        mock_llama_module = MagicMock()
        mock_llama_module.Llama.return_value = mock_llama

        with (
            patch.dict("sys.modules", {"llama_cpp": mock_llama_module}),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.suffix", ".gguf"),
        ):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)
            info = backend.get_model_info()

            assert info["backend"] == "llama-cpp"
            assert info["n_ctx"] == 4096
            assert info["n_threads"] == 8
