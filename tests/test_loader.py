"""Tests for the model loader module."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from inference.config import Settings
from inference.models.loader import (
    ModelLoader,
    ModelLoadError,
    ModelTooLargeError,
    estimate_parameters,
    get_torch_dtype,
    resolve_device,
)


class TestEstimateParameters:
    """Test estimate_parameters function."""

    def test_estimates_llama_style_model(self) -> None:
        """Test parameter estimation for LLaMA-style models."""
        config = MagicMock()
        config.hidden_size = 2048
        config.num_hidden_layers = 22
        config.vocab_size = 32000
        config.intermediate_size = 5632

        params = estimate_parameters(config)
        # Should be roughly 1.1B for TinyLlama
        assert params > 0
        assert params < 2_000_000_000  # Less than 2B

    def test_returns_zero_for_missing_attributes(self) -> None:
        """Test that missing config attributes return 0."""
        config = MagicMock()
        config.hidden_size = 0
        config.num_hidden_layers = 0
        config.vocab_size = 0

        params = estimate_parameters(config)
        assert params == 0

    def test_uses_default_intermediate_size(self) -> None:
        """Test that intermediate_size defaults to 4x hidden_size."""
        config = MagicMock()
        config.hidden_size = 256
        config.num_hidden_layers = 4
        config.vocab_size = 1000
        # Remove intermediate_size to trigger default
        del config.intermediate_size

        # Should still work with default
        params = estimate_parameters(config)
        assert params > 0


class TestGetTorchDtype:
    """Test get_torch_dtype function."""

    def test_cpu_returns_float32(self) -> None:
        """Test that CPU uses float32."""
        assert get_torch_dtype("cpu") == torch.float32

    def test_cuda_returns_float16(self) -> None:
        """Test that CUDA uses float16."""
        assert get_torch_dtype("cuda") == torch.float16

    def test_mps_returns_float16(self) -> None:
        """Test that MPS uses float16."""
        assert get_torch_dtype("mps") == torch.float16


class TestResolveDevice:
    """Test resolve_device function."""

    def test_explicit_cpu(self) -> None:
        """Test that explicit cpu setting returns cpu."""
        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self) -> None:
        """Test that explicit cuda setting returns cuda."""
        assert resolve_device("cuda") == "cuda"

    def test_explicit_mps(self) -> None:
        """Test that explicit mps setting returns mps."""
        assert resolve_device("mps") == "mps"

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_with_cuda_available(self, mock_cuda: MagicMock) -> None:
        """Test auto selects cuda when available."""
        assert resolve_device("auto") == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_auto_with_mps_available(self, mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
        """Test auto selects mps when cuda unavailable but mps available."""
        assert resolve_device("auto") == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
        """Test auto falls back to cpu when no GPU available."""
        assert resolve_device("auto") == "cpu"


class TestModelLoader:
    """Test ModelLoader class."""

    def test_init(self, mock_settings: Settings) -> None:
        """Test ModelLoader initialization."""
        with patch("inference.models.loader.resolve_device", return_value="cpu"):
            loader = ModelLoader(mock_settings)

        assert loader.settings == mock_settings
        assert loader.model_path == Path(mock_settings.model_path)
        assert loader.device == "cpu"

    def test_validate_model_path_nonexistent(self, mock_settings: Settings) -> None:
        """Test validation fails for nonexistent path."""
        mock_settings.model_path = "/nonexistent/path"
        loader = ModelLoader(mock_settings)

        with pytest.raises(ModelLoadError, match="does not exist"):
            loader.validate_model_path()

    def test_validate_model_path_missing_config(self, tmp_path: Any) -> None:
        """Test validation fails when config.json is missing."""
        settings = Settings(model_path=str(tmp_path))
        loader = ModelLoader(settings)

        with pytest.raises(ModelLoadError, match="No config.json found"):
            loader.validate_model_path()

    def test_validate_model_path_success(self, temp_model_dir: Path) -> None:
        """Test validation succeeds with valid model directory."""
        settings = Settings(model_path=str(temp_model_dir))
        loader = ModelLoader(settings)

        # Should not raise
        loader.validate_model_path()

    def test_validate_model_size_within_limit(
        self, mock_model_config: MagicMock, mock_settings: Settings
    ) -> None:
        """Test validation passes for small models."""
        loader = ModelLoader(mock_settings)
        # Should not raise
        loader.validate_model_size(mock_model_config)

    def test_validate_model_size_exceeds_limit(self, mock_settings: Settings) -> None:
        """Test validation fails for models exceeding limit."""
        mock_settings.max_parameters = 1000  # Very small limit

        config = MagicMock()
        config.hidden_size = 2048
        config.num_hidden_layers = 32
        config.vocab_size = 128000
        config.intermediate_size = 8192

        loader = ModelLoader(mock_settings)

        with pytest.raises(ModelTooLargeError, match="exceeds limit"):
            loader.validate_model_size(config)

    @patch("inference.models.loader.AutoConfig.from_pretrained")
    @patch("inference.models.loader.AutoModelForCausalLM.from_pretrained")
    @patch("inference.models.loader.AutoTokenizer.from_pretrained")
    def test_load_success(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_config: MagicMock,
        temp_model_dir: Path,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model_config: MagicMock,
    ) -> None:
        """Test successful model loading."""
        mock_auto_config.return_value = mock_model_config
        mock_auto_model.return_value = mock_model
        mock_auto_tokenizer.return_value = mock_tokenizer

        settings = Settings(model_path=str(temp_model_dir), device="cpu")
        loader = ModelLoader(settings)

        model, tokenizer = loader.load()

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once_with("cpu")

    @patch("inference.models.loader.AutoConfig.from_pretrained")
    @patch("inference.models.loader.AutoModelForCausalLM.from_pretrained")
    @patch("inference.models.loader.AutoTokenizer.from_pretrained")
    def test_load_sets_pad_token(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_config: MagicMock,
        temp_model_dir: Path,
        mock_model: MagicMock,
        mock_model_config: MagicMock,
    ) -> None:
        """Test that pad_token is set to eos_token if missing."""
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"

        mock_auto_config.return_value = mock_model_config
        mock_auto_model.return_value = mock_model
        mock_auto_tokenizer.return_value = tokenizer

        settings = Settings(model_path=str(temp_model_dir), device="cpu")
        loader = ModelLoader(settings)

        _, loaded_tokenizer = loader.load()

        assert loaded_tokenizer.pad_token == "</s>"

    @patch("inference.models.loader.AutoConfig.from_pretrained")
    @patch("inference.models.loader.AutoModelForCausalLM.from_pretrained")
    @patch("inference.models.loader.AutoTokenizer.from_pretrained")
    @patch("inference.models.loader.resolve_device", return_value="cuda")
    def test_load_with_cuda_uses_device_map(
        self,
        mock_resolve: MagicMock,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_config: MagicMock,
        temp_model_dir: Path,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model_config: MagicMock,
    ) -> None:
        """Test that CUDA loading uses device_map=auto."""
        mock_auto_config.return_value = mock_model_config
        mock_auto_model.return_value = mock_model
        mock_auto_tokenizer.return_value = mock_tokenizer

        settings = Settings(model_path=str(temp_model_dir), device="cuda")
        loader = ModelLoader(settings)

        loader.load()

        # Check that device_map was passed
        call_kwargs = mock_auto_model.call_args.kwargs
        assert call_kwargs.get("device_map") == "auto"
