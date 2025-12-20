"""Tests for the CPU optimizer module."""

import os
from unittest.mock import MagicMock, patch

from inference.config import Settings
from inference.engine.cpu_optimizer import CPUOptimizer


class TestCPUOptimizer:
    """Test CPUOptimizer class."""

    def test_init(self, mock_settings: Settings) -> None:
        """Test CPUOptimizer initialization."""
        optimizer = CPUOptimizer(mock_settings)

        assert optimizer.settings == mock_settings
        assert optimizer.num_threads == mock_settings.num_threads

    @patch.dict(os.environ, {}, clear=True)
    @patch("torch.set_num_threads")
    @patch("torch.set_grad_enabled")
    def test_setup_environment(
        self,
        mock_grad: MagicMock,
        mock_threads: MagicMock,
        mock_settings: Settings,
    ) -> None:
        """Test environment setup configures threads correctly."""
        mock_settings.num_threads = 8
        optimizer = CPUOptimizer(mock_settings)

        optimizer.setup_environment()

        mock_threads.assert_called_once_with(8)
        mock_grad.assert_called_once_with(False)

        # Check environment variables
        assert os.environ.get("MKL_NUM_THREADS") == "8"
        assert os.environ.get("OMP_NUM_THREADS") == "8"
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "8"

    @patch.dict(os.environ, {"MKL_NUM_THREADS": "16"}, clear=False)
    @patch("torch.set_num_threads")
    @patch("torch.set_grad_enabled")
    def test_setup_environment_preserves_existing_env(
        self,
        mock_grad: MagicMock,
        mock_threads: MagicMock,
        mock_settings: Settings,
    ) -> None:
        """Test that existing environment variables are preserved."""
        mock_settings.num_threads = 8
        optimizer = CPUOptimizer(mock_settings)

        optimizer.setup_environment()

        # Existing value should be preserved (setdefault behavior)
        assert os.environ.get("MKL_NUM_THREADS") == "16"

    def test_optimize_model_eval_mode(self, mock_settings: Settings, mock_model: MagicMock) -> None:
        """Test that optimize_model puts model in eval mode."""
        optimizer = CPUOptimizer(mock_settings)

        result = optimizer.optimize_model(mock_model)

        mock_model.eval.assert_called()
        assert result == mock_model

    def test_optimize_model_no_quantization(
        self, mock_settings: Settings, mock_model: MagicMock
    ) -> None:
        """Test that no quantization is applied when disabled."""
        mock_settings.quantization = "none"
        optimizer = CPUOptimizer(mock_settings)

        with patch.object(optimizer, "_apply_int8_quantization") as mock_quant:
            optimizer.optimize_model(mock_model)
            mock_quant.assert_not_called()

    def test_optimize_model_with_int8_quantization(
        self, mock_settings: Settings, mock_model: MagicMock
    ) -> None:
        """Test that int8 quantization is applied when enabled."""
        mock_settings.quantization = "int8"
        optimizer = CPUOptimizer(mock_settings)

        with patch("torch.quantization.quantize_dynamic", return_value=mock_model) as mock_quant:
            result = optimizer.optimize_model(mock_model)

            mock_quant.assert_called_once()
            assert result == mock_model

    def test_optimize_model_no_torch_compile(
        self, mock_settings: Settings, mock_model: MagicMock
    ) -> None:
        """Test that torch.compile is not applied when disabled."""
        mock_settings.enable_torch_compile = False
        optimizer = CPUOptimizer(mock_settings)

        with patch.object(optimizer, "_apply_torch_compile") as mock_compile:
            optimizer.optimize_model(mock_model)
            mock_compile.assert_not_called()

    def test_optimize_model_with_torch_compile(
        self, mock_settings: Settings, mock_model: MagicMock
    ) -> None:
        """Test that torch.compile is applied when enabled."""
        mock_settings.enable_torch_compile = True
        optimizer = CPUOptimizer(mock_settings)

        with patch("torch.compile", return_value=mock_model) as mock_compile:
            result = optimizer.optimize_model(mock_model)

            mock_compile.assert_called_once_with(
                mock_model,
                backend="inductor",
                mode="reduce-overhead",
            )
            assert result == mock_model

    def test_apply_int8_quantization_handles_error(
        self,
        mock_settings: Settings,
        mock_model: MagicMock,
    ) -> None:
        """Test that quantization errors are handled gracefully."""
        mock_settings.quantization = "int8"
        optimizer = CPUOptimizer(mock_settings)

        err = Exception("Quantization failed")
        with patch("torch.quantization.quantize_dynamic", side_effect=err):
            # Should not raise, just log warning
            result = optimizer._apply_int8_quantization(mock_model)
            assert result == mock_model

    def test_apply_torch_compile_handles_error(
        self,
        mock_settings: Settings,
        mock_model: MagicMock,
    ) -> None:
        """Test that torch.compile errors are handled gracefully."""
        mock_settings.enable_torch_compile = True
        optimizer = CPUOptimizer(mock_settings)

        with patch("torch.compile", side_effect=Exception("Compile failed")):
            # Should not raise, just log warning
            result = optimizer._apply_torch_compile(mock_model)
            assert result == mock_model
