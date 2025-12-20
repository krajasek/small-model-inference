"""CPU-specific optimizations for inference."""

import logging
import os

import torch
from transformers import PreTrainedModel

from ..config import Settings

logger = logging.getLogger(__name__)


class CPUOptimizer:
    """Applies CPU-specific optimizations for inference."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.num_threads = settings.num_threads

    def setup_environment(self) -> None:
        """Configure environment for optimal CPU inference."""
        logger.info(f"Setting up CPU environment with {self.num_threads} threads")

        # Set PyTorch thread count
        torch.set_num_threads(self.num_threads)

        # Set environment variables for Intel MKL and OpenMP
        os.environ.setdefault("MKL_NUM_THREADS", str(self.num_threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(self.num_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(self.num_threads))

        # Disable gradient computation globally for inference
        torch.set_grad_enabled(False)

        logger.info("CPU environment configured")

    def optimize_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply CPU optimizations to model."""
        model.eval()

        # Apply dynamic int8 quantization if enabled
        if self.settings.quantization == "int8":
            model = self._apply_int8_quantization(model)

        # Apply torch.compile if enabled (PyTorch 2.x)
        if self.settings.enable_torch_compile:
            model = self._apply_torch_compile(model)

        return model

    def _apply_int8_quantization(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply dynamic int8 quantization for CPU."""
        logger.info("Applying dynamic int8 quantization")
        try:
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            logger.info("Int8 quantization applied successfully")
        except Exception as e:
            logger.warning(f"Failed to apply int8 quantization: {e}")
        return model

    def _apply_torch_compile(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply torch.compile optimization."""
        logger.info("Applying torch.compile with inductor backend")
        try:
            model = torch.compile(
                model,
                backend="inductor",
                mode="reduce-overhead",
            )
            logger.info("torch.compile applied successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed, using eager mode: {e}")
        return model
