"""CPU-specific optimizations for inference."""

import logging
import os
from typing import Any

import torch
from transformers import PreTrainedModel

from ..config import Settings

logger = logging.getLogger(__name__)


def is_bitsandbytes_available() -> bool:
    """Check if bitsandbytes is available for int4 quantization."""
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


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

        # Enable Intel MKL optimizations if available
        os.environ.setdefault("MKL_DYNAMIC", "FALSE")

        # Disable gradient computation globally for inference
        torch.set_grad_enabled(False)

        logger.info("CPU environment configured")

    def optimize_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply CPU optimizations to model."""
        model.eval()

        # Apply quantization based on settings
        if self.settings.quantization == "int8":
            model = self._apply_int8_quantization(model)
        elif self.settings.quantization == "int4":
            model = self._apply_int4_quantization(model)

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

    def _apply_int4_quantization(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply int4 quantization using bitsandbytes or fallback."""
        logger.info("Applying int4 quantization")

        if is_bitsandbytes_available():
            try:
                # Note: int4 via bitsandbytes requires loading with quantization config
                # This is a post-hoc conversion which may have limitations
                logger.warning(
                    "int4 quantization is best applied during model loading. "
                    "Consider using INFERENCE_QUANTIZATION=int4 with model reload."
                )
                # Fallback to int8 for post-hoc quantization
                return self._apply_int8_quantization(model)
            except Exception as e:
                logger.warning(f"Failed to apply int4 quantization: {e}")
                return model
        else:
            logger.warning(
                "bitsandbytes not available for int4 quantization. "
                "Falling back to int8 quantization."
            )
            return self._apply_int8_quantization(model)

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


def get_quantization_config(quantization: str) -> dict[str, Any] | None:
    """Get quantization config for model loading.

    This should be passed to AutoModelForCausalLM.from_pretrained().
    """
    if quantization == "int4":
        if not is_bitsandbytes_available():
            logger.warning("bitsandbytes not available. Install with: pip install bitsandbytes")
            return None

        try:
            from transformers import BitsAndBytesConfig

            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,  # CPU compatible
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            }
        except Exception as e:
            logger.warning(f"Failed to create int4 config: {e}")
            return None

    elif quantization == "int8_load":
        # int8 via bitsandbytes at load time (different from dynamic int8)
        if not is_bitsandbytes_available():
            return None

        try:
            from transformers import BitsAndBytesConfig

            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            }
        except Exception as e:
            logger.warning(f"Failed to create int8 load config: {e}")
            return None

    return None
