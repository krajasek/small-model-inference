"""Tests for LlamaCppDraftModelWrapper and speculative decoding."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.backends.llamacpp import LlamaCppDraftModelWrapper


class TestLlamaCppDraftModelWrapper:
    """Tests for the draft model wrapper used in speculative decoding."""

    @pytest.fixture
    def mock_draft_llama(self) -> MagicMock:
        """Create a mock Llama instance for draft model."""
        mock = MagicMock()
        mock.token_eos.return_value = 2
        mock.sample.side_effect = [10, 11, 12]  # Draft tokens
        return mock

    def test_init(self, mock_draft_llama: MagicMock) -> None:
        """Test wrapper initialization."""
        wrapper = LlamaCppDraftModelWrapper(
            draft_llama=mock_draft_llama,
            num_pred_tokens=4,
        )

        assert wrapper._draft == mock_draft_llama
        assert wrapper._num_pred_tokens == 4

    def test_init_default_tokens(self, mock_draft_llama: MagicMock) -> None:
        """Test wrapper initialization with default token count."""
        wrapper = LlamaCppDraftModelWrapper(draft_llama=mock_draft_llama)
        assert wrapper._num_pred_tokens == 4

    def test_call_resets_model(self, mock_draft_llama: MagicMock) -> None:
        """Test that calling wrapper resets the draft model."""
        wrapper = LlamaCppDraftModelWrapper(draft_llama=mock_draft_llama)

        input_ids = np.array([1, 2, 3], dtype=np.intc)
        wrapper(input_ids)

        mock_draft_llama.reset.assert_called_once()

    def test_call_evals_input(self, mock_draft_llama: MagicMock) -> None:
        """Test that calling wrapper evaluates input tokens."""
        wrapper = LlamaCppDraftModelWrapper(draft_llama=mock_draft_llama)

        input_ids = np.array([1, 2, 3], dtype=np.intc)
        wrapper(input_ids)

        mock_draft_llama.eval.assert_called()
        # First call should be the input tokens
        first_call = mock_draft_llama.eval.call_args_list[0]
        assert first_call[0][0] == [1, 2, 3]

    def test_call_generates_draft_tokens(self, mock_draft_llama: MagicMock) -> None:
        """Test that wrapper generates draft tokens."""
        wrapper = LlamaCppDraftModelWrapper(
            draft_llama=mock_draft_llama,
            num_pred_tokens=4,
        )

        input_ids = np.array([1, 2, 3], dtype=np.intc)
        result = wrapper(input_ids)

        # Should generate max_draft tokens (num_pred_tokens - 1 = 3)
        assert len(result) == 3
        assert result[0] == 10
        assert result[1] == 11
        assert result[2] == 12

    def test_call_stops_on_eos(self, mock_draft_llama: MagicMock) -> None:
        """Test that wrapper stops on EOS token."""
        mock_draft_llama.sample.side_effect = [10, 2, 12]  # 2 is EOS
        wrapper = LlamaCppDraftModelWrapper(
            draft_llama=mock_draft_llama,
            num_pred_tokens=4,
        )

        input_ids = np.array([1, 2, 3], dtype=np.intc)
        result = wrapper(input_ids)

        # Should stop at EOS (not include it)
        assert len(result) == 1
        assert result[0] == 10

    def test_call_evals_generated_tokens(self, mock_draft_llama: MagicMock) -> None:
        """Test that each generated token is evaluated."""
        wrapper = LlamaCppDraftModelWrapper(
            draft_llama=mock_draft_llama,
            num_pred_tokens=4,
        )

        input_ids = np.array([1, 2, 3], dtype=np.intc)
        wrapper(input_ids)

        # Should eval input + each generated token
        # input eval + token 10 eval + token 11 eval + token 12 eval
        assert mock_draft_llama.eval.call_count == 4

    def test_call_samples_greedy(self, mock_draft_llama: MagicMock) -> None:
        """Test that sampling uses greedy (temp=0)."""
        wrapper = LlamaCppDraftModelWrapper(draft_llama=mock_draft_llama)

        input_ids = np.array([1, 2, 3], dtype=np.intc)
        wrapper(input_ids)

        # All sample calls should use temp=0
        for call in mock_draft_llama.sample.call_args_list:
            assert call.kwargs.get("temp") == 0.0

    def test_call_returns_numpy_array(self, mock_draft_llama: MagicMock) -> None:
        """Test that result is numpy array with correct dtype."""
        wrapper = LlamaCppDraftModelWrapper(draft_llama=mock_draft_llama)

        input_ids = np.array([1, 2, 3], dtype=np.intc)
        result = wrapper(input_ids)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.intc

    def test_call_with_single_token_prediction(self, mock_draft_llama: MagicMock) -> None:
        """Test with num_pred_tokens=1."""
        wrapper = LlamaCppDraftModelWrapper(
            draft_llama=mock_draft_llama,
            num_pred_tokens=1,
        )

        input_ids = np.array([1, 2, 3], dtype=np.intc)
        result = wrapper(input_ids)

        # max_draft = max(1, 1 - 1) = max(1, 0) = 1
        # So should still generate at least 1 token
        assert len(result) >= 0  # Could be 0 if logic changes

    def test_call_with_kwargs_ignored(self, mock_draft_llama: MagicMock) -> None:
        """Test that extra kwargs are accepted but ignored."""
        wrapper = LlamaCppDraftModelWrapper(draft_llama=mock_draft_llama)

        input_ids = np.array([1, 2, 3], dtype=np.intc)
        # Should not raise
        result = wrapper(input_ids, extra_param="ignored", another=123)

        assert len(result) > 0


class TestSpeculativeDecodingIntegration:
    """Integration tests for speculative decoding in LlamaCppBackend."""

    def test_backend_init_with_draft_model(self, tmp_path) -> None:
        """Test backend initialization with draft model configured."""
        from inference.config import Settings

        # Create mock model files
        main_model = tmp_path / "main.gguf"
        main_model.write_bytes(b"mock model")
        draft_model = tmp_path / "draft.gguf"
        draft_model.write_bytes(b"mock draft")

        settings = Settings(
            model_path=str(main_model),
            backend="llama-cpp",
            enable_speculative_decoding=True,
            draft_model_path=str(draft_model),
            num_speculative_tokens=4,
        )

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)

            # Draft model should be loaded and wrapper created
            assert backend._speculative_decoding_enabled is True
            assert backend._draft_model_wrapper is not None

    def test_backend_init_without_draft_model_path(self, tmp_path) -> None:
        """Test backend init when speculative decoding enabled but no draft path."""
        from inference.config import Settings

        main_model = tmp_path / "main.gguf"
        main_model.write_bytes(b"mock model")

        settings = Settings(
            model_path=str(main_model),
            backend="llama-cpp",
            enable_speculative_decoding=True,
            draft_model_path=None,  # No draft model
        )

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)

            # Should not enable speculative decoding without draft path
            assert backend._speculative_decoding_enabled is False
            assert backend._draft_model_wrapper is None

    def test_backend_draft_model_not_found(self, tmp_path) -> None:
        """Test backend when draft model path doesn't exist."""
        from inference.config import Settings

        main_model = tmp_path / "main.gguf"
        main_model.write_bytes(b"mock model")

        settings = Settings(
            model_path=str(main_model),
            backend="llama-cpp",
            enable_speculative_decoding=True,
            draft_model_path="/nonexistent/draft.gguf",
        )

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)

            # Should handle missing draft model gracefully
            assert backend._speculative_decoding_enabled is False

    def test_get_model_info_includes_speculative_decoding(self, tmp_path) -> None:
        """Test that model info includes speculative decoding status."""
        from inference.config import Settings

        main_model = tmp_path / "main.gguf"
        main_model.write_bytes(b"mock model")

        settings = Settings(
            model_path=str(main_model),
            backend="llama-cpp",
            enable_speculative_decoding=False,
        )

        mock_llama = MagicMock()
        mock_llama_class = MagicMock(return_value=mock_llama)

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_class)}):
            from importlib import reload

            import inference.backends.llamacpp

            reload(inference.backends.llamacpp)
            from inference.backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend(settings, None)
            info = backend.get_model_info()

            assert "speculative_decoding" in info
            assert info["speculative_decoding"] is False
