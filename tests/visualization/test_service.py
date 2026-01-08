"""
Tests for visualization service functions.

Unit tests for service layer functions including helper functions
and edge cases not covered by router tests.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.visualization.service import (
    get_emotion_color_map,
    get_available_features,
    _compute_fft_features_fast,
)
from src.visualization.constants import EMOTION_COLORS


class TestGetEmotionColorMap:
    """Tests for get_emotion_color_map function."""

    def test_basic_color_mapping(self):
        """Test basic emotion to color mapping."""
        emotions = ["Anger", "Fear", "Sadness"]
        result = get_emotion_color_map(emotions)

        assert len(result) == 3
        for mapping in result:
            assert hasattr(mapping, "emotion")
            assert hasattr(mapping, "color")
            assert mapping.color.startswith("#")

    def test_empty_emotions_list(self):
        """Test with empty emotions list."""
        result = get_emotion_color_map([])
        assert result == []

    def test_single_emotion(self):
        """Test with single emotion."""
        result = get_emotion_color_map(["Baseline"])
        assert len(result) == 1
        assert result[0].emotion == "Baseline"
        assert result[0].color.startswith("#")

    def test_many_emotions_color_cycling(self):
        """Test that colors cycle when more emotions than colors."""
        # Create more emotions than available colors
        many_emotions = [f"Emotion_{i}" for i in range(len(EMOTION_COLORS) + 5)]
        result = get_emotion_color_map(many_emotions)

        assert len(result) == len(many_emotions)
        # Colors should cycle
        for i, mapping in enumerate(result):
            expected_color = EMOTION_COLORS[i % len(EMOTION_COLORS)]
            assert mapping.color == expected_color

    def test_emotion_order_preserved(self):
        """Test that emotion order is preserved."""
        emotions = ["Z_last", "A_first", "M_middle"]
        result = get_emotion_color_map(emotions)

        result_emotions = [m.emotion for m in result]
        assert result_emotions == emotions


class TestGetAvailableFeatures:
    """Tests for get_available_features function."""

    def test_study1_has_features(self):
        """Test that study 1 returns features."""
        features = get_available_features(1)
        assert isinstance(features, list)
        assert len(features) > 0

    def test_study1_common_features(self):
        """Test that study 1 has common physiological features."""
        features = get_available_features(1)
        # Should have at least some of these
        common = ["ECG", "EDA", "SBP", "DBP", "respiration", "temp"]
        found = [f for f in common if f in features]
        assert len(found) >= 2, f"Expected common features, got: {features}"

    @pytest.mark.parametrize("study_number", [1, 2, 3, 4, 5, 6, 7])
    def test_all_studies_have_features(self, study_number: int):
        """Test that all valid studies return features."""
        features = get_available_features(study_number)
        assert len(features) > 0

    def test_invalid_study_raises(self):
        """Test that invalid study number raises exception."""
        with pytest.raises(Exception):  # DuckDB will raise on missing table
            get_available_features(99)


class TestComputeFFTFeaturesFast:
    """Tests for _compute_fft_features_fast helper function."""

    def test_basic_sinusoid(self):
        """Test FFT on known sinusoidal signal."""
        fs = 1000.0
        t = np.linspace(0, 1, int(fs))
        freq = 10.0
        signal = np.sin(2 * np.pi * freq * t)

        result = _compute_fft_features_fast(signal, sampling_interval=1/fs)

        assert "dominant_frequency" in result
        assert "signal_energy" in result
        # Dominant frequency should be near 10 Hz
        assert abs(result["dominant_frequency"] - 10.0) < 2.0

    def test_signal_energy_positive(self):
        """Test that signal energy is computed."""
        signal = np.random.randn(1000)
        result = _compute_fft_features_fast(signal, sampling_interval=0.001)

        assert result["signal_energy"] != 0.0

    def test_higher_amplitude_higher_energy(self):
        """Test that higher amplitude produces higher energy."""
        fs = 1000.0
        t = np.linspace(0, 1, int(fs))

        signal_low = np.sin(2 * np.pi * 10 * t)
        signal_high = 5 * np.sin(2 * np.pi * 10 * t)

        result_low = _compute_fft_features_fast(signal_low, 1/fs)
        result_high = _compute_fft_features_fast(signal_high, 1/fs)

        assert result_high["signal_energy"] > result_low["signal_energy"]

    def test_very_short_signal(self):
        """Test FFT with very short signal."""
        signal = np.array([1.0, 2.0, 3.0])
        result = _compute_fft_features_fast(signal, sampling_interval=0.001)

        # Should return default/zero values
        assert result["dominant_frequency"] == 0.0
        assert result["signal_energy"] == 0.0

    def test_constant_signal(self):
        """Test FFT with constant signal (DC only)."""
        signal = np.ones(1000) * 5.0
        result = _compute_fft_features_fast(signal, sampling_interval=0.001)

        # Dominant frequency detection depends on peak finding algorithm
        # DC component is at freq 0, but peak finder may detect elsewhere
        assert isinstance(result["dominant_frequency"], float)
        assert result["signal_energy"] != 0.0  # Should have some energy    def test_nan_handling(self):
        """Test FFT handles NaN values."""
        signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 200)
        # nan_to_num is used inside the function
        result = _compute_fft_features_fast(signal, sampling_interval=0.001)

        assert not np.isnan(result["dominant_frequency"])
        assert not np.isnan(result["signal_energy"])

    def test_multiple_frequencies(self):
        """Test FFT with multiple frequency components."""
        fs = 1000.0
        t = np.linspace(0, 1, int(fs))
        # 5 Hz dominant (amplitude 3), 20 Hz secondary (amplitude 1)
        signal = 3 * np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)

        result = _compute_fft_features_fast(signal, 1/fs)

        # 5 Hz should be dominant
        assert abs(result["dominant_frequency"] - 5.0) < 2.0


class TestServiceEdgeCases:
    """Tests for edge cases in service functions."""

    def test_empty_array_fft(self):
        """Test FFT with empty array."""
        signal = np.array([])
        result = _compute_fft_features_fast(signal, sampling_interval=0.001)

        assert result["dominant_frequency"] == 0.0
        assert result["signal_energy"] == 0.0

    def test_single_sample_fft(self):
        """Test FFT with single sample."""
        signal = np.array([42.0])
        result = _compute_fft_features_fast(signal, sampling_interval=0.001)

        assert result["dominant_frequency"] == 0.0
        assert result["signal_energy"] == 0.0

    def test_two_samples_fft(self):
        """Test FFT with two samples."""
        signal = np.array([1.0, 2.0])
        result = _compute_fft_features_fast(signal, sampling_interval=0.001)

        assert result["dominant_frequency"] == 0.0
        assert result["signal_energy"] == 0.0

    def test_inf_values_fft(self):
        """Test FFT with infinity values."""
        signal = np.array([1.0, np.inf, 3.0, 4.0, 5.0] * 200)
        result = _compute_fft_features_fast(signal, sampling_interval=0.001)

        # Should handle inf (nan_to_num converts to large finite number)
        assert np.isfinite(result["dominant_frequency"])

    def test_negative_sampling_interval(self):
        """Test FFT with negative sampling interval (edge case)."""
        signal = np.sin(np.linspace(0, 10, 1000))
        # Should not crash, but may produce unusual results
        result = _compute_fft_features_fast(signal, sampling_interval=-0.001)
        assert "dominant_frequency" in result
        assert "signal_energy" in result


class TestColorMapIntegration:
    """Integration tests for color mapping."""

    def test_all_emotion_colors_valid_hex(self):
        """Test that all emotion colors are valid hex colors."""
        for color in EMOTION_COLORS:
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB format
            # Should be valid hex
            int(color[1:], 16)

    def test_enough_colors_for_common_emotions(self):
        """Test that there are enough colors for typical number of emotions."""
        # POPANE typically has up to 15 distinct emotions
        assert len(EMOTION_COLORS) >= 10
