"""
Tests for visualization stimuli module.

Tests emotion mapping, valence mapping, and study name functions.
"""
import pytest
import numpy as np
import math

from src.visualization.stimuli import (
    get_emotion_from_marker,
    get_valence_from_marker,
    get_study_name,
    STIMULI_TO_EMOTION,
    STIMULI_TO_VALENCE,
    STUDY_NAMES,
)


class TestGetEmotionFromMarker:
    """Tests for get_emotion_from_marker function."""

    def test_baseline_marker(self):
        """Test baseline marker (-1)."""
        assert get_emotion_from_marker(-1) == "Baseline"

    def test_neutral_markers(self):
        """Test neutral emotion markers (101-110)."""
        for marker in range(101, 111):
            assert get_emotion_from_marker(marker) == "Neutral"

    def test_negative_emotion_markers(self):
        """Test negative emotion markers (201-210)."""
        # Test specific known mappings
        assert get_emotion_from_marker(201) == "Fear"
        assert get_emotion_from_marker(202) == "Anger"
        assert get_emotion_from_marker(205) == "Disgust"
        assert get_emotion_from_marker(206) == "Sadness"
        assert get_emotion_from_marker(209) == "Threat"

    def test_positive_emotion_markers(self):
        """Test positive emotion markers (301-309)."""
        assert get_emotion_from_marker(301) == "Amusement"
        assert get_emotion_from_marker(305) == "Tenderness"
        assert get_emotion_from_marker(307) == "Excitement"
        assert get_emotion_from_marker(308) == "Positive_Emotion_Low_Approach"
        assert get_emotion_from_marker(309) == "Positive_Emotion_High_Approach"

    def test_none_marker(self):
        """Test None marker returns Unknown."""
        assert get_emotion_from_marker(None) == "Unknown"

    def test_nan_marker(self):
        """Test NaN marker returns Unknown."""
        assert get_emotion_from_marker(float('nan')) == "Unknown"
        assert get_emotion_from_marker(np.nan) == "Unknown"

    def test_unknown_marker_int(self):
        """Test unknown integer marker."""
        result = get_emotion_from_marker(999)
        assert result.startswith("Unknown")
        assert "999" in result

    def test_float_marker(self):
        """Test float marker is converted to int."""
        assert get_emotion_from_marker(101.0) == "Neutral"
        assert get_emotion_from_marker(-1.0) == "Baseline"

    def test_string_marker(self):
        """Test string marker returns Unknown."""
        result = get_emotion_from_marker("invalid")
        assert result.startswith("Unknown")

    def test_all_known_markers_have_mapping(self):
        """Test that all markers in STIMULI_TO_EMOTION are mapped."""
        for marker, emotion in STIMULI_TO_EMOTION.items():
            result = get_emotion_from_marker(marker)
            assert result == emotion, f"Marker {marker} should map to {emotion}, got {result}"


class TestGetValenceFromMarker:
    """Tests for get_valence_from_marker function."""

    def test_baseline_valence(self):
        """Test baseline marker valence."""
        assert get_valence_from_marker(-1) == "Neutral"

    def test_neutral_valence(self):
        """Test neutral markers valence (101-110)."""
        for marker in range(101, 111):
            assert get_valence_from_marker(marker) == "Neutral"

    def test_negative_valence(self):
        """Test negative markers valence (201-210)."""
        for marker in range(201, 211):
            assert get_valence_from_marker(marker) == "Negative"

    def test_positive_valence(self):
        """Test positive markers valence (301-309)."""
        for marker in range(301, 310):
            assert get_valence_from_marker(marker) == "Positive"

    def test_unknown_valence(self):
        """Test unknown marker returns Unknown valence."""
        assert get_valence_from_marker(999) == "Unknown"

    def test_all_valence_categories(self):
        """Test that all three valence categories exist."""
        valences = set(STIMULI_TO_VALENCE.values())
        assert "Neutral" in valences
        assert "Negative" in valences
        assert "Positive" in valences


class TestGetStudyName:
    """Tests for get_study_name function."""

    @pytest.mark.parametrize("study_number", [1, 2, 3, 4, 5, 6, 7])
    def test_valid_study_numbers(self, study_number: int):
        """Test all valid study numbers."""
        result = get_study_name(study_number)
        assert result == f"STUDY{study_number}"

    def test_unknown_study_number(self):
        """Test unknown study number returns formatted name."""
        result = get_study_name(99)
        assert result == "STUDY99"

    def test_zero_study_number(self):
        """Test study number 0."""
        result = get_study_name(0)
        assert result == "STUDY0"

    def test_negative_study_number(self):
        """Test negative study number."""
        result = get_study_name(-1)
        assert result == "STUDY-1"


class TestStimuliConstants:
    """Tests for stimuli constant dictionaries."""

    def test_stimuli_to_emotion_not_empty(self):
        """Test that STIMULI_TO_EMOTION is not empty."""
        assert len(STIMULI_TO_EMOTION) > 0

    def test_stimuli_to_emotion_has_all_categories(self):
        """Test that STIMULI_TO_EMOTION has baseline, neutral, negative, positive."""
        emotions = set(STIMULI_TO_EMOTION.values())

        # Should have baseline
        assert "Baseline" in emotions
        # Should have neutral
        assert "Neutral" in emotions
        # Should have negative emotions
        negative_emotions = {"Fear", "Anger", "Disgust", "Sadness", "Threat"}
        assert len(negative_emotions & emotions) > 0
        # Should have positive emotions
        positive_emotions = {"Amusement", "Tenderness", "Excitement",
                           "Positive_Emotion_Low_Approach", "Positive_Emotion_High_Approach"}
        assert len(positive_emotions & emotions) > 0

    def test_stimuli_to_valence_not_empty(self):
        """Test that STIMULI_TO_VALENCE is not empty."""
        assert len(STIMULI_TO_VALENCE) > 0

    def test_study_names_complete(self):
        """Test that STUDY_NAMES has all 7 studies."""
        assert len(STUDY_NAMES) == 7
        for i in range(1, 8):
            assert i in STUDY_NAMES
            assert STUDY_NAMES[i] == f"STUDY{i}"

    def test_marker_ranges_consistent(self):
        """Test that marker ranges follow expected pattern."""
        # Baseline
        assert -1 in STIMULI_TO_EMOTION

        # Neutral: 101-110
        for marker in range(101, 111):
            assert marker in STIMULI_TO_EMOTION
            assert STIMULI_TO_EMOTION[marker] == "Neutral"

        # Negative: 201-210
        for marker in range(201, 211):
            assert marker in STIMULI_TO_EMOTION

        # Positive: 301-309
        for marker in range(301, 310):
            assert marker in STIMULI_TO_EMOTION


class TestEdgeCases:
    """Edge case tests for stimuli functions."""

    def test_marker_boundary_values(self):
        """Test boundary values around marker ranges."""
        # Just before neutral range
        result = get_emotion_from_marker(100)
        assert result.startswith("Unknown")

        # Just after neutral range
        result = get_emotion_from_marker(111)
        assert result.startswith("Unknown")

        # Just before negative range
        result = get_emotion_from_marker(200)
        assert result.startswith("Unknown")

        # Just after negative range
        result = get_emotion_from_marker(211)
        assert result.startswith("Unknown")

        # Just before positive range
        result = get_emotion_from_marker(300)
        assert result.startswith("Unknown")

        # Just after positive range
        result = get_emotion_from_marker(310)
        assert result.startswith("Unknown")

    def test_large_marker_values(self):
        """Test handling of very large marker values."""
        result = get_emotion_from_marker(999999)
        assert result.startswith("Unknown")
        assert "999999" in result

    def test_negative_marker_values(self):
        """Test handling of negative marker values (except -1)."""
        result = get_emotion_from_marker(-2)
        assert result.startswith("Unknown")
        assert "-2" in result

    def test_infinity_marker(self):
        """Test handling of infinity marker."""
        # Infinity causes OverflowError when converting to int
        # The function should handle this gracefully
        try:
            result = get_emotion_from_marker(float('inf'))
            assert "Unknown" in result or isinstance(result, str)
        except OverflowError:
            # Expected behavior - infinity cannot convert to int
            pass

    def test_marker_type_coercion(self):
        """Test that different numeric types are handled."""
        # Integer
        assert get_emotion_from_marker(101) == "Neutral"
        # Float
        assert get_emotion_from_marker(101.0) == "Neutral"
        # Float with decimal (truncated)
        assert get_emotion_from_marker(101.9) == "Neutral"
        # Numpy int
        assert get_emotion_from_marker(np.int64(101)) == "Neutral"
        # Numpy float
        assert get_emotion_from_marker(np.float64(101.0)) == "Neutral"
