"""Stimuli metadata for emotion mapping.

The marker field in the database corresponds to Stimuli ID.
This mapping provides the emotion name for each stimuli.
"""
import numpy as np

# Stimuli ID to Emotion mapping from metadata.xlsx
# Convention: 1XX = neutral, 2XX = negative, 3XX = positive
STIMULI_TO_EMOTION = {
    -1: "Baseline",
    101: "Neutral",
    102: "Neutral",
    103: "Neutral",
    104: "Neutral",
    105: "Neutral",
    106: "Neutral",
    107: "Neutral",
    108: "Neutral",
    109: "Neutral",
    110: "Neutral",
    201: "Fear",
    202: "Anger",
    203: "Anger",
    204: "Anger",
    205: "Disgust",
    206: "Sadness",
    207: "Sadness",
    208: "Anger",
    209: "Threat",
    210: "Fear",
    301: "Amusement",
    302: "Amusement",
    303: "Amusement",
    304: "Amusement",
    305: "Tenderness",
    306: "Tenderness",
    307: "Excitement",
    308: "Positive_Emotion_Low_Approach",
    309: "Positive_Emotion_High_Approach",
}

# Valence mapping for broader categorization
STIMULI_TO_VALENCE = {
    -1: "Neutral",
    **{k: "Neutral" for k in range(101, 111)},
    **{k: "Negative" for k in range(201, 211)},
    **{k: "Positive" for k in range(301, 310)},
}

# Study names
STUDY_NAMES = {
    1: "STUDY1",
    2: "STUDY2",
    3: "STUDY3",
    4: "STUDY4",
    5: "STUDY5",
    6: "STUDY6",
    7: "STUDY7",
}


def get_emotion_from_marker(marker) -> str:
    """Get emotion name from marker/stimuli ID.

    Handles None, NaN, and non-numeric values safely.
    """
    if marker is None:
        return "Unknown"
    try:
        # Check for NaN (float type)
        if isinstance(marker, float) and np.isnan(marker):
            return "Unknown"
        marker_int = int(marker)
        return STIMULI_TO_EMOTION.get(marker_int, f"Unknown_{marker_int}")
    except (ValueError, TypeError):
        return f"Unknown_{marker}"


def get_valence_from_marker(marker: int) -> str:
    """Get valence category from marker/stimuli ID."""
    return STIMULI_TO_VALENCE.get(int(marker), "Unknown")


def get_study_name(study_number: int) -> str:
    """Get study name from study number."""
    return STUDY_NAMES.get(study_number, f"STUDY{study_number}")
