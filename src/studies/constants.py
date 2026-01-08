"""
Studies module constants
"""

# Study column definitions per study number
STUDY_COLUMNS = {
    1: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'EDA', 'temp', 'respiration', 'SBP', 'DBP', 'marker'],
    2: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'EDA', 'SBP', 'DBP', 'CO', 'TPR', 'marker'],
    3: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'EDA', 'dzdt', 'dz', 'z0', 'SBP', 'DBP', 'CO', 'TPR', 'marker'],
    4: ['Subject_ID', 'timestamp', 'ECG', 'EDA', 'SBP', 'DBP', 'CO', 'TPR', 'marker'],
    5: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'EDA', 'SBP', 'DBP', 'CO', 'TPR', 'marker'],
    6: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'dzdt', 'dz', 'z0', 'EDA', 'SBP', 'DBP', 'CO', 'TPR', 'marker'],
    7: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'dzdt', 'dz', 'z0', 'marker'],
}

# Study names/descriptions
STUDY_NAMES = {
    1: "Study 1 - Cardiovascular & Respiratory",
    2: "Study 2 - Cardiovascular Output",
    3: "Study 3 - Impedance Cardiography",
    4: "Study 4 - Basic Cardiovascular",
    5: "Study 5 - Cardiovascular Output Extended",
    6: "Study 6 - Full Impedance",
    7: "Study 7 - Core Impedance",
}


class ErrorCode:
    """Module-specific error codes"""
    STUDY_NOT_FOUND = "STUDY_NOT_FOUND"
    INVALID_STUDY_NUMBER = "INVALID_STUDY_NUMBER"
    SUBJECT_NOT_FOUND = "SUBJECT_NOT_FOUND"
    DATA_FETCH_ERROR = "DATA_FETCH_ERROR"
