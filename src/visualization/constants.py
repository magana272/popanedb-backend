"""Constants for visualization module."""

# Feature configuration matching POPANEpy
FEATURE_CONFIG = {
    'ECG': {
        'title': 'ECG Signal',
        'unit': 'mV',
        'color': '#1f77b4'  # blue
    },
    'EDA': {
        'title': 'EDA Signal',
        'unit': 'µS',
        'color': '#2ca02c'  # green
    },
    'SBP': {
        'title': 'Systolic Blood Pressure',
        'unit': 'mmHg',
        'color': '#d62728'  # red
    },
    'DBP': {
        'title': 'Diastolic Blood Pressure',
        'unit': 'mmHg',
        'color': '#9467bd'  # purple
    },
    'respiration': {
        'title': 'Respiration Signal',
        'unit': 'breaths/min',
        'color': '#ff7f0e'  # orange
    },
    'temp': {
        'title': 'Temperature Signal',
        'unit': '°C',
        'color': '#8c564b'  # brown
    },
    'CO': {
        'title': 'Cardiac Output',
        'unit': 'l/min',
        'color': '#e377c2'  # pink
    },
    'TPR': {
        'title': 'Total Peripheral Resistance',
        'unit': 'mmHg*min/l',
        'color': '#17becf'  # cyan
    },
    'dz': {
        'title': 'dz Signal',
        'unit': 'ohm',
        'color': '#bcbd22'  # yellow-green
    },
    'z0': {
        'title': 'z0 Signal',
        'unit': 'ohm',
        'color': '#7f7f7f'  # gray
    },
    'dzdt': {
        'title': 'dzdt Signal',
        'unit': 'ohm/s',
        'color': '#17becf'  # teal
    },
    'affect': {
        'title': 'Affect Signal',
        'unit': '',
        'color': '#2ca02c'  # green
    }
}

ALL_FEATURES = list(FEATURE_CONFIG.keys())

# HUSL color palette for emotions (generated to match seaborn's husl palette)
# These are hex colors from seaborn's husl palette
EMOTION_COLORS = [
    '#f77189',  # Pink-red
    '#dc8932',  # Orange
    '#ae9d31',  # Olive
    '#77ab31',  # Green
    '#33b07a',  # Teal
    '#36ada4',  # Cyan
    '#38a9c5',  # Sky blue
    '#39a7d0',  # Light blue
    '#6e9bf4',  # Blue
    '#a48cf4',  # Purple
    '#cc7af4',  # Magenta
    '#f565cc',  # Pink
]
