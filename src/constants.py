"""
Global constants for the POPANE API
"""

from enum import Enum


class Environment(str, Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


# Valid study numbers in the database
VALID_STUDY_NUMBERS = range(1, 8)
MIN_STUDY_NUMBER = 1
MAX_STUDY_NUMBER = 7
