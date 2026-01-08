"""
Studies module dependencies for FastAPI routes
"""

from fastapi import Path

from src.constants import MIN_STUDY_NUMBER, MAX_STUDY_NUMBER
from src.studies.exceptions import InvalidStudyNumberError


async def valid_study_number(
    study_number: int = Path(..., description="Study number (1-7)")
) -> int:
    """
    Dependency to validate study number is within valid range.

    Can be reused across multiple routes that need study validation.
    """
    if study_number < MIN_STUDY_NUMBER or study_number > MAX_STUDY_NUMBER:
        raise InvalidStudyNumberError(study_number)
    return study_number
