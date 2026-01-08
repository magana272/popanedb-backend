"""
Studies module exceptions
"""

from fastapi import HTTPException, status

from src.studies.constants import ErrorCode


class StudyNotFoundError(HTTPException):
    """Raised when a study is not found"""
    def __init__(self, study_number: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.STUDY_NOT_FOUND,
                "message": f"Study {study_number} not found"
            }
        )


class InvalidStudyNumberError(HTTPException):
    """Raised when study number is out of valid range"""
    def __init__(self, study_number: int):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_STUDY_NUMBER,
                "message": f"Study number must be between 1 and 7, got {study_number}"
            }
        )


class SubjectNotFoundError(HTTPException):
    """Raised when a subject is not found in a study"""
    def __init__(self, study_number: int, subject_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.SUBJECT_NOT_FOUND,
                "message": f"Subject {subject_id} not found in study {study_number}"
            }
        )


class DataFetchError(HTTPException):
    """Raised when data fetching fails"""
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": ErrorCode.DATA_FETCH_ERROR,
                "message": message
            }
        )
