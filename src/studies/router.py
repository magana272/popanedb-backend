"""
Studies module router - API endpoints for study data access
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query

from src.studies import service
from src.studies.dependencies import valid_study_number
from src.studies.schemas import (
    StudyMetadata,
    SubjectInfo,
    StudyStats,
    ColumnInfo,
    EmotionSummaryResponse,
)

router = APIRouter(prefix="/api", tags=["Studies"])


@router.get(
    "/studies",
    response_model=List[StudyMetadata],
    summary="Get all studies",
    description="Returns metadata for all available studies including row counts and subject counts"
)
def get_studies() -> List[StudyMetadata]:
    """
    Get list of available studies with metadata.

    Note: Using sync function since DuckDB operations are CPU-bound
    and will run in threadpool automatically.
    """
    return service.get_all_studies()


@router.get(
    "/study/{study_number}/subjects",
    response_model=List[SubjectInfo],
    summary="Get study subjects",
    description="Returns all subjects for a specific study with their record counts"
)
def get_subjects(
    study_number: int = Depends(valid_study_number)
) -> List[SubjectInfo]:
    """Get list of subjects for a study."""
    return service.get_study_subjects(study_number)


@router.get(
    "/study/{study_number}/data",
    summary="Get study data",
    description="Returns physiological data from a study with optional filtering by subjects and time range"
)
def get_study_data(
    study_number: int = Depends(valid_study_number),
    subject_id: Optional[int] = Query(None, description="Filter by single subject ID"),
    subject_ids: List[int] = Query(None, description="Filter by multiple subject IDs"),
    start_time: Optional[float] = Query(None, description="Start timestamp"),
    end_time: Optional[float] = Query(None, description="End timestamp"),
    limit: int = Query(1000, ge=1, le=100000, description="Maximum rows to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    Get data from a study with optional filtering.

    Supports:
    - Multiple subject selection via subject_ids parameter
    - Time range filtering
    - Pagination with limit/offset
    """
    return service.get_study_data(
        study_number=study_number,
        subject_id=subject_id,
        subject_ids=subject_ids,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset
    )


@router.get(
    "/study/{study_number}/stats",
    response_model=StudyStats,
    summary="Get study statistics",
    description="Returns statistical summary for a study including record counts and timestamp range"
)
def get_study_stats(
    study_number: int = Depends(valid_study_number)
) -> StudyStats:
    """Get statistics for a study."""
    return service.get_study_statistics(study_number)


@router.get(
    "/study/{study_number}/columns",
    response_model=List[ColumnInfo],
    summary="Get study columns",
    description="Returns column information (name and data type) for a study"
)
def get_study_columns(
    study_number: int = Depends(valid_study_number)
) -> List[ColumnInfo]:
    """Get column information for a study."""
    return service.get_study_column_info(study_number)


@router.get(
    "/study/{study_number}/emotions/summary",
    response_model=EmotionSummaryResponse,
    summary="Get emotion summary across subjects",
    description="Returns emotion counts and subject lists for efficient emotion chart rendering"
)
def get_emotion_summary(
    study_number: int = Depends(valid_study_number),
    subjects: Optional[str] = Query(None, description="Comma-separated subject IDs (e.g., '1,2,3,4,5')")
) -> EmotionSummaryResponse:
    """
    Get emotion summary for multiple subjects in a single call.

    Returns:
    - emotion_counts: How many subjects experienced each emotion
    - subjects_per_emotion: Which specific subjects experienced each emotion
    - total_subjects: Total subjects queried

    This replaces N individual getEmotionColoredSignals calls with a single efficient query.
    """
    subject_ids = [int(s.strip()) for s in subjects.split(',')] if subjects else None
    return service.get_emotion_summary(study_number, subject_ids)
