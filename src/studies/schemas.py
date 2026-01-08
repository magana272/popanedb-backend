"""
Studies module Pydantic schemas
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

from src.studies.constants import STUDY_COLUMNS


class SubjectInfo(BaseModel):
    """Subject information within a study"""
    id: int = Field(..., description="Subject ID")
    study_number: int = Field(..., alias="studyNumber", description="Study number")
    record_count: int = Field(..., alias="recordCount", description="Number of records for this subject")

    class Config:
        populate_by_name = True


class StudyMetadata(BaseModel):
    """Metadata about a study"""
    study_number: int = Field(..., alias="studyNumber")
    table_name: str = Field(..., alias="tableName")
    columns: List[str]
    row_count: int = Field(0, alias="rowCount")
    subject_count: int = Field(0, alias="subjectCount")
    error: Optional[str] = None

    class Config:
        populate_by_name = True


class StudyStats(BaseModel):
    """Statistical information about a study"""
    study_number: int = Field(..., alias="studyNumber")
    total_records: int = Field(..., alias="totalRecords")
    subjects: int
    min_timestamp: Optional[float] = Field(None, alias="minTimestamp")
    max_timestamp: Optional[float] = Field(None, alias="maxTimestamp")

    class Config:
        populate_by_name = True


class ColumnInfo(BaseModel):
    """Information about a database column"""
    name: str
    type: str


class StudyDataRow(BaseModel):
    """A single row of study data - dynamic based on study"""
    # Using dict for flexibility since columns vary by study

    class Config:
        extra = "allow"


class StudyDataQuery(BaseModel):
    """Query parameters for fetching study data"""
    subject_id: Optional[int] = Field(None, description="Filter by single subject ID")
    subject_ids: Optional[List[int]] = Field(None, description="Filter by multiple subject IDs")
    start_time: Optional[float] = Field(None, description="Start timestamp filter")
    end_time: Optional[float] = Field(None, description="End timestamp filter")
    limit: int = Field(1000, ge=1, le=10000, description="Maximum rows to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class StudyDataResponse(BaseModel):
    """Response containing study data"""
    data: List[Dict[str, Any]]
    count: int
    study_number: int = Field(..., alias="studyNumber")

    class Config:
        populate_by_name = True


class EmotionSummaryResponse(BaseModel):
    """Response containing emotion summary across subjects"""
    emotion_counts: Dict[str, int] = Field(
        ...,
        alias="emotionCounts",
        description="Count of subjects that experienced each emotion"
    )
    subjects_per_emotion: Dict[str, List[int]] = Field(
        ...,
        alias="subjectsPerEmotion",
        description="List of subject IDs for each emotion"
    )
    total_subjects: int = Field(
        ...,
        alias="totalSubjects",
        description="Total number of subjects queried"
    )
    study_number: int = Field(..., alias="studyNumber")

    class Config:
        populate_by_name = True
