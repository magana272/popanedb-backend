"""
Studies module business logic / service layer
"""

import math
from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.database import get_db_connection
from src.studies.constants import STUDY_COLUMNS
from src.studies.schemas import StudyMetadata, SubjectInfo, StudyStats, ColumnInfo, EmotionSummaryResponse
from src.studies.exceptions import DataFetchError
from src.visualization.stimuli import get_emotion_from_marker


def get_all_studies() -> List[StudyMetadata]:
    """
    Fetch metadata for all studies.

    Returns list of StudyMetadata with row counts and subject counts.
    """
    studies = []

    with get_db_connection() as conn:
        for study_num in range(1, 8):
            table_name = f"study{study_num}"
            try:
                # Get row count
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                row_count = result[0] if result else 0

                # Get subject count
                subject_result = conn.execute(
                    f"SELECT COUNT(DISTINCT Subject_ID) FROM {table_name}"
                ).fetchone()
                subject_count = subject_result[0] if subject_result else 0

                studies.append(StudyMetadata(
                    studyNumber=study_num,
                    tableName=table_name,
                    columns=STUDY_COLUMNS.get(study_num, []),
                    rowCount=row_count,
                    subjectCount=subject_count
                ))
            except Exception as e:
                # Table might not exist - include error info
                studies.append(StudyMetadata(
                    studyNumber=study_num,
                    tableName=table_name,
                    columns=STUDY_COLUMNS.get(study_num, []),
                    rowCount=0,
                    subjectCount=0,
                    error=str(e)
                ))

    return studies


def get_study_subjects(study_number: int) -> List[SubjectInfo]:
    """
    Get all subjects for a specific study with their record counts.
    """
    table_name = f"study{study_number}"

    try:
        with get_db_connection() as conn:
            result = conn.execute(f"""
                SELECT Subject_ID, COUNT(*) as record_count
                FROM {table_name}
                GROUP BY Subject_ID
                ORDER BY Subject_ID
            """).fetchall()

            return [
                SubjectInfo(
                    id=row[0],
                    studyNumber=study_number,
                    recordCount=row[1]
                )
                for row in result
            ]
    except Exception as e:
        raise DataFetchError(f"Failed to fetch subjects: {str(e)}")


def get_study_data(
    study_number: int,
    subject_id: Optional[int] = None,
    subject_ids: Optional[List[int]] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    limit: int = 1000,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Fetch study data with optional filtering.

    Supports filtering by:
    - Single subject ID
    - Multiple subject IDs
    - Time range (start_time, end_time)
    - Pagination (limit, offset)
    """
    table_name = f"study{study_number}"

    # Build query conditions
    conditions = []
    params = []

    # Handle multiple subject IDs (takes precedence over single subject_id)
    if subject_ids and len(subject_ids) > 0:
        placeholders = ', '.join(['?' for _ in subject_ids])
        conditions.append(f"Subject_ID IN ({placeholders})")
        params.extend(subject_ids)
    elif subject_id is not None:
        conditions.append("Subject_ID = ?")
        params.append(subject_id)

    if start_time is not None:
        conditions.append("timestamp >= ?")
        params.append(start_time)

    if end_time is not None:
        conditions.append("timestamp <= ?")
        params.append(end_time)

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    # Use window function to apply limit per subject AND per emotion (marker)
    query = f"""
        SELECT * FROM (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY Subject_ID, marker ORDER BY timestamp) as row_num
            FROM {table_name}
            {where_clause}
        ) ranked
        WHERE row_num <= {limit}
        ORDER BY Subject_ID, timestamp
    """

    try:
        with get_db_connection() as conn:
            result = conn.execute(query, params).fetchall()
            columns = [desc[0] for desc in conn.description]

            # Convert to list of dicts (exclude row_num column)
            data = []
            for row in result:
                row_dict = dict(zip(columns, row))
                row_dict.pop('row_num', None)  # Remove the row_num column
                # Sanitize NaN/Inf values for JSON serialization
                for key, value in row_dict.items():
                    if isinstance(value, float):
                        if math.isnan(value) or math.isinf(value):
                            row_dict[key] = None
                data.append(row_dict)

            # Calculate time_offset per subject (POPANEpy pattern)
            # time_offset = timestamp - first_timestamp_for_subject
            if data:
                # Group by subject and calculate offset
                subject_first_ts = {}
                for row in data:
                    subj = row.get('Subject_ID')
                    ts = row.get('timestamp')
                    if subj not in subject_first_ts and ts is not None:
                        subject_first_ts[subj] = ts

                # Add time_offset to each row
                for row in data:
                    subj = row.get('Subject_ID')
                    ts = row.get('timestamp')
                    if subj in subject_first_ts and ts is not None:
                        row['time_offset'] = ts - subject_first_ts[subj]
                    else:
                        row['time_offset'] = 0.0

            return data
    except Exception as e:
        raise DataFetchError(f"Failed to fetch study data: {str(e)}")


def get_study_statistics(study_number: int) -> StudyStats:
    """
    Get statistical summary for a study.
    """
    table_name = f"study{study_number}"

    try:
        with get_db_connection() as conn:
            result = conn.execute(f"""
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT Subject_ID) as subject_count,
                    MIN(timestamp) as min_timestamp,
                    MAX(timestamp) as max_timestamp
                FROM {table_name}
            """).fetchone()
            if result is None:
                raise DataFetchError(f"No data found for study {study_number}")
            return StudyStats(
                studyNumber=study_number,
                totalRecords=result[0],
                subjects=result[1],
                minTimestamp=result[2],
                maxTimestamp=result[3]
            )
    except Exception as e:
        raise DataFetchError(f"Failed to fetch study statistics: {str(e)}")


def get_study_column_info(study_number: int) -> List[ColumnInfo]:
    """
    Get column information (name and type) for a study table.
    """
    table_name = f"study{study_number}"

    try:
        with get_db_connection() as conn:
            result = conn.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """).fetchall()

            return [ColumnInfo(name=row[0], type=row[1]) for row in result]
    except Exception as e:
        raise DataFetchError(f"Failed to fetch column info: {str(e)}")


def get_emotion_summary(
    study_number: int,
    subject_ids: Optional[List[int]] = None
) -> EmotionSummaryResponse:
    """
    Get emotion summary across multiple subjects.

    Returns counts of subjects per emotion and which subjects experienced each emotion.
    This is much more efficient than calling get_emotion_colored_signals for each subject.
    """
    table_name = f"study{study_number}"

    try:
        with get_db_connection() as conn:
            # Build query - get distinct emotions per subject
            if subject_ids:
                placeholders = ', '.join(['?' for _ in subject_ids])
                query = f"""
                    SELECT DISTINCT Subject_ID, marker
                    FROM {table_name}
                    WHERE Subject_ID IN ({placeholders})
                    AND marker IS NOT NULL
                """
                df = conn.execute(query, subject_ids).fetchdf()
                queried_subjects = set(subject_ids)
            else:
                # Get all subjects if none specified
                query = f"""
                    SELECT DISTINCT Subject_ID, marker
                    FROM {table_name}
                    WHERE marker IS NOT NULL
                """
                df = conn.execute(query).fetchdf()
                queried_subjects = set(df['Subject_ID'].unique())

            if len(df) == 0:
                return EmotionSummaryResponse(
                    emotionCounts={},
                    subjectsPerEmotion={},
                    totalSubjects=len(queried_subjects),
                    studyNumber=study_number
                )

            # Map markers to emotions
            df['emotion'] = df['marker'].apply(get_emotion_from_marker)

            # Build subjects_per_emotion dict
            subjects_per_emotion: Dict[str, List[int]] = defaultdict(list)

            for _, row in df.iterrows():
                subject_id = int(row['Subject_ID'])
                emotion = row['emotion']
                if subject_id not in subjects_per_emotion[emotion]:
                    subjects_per_emotion[emotion].append(subject_id)

            # Sort subject lists and convert to regular dict
            subjects_per_emotion = {
                emotion: sorted(subjects)
                for emotion, subjects in sorted(subjects_per_emotion.items())
            }

            # Build emotion_counts from subjects_per_emotion
            emotion_counts = {
                emotion: len(subjects)
                for emotion, subjects in subjects_per_emotion.items()
            }

            return EmotionSummaryResponse(
                emotionCounts=emotion_counts,
                subjectsPerEmotion=subjects_per_emotion,
                totalSubjects=len(queried_subjects),
                studyNumber=study_number
            )
    except Exception as e:
        raise DataFetchError(f"Failed to fetch emotion summary: {str(e)}")
