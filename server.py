"""
POPANE Emotion Database API Server
FastAPI backend to serve data from the DuckDB database
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import duckdb
import os

load_dotenv()

app = FastAPI(
    title="POPANE Emotion Database API",
    description="API for accessing physiological emotion data",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path
DB_PATH = os.getenv("DATABASE_PATH")

if DB_PATH is None:
    raise RuntimeError("Environment variable DATABASE_PATH not set")

# Study column definitions
STUDY_COLUMNS = {
    1: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'EDA', 'temp', 'respiration', 'SBP', 'DBP', 'marker'],
    2: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'EDA', 'SBP', 'DBP', 'CO', 'TPR', 'marker'],
    3: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'EDA', 'dzdt', 'dz', 'z0', 'SBP', 'DBP', 'CO', 'TPR', 'marker'],
    4: ['Subject_ID', 'timestamp', 'ECG', 'EDA', 'SBP', 'DBP', 'CO', 'TPR', 'marker'],
    5: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'EDA', 'SBP', 'DBP', 'CO', 'TPR', 'marker'],
    6: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'dzdt', 'dz', 'z0', 'EDA', 'SBP', 'DBP', 'CO', 'TPR', 'marker'],
    7: ['Subject_ID', 'timestamp', 'affect', 'ECG', 'dzdt', 'dz', 'z0', 'marker'],
}


def get_db_connection():
    """Create a read-only connection to the database"""
    if not os.path.exists(DB_PATH): # type: ignore
        raise HTTPException(status_code=500, detail=f"Database not found at {DB_PATH}")
    return duckdb.connect(database=DB_PATH, read_only=True) # type: ignore


@app.get("/")
async def root():
    """API health check"""
    return {"status": "ok", "message": "POPANE Emotion Database API"}


@app.get("/api/studies")
async def get_studies():
    """Get list of available studies with metadata"""
    studies = []
    try:
        conn = get_db_connection()
        for study_num in range(1, 8):
            table_name = f"study{study_num}"
            try:
                # Check if table exists and get row count
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                row_count = result[0] if result else 0

                # Get subject count
                subject_result = conn.execute(
                    f"SELECT COUNT(DISTINCT Subject_ID) FROM {table_name}"
                ).fetchone()
                subject_count = subject_result[0] if subject_result else 0

                studies.append({
                    "studyNumber": study_num,
                    "tableName": table_name,
                    "columns": STUDY_COLUMNS.get(study_num, []),
                    "rowCount": row_count,
                    "subjectCount": subject_count
                })
            except Exception as e:
                # Table might not exist
                studies.append({
                    "studyNumber": study_num,
                    "tableName": table_name,
                    "columns": STUDY_COLUMNS.get(study_num, []),
                    "rowCount": 0,
                    "subjectCount": 0,
                    "error": str(e)
                })
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return studies


@app.get("/api/study/{study_number}/subjects")
async def get_subjects(study_number: int):
    """Get list of subjects for a study"""
    if study_number < 1 or study_number > 7:
        raise HTTPException(status_code=400, detail="Study number must be between 1 and 7")

    try:
        conn = get_db_connection()
        table_name = f"study{study_number}"

        # Get distinct subjects with their record counts
        result = conn.execute(f"""
            SELECT Subject_ID, COUNT(*) as record_count
            FROM {table_name}
            GROUP BY Subject_ID
            ORDER BY Subject_ID
        """).fetchall()

        conn.close()

        subjects = [
            {"id": row[0], "studyNumber": study_number, "recordCount": row[1]}
            for row in result
        ]

        return subjects
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/study/{study_number}/data")
async def get_study_data(
    study_number: int,
    subject_id: Optional[int] = Query(None, description="Filter by single subject ID"),
    subject_ids: List[int] = Query(None, description="Filter by multiple subject IDs"),
    start_time: Optional[float] = Query(None, description="Start timestamp"),
    end_time: Optional[float] = Query(None, description="End timestamp"),
    limit: int = Query(1000, description="Maximum rows to return", le=10000),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get data from a study with optional filtering. Supports multiple subjects."""
    if study_number < 1 or study_number > 7:
        raise HTTPException(status_code=400, detail="Study number must be between 1 and 7")

    try:
        conn = get_db_connection()
        table_name = f"study{study_number}"

        # Build query with filters
        conditions = []
        params = []

        # Handle multiple subject IDs
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

        query = f"""
            SELECT *
            FROM {table_name}
            {where_clause}
            ORDER BY timestamp
            LIMIT {limit} OFFSET {offset}
        """

        result = conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in conn.description]

        conn.close()

        # Convert to list of dicts
        data = [dict(zip(columns, row)) for row in result]

        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/study/{study_number}/stats")
async def get_study_stats(study_number: int):
    """Get statistics for a study"""
    if study_number < 1 or study_number > 7:
        raise HTTPException(status_code=400, detail="Study number must be between 1 and 7")

    try:
        conn = get_db_connection()
        table_name = f"study{study_number}"

        # Get basic stats
        result = conn.execute(f"""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT Subject_ID) as subject_count,
                MIN(timestamp) as min_timestamp,
                MAX(timestamp) as max_timestamp
            FROM {table_name}
        """).fetchone()

        conn.close()

        return {
            "studyNumber": study_number,
            "totalRecords": result[0], # type: ignore
            "subjects": result[1], # type: ignore
            "minTimestamp": result[2], # type: ignore
            "maxTimestamp": result[3] # type: ignore
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/study/{study_number}/columns")
async def get_study_columns(study_number: int):
    """Get column information for a study"""
    if study_number < 1 or study_number > 7:
        raise HTTPException(status_code=400, detail="Study number must be between 1 and 7")

    try:
        conn = get_db_connection()
        table_name = f"study{study_number}"

        # Get column info from the database
        result = conn.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """).fetchall()

        conn.close()

        columns = [{"name": row[0], "type": row[1]} for row in result]

        return columns
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
