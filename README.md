# POPANE Backend API

A FastAPI backend for serving physiological emotion data from the POPANE database.

## Overview

This API provides endpoints for accessing and visualizing physiological data across 7 emotion studies. It serves data from a DuckDB database containing ECG, EDA, blood pressure, respiration, and other physiological signals.

## Setup

### 1. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\\Scripts\\activate
```

### 2. Install Dependencies

```bash
# Quick install (base requirements)
pip install -r requirements/base.txt

# Or for development (includes testing tools)
pip install -r requirements/dev.txt

# Or for production
pip install -r requirements/prod.txt
```

### 3. Configure Environment

Create a `.env` file (optional - defaults are provided):

```env
APP_NAME=POPANE API
APP_VERSION=1.0.0
ENVIRONMENT=DEVELOPMENT
DEBUG=True
DATABASE_DIR=\Users\path\to\popanebackend\popanedb-backend\db\
DATABASE_PATH=/path/to/popane_emotion.db
HOST=0.0.0.0
PORT=8000
```

### 4. Run the Server

```bash

./run.sh

# Or directly with Uvicorn

uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

## API Documentation

Interactive documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Endpoints

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check - returns API status |

**Response:**
```json
{
  "status": "healthy",
  "name": "POPANE API",
  "version": "1.0.0"
}
```

---

### Studies API (`/api`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/studies` | Get all available studies |
| GET | `/api/study/{study_number}/subjects` | Get subjects for a study |
| GET | `/api/study/{study_number}/data` | Get physiological data |
| GET | `/api/study/{study_number}/stats` | Get study statistics |
| GET | `/api/study/{study_number}/columns` | Get column information |
| GET | `/api/study/{study_number}/emotions/summary` | Get emotion summary across subjects |

#### GET `/api/studies`

Returns metadata for all available studies (1-7).

**Response:**
```json
[
  {
    "study_number": 1,
    "name": "Study 1",
    "description": "...",
    "record_count": 12345
  }
]
```

#### GET `/api/study/{study_number}/subjects`

Returns subjects in a study with record counts.

**Parameters:**
- `study_number` (path): Study number (1-7)

**Response:**
```json
[
  {
    "subject_id": "S001",
    "record_count": 5000
  }
]
```

#### GET `/api/study/{study_number}/data`

Returns physiological data with optional filtering.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `study_number` | path | Yes | Study number (1-7) |
| `subject_id` | query | No | Filter by single subject |
| `subject_ids` | query | No | Filter by multiple subjects (list) |
| `start_time` | query | No | Start timestamp filter |
| `end_time` | query | No | End timestamp filter |
| `limit` | query | No | Max records (default: 1000, max: 10000) |
| `offset` | query | No | Pagination offset (default: 0) |

#### GET `/api/study/{study_number}/stats`

Returns statistical summary for the study.

#### GET `/api/study/{study_number}/columns`

Returns column definitions for the study.

#### GET `/api/study/{study_number}/emotions/summary`

Returns emotion summary across subjects for efficient chart rendering.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `study_number` | path | Yes | - | Study number (1-7) |
| `subjects` | query | No | all | Comma-separated subject IDs (e.g., '1,2,3,4,5') |

**Response:**
```json
{
  "study_number": 1,
  "emotion_counts": {"happy": 45, "sad": 42, "neutral": 50},
  "subjects_per_emotion": {"happy": [1,2,3], "sad": [1,4,5]},
  "total_subjects": 50
}
```

---

### Visualization API (`/viz`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/viz/features/{study_number}` | Get available features |
| GET | `/viz/signals/{study_number}/{subject_id}` | Get emotion-colored signals |
| GET | `/viz/pca/{study_number}` | Get PCA analysis |
| GET | `/viz/pca/frequency/{study_number}` | Get PCA on FFT-derived features |
| GET | `/viz/analysis/{study_number}/{subject_id}` | Get time/frequency analysis |
| GET | `/viz/spectrum/{study_number}/{subject_id}` | Get power spectrum (Welch PSD) |
| GET | `/viz/fft/{study_number}/{subject_id}` | Get FFT frequency spectrum |

#### GET `/viz/features/{study_number}`

Returns available physiological features for a study.

**Response:**
```json
[
  {
    "feature": "ECG",
    "title": "ECG Signal",
    "unit": "mV",
    "color": "#1f77b4"
  }
]
```

#### GET `/viz/signals/{study_number}/{subject_id}`

Returns signal data with emotion-colored segments for visualization.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `study_number` | path | Yes | - | Study number (1-7) |
| `subject_id` | path | Yes | - | Subject ID |
| `features` | query | No | ["ECG"] | Comma-separated features to include |
| `start_time` | query | No | 0.0 | Start time in seconds |
| `end_time` | query | No | 10.0 | End time in seconds |
| `seconds_per_emotion` | query | No | - | Fetch first N seconds of each distinct emotion (overrides start_time/end_time) |

**Notes:**
- When `seconds_per_emotion` is provided, the endpoint fetches the first N seconds of each distinct emotion and normalizes `time_offset` sequentially on one plot
- This mode is useful for comparing the same time window across different emotional states

#### GET `/viz/pca/{study_number}`

Returns PCA dimensionality reduction results.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `study_number` | path | Yes | - | Study number (1-7) |
| `subject_ids` | query | No | all | Comma-separated subject IDs |
| `features` | query | No | all | Comma-separated features for PCA (min 2) |
| `sample_size` | query | No | 1000 | Max samples (100-100000) |

#### GET `/viz/pca/frequency/{study_number}`

Returns PCA on FFT-derived and time-domain features across all subjects.

For each (subject, emotion) combination, computes per signal:
- **dominant_frequency**: Peak frequency from FFT
- **signal_energy**: Log2 of total FFT energy
- **mean, std, min, max, range**: Time-domain statistics
- **vlf_power, lf_power, hf_power**: Welch PSD frequency band powers
- **lf_nu, hf_nu, lf_hf_ratio**: Normalized HRV metrics

Each point in the PCA represents one (subject, emotion) pair, colored by emotion.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `study_number` | path | Yes | - | Study number (1-7) |
| `features` | query | No | all | Comma-separated signal features (ECG, EDA, DBP, etc.) |
| `subject_ids` | query | No | all | Comma-separated subject IDs |

**Example:** `/viz/pca/frequency/1?features=ECG,EDA,DBP`

#### GET `/viz/analysis/{study_number}/{subject_id}`

Returns time and frequency domain analysis features.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `study_number` | path | Yes | - | Study number (1-7) |
| `subject_id` | path | Yes | - | Subject ID |
| `feature` | query | No | "ECG" | Feature to analyze |
| `sampling_rate` | query | No | 1.0 | Signal sampling rate (Hz) |

#### GET `/viz/spectrum/{study_number}/{subject_id}`

Returns power spectral density (PSD) for frequency analysis.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `study_number` | path | Yes | - | Study number (1-7) |
| `subject_id` | path | Yes | - | Subject ID |
| `feature` | query | No | "ECG" | Feature to analyze |
| `sampling_rate` | query | No | 1.0 | Signal sampling rate (Hz) |
| `max_freq` | query | No | 0.5 | Max frequency to display (Hz) |

**Frequency Bands:**
- **VLF**: 0.003-0.04 Hz (thermoregulation, hormonal)
- **LF**: 0.04-0.15 Hz (sympathetic + parasympathetic)
- **HF**: 0.15-0.4 Hz (parasympathetic, respiratory)

#### GET `/viz/fft/{study_number}/{subject_id}`

Returns FFT-based frequency spectrum for a signal, grouped by emotion.

Supports single subject or multiple subjects (comma-separated in path).
When multiple subjects are provided, data is aggregated across all subjects.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `study_number` | path | Yes | - | Study number (1-7) |
| `subject_id` | path | Yes | - | Subject ID or comma-separated IDs (e.g., `120` or `120,121,122`) |
| `feature` | query | No | "DBP" | Signal to analyze (ECG, EDA, DBP, SBP, etc.) |
| `max_freq` | query | No | 2.0 | Maximum frequency to display (Hz) |

**Response includes per-emotion:**
- `dominant_frequency`: Peak frequency in Hz (via peak finding)
- `signal_energy`: Log2 of total signal energy
- `spectrum`: Array of `{frequency, magnitude}` points for plotting

**Examples:**
- Single subject: `/viz/fft/1/120?feature=ECG&max_freq=2.0`
- Multiple subjects: `/viz/fft/1/120,121,122?feature=ECG&max_freq=2.0`

---

## Quick Reference

### Available Features by Study

| Feature | Description | Unit | Studies |
|---------|-------------|------|---------|
| ECG | Electrocardiogram | mV | 1-7 |
| EDA | Electrodermal Activity | µS | 1-6 |
| SBP | Systolic Blood Pressure | mmHg | 1-6 |
| DBP | Diastolic Blood Pressure | mmHg | 1-6 |
| respiration | Respiration Signal | breaths/min | 1 |
| temp | Temperature | °C | 1 |
| CO | Cardiac Output | l/min | 2-6 |
| TPR | Total Peripheral Resistance | mmHg*min/l | 2-6 |
| dz | dz Signal | ohm | 3, 6-7 |
| z0 | z0 Signal | ohm | 3, 6-7 |
| dzdt | dzdt Signal | ohm/s | 3, 6-7 |
| affect | Emotion Label | - | 1-3, 5-7 |

### Study Column Definitions

| Study | Columns |
|-------|---------|
| 1 | Subject_ID, timestamp, affect, ECG, EDA, temp, respiration, SBP, DBP, marker |
| 2 | Subject_ID, timestamp, affect, ECG, EDA, SBP, DBP, CO, TPR, marker |
| 3 | Subject_ID, timestamp, affect, ECG, EDA, dzdt, dz, z0, SBP, DBP, CO, TPR, marker |
| 4 | Subject_ID, timestamp, ECG, EDA, SBP, DBP, CO, TPR, marker |
| 5 | Subject_ID, timestamp, affect, ECG, EDA, SBP, DBP, CO, TPR, marker |
| 6 | Subject_ID, timestamp, affect, ECG, dzdt, dz, z0, EDA, SBP, DBP, CO, TPR, marker |
| 7 | Subject_ID, timestamp, affect, ECG, dzdt, dz, z0, marker |

---

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `APP_NAME` | Application name | POPANE API |
| `APP_VERSION` | API version | 1.0.0 |
| `ENVIRONMENT` | Environment mode | DEVELOPMENT |
| `DEBUG` | Enable debug mode | True |
| `DATABASE_PATH` | Path to DuckDB file | /Users/.../popane_emotion.db |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `CORS_ORIGINS` | Allowed origins | localhost:3000,3001,3002 |
| `DEFAULT_LIMIT` | Default query limit | 1000 |
| `MAX_LIMIT` | Maximum query limit | 10000 |

---

## Project Structure

```
backend/
├── server.py           # Legacy standalone server
├── run.sh              # Server launch script
├── requirements.txt    # Base dependencies
├── requirements/       # Environment-specific requirements
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── src/
│   ├── main.py         # FastAPI app factory
│   ├── config.py       # Pydantic settings
│   ├── database.py     # DuckDB connection
│   ├── constants.py    # Global constants
│   ├── exceptions.py   # Custom exceptions
│   ├── pagination.py   # Pagination utilities
│   ├── studies/        # Studies API module
│   │   ├── router.py   # /api endpoints
│   │   ├── service.py  # Business logic
│   │   └── schemas.py  # Pydantic models
│   └── visualization/  # Visualization API module
│       ├── router.py   # /viz endpoints
│       ├── service.py  # Business logic
│       ├── features.py # Feature extraction
│       ├── stimuli.py  # Stimuli handling
│       └── constants.py # Feature configs
└── tests/              # Test suite
```

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/studies/test_router.py
```
