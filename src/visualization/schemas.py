"""Pydantic schemas for visualization endpoints."""
from typing import Optional
from pydantic import BaseModel, Field


class EmotionColorMap(BaseModel):
    """Mapping of emotions to colors."""
    emotion: str
    color: str
    recording_id: Optional[str] = None


class SignalDataPoint(BaseModel):
    """Single data point with time offset."""
    time_offset: float = Field(..., description="Time offset from start (seconds)")
    value: float
    emotion: str
    color: str


class SignalSeries(BaseModel):
    """A complete signal series for visualization."""
    feature: str
    unit: str
    title: str
    data_points: list[dict]  # [{time_offset, value, emotion, color}]


class EmotionColoredSignalsRequest(BaseModel):
    """Request for emotion-colored signals."""
    study_number: int = Field(..., ge=1, le=7)
    subject_id: int = Field(..., ge=1)
    features: Optional[list[str]] = None  # If None, return all available
    start_time: Optional[float] = Field(default=0.0, description="Start time in seconds")
    end_time: Optional[float] = Field(default=10.0, description="End time in seconds")


class EmotionColoredSignalsResponse(BaseModel):
    """Response with emotion-colored signal data."""
    study_number: int
    subject_id: int
    study_name: str
    emotions: list[EmotionColorMap]
    signals: list[SignalSeries]
    available_features: list[str]
    time_range: dict  # {min: float, max: float}


class PCADataPoint(BaseModel):
    """Single PCA data point."""
    pc1: float
    pc2: float
    emotion: str
    color: str
    subject_id: int


class PCARequest(BaseModel):
    """Request for PCA analysis."""
    study_number: int = Field(..., ge=1, le=7)
    subject_ids: Optional[list[int]] = None  # If None, use all subjects
    features: Optional[list[str]] = None  # Features to use for PCA
    sample_size: int = Field(default=1000, description="Number of samples per subject for PCA")


class PCAResponse(BaseModel):
    """Response with PCA results."""
    study_number: int
    data_points: list[PCADataPoint]
    explained_variance: list[float]  # [PC1 variance ratio, PC2 variance ratio]
    features_used: list[str]
    emotions: list[EmotionColorMap]


class FeatureInfo(BaseModel):
    """Information about available features."""
    name: str
    unit: str
    title: str
    default_color: str


# ============================================================================
# Time-Domain and Frequency-Domain Feature Analysis Schemas
# ============================================================================

class TimeDomainMetrics(BaseModel):
    """Time-domain metrics for a signal segment."""
    # Basic statistics
    mean: float
    std: float
    min_val: float
    max_val: float
    range_val: float

    # HRV-specific (optional, for RR intervals)
    rmssd: Optional[float] = None
    sdnn: Optional[float] = None
    sdsd: Optional[float] = None
    pnn50: Optional[float] = None
    pnn20: Optional[float] = None

    # Metadata
    n_samples: int
    duration_sec: float


class FrequencyDomainMetrics(BaseModel):
    """Frequency-domain metrics for a signal segment."""
    # Power in frequency bands
    vlf_power: float = Field(description="Very Low Frequency power (0.003-0.04 Hz)")
    lf_power: float = Field(description="Low Frequency power (0.04-0.15 Hz)")
    hf_power: float = Field(description="High Frequency power (0.15-0.4 Hz)")
    total_power: float

    # Normalized units
    lf_nu: float = Field(description="LF in normalized units (%)")
    hf_nu: float = Field(description="HF in normalized units (%)")

    # Ratios
    lf_hf_ratio: float = Field(description="LF/HF ratio (sympathovagal balance)")

    # Peak frequencies
    vlf_peak: float
    lf_peak: float
    hf_peak: float

    # Metadata
    n_samples: int
    sampling_rate: float


class EmotionFeatures(BaseModel):
    """Features computed for a specific emotion segment."""
    emotion: str
    color: str
    time_domain: TimeDomainMetrics
    frequency_domain: FrequencyDomainMetrics


class FeatureAnalysisResponse(BaseModel):
    """Response with time and frequency domain features per emotion."""
    study_number: int
    subject_id: int
    feature: str  # Which signal was analyzed (ECG, EDA, etc.)
    feature_title: str
    feature_unit: str
    emotions: list[EmotionFeatures]
    # Summary across all emotions
    summary: dict = Field(default_factory=dict)


class PowerSpectrumPoint(BaseModel):
    """Single point in power spectrum."""
    frequency: float
    power: float


class PowerSpectrumResponse(BaseModel):
    """Power spectrum data for visualization."""
    study_number: int
    subject_id: int
    feature: str
    emotions: list[EmotionColorMap]
    spectra: list[dict]  # [{emotion, color, data: [{frequency, power}]}]
    frequency_bands: dict = Field(
        default_factory=lambda: {
            "vlf": {"min": 0.003, "max": 0.04, "label": "VLF"},
            "lf": {"min": 0.04, "max": 0.15, "label": "LF"},
            "hf": {"min": 0.15, "max": 0.4, "label": "HF"}
        }
    )


class FFTSpectrumPoint(BaseModel):
    """Single point in FFT spectrum."""
    frequency: float
    magnitude: float


class EmotionFFTSpectrum(BaseModel):
    """FFT spectrum for one emotion."""
    emotion: str
    color: str
    dominant_frequency: float
    signal_energy: float
    spectrum: list[FFTSpectrumPoint]
    n_samples: int


class FFTSpectrumResponse(BaseModel):
    """FFT frequency spectrum response for visualization."""
    study_number: int
    subject_id: int | str  # Can be single int or comma-separated string for multiple subjects
    feature: str
    feature_title: str
    sampling_interval: float
    max_frequency: float
    emotions: list[EmotionFFTSpectrum]


class FeatureMatrixRow(BaseModel):
    """Single row in feature matrix (one subject-emotion combination)."""
    subject_id: int
    emotion: str
    color: str
    # Features are flattened directly onto the row as additional fields
    model_config = {"extra": "allow"}


class FeatureMatrixResponse(BaseModel):
    """Feature matrix response for table display and PCA."""
    study_number: int
    columns: list[str]  # Column names in order
    rows: list[dict]  # Each row is a flat dict with subject_id, emotion, color, and all feature values
    signals_used: list[str]
    n_subjects: int
    n_emotions: int
    n_features: int
