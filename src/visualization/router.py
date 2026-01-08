"""Router for visualization endpoints."""
from typing import Optional
from fastapi import APIRouter, Query, HTTPException, Response

from src.visualization.schemas import (
    EmotionColoredSignalsResponse,
    PCAResponse,
    FeatureInfo,
    FeatureAnalysisResponse,
    PowerSpectrumResponse,
    FFTSpectrumResponse,
    FeatureMatrixResponse,
)
from src.visualization.service import (
    get_emotion_colored_signals,
    get_pca_analysis,
    get_available_features,
    get_feature_analysis,
    get_power_spectrum,
    get_fft_spectrum,
    get_frequency_pca,
    get_feature_matrix,
)
from src.visualization.constants import FEATURE_CONFIG

router = APIRouter(prefix="/viz", tags=["visualization"])

# Cache duration: 1 hour for computed features (they don't change)
CACHE_MAX_AGE = 3600


@router.get("/features/{study_number}", response_model=list[FeatureInfo])
async def get_features(study_number: int):
    """Get available features for a study."""
    try:
        features = get_available_features(study_number)
        return [
            FeatureInfo(
                name=f,
                unit=FEATURE_CONFIG.get(f, {}).get('unit', ''),
                title=FEATURE_CONFIG.get(f, {}).get('title', f),
                default_color=FEATURE_CONFIG.get(f, {}).get('color', '#1f77b4')
            )
            for f in features
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/{study_number}/{subject_id}", response_model=EmotionColoredSignalsResponse)
async def get_signals(
    study_number: int,
    subject_id: int,
    features: Optional[str] = Query(default=None, description="Comma-separated list of features"),
    start_time: float = Query(default=0.0, description="Start time in seconds"),
    end_time: float = Query(default=10.0, description="End time in seconds"),
    seconds_per_emotion: Optional[int] = Query(default=None, description="Fetch first N seconds of each distinct emotion (overrides start_time/end_time)"),
):
    """
    Get emotion-colored signal data for a subject.

    Implements POPANEpy visualization patterns:
    - Timestamps offset to start at 0
    - Signals colored by emotion using HUSL palette

    If seconds_per_emotion is provided, fetches the first N seconds of each
    distinct emotion and normalizes time_offset sequentially on one plot.
    """
    try:
        feature_list = features.split(',') if features else None
        return get_emotion_colored_signals(
            study_number=study_number,
            subject_id=subject_id,
            features=feature_list,
            start_time=start_time,
            end_time=end_time,
            seconds_per_emotion=seconds_per_emotion
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pca/{study_number}", response_model=PCAResponse)
async def get_pca(
    study_number: int,
    subject_ids: Optional[str] = Query(default=None, description="Comma-separated subject IDs"),
    features: Optional[str] = Query(default=None, description="Comma-separated feature names (minimum 2 required)"),
    sample_size: int = Query(default=1000, ge=100, le=100000, description="Number of samples for PCA"),
):
    """
    Get PCA analysis of physiological data colored by emotion.

    Implements POPANEpy analysis patterns:
    - StandardScaler for feature normalization
    - PCA with 2 components
    - Points colored by emotion

    Note: At least 2 features are required for PCA dimensionality reduction.
    """
    try:
        subject_id_list = [int(s) for s in subject_ids.split(',')] if subject_ids else None
        feature_list = features.split(',') if features else None

        # Validate minimum features before calling service
        if feature_list and len(feature_list) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 features are required for PCA. Please select more features."
            )

        return get_pca_analysis(
            study_number=study_number,
            subject_ids=subject_id_list,
            features=feature_list,
            sample_size=sample_size
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PCA error: {str(e)}")


# ============================================================================
# Time-Domain and Frequency-Domain Feature Analysis Endpoints
# ============================================================================

@router.get("/analysis/{study_number}/{subject_id}", response_model=FeatureAnalysisResponse)
async def get_analysis(
    study_number: int,
    subject_id: int,
    feature: str = Query(default='ECG', description="Signal feature to analyze"),
    sampling_rate: float = Query(default=1000.0, description="Sampling rate in Hz"),
):
    """
    Get time-domain and frequency-domain features for a signal, grouped by emotion.

    Time-domain features include:
    - Basic statistics: mean, std, min, max, range
    - HRV metrics (for RR intervals): RMSSD, SDNN, SDSD, pNN50, pNN20

    Frequency-domain features include:
    - Power bands: VLF (0.003-0.04 Hz), LF (0.04-0.15 Hz), HF (0.15-0.4 Hz)
    - Normalized units: LFnu, HFnu
    - LF/HF ratio (sympathovagal balance indicator)
    - Peak frequencies in each band
    """
    try:
        return get_feature_analysis(
            study_number=study_number,
            subject_id=subject_id,
            feature=feature,
            sampling_rate=sampling_rate
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/spectrum/{study_number}/{subject_id}", response_model=PowerSpectrumResponse)
async def get_spectrum(
    study_number: int,
    subject_id: int,
    feature: str = Query(default='ECG', description="Signal feature to analyze"),
    sampling_rate: float = Query(default=1000.0, description="Sampling rate in Hz"),
    max_freq: float = Query(default=0.5, description="Maximum frequency to display (Hz)"),
):
    """
    Get power spectrum (PSD) for a signal, grouped by emotion.

    Returns Welch's power spectral density estimate for each emotion segment,
    useful for visualizing frequency content differences between emotional states.

    Frequency bands are annotated:
    - VLF: 0.003-0.04 Hz (thermoregulation, hormonal)
    - LF: 0.04-0.15 Hz (mix of sympathetic and parasympathetic)
    - HF: 0.15-0.4 Hz (parasympathetic, respiratory)
    """
    try:
        return get_power_spectrum(
            study_number=study_number,
            subject_id=subject_id,
            feature=feature,
            sampling_rate=sampling_rate,
            max_freq=max_freq
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Spectrum error: {str(e)}")


@router.get("/fft/{study_number}/{subject_id}", response_model=FFTSpectrumResponse)
async def get_fft(
    study_number: int,
    subject_id: str,  # Can be single ID or comma-separated list
    feature: str = Query(default="DBP", description="Signal to analyze (ECG, EDA, DBP, SBP, etc.)"),
    max_freq: float = Query(default=2.0, description="Maximum frequency to display (Hz)"),
):
    """
    Get FFT-based frequency spectrum for a signal, grouped by emotion.

    Supports single subject or multiple subjects (comma-separated in path).
    When multiple subjects are provided, data is aggregated across all subjects.

    Implements POPANEpy frequency_analysis pattern:
    - FFT computed for each emotion segment
    - Dominant frequency detected via peak finding
    - Signal energy calculated (log2 scale)
    - Normalized magnitude spectrum (2/N * |FFT|) for visualization

    Returns spectrum data suitable for plotting frequency vs magnitude
    with each emotion as a separate line.

    Example response includes per-emotion:
    - dominant_frequency: Peak frequency in Hz
    - signal_energy: Log2 of total signal energy
    - spectrum: Array of {frequency, magnitude} points
    """
    try:
        # Parse subject_id - can be single ID or comma-separated list
        subject_ids = [int(s.strip()) for s in subject_id.split(',')]

        return get_fft_spectrum(
            study_number=study_number,
            subject_ids=subject_ids,
            feature=feature,
            max_freq=max_freq
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"FFT error: {str(e)}")


@router.get("/pca/frequency/{study_number}", response_model=PCAResponse)
async def get_frequency_pca_endpoint(
    study_number: int,
    response: Response,
    features: Optional[str] = Query(default=None, description="Comma-separated signal features (ECG, EDA, DBP, etc.)"),
    subject_ids: Optional[str] = Query(default=None, description="Comma-separated subject IDs (default: all subjects)"),
):
    """
    PCA on FFT-derived and time-domain features across all subjects.

    For each (subject, emotion) combination, computes per signal:
    - dominant_frequency: Peak frequency from FFT
    - signal_energy: Log2 of total FFT energy
    - mean: Time-domain mean
    - std: Time-domain standard deviation

    This creates a rich feature space capturing both frequency and amplitude
    characteristics of each emotional state per subject.

    Each point in the resulting PCA represents one (subject, emotion) pair,
    colored by emotion.

    Use cases:
    - Visualize how different emotions cluster in frequency/amplitude space
    - Compare physiological responses across subjects and emotions
    - Identify subjects with atypical emotional responses

    Example: /viz/pca/frequency/1?features=ECG,EDA,DBP
    """
    try:
        feature_list = features.split(',') if features else None
        subject_id_list = [int(s) for s in subject_ids.split(',')] if subject_ids else None

        result = get_frequency_pca(
            study_number=study_number,
            features=feature_list,
            subject_ids=subject_id_list,
        )

        # Cache for 1 hour - data is static
        response.headers["Cache-Control"] = f"public, max-age={CACHE_MAX_AGE}"
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Frequency PCA error: {str(e)}")


@router.get("/features/matrix/{study_number}", response_model=FeatureMatrixResponse)
async def get_feature_matrix_endpoint(
    study_number: int,
    response: Response,
    features: Optional[str] = Query(default=None, description="Comma-separated signal features (ECG, EDA, DBP, etc.)"),
    subject_ids: Optional[str] = Query(default=None, description="Comma-separated subject IDs (default: all subjects)"),
):
    """
    Get raw feature matrix for table display.

    Returns the same features computed for PCA but in raw table format:
    - Each row is a (subject_id, emotion) combination
    - Each column is a computed feature (e.g., ECG_dom_freq, ECG_energy, ECG_mean, etc.)

    For each signal, computes 14 features:
    - FFT: dom_freq, energy
    - Time-domain: mean, std, min, max, range
    - Frequency-domain (Welch PSD): vlf_power, lf_power, hf_power, total_power, lf_nu, hf_nu, lf_hf_ratio

    Use cases:
    - Display raw feature values in a table
    - Export data for external analysis
    - Inspect feature values before/after PCA

    Example: /viz/features/matrix/1?features=ECG,EDA
    """
    try:
        feature_list = features.split(',') if features else None
        subject_id_list = [int(s) for s in subject_ids.split(',')] if subject_ids else None

        result = get_feature_matrix(
            study_number=study_number,
            features=feature_list,
            subject_ids=subject_id_list,
        )

        response.headers["Cache-Control"] = f"public, max-age={CACHE_MAX_AGE}"
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feature matrix error: {str(e)}")
