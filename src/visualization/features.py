"""
Time-domain and frequency-domain feature extraction for physiological signals.

Implements standard HRV (Heart Rate Variability) analysis metrics following
Task Force of ESC/NASPE guidelines and Neurokit2/HeartPy conventions.

Time Domain Features:
- Mean, Std, Min, Max (basic statistics)
- RMSSD: Root Mean Square of Successive Differences
- SDNN: Standard Deviation of NN intervals
- pNN50: Percentage of successive RR intervals differing by >50ms
- SDSD: Standard Deviation of Successive Differences

Frequency Domain Features (via Welch's method):
- VLF: Very Low Frequency power (0.003-0.04 Hz)
- LF: Low Frequency power (0.04-0.15 Hz)
- HF: High Frequency power (0.15-0.4 Hz)
- LF/HF ratio: Sympathovagal balance indicator
- Total Power: Sum of all frequency bands
- Normalized LF/HF (LFnu, HFnu)
"""
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


@dataclass
class TimeDomainFeatures:
    """Time-domain features for a signal segment."""
    # Basic statistics
    mean: float
    std: float
    min_val: float
    max_val: float
    range_val: float

    # HRV-specific (for RR intervals)
    rmssd: Optional[float] = None  # Root Mean Square of Successive Differences
    sdnn: Optional[float] = None   # Standard Deviation of NN intervals
    sdsd: Optional[float] = None   # Standard Deviation of Successive Differences
    pnn50: Optional[float] = None  # Percentage of NN50
    pnn20: Optional[float] = None  # Percentage of NN20

    # Signal quality
    n_samples: int = 0
    duration_sec: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrequencyDomainFeatures:
    """Frequency-domain features for a signal segment."""
    # Power in frequency bands (ms^2 or µV^2 depending on signal)
    vlf_power: float = 0.0    # Very Low Frequency: 0.003-0.04 Hz
    lf_power: float = 0.0     # Low Frequency: 0.04-0.15 Hz
    hf_power: float = 0.0     # High Frequency: 0.15-0.4 Hz
    total_power: float = 0.0  # Total spectral power

    # Normalized units (percentage of LF+HF)
    lf_nu: float = 0.0        # LF in normalized units
    hf_nu: float = 0.0        # HF in normalized units

    # Ratios
    lf_hf_ratio: float = 0.0  # LF/HF ratio (sympathovagal balance)

    # Peak frequencies
    vlf_peak: float = 0.0     # Peak frequency in VLF band
    lf_peak: float = 0.0      # Peak frequency in LF band
    hf_peak: float = 0.0      # Peak frequency in HF band

    # Metadata
    n_samples: int = 0
    sampling_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_time_domain_features(
    signal_values: np.ndarray,
    sampling_rate: float = 1000.0,
    is_rr_intervals: bool = False
) -> TimeDomainFeatures:
    """
    Compute time-domain features from a signal.

    Args:
        signal_values: Array of signal values (or RR intervals in ms)
        sampling_rate: Sampling rate in Hz
        is_rr_intervals: If True, compute HRV-specific metrics (RMSSD, SDNN, etc.)

    Returns:
        TimeDomainFeatures dataclass with computed metrics
    """
    if len(signal_values) < 2:
        return TimeDomainFeatures(
            mean=float(signal_values[0]) if len(signal_values) == 1 else 0.0,
            std=0.0,
            min_val=float(signal_values[0]) if len(signal_values) == 1 else 0.0,
            max_val=float(signal_values[0]) if len(signal_values) == 1 else 0.0,
            range_val=0.0,
            n_samples=len(signal_values),
            duration_sec=len(signal_values) / sampling_rate
        )

    # Basic statistics
    mean_val = float(np.nanmean(signal_values))
    std_val = float(np.nanstd(signal_values, ddof=1))
    min_val = float(np.nanmin(signal_values))
    max_val = float(np.nanmax(signal_values))
    range_val = max_val - min_val

    # Duration
    n_samples = len(signal_values)
    duration_sec = n_samples / sampling_rate

    # HRV-specific metrics (only for RR intervals)
    rmssd = None
    sdnn = None
    sdsd = None
    pnn50 = None
    pnn20 = None

    if is_rr_intervals and len(signal_values) > 2:
        # Successive differences
        diff_rr = np.diff(signal_values)

        # RMSSD: Root Mean Square of Successive Differences
        rmssd = float(np.sqrt(np.nanmean(diff_rr ** 2)))

        # SDNN: Standard Deviation of NN intervals
        sdnn = std_val  # Same as std for NN intervals

        # SDSD: Standard Deviation of Successive Differences
        sdsd = float(np.nanstd(diff_rr, ddof=1))

        # pNN50: Percentage of successive intervals > 50ms
        nn50 = np.sum(np.abs(diff_rr) > 50)
        pnn50 = float(100 * nn50 / len(diff_rr))

        # pNN20: Percentage of successive intervals > 20ms
        nn20 = np.sum(np.abs(diff_rr) > 20)
        pnn20 = float(100 * nn20 / len(diff_rr))

    return TimeDomainFeatures(
        mean=mean_val,
        std=std_val,
        min_val=min_val,
        max_val=max_val,
        range_val=range_val,
        rmssd=rmssd,
        sdnn=sdnn,
        sdsd=sdsd,
        pnn50=pnn50,
        pnn20=pnn20,
        n_samples=n_samples,
        duration_sec=duration_sec
    )


def compute_frequency_domain_features(
    signal_values: np.ndarray,
    sampling_rate: float = 1000.0,
    vlf_band: tuple = (0.003, 0.04),
    lf_band: tuple = (0.04, 0.15),
    hf_band: tuple = (0.15, 0.4),
    method: str = 'welch'
) -> FrequencyDomainFeatures:
    """
    Compute frequency-domain features using Welch's method.

    Args:
        signal_values: Array of signal values
        sampling_rate: Sampling rate in Hz
        vlf_band: VLF frequency band (Hz)
        lf_band: LF frequency band (Hz)
        hf_band: HF frequency band (Hz)
        method: PSD method ('welch' or 'periodogram')

    Returns:
        FrequencyDomainFeatures dataclass with computed metrics
    """
    n_samples = len(signal_values)

    # Need at least a few seconds of data for meaningful frequency analysis
    min_samples = int(sampling_rate * 4)  # At least 4 seconds
    if n_samples < min_samples:
        return FrequencyDomainFeatures(
            n_samples=n_samples,
            sampling_rate=sampling_rate
        )

    # Remove mean (detrend)
    signal_detrended = signal_values - np.nanmean(signal_values)

    # Handle NaN values
    signal_clean = np.nan_to_num(signal_detrended, nan=0.0)

    # Compute PSD using Welch's method
    if method == 'welch':
        # For low frequency analysis, we need longer segments
        # To resolve 0.003 Hz, need at least 1/0.003 = 333 seconds
        # nperseg determines frequency resolution: df = fs / nperseg
        # For df = 0.001 Hz at fs = 1000 Hz, need nperseg = 1,000,000
        # Compromise: use larger segments for better low-freq resolution

        # Target frequency resolution of ~0.001 Hz (or as good as data allows)
        target_freq_res = 0.005  # 0.005 Hz resolution
        ideal_nperseg = int(sampling_rate / target_freq_res)

        # But can't exceed signal length / 2 (need at least 2 segments)
        max_nperseg = n_samples // 2
        nperseg = min(ideal_nperseg, max_nperseg)

        # Minimum nperseg for stability
        nperseg = max(nperseg, 256)

        freqs, psd = scipy_signal.welch(
            signal_clean,
            fs=sampling_rate,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling='density'
        )
    else:
        # Periodogram
        freqs, psd = scipy_signal.periodogram(
            signal_clean,
            fs=sampling_rate,
            scaling='density'
        )

    # Helper function to compute band power
    def band_power(freq_low: float, freq_high: float) -> tuple:
        """Compute power in frequency band and peak frequency."""
        idx = np.logical_and(freqs >= freq_low, freqs < freq_high)
        if not np.any(idx):
            return 0.0, 0.0

        band_psd = psd[idx]
        band_freqs = freqs[idx]

        # Total power in band (integrate PSD using trapezoidal rule)
        # Use scipy.integrate.trapezoid for NumPy 2.0+ compatibility
        from scipy import integrate
        power = float(integrate.trapezoid(band_psd, band_freqs))

        # Peak frequency
        peak_idx = np.argmax(band_psd)
        peak_freq = float(band_freqs[peak_idx])

        return power, peak_freq

    # Compute power in each band
    vlf_power, vlf_peak = band_power(*vlf_band)
    lf_power, lf_peak = band_power(*lf_band)
    hf_power, hf_peak = band_power(*hf_band)

    # Total power
    total_power = vlf_power + lf_power + hf_power

    # Normalized units (LF and HF as percentage of LF+HF)
    lf_hf_sum = lf_power + hf_power
    lf_nu = float(100 * lf_power / lf_hf_sum) if lf_hf_sum > 0 else 0.0
    hf_nu = float(100 * hf_power / lf_hf_sum) if lf_hf_sum > 0 else 0.0

    # LF/HF ratio
    lf_hf_ratio = float(lf_power / hf_power) if hf_power > 0 else 0.0


    return FrequencyDomainFeatures(
        vlf_power=vlf_power,
        lf_power=lf_power,
        hf_power=hf_power,
        total_power=total_power,
        lf_nu=lf_nu,
        hf_nu=hf_nu,
        lf_hf_ratio=lf_hf_ratio,
        vlf_peak=vlf_peak,
        lf_peak=lf_peak,
        hf_peak=hf_peak,
        n_samples=n_samples,
        sampling_rate=sampling_rate
    )


def compute_all_features(
    signal_values: np.ndarray,
    sampling_rate: float = 1000.0,
    is_rr_intervals: bool = False
) -> dict:
    """
    Compute both time-domain and frequency-domain features.

    Args:
        signal_values: Array of signal values
        sampling_rate: Sampling rate in Hz
        is_rr_intervals: If True, compute HRV-specific metrics

    Returns:
        Dictionary with 'time_domain' and 'frequency_domain' feature dicts
    """
    time_features = compute_time_domain_features(
        signal_values, sampling_rate, is_rr_intervals
    )

    freq_features = compute_frequency_domain_features(
        signal_values, sampling_rate
    )

    return {
        'time_domain': time_features.to_dict(),
        'frequency_domain': freq_features.to_dict()
    }


# ============================================================================
# FFT-based Frequency Spectrum Analysis
# ============================================================================

@dataclass
class EmotionFrequencySpectrum:
    """Frequency spectrum data for one emotion."""
    emotion: str
    color: str
    dominant_frequency: float
    signal_energy: float
    spectrum: List[Dict[str, float]]  # [{frequency, magnitude}, ...]
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


def get_dominant_frequency(xf: np.ndarray, yf: np.ndarray) -> float:
    """Find dominant frequency from FFT results using peak detection."""
    N = len(yf)
    magnitude = np.abs(yf[:N // 2])

    if len(magnitude) == 0:
        return 0.0

    # Find peaks in the magnitude spectrum
    try:
        peaks, _ = find_peaks(magnitude, height=0)

        if len(peaks) == 0:
            # No peaks found, return frequency of max magnitude
            return float(xf[np.argmax(magnitude)])

        # Return frequency of highest peak
        dominant_idx = peaks[np.argmax(magnitude[peaks])]
        return float(xf[dominant_idx])
    except Exception:
        return float(xf[np.argmax(magnitude)]) if len(xf) > 0 else 0.0


def calculate_signal_energy(yf: np.ndarray) -> float:
    """
    Calculate signal energy from FFT results.

    Signal energy gives an overall measure of the intensity of variations.
    Calculated as the sum of squared magnitudes of FFT results (Parseval's theorem).
    """
    if len(yf) == 0:
        return 0.0
    return float(np.sum(np.abs(yf) ** 2) / len(yf))


def compute_fft_spectrum(
    signal_values: np.ndarray,
    sampling_interval: float,
    max_freq: float = 2.0
) -> Dict[str, Any]:
    """
    Compute FFT-based frequency spectrum for visualization.

    Args:
        signal_values: Array of signal values
        sampling_interval: Time between samples (T = 1/fs)
        max_freq: Maximum frequency to include in output (Hz)

    Returns:
        Dictionary with spectrum data, dominant frequency, and signal energy
    """
    N = len(signal_values)

    if N < 4:
        return {
            'spectrum': [],
            'dominant_frequency': 0.0,
            'signal_energy': 0.0,
            'n_samples': N
        }

    # Remove NaN values
    signal_clean = np.nan_to_num(signal_values, nan=0.0)

    # Compute FFT
    yf = fft(signal_clean)
    xf = fftfreq(N, sampling_interval)[:N // 2]

    # Compute normalized magnitude (2/N * |FFT|)
    magnitude = 2.0 / N * np.abs(yf[:N // 2]) # type: ignore

    # Filter to max frequency
    freq_mask = (xf <= max_freq) & (xf >= 0.0)
    xf_filtered = xf[freq_mask]
    magnitude_filtered = magnitude[freq_mask]

    # Get dominant frequency and signal energy
    dominant_freq = get_dominant_frequency(xf, yf) # type: ignore
    raw_energy = calculate_signal_energy(yf) # type: ignore
    signal_energy = float(np.log2(raw_energy)) if raw_energy > 0 else 0.0

    # Build spectrum points (downsample if too many points for frontend)
    max_points = 500
    if len(xf_filtered) > max_points:
        indices = np.linspace(0, len(xf_filtered) - 1, max_points, dtype=int)
        xf_filtered = xf_filtered[indices]
        magnitude_filtered = magnitude_filtered[indices]

    spectrum = [
        {'frequency': float(f), 'magnitude': float(m)}
        for f, m in zip(xf_filtered, magnitude_filtered)
    ]

    return {
        'spectrum': spectrum,
        'dominant_frequency': dominant_freq,
        'signal_energy': signal_energy,
        'n_samples': N
    }
