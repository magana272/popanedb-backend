"""
Comprehensive tests for visualization feature extraction.

Tests FFT, time-domain, and frequency-domain feature computation
to ensure PCA produces valid 2D scatter plots (not straight lines).
"""
import pytest
import numpy as np
from src.visualization.features import (
    compute_time_domain_features,
    compute_frequency_domain_features,
    compute_fft_spectrum,
    get_dominant_frequency,
    calculate_signal_energy,
    TimeDomainFeatures,
    FrequencyDomainFeatures,
)


class TestTimeDomainFeatures:
    """Tests for time-domain feature extraction."""

    def test_basic_statistics(self):
        """Test basic statistics computation."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        features = compute_time_domain_features(signal, sampling_rate=1000.0)

        assert features.mean == pytest.approx(3.0, rel=1e-5)
        assert features.min_val == pytest.approx(1.0, rel=1e-5)
        assert features.max_val == pytest.approx(5.0, rel=1e-5)
        assert features.range_val == pytest.approx(4.0, rel=1e-5)
        assert features.n_samples == 5
        assert features.std > 0  # Should have some variance

    def test_sinusoidal_signal(self):
        """Test with a known sinusoidal signal."""
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
        features = compute_time_domain_features(signal, sampling_rate=1000.0)

        # Mean of sine wave should be ~0
        assert abs(features.mean) < 0.01
        # Std of sine wave with amplitude 1 is 1/sqrt(2) ≈ 0.707
        assert features.std == pytest.approx(0.707, rel=0.05)
        assert features.min_val == pytest.approx(-1.0, rel=0.01)
        assert features.max_val == pytest.approx(1.0, rel=0.01)

    def test_empty_signal(self):
        """Test with minimal signal."""
        signal = np.array([42.0])
        features = compute_time_domain_features(signal, sampling_rate=1000.0)

        assert features.mean == 42.0
        assert features.std == 0.0
        assert features.n_samples == 1

    def test_hrv_metrics(self):
        """Test HRV-specific metrics for RR intervals."""
        # Simulated RR intervals in ms (typical heart rate ~60-100 bpm)
        rr_intervals = np.array([800, 850, 820, 900, 810, 870, 830, 890, 815, 860])
        features = compute_time_domain_features(
            rr_intervals, sampling_rate=1.0, is_rr_intervals=True
        )

        assert features.rmssd is not None
        assert features.sdnn is not None
        assert features.sdsd is not None
        assert features.pnn50 is not None
        assert features.pnn20 is not None
        assert features.rmssd > 0
        assert features.sdnn > 0


class TestFrequencyDomainFeatures:
    """Tests for frequency-domain (Welch PSD) feature extraction.

    Note: For HRV frequency analysis, we need:
    - Low sampling rates (4-10 Hz) to get good frequency resolution in 0.04-0.4 Hz range
    - Long signals (>5 minutes ideal) to resolve VLF (<0.04 Hz)

    At 4 Hz sampling with 60 seconds: freq resolution = 4/60 = 0.067 Hz
    At 4 Hz sampling with 300 seconds: freq resolution = 4/300 = 0.013 Hz
    """

    def test_single_frequency(self):
        """Test with a pure sinusoidal signal at HRV-appropriate sampling."""
        fs = 4.0  # 4 Hz (typical for RR interval analysis)
        duration = 300  # 5 minutes
        t = np.linspace(0, duration, int(fs * duration))
        freq = 0.1  # 0.1 Hz (in LF band)
        signal = np.sin(2 * np.pi * freq * t)

        features = compute_frequency_domain_features(signal, sampling_rate=fs)

        # LF band should have most power for 0.1 Hz signal
        assert features.lf_power > features.vlf_power
        assert features.lf_power > features.hf_power
        assert features.total_power > 0
        # LF peak should be near 0.1 Hz
        assert 0.05 < features.lf_peak < 0.15

    def test_hf_signal(self):
        """Test with signal in HF band."""
        fs = 4.0  # 4 Hz sampling
        duration = 300  # 5 minutes
        t = np.linspace(0, duration, int(fs * duration))
        freq = 0.25  # 0.25 Hz (in HF band: 0.15-0.4 Hz)
        signal = np.sin(2 * np.pi * freq * t)

        features = compute_frequency_domain_features(signal, sampling_rate=fs)

        # HF band should dominate
        assert features.hf_power > features.lf_power
        assert features.hf_nu > features.lf_nu
        assert 0.15 < features.hf_peak < 0.4

    def test_lf_hf_ratio(self):
        """Test LF/HF ratio computation."""
        fs = 4.0  # 4 Hz sampling
        duration = 300  # 5 minutes
        t = np.linspace(0, duration, int(fs * duration))
        # Mix of LF (0.1 Hz) and HF (0.25 Hz) components
        signal = 2 * np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 0.25 * t)

        features = compute_frequency_domain_features(signal, sampling_rate=fs)

        # LF should dominate (amplitude 2 vs 1)
        assert features.lf_hf_ratio > 1.0
        assert features.lf_nu > features.hf_nu

    def test_normalized_units(self):
        """Test that normalized units sum to ~100%."""
        fs = 4.0  # 4 Hz sampling
        duration = 300  # 5 minutes
        t = np.linspace(0, duration, int(fs * duration))
        signal = np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 0.25 * t)

        features = compute_frequency_domain_features(signal, sampling_rate=fs)

        # LFnu + HFnu should be ~100%
        assert features.lf_nu + features.hf_nu == pytest.approx(100.0, rel=0.01)

    def test_short_signal(self):
        """Test with signal too short for frequency analysis."""
        signal = np.array([1.0, 2.0, 3.0])  # Only 3 samples
        features = compute_frequency_domain_features(signal, sampling_rate=1000.0)

        # Should return default values
        assert features.total_power == 0.0
        assert features.n_samples == 3

    def test_metadata_preservation(self):
        """Test that metadata is correctly preserved."""
        fs = 4.0
        duration = 300
        t = np.linspace(0, duration, int(fs * duration))
        signal = np.sin(2 * np.pi * 0.1 * t)

        features = compute_frequency_domain_features(signal, sampling_rate=fs)

        assert features.n_samples == int(fs * duration)
        assert features.sampling_rate == fs


class TestFFTSpectrum:
    """Tests for FFT-based spectrum computation."""

    def test_dominant_frequency_detection(self):
        """Test that dominant frequency is correctly detected."""
        fs = 1000.0
        t = np.linspace(0, 1, int(fs))
        freq = 10.0  # 10 Hz
        signal = np.sin(2 * np.pi * freq * t)

        result = compute_fft_spectrum(signal, sampling_interval=1/fs, max_freq=50.0)

        # Dominant frequency should be ~10 Hz
        assert result['dominant_frequency'] == pytest.approx(10.0, rel=0.1)
        assert result['signal_energy'] != 0.0
        assert result['n_samples'] == int(fs)

    def test_multiple_frequencies(self):
        """Test with multiple frequency components."""
        fs = 1000.0
        t = np.linspace(0, 1, int(fs))
        # 5 Hz dominant, 20 Hz secondary
        signal = 3 * np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)

        result = compute_fft_spectrum(signal, sampling_interval=1/fs, max_freq=50.0)

        # 5 Hz should be dominant (higher amplitude)
        assert result['dominant_frequency'] == pytest.approx(5.0, rel=0.2)

    def test_spectrum_output(self):
        """Test that spectrum output is properly formatted."""
        fs = 1000.0
        t = np.linspace(0, 1, int(fs))
        signal = np.sin(2 * np.pi * 10 * t)

        result = compute_fft_spectrum(signal, sampling_interval=1/fs, max_freq=50.0)

        assert 'spectrum' in result
        assert len(result['spectrum']) > 0
        # Check spectrum point structure
        point = result['spectrum'][0]
        assert 'frequency' in point
        assert 'magnitude' in point
        assert point['frequency'] >= 0
        assert point['magnitude'] >= 0

    def test_max_freq_filtering(self):
        """Test that max_freq parameter filters spectrum correctly."""
        fs = 1000.0
        t = np.linspace(0, 1, int(fs))
        signal = np.sin(2 * np.pi * 10 * t)

        result = compute_fft_spectrum(signal, sampling_interval=1/fs, max_freq=5.0)

        # All frequencies should be <= max_freq
        for point in result['spectrum']:
            assert point['frequency'] <= 5.0

    def test_signal_energy(self):
        """Test signal energy computation."""
        # Higher amplitude = higher energy
        fs = 1000.0
        t = np.linspace(0, 1, int(fs))

        signal_low = np.sin(2 * np.pi * 10 * t)
        signal_high = 5 * np.sin(2 * np.pi * 10 * t)

        result_low = compute_fft_spectrum(signal_low, sampling_interval=1/fs)
        result_high = compute_fft_spectrum(signal_high, sampling_interval=1/fs)

        # Higher amplitude should have higher energy
        assert result_high['signal_energy'] > result_low['signal_energy']


class TestFeatureVariance:
    """Tests to ensure features have enough variance for PCA."""

    def test_different_signals_produce_different_features(self):
        """Verify that different signals produce different feature vectors."""
        fs = 4.0  # 4 Hz sampling (HRV-appropriate)
        duration = 300  # 5 minutes
        t = np.linspace(0, duration, int(fs * duration))

        # Signal 1: Low frequency, low amplitude
        signal1 = np.sin(2 * np.pi * 0.1 * t)

        # Signal 2: High frequency, high amplitude
        signal2 = 3 * np.sin(2 * np.pi * 0.3 * t)

        # Signal 3: Mixed frequencies
        signal3 = np.sin(2 * np.pi * 0.1 * t) + 2 * np.sin(2 * np.pi * 0.25 * t)

        features1 = compute_frequency_domain_features(signal1, fs)
        features2 = compute_frequency_domain_features(signal2, fs)
        features3 = compute_frequency_domain_features(signal3, fs)

        # Features should differ significantly
        assert features1.total_power != features2.total_power
        assert features1.lf_hf_ratio != features3.lf_hf_ratio
        assert features2.hf_power != features3.hf_power

    def test_feature_matrix_variance(self):
        """Test that feature matrix has variance across samples."""
        fs = 4.0  # 4 Hz sampling (HRV-appropriate)
        duration = 300  # 5 minutes
        t = np.linspace(0, duration, int(fs * duration))
        np.random.seed(42)  # For reproducibility

        # Generate multiple "emotion" signals
        signals = [
            np.sin(2 * np.pi * 0.1 * t),  # Baseline (LF dominant)
            2 * np.sin(2 * np.pi * 0.25 * t) + np.random.normal(0, 0.1, len(t)),  # Stress (HF + noise)
            np.sin(2 * np.pi * 0.08 * t) + 0.5 * np.sin(2 * np.pi * 0.25 * t),  # Relaxed (LF + some HF)
            3 * np.sin(2 * np.pi * 0.3 * t),  # Excited (high HF)
        ]

        feature_matrix = []
        for signal in signals:
            td = compute_time_domain_features(signal, fs)
            fd = compute_frequency_domain_features(signal, fs)
            fft = compute_fft_spectrum(signal, 1/fs)

            feature_vector = [
                fft['dominant_frequency'],
                fft['signal_energy'],
                td.mean,
                td.std,
                fd.lf_power,
                fd.hf_power,
                fd.lf_hf_ratio,
            ]
            feature_matrix.append(feature_vector)

        X = np.array(feature_matrix)

        # Check that most columns have variance (some might be zero if signals are similar)
        col_variances = np.var(X, axis=0)
        nonzero_variance_cols = np.sum(col_variances > 1e-10)
        assert nonzero_variance_cols >= 5, f"Only {nonzero_variance_cols} columns have variance"

        # PCA sanity check: should be able to reduce to 2D
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # Filter out zero-variance columns
        valid_cols = col_variances > 1e-10
        X_filtered = X[:, valid_cols]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)

        n_components = min(2, X_filtered.shape[1], X_filtered.shape[0])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Should not be a straight line
        pc1_var = np.var(X_pca[:, 0])
        assert pc1_var > 0, "PC1 has zero variance"

        if n_components > 1:
            # PC2 should have some variance (not a straight line)
            assert pca.explained_variance_ratio_[1] > 0.001, "PC2 explains too little variance - possible straight line"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        features = compute_time_domain_features(signal, sampling_rate=1000.0)

        # Should compute stats ignoring NaN
        assert not np.isnan(features.mean)
        assert not np.isnan(features.std)

    def test_constant_signal(self):
        """Test with constant signal (zero variance)."""
        signal = np.ones(1000) * 42.0
        features = compute_time_domain_features(signal, sampling_rate=1000.0)

        assert features.mean == 42.0
        assert features.std == pytest.approx(0.0, abs=1e-10)
        assert features.range_val == 0.0

    def test_very_short_signal_fft(self):
        """Test FFT with very short signal."""
        signal = np.array([1.0, 2.0])
        result = compute_fft_spectrum(signal, sampling_interval=0.001)

        # Should return empty/zero results
        assert result['n_samples'] == 2
        assert len(result['spectrum']) == 0 or result['dominant_frequency'] == 0.0

    def test_negative_values(self):
        """Test with negative signal values."""
        signal = np.array([-5.0, -3.0, -1.0, -4.0, -2.0])
        features = compute_time_domain_features(signal)

        assert features.mean == pytest.approx(-3.0, rel=1e-5)
        assert features.min_val == -5.0
        assert features.max_val == -1.0


class TestIntegration:
    """Integration tests combining multiple feature types."""

    def test_full_feature_pipeline(self):
        """Test complete feature extraction pipeline."""
        fs = 4.0  # 4 Hz sampling (HRV-appropriate)
        duration = 300  # 5 minutes
        t = np.linspace(0, duration, int(fs * duration))
        np.random.seed(42)

        # Realistic physiological-like signal
        signal = (
            0.5 * np.sin(2 * np.pi * 0.05 * t) +  # VLF component
            1.0 * np.sin(2 * np.pi * 0.1 * t) +  # LF component
            0.3 * np.sin(2 * np.pi * 0.25 * t) +  # HF component (respiration)
            0.1 * np.random.normal(0, 1, len(t))  # Noise
        )

        # Time-domain
        td = compute_time_domain_features(signal, fs)
        assert td.n_samples == int(fs * duration)
        assert td.duration_sec == pytest.approx(duration, rel=0.01)

        # Frequency-domain
        fd = compute_frequency_domain_features(signal, fs)
        assert fd.total_power > 0
        assert fd.lf_power > fd.hf_power  # LF should dominate

        # FFT
        fft = compute_fft_spectrum(signal, 1/fs, max_freq=2.0)
        assert fft['signal_energy'] != 0
        assert len(fft['spectrum']) > 0

        # Build feature vector (like PCA would use)
        feature_vector = [
            fft['dominant_frequency'],
            fft['signal_energy'],
            td.mean,
            td.std,
            td.min_val,
            td.max_val,
            td.range_val,
            fd.vlf_power,
            fd.lf_power,
            fd.hf_power,
            fd.total_power,
            fd.lf_nu,
            fd.hf_nu,
            fd.lf_hf_ratio,
        ]

        # All features should be finite numbers
        for i, val in enumerate(feature_vector):
            assert np.isfinite(val), f"Feature {i} is not finite: {val}"
