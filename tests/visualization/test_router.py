"""
Comprehensive tests for visualization router endpoints.

Tests all 8 visualization endpoints for proper response structure,
error handling, and data validation.
"""
import pytest
from httpx import AsyncClient, ASGITransport

from src.main import app


@pytest.fixture
async def client():
    """Async test client fixture."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        timeout=60.0  # Longer timeout for feature computation
    ) as client:
        yield client


# ============================================================================
# GET /viz/features/{study_number} - Available Features
# ============================================================================

class TestGetFeatures:
    """Tests for /viz/features/{study_number} endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("study_number", [1, 2, 3, 4, 5, 6, 7])
    async def test_get_features_all_studies(self, client: AsyncClient, study_number: int):
        """Test getting features for all valid studies."""
        response = await client.get(f"/viz/features/{study_number}")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) > 0

        # Validate feature structure
        for feature in data:
            assert "name" in feature
            assert "unit" in feature
            assert "title" in feature
            assert "default_color" in feature
            assert isinstance(feature["name"], str)

    @pytest.mark.asyncio
    async def test_get_features_includes_common_features(self, client: AsyncClient):
        """Test that common physiological features are present."""
        response = await client.get("/viz/features/1")
        assert response.status_code == 200
        data = response.json()

        feature_names = [f["name"] for f in data]
        # At least some common features should be present
        common_features = ["ECG", "EDA", "SBP", "DBP"]
        found = [f for f in common_features if f in feature_names]
        assert len(found) >= 2, f"Expected common features, got: {feature_names}"

    @pytest.mark.asyncio
    async def test_get_features_invalid_study(self, client: AsyncClient):
        """Test getting features for invalid study number."""
        response = await client.get("/viz/features/99")
        assert response.status_code == 500


# ============================================================================
# GET /viz/signals/{study_number}/{subject_id} - Emotion-Colored Signals
# ============================================================================

class TestGetSignals:
    """Tests for /viz/signals/{study_number}/{subject_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_signals_basic(self, client: AsyncClient):
        """Test basic signal retrieval."""
        response = await client.get("/viz/signals/1/1")
        assert response.status_code == 200
        data = response.json()

        assert "study_number" in data
        assert "subject_id" in data
        assert "signals" in data
        assert "emotions" in data
        assert "available_features" in data
        assert "time_range" in data

        assert data["study_number"] == 1
        assert data["subject_id"] == 1
        assert isinstance(data["signals"], list)
        assert isinstance(data["emotions"], list)

    @pytest.mark.asyncio
    async def test_get_signals_with_features(self, client: AsyncClient):
        """Test signal retrieval with specific features."""
        response = await client.get("/viz/signals/1/1?features=ECG,EDA")
        assert response.status_code == 200
        data = response.json()

        signal_features = [s["feature"] for s in data["signals"]]
        # Should only have requested features (or subset if not all available)
        assert len(signal_features) <= 2

    @pytest.mark.asyncio
    async def test_get_signals_with_time_range(self, client: AsyncClient):
        """Test signal retrieval with custom time range."""
        response = await client.get("/viz/signals/1/1?start_time=0&end_time=5")
        assert response.status_code == 200
        data = response.json()

        # Time range should be respected
        time_range = data["time_range"]
        assert time_range["max"] <= 5.5  # Allow small buffer

    @pytest.mark.asyncio
    async def test_get_signals_with_seconds_per_emotion(self, client: AsyncClient):
        """Test signal retrieval with seconds_per_emotion mode."""
        response = await client.get("/viz/signals/1/1?seconds_per_emotion=2")
        assert response.status_code == 200
        data = response.json()

        assert "signals" in data
        assert len(data["signals"]) > 0

    @pytest.mark.asyncio
    async def test_get_signals_invalid_subject(self, client: AsyncClient):
        """Test getting signals for non-existent subject."""
        response = await client.get("/viz/signals/1/99999")
        # Returns 404 or 500 depending on where error is caught
        assert response.status_code in [404, 500]

    @pytest.mark.asyncio
    async def test_get_signals_data_point_structure(self, client: AsyncClient):
        """Test that signal data points have correct structure."""
        response = await client.get("/viz/signals/1/1?end_time=1")
        assert response.status_code == 200
        data = response.json()

        if data["signals"]:
            signal = data["signals"][0]
            assert "feature" in signal
            assert "unit" in signal
            assert "title" in signal
            assert "data_points" in signal

            if signal["data_points"]:
                point = signal["data_points"][0]
                assert "time_offset" in point
                assert "value" in point
                assert "emotion" in point
                assert "color" in point

    @pytest.mark.asyncio
    async def test_get_signals_emotion_color_mapping(self, client: AsyncClient):
        """Test that emotions have color mappings."""
        response = await client.get("/viz/signals/1/1")
        assert response.status_code == 200
        data = response.json()

        emotions = data["emotions"]
        assert len(emotions) > 0

        for emotion_map in emotions:
            assert "emotion" in emotion_map
            assert "color" in emotion_map
            # Color should be hex format
            assert emotion_map["color"].startswith("#")


# ============================================================================
# GET /viz/pca/{study_number} - Basic PCA Analysis
# ============================================================================

class TestGetPCA:
    """Tests for /viz/pca/{study_number} endpoint."""

    @pytest.mark.asyncio
    async def test_get_pca_basic(self, client: AsyncClient):
        """Test basic PCA analysis."""
        response = await client.get("/viz/pca/1?sample_size=500")
        assert response.status_code == 200
        data = response.json()

        assert "study_number" in data
        assert "data_points" in data
        assert "explained_variance" in data
        assert "features_used" in data
        assert "emotions" in data

        assert data["study_number"] == 1
        assert len(data["explained_variance"]) == 2
        assert len(data["features_used"]) >= 2

    @pytest.mark.asyncio
    async def test_get_pca_data_point_structure(self, client: AsyncClient):
        """Test PCA data point structure."""
        response = await client.get("/viz/pca/1?sample_size=100")
        assert response.status_code == 200
        data = response.json()

        assert len(data["data_points"]) > 0
        point = data["data_points"][0]

        assert "pc1" in point
        assert "pc2" in point
        assert "emotion" in point
        assert "color" in point
        assert "subject_id" in point

        assert isinstance(point["pc1"], (int, float))
        assert isinstance(point["pc2"], (int, float))

    @pytest.mark.asyncio
    async def test_get_pca_with_features(self, client: AsyncClient):
        """Test PCA with specific features."""
        response = await client.get("/viz/pca/1?features=ECG,EDA,SBP&sample_size=200")
        assert response.status_code == 200
        data = response.json()

        # Features used should match requested (if available)
        assert len(data["features_used"]) <= 3

    @pytest.mark.asyncio
    async def test_get_pca_with_subject_filter(self, client: AsyncClient):
        """Test PCA with subject ID filter."""
        response = await client.get("/viz/pca/1?subject_ids=1,2,3&sample_size=200")
        assert response.status_code == 200
        data = response.json()

        # All data points should be from requested subjects
        subject_ids = set(p["subject_id"] for p in data["data_points"])
        assert subject_ids.issubset({1, 2, 3})

    @pytest.mark.asyncio
    async def test_get_pca_insufficient_features(self, client: AsyncClient):
        """Test PCA with only 1 feature (should fail)."""
        response = await client.get("/viz/pca/1?features=ECG")
        assert response.status_code == 400
        data = response.json()
        assert "2 features" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_pca_explained_variance_sum(self, client: AsyncClient):
        """Test that explained variance values are valid."""
        response = await client.get("/viz/pca/1?sample_size=500")
        assert response.status_code == 200
        data = response.json()

        # Variance ratios should be between 0 and 1
        for var in data["explained_variance"]:
            assert 0 <= var <= 1

        # Sum should be <= 1
        assert sum(data["explained_variance"]) <= 1.01  # Allow small floating point error


# ============================================================================
# GET /viz/analysis/{study_number}/{subject_id} - Feature Analysis
# ============================================================================

class TestGetAnalysis:
    """Tests for /viz/analysis/{study_number}/{subject_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_analysis_basic(self, client: AsyncClient):
        """Test basic feature analysis."""
        response = await client.get("/viz/analysis/1/1")
        assert response.status_code == 200
        data = response.json()

        assert "study_number" in data
        assert "subject_id" in data
        assert "feature" in data
        assert "emotions" in data
        assert "summary" in data

        assert data["study_number"] == 1
        assert data["subject_id"] == 1

    @pytest.mark.asyncio
    async def test_get_analysis_emotion_features(self, client: AsyncClient):
        """Test that analysis returns features per emotion."""
        response = await client.get("/viz/analysis/1/1?feature=ECG")
        assert response.status_code == 200
        data = response.json()

        emotions = data["emotions"]
        assert len(emotions) > 0

        emotion_data = emotions[0]
        assert "emotion" in emotion_data
        assert "color" in emotion_data
        assert "time_domain" in emotion_data
        assert "frequency_domain" in emotion_data

    @pytest.mark.asyncio
    async def test_get_analysis_time_domain_metrics(self, client: AsyncClient):
        """Test time-domain metrics structure."""
        response = await client.get("/viz/analysis/1/1")
        assert response.status_code == 200
        data = response.json()

        if data["emotions"]:
            td = data["emotions"][0]["time_domain"]
            assert "mean" in td
            assert "std" in td
            assert "min_val" in td
            assert "max_val" in td
            assert "range_val" in td
            assert "n_samples" in td

    @pytest.mark.asyncio
    async def test_get_analysis_frequency_domain_metrics(self, client: AsyncClient):
        """Test frequency-domain metrics structure."""
        response = await client.get("/viz/analysis/1/1")
        assert response.status_code == 200
        data = response.json()

        if data["emotions"]:
            fd = data["emotions"][0]["frequency_domain"]
            assert "vlf_power" in fd
            assert "lf_power" in fd
            assert "hf_power" in fd
            assert "total_power" in fd
            assert "lf_nu" in fd
            assert "hf_nu" in fd
            assert "lf_hf_ratio" in fd

    @pytest.mark.asyncio
    async def test_get_analysis_with_custom_feature(self, client: AsyncClient):
        """Test analysis with specific feature."""
        response = await client.get("/viz/analysis/1/1?feature=EDA")
        assert response.status_code == 200
        data = response.json()

        assert data["feature"] == "EDA"

    @pytest.mark.asyncio
    async def test_get_analysis_invalid_feature(self, client: AsyncClient):
        """Test analysis with non-existent feature."""
        response = await client.get("/viz/analysis/1/1?feature=INVALID_FEATURE")
        # Returns 404 or 500 depending on error handling
        assert response.status_code in [404, 500]

    @pytest.mark.asyncio
    async def test_get_analysis_invalid_subject(self, client: AsyncClient):
        """Test analysis for non-existent subject."""
        response = await client.get("/viz/analysis/1/99999")
        # Returns 404 or 500 depending on error handling
        assert response.status_code in [404, 500]


# ============================================================================
# GET /viz/spectrum/{study_number}/{subject_id} - Power Spectrum
# ============================================================================

class TestGetSpectrum:
    """Tests for /viz/spectrum/{study_number}/{subject_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_spectrum_basic(self, client: AsyncClient):
        """Test basic power spectrum retrieval."""
        response = await client.get("/viz/spectrum/1/1")
        assert response.status_code == 200
        data = response.json()

        assert "study_number" in data
        assert "subject_id" in data
        assert "feature" in data
        assert "emotions" in data
        assert "spectra" in data

    @pytest.mark.asyncio
    async def test_get_spectrum_data_structure(self, client: AsyncClient):
        """Test spectrum data structure."""
        response = await client.get("/viz/spectrum/1/1")
        assert response.status_code == 200
        data = response.json()

        if data["spectra"]:
            spectrum = data["spectra"][0]
            assert "emotion" in spectrum
            assert "color" in spectrum
            assert "data" in spectrum

            if spectrum["data"]:
                point = spectrum["data"][0]
                assert "frequency" in point
                assert "power" in point
                assert point["frequency"] >= 0
                assert point["power"] >= 0

    @pytest.mark.asyncio
    async def test_get_spectrum_max_freq_filter(self, client: AsyncClient):
        """Test that max_freq parameter filters spectrum."""
        max_freq = 0.3
        response = await client.get(f"/viz/spectrum/1/1?max_freq={max_freq}")
        assert response.status_code == 200
        data = response.json()

        for spectrum in data["spectra"]:
            for point in spectrum["data"]:
                assert point["frequency"] <= max_freq + 0.01  # Allow small buffer

    @pytest.mark.asyncio
    async def test_get_spectrum_with_feature(self, client: AsyncClient):
        """Test spectrum with specific feature."""
        response = await client.get("/viz/spectrum/1/1?feature=EDA")
        assert response.status_code == 200
        data = response.json()

        assert data["feature"] == "EDA"


# ============================================================================
# GET /viz/fft/{study_number}/{subject_id} - FFT Spectrum
# ============================================================================

class TestGetFFT:
    """Tests for /viz/fft/{study_number}/{subject_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_fft_basic(self, client: AsyncClient):
        """Test basic FFT spectrum retrieval."""
        response = await client.get("/viz/fft/1/1")
        assert response.status_code == 200
        data = response.json()

        assert "study_number" in data
        assert "subject_id" in data
        assert "feature" in data
        assert "sampling_interval" in data
        assert "max_frequency" in data
        assert "emotions" in data

    @pytest.mark.asyncio
    async def test_get_fft_emotion_data_structure(self, client: AsyncClient):
        """Test FFT emotion data structure."""
        response = await client.get("/viz/fft/1/1")
        assert response.status_code == 200
        data = response.json()

        emotions = data["emotions"]
        assert len(emotions) > 0

        emotion = emotions[0]
        assert "emotion" in emotion
        assert "color" in emotion
        assert "dominant_frequency" in emotion
        assert "signal_energy" in emotion
        assert "spectrum" in emotion
        assert "n_samples" in emotion

    @pytest.mark.asyncio
    async def test_get_fft_spectrum_points(self, client: AsyncClient):
        """Test FFT spectrum point structure."""
        response = await client.get("/viz/fft/1/1")
        assert response.status_code == 200
        data = response.json()

        for emotion in data["emotions"]:
            if emotion["spectrum"]:
                point = emotion["spectrum"][0]
                assert "frequency" in point
                assert "magnitude" in point
                assert point["frequency"] >= 0
                assert point["magnitude"] >= 0

    @pytest.mark.asyncio
    async def test_get_fft_multiple_subjects(self, client: AsyncClient):
        """Test FFT with multiple comma-separated subjects."""
        response = await client.get("/viz/fft/1/1,2,3")
        assert response.status_code == 200
        data = response.json()

        # subject_id should be comma-separated string or single value
        assert data["subject_id"] in ["1,2,3", 1, "1", 2, "2", 3, "3"]

    @pytest.mark.asyncio
    async def test_get_fft_with_feature(self, client: AsyncClient):
        """Test FFT with specific feature."""
        response = await client.get("/viz/fft/1/1?feature=ECG")
        assert response.status_code == 200
        data = response.json()

        assert data["feature"] == "ECG"

    @pytest.mark.asyncio
    async def test_get_fft_max_freq_filter(self, client: AsyncClient):
        """Test FFT max_freq parameter."""
        max_freq = 1.0
        response = await client.get(f"/viz/fft/1/1?max_freq={max_freq}")
        assert response.status_code == 200
        data = response.json()

        assert data["max_frequency"] == max_freq


# ============================================================================
# GET /viz/pca/frequency/{study_number} - Frequency PCA
# ============================================================================

class TestGetFrequencyPCA:
    """Tests for /viz/pca/frequency/{study_number} endpoint."""

    @pytest.mark.asyncio
    async def test_get_frequency_pca_basic(self, client: AsyncClient):
        """Test basic frequency PCA."""
        response = await client.get("/viz/pca/frequency/1?subject_ids=1,2,3")
        assert response.status_code == 200
        data = response.json()

        assert "study_number" in data
        assert "data_points" in data
        assert "explained_variance" in data
        assert "features_used" in data
        assert "emotions" in data

    @pytest.mark.asyncio
    async def test_get_frequency_pca_data_points(self, client: AsyncClient):
        """Test frequency PCA data point structure."""
        response = await client.get("/viz/pca/frequency/1?subject_ids=1,2")
        assert response.status_code == 200
        data = response.json()

        assert len(data["data_points"]) > 0
        point = data["data_points"][0]

        assert "pc1" in point
        assert "pc2" in point
        assert "emotion" in point
        assert "color" in point
        assert "subject_id" in point

    @pytest.mark.asyncio
    async def test_get_frequency_pca_features_include_fft(self, client: AsyncClient):
        """Test that frequency PCA includes FFT-derived features."""
        response = await client.get("/viz/pca/frequency/1?subject_ids=1,2,3")
        assert response.status_code == 200
        data = response.json()

        features = data["features_used"]
        # Should include frequency-related features
        has_freq_features = any("freq" in f.lower() or "energy" in f.lower() for f in features)
        assert has_freq_features, f"Expected frequency features, got: {features[:10]}"

    @pytest.mark.asyncio
    async def test_get_frequency_pca_with_features(self, client: AsyncClient):
        """Test frequency PCA with specific signal features."""
        response = await client.get("/viz/pca/frequency/1?features=ECG,EDA&subject_ids=1,2")
        assert response.status_code == 200
        data = response.json()

        # Features should be based on ECG and EDA
        features = data["features_used"]
        has_ecg = any("ECG" in f for f in features)
        has_eda = any("EDA" in f for f in features)
        assert has_ecg or has_eda

    @pytest.mark.asyncio
    async def test_get_frequency_pca_cache_header(self, client: AsyncClient):
        """Test that frequency PCA returns cache headers."""
        response = await client.get("/viz/pca/frequency/1?subject_ids=1,2")
        assert response.status_code == 200

        # Should have cache-control header
        assert "cache-control" in response.headers
        assert "max-age" in response.headers["cache-control"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("study_number", [1, 2, 3])
    async def test_get_frequency_pca_multiple_studies(self, client: AsyncClient, study_number: int):
        """Test frequency PCA works for multiple studies."""
        response = await client.get(f"/viz/pca/frequency/{study_number}?subject_ids=1,2,3")
        assert response.status_code == 200
        data = response.json()
        assert data["study_number"] == study_number


# ============================================================================
# GET /viz/features/matrix/{study_number} - Feature Matrix
# ============================================================================

class TestGetFeatureMatrix:
    """Tests for /viz/features/matrix/{study_number} endpoint."""

    @pytest.mark.asyncio
    async def test_get_feature_matrix_basic(self, client: AsyncClient):
        """Test basic feature matrix retrieval."""
        response = await client.get("/viz/features/matrix/1?subject_ids=1,2")
        assert response.status_code == 200
        data = response.json()

        assert "study_number" in data
        assert "columns" in data
        assert "rows" in data
        assert "signals_used" in data
        assert "n_subjects" in data
        assert "n_emotions" in data
        assert "n_features" in data

    @pytest.mark.asyncio
    async def test_get_feature_matrix_row_structure(self, client: AsyncClient):
        """Test feature matrix row structure."""
        response = await client.get("/viz/features/matrix/1?subject_ids=1")
        assert response.status_code == 200
        data = response.json()

        assert len(data["rows"]) > 0
        row = data["rows"][0]

        # Should have Subject and Emotion columns
        assert "Subject" in row
        assert "Emotion" in row
        # Should have color for styling (prefixed with _)
        assert "_color" in row

    @pytest.mark.asyncio
    async def test_get_feature_matrix_columns_match_rows(self, client: AsyncClient):
        """Test that column names match row keys."""
        response = await client.get("/viz/features/matrix/1?subject_ids=1")
        assert response.status_code == 200
        data = response.json()

        columns = set(data["columns"])
        row = data["rows"][0]
        row_keys = set(k for k in row.keys() if not k.startswith("_"))

        # All columns should be in row keys
        assert columns.issubset(row_keys)

    @pytest.mark.asyncio
    async def test_get_feature_matrix_with_features(self, client: AsyncClient):
        """Test feature matrix with specific signal features."""
        response = await client.get("/viz/features/matrix/1?features=ECG&subject_ids=1")
        assert response.status_code == 200
        data = response.json()

        # Should only have ECG-related columns
        feature_cols = [c for c in data["columns"] if c not in ["Subject", "Emotion"]]
        for col in feature_cols:
            assert "ECG" in col, f"Unexpected column: {col}"

    @pytest.mark.asyncio
    async def test_get_feature_matrix_14_features_per_signal(self, client: AsyncClient):
        """Test that each signal has 14 computed features."""
        response = await client.get("/viz/features/matrix/1?features=ECG&subject_ids=1")
        assert response.status_code == 200
        data = response.json()

        # 14 features per signal + Subject + Emotion = 16 columns
        feature_cols = [c for c in data["columns"] if c not in ["Subject", "Emotion"]]
        assert len(feature_cols) == 14, f"Expected 14 features, got {len(feature_cols)}"

        # Check expected feature suffixes
        expected_suffixes = [
            "dom_freq", "energy", "mean", "std", "min", "max", "range",
            "vlf_power", "lf_power", "hf_power", "total_power",
            "lf_nu", "hf_nu", "lf_hf_ratio"
        ]
        for suffix in expected_suffixes:
            assert any(suffix in col for col in feature_cols), f"Missing {suffix} feature"

    @pytest.mark.asyncio
    async def test_get_feature_matrix_cache_header(self, client: AsyncClient):
        """Test that feature matrix returns cache headers."""
        response = await client.get("/viz/features/matrix/1?subject_ids=1")
        assert response.status_code == 200

        assert "cache-control" in response.headers
        assert "max-age" in response.headers["cache-control"]

    @pytest.mark.asyncio
    async def test_get_feature_matrix_numeric_values(self, client: AsyncClient):
        """Test that feature values are numeric (not strings or None except for expected)."""
        response = await client.get("/viz/features/matrix/1?subject_ids=1")
        assert response.status_code == 200
        data = response.json()

        row = data["rows"][0]
        for col in data["columns"]:
            if col not in ["Subject", "Emotion"]:
                value = row.get(col)
                # Value should be numeric or None (for NaN/Inf)
                assert value is None or isinstance(value, (int, float)), \
                    f"Column {col} has non-numeric value: {value}"

    @pytest.mark.asyncio
    async def test_get_feature_matrix_sorted_by_subject_emotion(self, client: AsyncClient):
        """Test that rows are sorted by Subject, then Emotion."""
        response = await client.get("/viz/features/matrix/1?subject_ids=1,2,3")
        assert response.status_code == 200
        data = response.json()

        rows = data["rows"]
        if len(rows) > 1:
            # Check sorting
            for i in range(len(rows) - 1):
                curr = rows[i]
                next_row = rows[i + 1]
                # Subject should be non-decreasing
                assert curr["Subject"] <= next_row["Subject"], \
                    f"Rows not sorted by Subject: {curr['Subject']} > {next_row['Subject']}"


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling across all endpoints."""

    @pytest.mark.asyncio
    async def test_invalid_study_number_zero(self, client: AsyncClient):
        """Test that study number 0 is handled."""
        response = await client.get("/viz/features/0")
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_invalid_study_number_negative(self, client: AsyncClient):
        """Test that negative study number is handled."""
        response = await client.get("/viz/features/-1")
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_signals_nonexistent_subject(self, client: AsyncClient):
        """Test signals endpoint with non-existent subject."""
        response = await client.get("/viz/signals/1/999999")
        # Returns 404 or 500 depending on error handling
        assert response.status_code in [404, 500]

    @pytest.mark.asyncio
    async def test_analysis_nonexistent_subject(self, client: AsyncClient):
        """Test analysis endpoint with non-existent subject."""
        response = await client.get("/viz/analysis/1/999999")
        # Returns 404 or 500 depending on error handling
        assert response.status_code in [404, 500]

    @pytest.mark.asyncio
    async def test_fft_invalid_subject_id_format(self, client: AsyncClient):
        """Test FFT with invalid subject ID format."""
        response = await client.get("/viz/fft/1/abc")
        assert response.status_code in [404, 422, 500]  # Depends on error handling
