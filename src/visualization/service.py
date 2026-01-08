"""Service layer for visualization operations."""
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Lazy imports for PCA to speed up server startup
_PCA = None
_StandardScaler = None

def _get_sklearn_imports():
    """Lazy load sklearn to improve startup time."""
    global _PCA, _StandardScaler
    if _PCA is None:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        _PCA = PCA
        _StandardScaler = StandardScaler
    return _PCA, _StandardScaler

from src.database import get_db_connection
from src.visualization.constants import FEATURE_CONFIG, ALL_FEATURES, EMOTION_COLORS
from src.visualization.schemas import (
    EmotionColorMap,
    EmotionColoredSignalsResponse,
    SignalSeries,
    PCAResponse,
    PCADataPoint,
)
from src.visualization.stimuli import get_emotion_from_marker, get_study_name


def get_emotion_color_map(emotions: list[str]) -> list[EmotionColorMap]:
    """Generate emotion to color mapping using HUSL palette."""
    return [
        EmotionColorMap(
            emotion=emotion,
            color=EMOTION_COLORS[i % len(EMOTION_COLORS)]
        )
        for i, emotion in enumerate(emotions)
    ]


def get_available_features(study_number: int) -> list[str]:
    """Get list of features available in a study."""
    with get_db_connection() as conn:
        # Get column names from table
        result = conn.execute(f"DESCRIBE study{study_number}").fetchall()
        columns = [row[0] for row in result]

        # Filter to known features
        return [col for col in columns if col in ALL_FEATURES]


def get_emotion_colored_signals(
    study_number: int,
    subject_id: int,
    features: Optional[list[str]] = None,
    start_time: float = 0.0,
    end_time: float = 10.0,
    seconds_per_emotion: Optional[int] = None
) -> EmotionColoredSignalsResponse:
    """
    Get signal data colored by emotion.

    Implements POPANEpy pattern:
    - time_offset = timestamp - timestamp.iloc[0] (move timestamp to 0)
    - Color signals by emotion (derived from marker/stimuli)

    If seconds_per_emotion is provided, fetches first N seconds of each distinct
    emotion and normalizes time_offset sequentially (ignoring start_time/end_time).
    """
    with get_db_connection() as conn:
        # Get all available features if not specified
        available_features = get_available_features(study_number)

        if features is None:
            features = available_features
        else:
            # Filter to only available features
            features = [f for f in features if f in available_features]

        if not features:
            features = ['ECG'] if 'ECG' in available_features else available_features[:1]

        # Build column list - use marker instead of EMOTION
        columns = ['timestamp', 'marker'] + features
        columns_str = ', '.join(columns)

        # Query data for subject
        query = f"""
            SELECT {columns_str}
            FROM study{study_number}
            WHERE Subject_ID = ?
            ORDER BY timestamp
        """

        df_result = conn.execute(query, [subject_id]).fetchdf()

        if len(df_result) == 0:
            raise ValueError(f"No data found for subject {subject_id} in study {study_number}")

        # Map marker to emotion using stimuli metadata
        df_result['EMOTION'] = df_result['marker'].apply(get_emotion_from_marker)
        study_name = get_study_name(study_number)

        # Calculate time_offset from the first timestamp (POPANEpy pattern)
        # time_offset = timestamp - timestamp.iloc[0]
        base_timestamp = df_result['timestamp'].iloc[0]
        df_result['time_offset'] = df_result['timestamp'] - base_timestamp

        # Handle seconds_per_emotion mode: fetch first N seconds of each emotion
        if seconds_per_emotion is not None and seconds_per_emotion > 0:
            # Get unique emotions in order of first appearance
            unique_emotions_ordered = df_result['EMOTION'].unique().tolist()

            emotion_dfs = []
            cumulative_offset = 0.0

            for emotion in unique_emotions_ordered:
                # Get all data for this emotion
                emotion_df = df_result[df_result['EMOTION'] == emotion].copy()

                if len(emotion_df) == 0:
                    continue

                # Calculate time within this emotion segment
                emotion_base_time = emotion_df['time_offset'].iloc[0]
                emotion_df['emotion_local_time'] = emotion_df['time_offset'] - emotion_base_time

                # Take first N seconds of this emotion
                emotion_df = emotion_df[emotion_df['emotion_local_time'] <= seconds_per_emotion].copy()

                if len(emotion_df) == 0:
                    continue

                # Normalize time_offset to be sequential across emotions
                emotion_df['time_offset'] = emotion_df['emotion_local_time'] + cumulative_offset

                # Update cumulative offset for next emotion
                cumulative_offset = emotion_df['time_offset'].max() + 0.001  # Small gap between emotions

                emotion_dfs.append(emotion_df)

            if emotion_dfs:
                df_result = pd.concat(emotion_dfs, ignore_index=True)
            else:
                raise ValueError(f"No data found for any emotion with {seconds_per_emotion}s window")
        else:
            # Filter by time range BEFORE processing emotions (original behavior)
            df_result = df_result[
                (df_result['time_offset'] >= start_time) &
                (df_result['time_offset'] <= end_time)
            ].copy()

        if len(df_result) == 0:
            raise ValueError(f"No data found in time range {start_time}-{end_time}s")

        # Get unique emotions and create color map
        unique_emotions = df_result['EMOTION'].unique().tolist()
        emotion_color_map = get_emotion_color_map(unique_emotions)
        emotion_to_color = {ec.emotion: ec.color for ec in emotion_color_map}

        # Process each feature
        signals: list[SignalSeries] = []

        for feature in features:
            if feature not in df_result.columns:
                continue

            feature_config = FEATURE_CONFIG.get(feature, {
                'title': feature,
                'unit': '',
                'color': '#1f77b4'
            })

            data_points = []

            # Add data points with emotion coloring
            for _, row in df_result.iterrows():
                emotion = row['EMOTION']
                color = emotion_to_color[emotion]

                # Use pd.notna() to handle all non-numeric types safely
                value = None
                if pd.notna(row[feature]):
                    try:
                        value = float(row[feature])
                    except (ValueError, TypeError):
                        value = None

                data_points.append({
                    'time_offset': float(row['time_offset']),
                    'value': value,
                    'emotion': emotion,
                    'color': color
                })

            signals.append(SignalSeries(
                feature=feature,
                unit=feature_config['unit'],
                title=feature_config['title'],
                data_points=data_points
            ))

        # Calculate overall time range
        all_times = [dp['time_offset'] for s in signals for dp in s.data_points if dp.get('time_offset') is not None]
        time_range = {
            'min': min(all_times) if all_times else 0,
            'max': max(all_times) if all_times else end_time
        }

        return EmotionColoredSignalsResponse(
            study_number=study_number,
            subject_id=subject_id,
            study_name=study_name,
            emotions=emotion_color_map,
            signals=signals,
            available_features=available_features,
            time_range=time_range
        )


def get_pca_analysis(
    study_number: int,
    subject_ids: Optional[list[int]] = None,
    features: Optional[list[str]] = None,
    sample_size: int = 1000
) -> PCAResponse:
    """
    Perform PCA analysis on physiological data.

    Implements POPANEpy pattern:
    - StandardScaler for feature scaling
    - PCA with 2 components
    - Color points by emotion
    """
    with get_db_connection() as conn:
        # Get available features
        available_features = get_available_features(study_number)

        if features is None:
            # Use numeric physiological features for PCA
            features = [f for f in ['ECG', 'EDA', 'SBP', 'DBP', 'respiration', 'temp']
                       if f in available_features]
        else:
            features = [f for f in features if f in available_features]

        if len(features) < 2:
            raise ValueError("At least 2 features required for PCA")

        # Build subject filter
        subject_filter = ""
        if subject_ids:
            subject_ids_str = ', '.join(str(s) for s in subject_ids)
            subject_filter = f"AND Subject_ID IN ({subject_ids_str})"

        # Query data with sampling - use marker instead of EMOTION
        columns_str = ', '.join(['Subject_ID', 'marker'] + features)

        query = f"""
            SELECT {columns_str}
            FROM study{study_number}
            WHERE 1=1 {subject_filter}
            ORDER BY RANDOM()
            LIMIT ?
        """

        df = conn.execute(query, [sample_size * 10]).fetchdf()  # Get more, then filter

        if len(df) == 0:
            raise ValueError(f"No data found for study {study_number}")

        # Map marker to emotion
        df['EMOTION'] = df['marker'].apply(get_emotion_from_marker)

        # Convert feature columns to numeric, coercing errors to NaN
        for feature in features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')

        # Drop rows with NaN values in feature columns
        df = df.dropna(subset=features)

        # Sample if still too large
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        # Get unique emotions and color map
        unique_emotions = df['EMOTION'].unique().tolist()
        emotion_color_map = get_emotion_color_map(unique_emotions)
        emotion_to_color = {ec.emotion: ec.color for ec in emotion_color_map}

        # Prepare data for PCA
        X = df[features].values
        y = df['EMOTION'].values
        subject_ids_arr = df['Subject_ID'].values

        # Lazy load sklearn
        PCA, StandardScaler = _get_sklearn_imports()

        # Scale features (POPANEpy pattern)
        scaler = StandardScaler() # type: ignore
        X_scaled = scaler.fit_transform(X)

        # Perform PCA (POPANEpy pattern)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)

        # Create data points
        data_points = []
        for i in range(len(pca_result)):
            emotion = str(y[i])
            data_points.append(PCADataPoint(
                pc1=float(pca_result[i, 0]),
                pc2=float(pca_result[i, 1]),
                emotion=emotion,
                color=emotion_to_color.get(emotion, '#808080'),
                subject_id=int(subject_ids_arr[i])
            ))

        return PCAResponse(
            study_number=study_number,
            data_points=data_points,
            explained_variance=[float(v) for v in pca.explained_variance_ratio_],
            features_used=features,
            emotions=emotion_color_map
        )


def get_frequency_pca(
    study_number: int,
    features: Optional[list[str]] = None,
    subject_ids: Optional[list[int]] = None,
) -> PCAResponse:
    """
    Compute PCA on comprehensive FFT and frequency-domain features across all subjects.

    OPTIMIZED: Uses single bulk query and vectorized groupby operations.

    For each (subject, emotion) combination, computes per signal:
    - FFT: dominant_frequency, signal_energy
    - Time-domain: mean, std, min, max, range
    - Frequency-domain (Welch): vlf_power, lf_power, hf_power, total_power,
                                lf_nu, hf_nu, lf_hf_ratio

    Args:
        study_number: Study number (1-7)
        features: Signal features to analyze (default: key physiological features)
        subject_ids: Specific subjects to include (default: all)

    Returns:
        PCAResponse with each point representing a subject-emotion combination
    """
    from src.visualization.features import compute_frequency_domain_features

    with get_db_connection() as conn:
        # Get available features
        available_features = get_available_features(study_number)

        if features is None:
            features = [f for f in ['ECG', 'EDA', 'SBP', 'DBP', 'dzdt', 'CO', 'TPR']
                       if f in available_features]
        else:
            features = [f for f in features if f in available_features]

        if not features:
            raise ValueError(f"No valid features found for study {study_number}")

        # Build subject filter for bulk query
        subject_filter = ""
        if subject_ids:
            subject_ids_str = ', '.join(str(s) for s in subject_ids)
            subject_filter = f"AND Subject_ID IN ({subject_ids_str})"

        # OPTIMIZED: Single bulk query for all subjects
        columns_str = ', '.join(['Subject_ID', 'timestamp', 'marker'] + features)
        query = f"""
            SELECT {columns_str}
            FROM study{study_number}
            WHERE 1=1 {subject_filter}
            ORDER BY Subject_ID, timestamp
        """
        df = conn.execute(query).fetchdf()

        if len(df) == 0:
            raise ValueError(f"No data found for study {study_number}")

        # Map marker to emotion (vectorized)
        df['EMOTION'] = df['marker'].apply(get_emotion_from_marker)

        # Calculate global sampling interval from first subject's data
        first_subject = df['Subject_ID'].iloc[0]
        first_subject_df = df[df['Subject_ID'] == first_subject]
        if len(first_subject_df) > 1:
            sampling_interval = float(first_subject_df['timestamp'].iloc[1] - first_subject_df['timestamp'].iloc[0])
        else:
            sampling_interval = 0.001
        sampling_rate = 1.0 / sampling_interval if sampling_interval > 0 else 1000.0

        # OPTIMIZED: Group by (Subject_ID, EMOTION) and compute features
        feature_rows = []
        row_metadata = []

        # Get unique (subject, emotion) combinations
        grouped = df.groupby(['Subject_ID', 'EMOTION'])

        for (subject_id, emotion), group_df in grouped:
            feature_vector = []
            valid_row = True

            for feature in features:
                signal_values = pd.to_numeric(group_df[feature], errors='coerce').dropna().values

                if len(signal_values) < 100:  # Need enough samples for frequency analysis
                    valid_row = False
                    break

                # Compute FFT features
                fft_result = _compute_fft_features_fast(signal_values, sampling_interval)

                # Compute time-domain features
                mean_val = float(np.nanmean(signal_values))
                std_val = float(np.nanstd(signal_values, ddof=1)) if len(signal_values) > 1 else 0.0
                min_val = float(np.nanmin(signal_values))
                max_val = float(np.nanmax(signal_values))
                range_val = max_val - min_val

                # Compute frequency-domain features (Welch PSD)
                freq_features = compute_frequency_domain_features(signal_values, sampling_rate)

                feature_vector.extend([
                    fft_result['dominant_frequency'],
                    fft_result['signal_energy'],
                    mean_val,
                    std_val,
                    min_val,
                    max_val,
                    range_val,
                    # Frequency-domain features (Welch)
                    freq_features.vlf_power,
                    freq_features.lf_power,
                    freq_features.hf_power,
                    freq_features.total_power,
                    freq_features.lf_nu,
                    freq_features.hf_nu,
                    freq_features.lf_hf_ratio,
                ])

            if valid_row and len(feature_vector) > 0:
                feature_rows.append(feature_vector)
                row_metadata.append({'subject_id': int(subject_id), 'emotion': str(emotion)})

        if len(feature_rows) < 3:
            raise ValueError(f"Insufficient data for PCA. Found only {len(feature_rows)} valid (subject, emotion) combinations.")

        # Convert to numpy array
        X = np.array(feature_rows)

        # Check for NaN/Inf and replace with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for zero variance columns and remove them
        col_std = np.std(X, axis=0)
        valid_cols = col_std > 1e-10
        if not np.all(valid_cols):
            X = X[:, valid_cols]

        if X.shape[1] < 2:
            raise ValueError("Not enough valid features with variance for PCA")

        # Get unique emotions and create color map
        unique_emotions = list(set(m['emotion'] for m in row_metadata))
        emotion_color_map = get_emotion_color_map(unique_emotions)
        emotion_to_color = {ec.emotion: ec.color for ec in emotion_color_map}

        # Lazy load sklearn
        PCA, StandardScaler = _get_sklearn_imports()

        # Scale features
        scaler = StandardScaler()  # type: ignore
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        n_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X_scaled)

        # Handle case where we only got 1 component
        if pca_result.shape[1] == 1:
            pca_result = np.column_stack([pca_result, np.zeros(pca_result.shape[0])])

        # Create data points
        data_points = [
            PCADataPoint(
                pc1=float(pca_result[i, 0]),
                pc2=float(pca_result[i, 1]),
                emotion=row_metadata[i]['emotion'],
                color=emotion_to_color.get(row_metadata[i]['emotion'], '#808080'),
                subject_id=row_metadata[i]['subject_id']
            )
            for i in range(len(pca_result))
        ]

        # Build feature names used (per signal)
        feature_names_used = []
        for f in features:
            feature_names_used.extend([
                f"{f}_dom_freq",
                f"{f}_energy",
                f"{f}_mean",
                f"{f}_std",
                f"{f}_min",
                f"{f}_max",
                f"{f}_range",
                f"{f}_vlf_power",
                f"{f}_lf_power",
                f"{f}_hf_power",
                f"{f}_total_power",
                f"{f}_lf_nu",
                f"{f}_hf_nu",
                f"{f}_lf_hf_ratio",
            ])

        # Filter to only valid columns
        if not np.all(valid_cols):
            feature_names_used = [f for i, f in enumerate(feature_names_used) if i < len(valid_cols) and valid_cols[i]]

        explained_var = [float(v) for v in pca.explained_variance_ratio_]
        if len(explained_var) == 1:
            explained_var.append(0.0)

        return PCAResponse(
            study_number=study_number,
            data_points=data_points,
            explained_variance=explained_var,
            features_used=feature_names_used,
            emotions=emotion_color_map
        )


def get_feature_matrix(
    study_number: int,
    features: Optional[list[str]] = None,
    subject_ids: Optional[list[int]] = None,
) -> dict:
    """
    Get the raw feature matrix for table display and analysis.

    Returns the same feature computation as get_frequency_pca but without
    performing PCA - just the raw feature values per (subject, emotion).

    OPTIMIZED: Uses ThreadPoolExecutor for parallel feature computation.

    Args:
        study_number: Study number (1-7)
        features: Signal features to analyze (default: key physiological features)
        subject_ids: Specific subjects to include (default: all)

    Returns:
        FeatureMatrixResponse with raw feature values
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.visualization.features import compute_frequency_domain_features
    from src.visualization.schemas import FeatureMatrixResponse

    with get_db_connection() as conn:
        # Get available features
        available_features = get_available_features(study_number)

        if features is None:
            features = [f for f in ['ECG', 'EDA', 'SBP', 'DBP', 'dzdt', 'CO', 'TPR']
                       if f in available_features]
        else:
            features = [f for f in features if f in available_features]

        if not features:
            raise ValueError(f"No valid features found for study {study_number}")

        # Build subject filter for bulk query
        subject_filter = ""
        if subject_ids:
            subject_ids_str = ', '.join(str(s) for s in subject_ids)
            subject_filter = f"AND Subject_ID IN ({subject_ids_str})"

        # Single bulk query for all subjects
        columns_str = ', '.join(['Subject_ID', 'timestamp', 'marker'] + features)
        query = f"""
            SELECT {columns_str}
            FROM study{study_number}
            WHERE 1=1 {subject_filter}
            ORDER BY Subject_ID, timestamp
        """
        df = conn.execute(query).fetchdf()

        if len(df) == 0:
            raise ValueError(f"No data found for study {study_number}")

        # Map marker to emotion (vectorized)
        df['EMOTION'] = df['marker'].apply(get_emotion_from_marker)

        # Calculate global sampling interval from first subject's data
        first_subject = df['Subject_ID'].iloc[0]
        first_subject_df = df[df['Subject_ID'] == first_subject]
        if len(first_subject_df) > 1:
            sampling_interval = float(first_subject_df['timestamp'].iloc[1] - first_subject_df['timestamp'].iloc[0])
        else:
            sampling_interval = 0.001
        sampling_rate = 1.0 / sampling_interval if sampling_interval > 0 else 1000.0

        # Build column names
        feature_columns = []
        for f in features:
            feature_columns.extend([
                f"{f}_dom_freq",
                f"{f}_energy",
                f"{f}_mean",
                f"{f}_std",
                f"{f}_min",
                f"{f}_max",
                f"{f}_range",
                f"{f}_vlf_power",
                f"{f}_lf_power",
                f"{f}_hf_power",
                f"{f}_total_power",
                f"{f}_lf_nu",
                f"{f}_hf_nu",
                f"{f}_lf_hf_ratio",
            ])

        # Get unique emotions and create color map
        unique_emotions = df['EMOTION'].unique().tolist()
        emotion_color_map = get_emotion_color_map(unique_emotions)
        emotion_to_color = {ec.emotion: ec.color for ec in emotion_color_map}

        # Pre-group data to avoid repeated groupby overhead
        grouped_data = {
            (subject_id, emotion): group_df
            for (subject_id, emotion), group_df in df.groupby(['Subject_ID', 'EMOTION'])
        }

        def process_group(key_group):
            """Process a single (subject_id, emotion) group - runs in thread."""
            (subject_id, emotion), group_df = key_group
            feature_dict = {}

            for feature in features:
                signal_values = pd.to_numeric(group_df[feature], errors='coerce').dropna().values

                if len(signal_values) < 100:
                    return None  # Skip this group

                # Compute FFT features
                fft_result = _compute_fft_features_fast(signal_values, sampling_interval)

                # Compute time-domain features (vectorized numpy)
                mean_val = float(np.nanmean(signal_values))
                std_val = float(np.nanstd(signal_values, ddof=1)) if len(signal_values) > 1 else 0.0
                min_val = float(np.nanmin(signal_values))
                max_val = float(np.nanmax(signal_values))
                range_val = max_val - min_val

                # Compute frequency-domain features (Welch PSD)
                freq_features = compute_frequency_domain_features(signal_values, sampling_rate)

                # Store in dict
                feature_dict[f"{feature}_dom_freq"] = round(fft_result['dominant_frequency'], 4)
                feature_dict[f"{feature}_energy"] = round(fft_result['signal_energy'], 4)
                feature_dict[f"{feature}_mean"] = round(mean_val, 4)
                feature_dict[f"{feature}_std"] = round(std_val, 4)
                feature_dict[f"{feature}_min"] = round(min_val, 4)
                feature_dict[f"{feature}_max"] = round(max_val, 4)
                feature_dict[f"{feature}_range"] = round(range_val, 4)
                feature_dict[f"{feature}_vlf_power"] = round(freq_features.vlf_power, 4)
                feature_dict[f"{feature}_lf_power"] = round(freq_features.lf_power, 4)
                feature_dict[f"{feature}_hf_power"] = round(freq_features.hf_power, 4)
                feature_dict[f"{feature}_total_power"] = round(freq_features.total_power, 4)
                feature_dict[f"{feature}_lf_nu"] = round(freq_features.lf_nu, 4)
                feature_dict[f"{feature}_hf_nu"] = round(freq_features.hf_nu, 4)
                feature_dict[f"{feature}_lf_hf_ratio"] = round(freq_features.lf_hf_ratio, 4)

            # Replace NaN/Inf with None
            for key, value in feature_dict.items():
                if np.isnan(value) or np.isinf(value):
                    feature_dict[key] = None

            return {
                'Subject': int(subject_id),
                'Emotion': str(emotion),
                '_color': emotion_to_color.get(str(emotion), '#808080'),  # Prefixed to exclude from columns
                **feature_dict
            }

        # Parallel processing with ThreadPoolExecutor
        rows = []
        max_workers = min(32, len(grouped_data))  # Cap at 32 threads

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_group, (key, group_df)): key
                for key, group_df in grouped_data.items()
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    rows.append(result)

        # Sort rows by Subject, then Emotion
        rows.sort(key=lambda r: (r['Subject'], r['Emotion']))

        # Get unique subjects
        unique_subjects = list(set(r['Subject'] for r in rows))

        return FeatureMatrixResponse(
            study_number=study_number,
            columns=['Subject', 'Emotion'] + feature_columns,
            rows=rows,
            signals_used=features,
            n_subjects=len(unique_subjects),
            n_emotions=len(unique_emotions),
            n_features=len(feature_columns)
        ) # type: ignore


def _compute_fft_features_fast(signal_values: np.ndarray, sampling_interval: float) -> dict:
    """
    Fast FFT feature computation - only returns dominant_frequency and signal_energy.
    Skips spectrum generation for performance.
    """
    from scipy.fft import fft, fftfreq
    from scipy.signal import find_peaks

    N = len(signal_values)
    if N < 4:
        return {'dominant_frequency': 0.0, 'signal_energy': 0.0}

    # Clean signal
    signal_clean = np.nan_to_num(signal_values, nan=0.0)

    # Compute FFT
    yf = fft(signal_clean)
    xf = fftfreq(N, sampling_interval)[:N // 2]
    magnitude = np.abs(yf[:N // 2])

    # Dominant frequency via peak finding
    try:
        peaks, _ = find_peaks(magnitude, height=0)
        if len(peaks) > 0:
            dominant_freq = float(xf[peaks[np.argmax(magnitude[peaks])]])
        else:
            dominant_freq = float(xf[np.argmax(magnitude)]) if len(xf) > 0 else 0.0
    except Exception:
        dominant_freq = float(xf[np.argmax(magnitude)]) if len(xf) > 0 else 0.0

    # Signal energy (log2 scale)
    raw_energy = float(np.sum(np.abs(yf) ** 2) / N)
    signal_energy = float(np.log2(raw_energy)) if raw_energy > 0 else 0.0

    return {'dominant_frequency': dominant_freq, 'signal_energy': signal_energy}


def get_feature_analysis(
    study_number: int,
    subject_id: int,
    feature: str = 'ECG',
    sampling_rate: float = 1000.0
) -> dict:
    """
    Compute time-domain and frequency-domain features for each emotion segment.

    Args:
        study_number: Study number (1-7)
        subject_id: Subject ID
        feature: Signal feature to analyze (ECG, EDA, etc.)
        sampling_rate: Sampling rate in Hz

    Returns:
        FeatureAnalysisResponse with metrics per emotion
    """
    from src.visualization.features import (
        compute_time_domain_features,
        compute_frequency_domain_features
    )
    from src.visualization.schemas import (
        FeatureAnalysisResponse,
        EmotionFeatures,
        TimeDomainMetrics,
        FrequencyDomainMetrics
    )
    from scipy import signal as scipy_signal

    with get_db_connection() as conn:
        available_features = get_available_features(study_number)

        if feature not in available_features:
            raise ValueError(f"Feature '{feature}' not available in study {study_number}")

        # Query data for subject
        query = f"""
            SELECT timestamp, marker, {feature}
            FROM study{study_number}
            WHERE Subject_ID = ?
            ORDER BY timestamp
        """

        df = conn.execute(query, [subject_id]).fetchdf()

        if len(df) == 0:
            raise ValueError(f"No data found for subject {subject_id} in study {study_number}")

        # Map marker to emotion
        df['EMOTION'] = df['marker'].apply(get_emotion_from_marker)

        # Convert feature to numeric
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

        # Get unique emotions
        unique_emotions = df['EMOTION'].unique().tolist()
        emotion_color_map = get_emotion_color_map(unique_emotions)
        emotion_to_color = {ec.emotion: ec.color for ec in emotion_color_map}

        # Compute features for each emotion segment
        emotion_features = []

        for emotion in unique_emotions:
            emotion_df = df[df['EMOTION'] == emotion]
            signal_values = emotion_df[feature].dropna().values

            if len(signal_values) < 10:
                continue  # Skip segments that are too short

            # Determine if this is RR interval data (for HRV-specific metrics)
            is_rr = feature.upper() in ['RR', 'IBI', 'NN']

            # Compute time-domain features
            time_features = compute_time_domain_features(
                signal_values,
                sampling_rate=sampling_rate,
                is_rr_intervals=is_rr
            )

            # Compute frequency-domain features
            freq_features = compute_frequency_domain_features(
                signal_values,
                sampling_rate=sampling_rate
            )

            emotion_features.append(EmotionFeatures(
                emotion=emotion,
                color=emotion_to_color.get(emotion, '#808080'),
                time_domain=TimeDomainMetrics(**time_features.to_dict()),
                frequency_domain=FrequencyDomainMetrics(**freq_features.to_dict())
            ))

        # Get feature config
        feature_config = FEATURE_CONFIG.get(feature, {})

        return FeatureAnalysisResponse(
            study_number=study_number,
            subject_id=subject_id,
            feature=feature,
            feature_title=feature_config.get('title', feature),
            feature_unit=feature_config.get('unit', ''),
            emotions=emotion_features,
            summary={
                'n_emotions': len(emotion_features),
                'total_samples': len(df),
                'sampling_rate': sampling_rate
            }
        ) # type: ignore


def get_power_spectrum(
    study_number: int,
    subject_id: int,
    feature: str = 'ECG',
    sampling_rate: float = 1000.0,
    max_freq: float = 0.5
) -> dict:
    """
    Compute power spectrum for each emotion segment.

    Args:
        study_number: Study number (1-7)
        subject_id: Subject ID
        feature: Signal feature to analyze
        sampling_rate: Sampling rate in Hz
        max_freq: Maximum frequency to include in spectrum (Hz)

    Returns:
        PowerSpectrumResponse with PSD data per emotion
    """
    from src.visualization.schemas import PowerSpectrumResponse
    from scipy import signal as scipy_signal

    with get_db_connection() as conn:
        available_features = get_available_features(study_number)

        if feature not in available_features:
            raise ValueError(f"Feature '{feature}' not available in study {study_number}")

        # Query data for subject
        query = f"""
            SELECT timestamp, marker, {feature}
            FROM study{study_number}
            WHERE Subject_ID = ?
            ORDER BY timestamp
        """

        df = conn.execute(query, [subject_id]).fetchdf()

        if len(df) == 0:
            raise ValueError(f"No data found for subject {subject_id} in study {study_number}")

        # Map marker to emotion
        df['EMOTION'] = df['marker'].apply(get_emotion_from_marker)

        # Convert feature to numeric
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

        # Get unique emotions
        unique_emotions = df['EMOTION'].unique().tolist()
        emotion_color_map = get_emotion_color_map(unique_emotions)
        emotion_to_color = {ec.emotion: ec.color for ec in emotion_color_map}

        # Compute PSD for each emotion
        spectra = []

        for emotion in unique_emotions:
            emotion_df = df[df['EMOTION'] == emotion]
            signal_values = emotion_df[feature].dropna().values

            if len(signal_values) < 256:  # Need enough samples for PSD
                continue

            # Detrend
            signal_detrended = signal_values - np.nanmean(signal_values)
            signal_clean = np.nan_to_num(signal_detrended, nan=0.0)

            # Compute PSD using Welch's method
            # Use larger nperseg for better frequency resolution
            # Target ~0.01 Hz resolution: nperseg = fs / 0.01 = 100000 at 1000 Hz
            target_freq_res = 0.01  # Hz
            ideal_nperseg = int(sampling_rate / target_freq_res)
            max_nperseg = len(signal_clean) // 2
            nperseg = min(ideal_nperseg, max_nperseg)
            nperseg = max(nperseg, 256)  # Minimum for stability

            freqs, psd = scipy_signal.welch(
                signal_clean,
                fs=sampling_rate,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                scaling='density'
            )

            # Filter to max frequency
            freq_mask = freqs <= max_freq
            freqs_filtered = freqs[freq_mask]
            psd_filtered = psd[freq_mask]

            # Convert to list of points
            spectrum_data = [
                {'frequency': float(f), 'power': float(p)}
                for f, p in zip(freqs_filtered, psd_filtered)
            ]

            spectra.append({
                'emotion': emotion,
                'color': emotion_to_color.get(emotion, '#808080'),
                'data': spectrum_data
            })

        return PowerSpectrumResponse(
            study_number=study_number,
            subject_id=subject_id,
            feature=feature,
            emotions=emotion_color_map,
            spectra=spectra
        ) # type: ignore


def get_fft_spectrum(
    study_number: int,
    subject_ids: list[int],
    feature: str = "DBP",
    max_freq: float = 2.0
) -> dict:
    """
    Get FFT-based frequency spectrum for a signal, grouped by emotion.

    This implements the frequency_analysis pattern from POPANEpy notebooks:
    - FFT for each emotion segment
    - Dominant frequency detection via peak finding
    - Signal energy calculation (log2 scale)
    - Normalized magnitude spectrum for visualization

    Args:
        study_number: Study number (1-7)
        subject_ids: List of Subject IDs to include
        feature: Signal to analyze (ECG, EDA, DBP, SBP, etc.)
        max_freq: Maximum frequency to display (Hz)

    Returns:
        FFTSpectrumResponse with spectrum data per emotion
    """
    from src.visualization.features import compute_fft_spectrum
    from src.visualization.schemas import FFTSpectrumResponse, EmotionFFTSpectrum, FFTSpectrumPoint

    with get_db_connection() as conn:
        # Check if feature exists
        available_features = get_available_features(study_number)
        if feature not in available_features:
            raise ValueError(f"Feature '{feature}' not available in study {study_number}. Available: {available_features}")

        # Query data for all requested subjects
        placeholders = ', '.join(['?' for _ in subject_ids])
        query = f"""
            SELECT Subject_ID, timestamp, marker, {feature}
            FROM study{study_number}
            WHERE Subject_ID IN ({placeholders})
            ORDER BY Subject_ID, timestamp
        """
        df = conn.execute(query, subject_ids).fetchdf()

        if len(df) == 0:
            raise ValueError(f"No data found for subjects {subject_ids} in study {study_number}")

        # Map marker to emotion
        df['EMOTION'] = df['marker'].apply(get_emotion_from_marker)

        # Calculate sampling interval from timestamps
        if len(df) > 1:
            sampling_interval = float(df['timestamp'].iloc[1] - df['timestamp'].iloc[0])
        else:
            sampling_interval = 0.001  # Default 1ms

        # Get unique emotions and create color map
        unique_emotions = df['EMOTION'].unique().tolist()
        emotion_color_map = get_emotion_color_map(unique_emotions)
        emotion_to_color = {ec.emotion: ec.color for ec in emotion_color_map}

        # Compute FFT spectrum for each emotion (aggregated across all subjects)
        emotion_spectra = []

        for emotion in unique_emotions:
            emotion_df = df[df['EMOTION'] == emotion]
            signal_values = emotion_df[feature].dropna().values

            if len(signal_values) < 10:  # Need minimum samples
                continue

            # Compute FFT spectrum
            fft_result = compute_fft_spectrum(
                signal_values,
                sampling_interval,
                max_freq
            )

            # Convert spectrum to Pydantic models
            spectrum_points = [
                FFTSpectrumPoint(frequency=p['frequency'], magnitude=p['magnitude'])
                for p in fft_result['spectrum']
            ]

            emotion_spectra.append(EmotionFFTSpectrum(
                emotion=emotion,
                color=emotion_to_color.get(emotion, '#808080'),
                dominant_frequency=fft_result['dominant_frequency'],
                signal_energy=fft_result['signal_energy'],
                spectrum=spectrum_points,
                n_samples=fft_result['n_samples']
            ))

        # Get feature config
        feature_config = FEATURE_CONFIG.get(feature, {'title': feature, 'unit': ''})

        # Return subject_id as comma-separated string if multiple, or single int
        subject_id_str = ','.join(map(str, subject_ids)) if len(subject_ids) > 1 else subject_ids[0]

        return FFTSpectrumResponse(
            study_number=study_number,
            subject_id=subject_id_str,
            feature=feature,
            feature_title=feature_config['title'],
            sampling_interval=sampling_interval,
            max_frequency=max_freq,
            emotions=emotion_spectra
        ) # type: ignore
