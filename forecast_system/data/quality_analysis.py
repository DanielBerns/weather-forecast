import json
from pathlib import Path
import numpy as np
import pandas as pd

def compute_column_stats(series):
    """
    Computes count, mean, std, min, 25%, 50%, 75%, max, nan count, and IQR outliers for a pandas Series.
    """
    s_clean = series.dropna()
    if len(s_clean) == 0:
        return {
            'count': 0, 'mean': 0.0, 'std': 0.0,
            'min': 0.0, 'q25': 0.0, 'median': 0.0, 'q75': 0.0, 'max': 0.0,
            'nan_count': int(series.isna().sum()),
            'nan_pct': 100.0,
            'outlier_count': 0,
            'outlier_pct': 0.0
        }

    q25, median, q75 = float(np.percentile(s_clean, 25)), float(np.percentile(s_clean, 50)), float(np.percentile(s_clean, 75))
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = s_clean[(s_clean < lower_bound) | (s_clean > upper_bound)]

    total_len = len(series)
    nan_cnt = int(series.isna().sum())

    return {
        'count': len(s_clean),
        'mean': round(float(s_clean.mean()), 2),
        'std': round(float(s_clean.std()), 2),
        'min': round(float(s_clean.min()), 2),
        'q25': round(q25, 2),
        'median': round(median, 2),
        'q75': round(q75, 2),
        'max': round(float(s_clean.max()), 2),
        'nan_count': nan_cnt,
        'nan_pct': round((nan_cnt / total_len) * 100.0, 2) if total_len > 0 else 0.0,
        'outlier_count': len(outliers),
        'outlier_pct': round((len(outliers) / len(s_clean)) * 100.0, 2) if len(s_clean) > 0 else 0.0
    }

def detect_bad_values(df, temp_min=-30.0, temp_max=50.0, dew_min=-40.0, dew_max=40.0):
    """
    Detects bad values: NaNs, physical threshold violations, and dew point depression anomalies.
    """
    total_records = len(df)
    nan_rows = df.isna().any(axis=1).sum()

    bad_temp = ((df['temp'] < temp_min) | (df['temp'] > temp_max)).sum() if 'temp' in df.columns else 0
    bad_dew = ((df['dew'] < dew_min) | (df['dew'] > dew_max)).sum() if 'dew' in df.columns else 0
    dew_above_temp = (df['dew'] > df['temp']).sum() if ('temp' in df.columns and 'dew' in df.columns) else 0

    return {
        'total_records': total_records,
        'nan_rows': int(nan_rows),
        'nan_row_pct': round((nan_rows / total_records) * 100.0, 2) if total_records > 0 else 0.0,
        'bad_temp_count': int(bad_temp),
        'bad_dew_count': int(bad_dew),
        'dew_above_temp_count': int(dew_above_temp)
    }

def compute_histogram_bins(series, num_bins=30, val_range=None):
    """
    Computes binned histogram counts and bin edges for a series.
    """
    s_clean = series.dropna()
    if len(s_clean) == 0:
        return {'bin_centers': [], 'counts': [], 'density': []}

    if val_range is None:
        val_range = (s_clean.min(), s_clean.max())

    counts, bin_edges = np.histogram(s_clean, bins=num_bins, range=val_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = counts.sum()
    density = (counts / total).tolist() if total > 0 else [0.0] * num_bins

    return {
        'bin_centers': [round(float(c), 2) for c in bin_centers],
        'counts': counts.tolist(),
        'density': [round(float(d), 4) for d in density]
    }

def analyze_dataset_splits(X_clean, df_combined=None, temp_min=-30.0, temp_max=50.0, dew_min=-40.0, dew_max=40.0):
    """
    Analyzes data properties across Train (<=2024), Validation (2025), and Test (2026) splits.
    """
    train_mask = X_clean.index < '2025-01-01'
    val_mask = (X_clean.index >= '2025-01-01') & (X_clean.index < '2026-01-01')
    test_mask = X_clean.index >= '2026-01-01'

    splits = {
        'Train (<=2024)': X_clean.loc[train_mask],
        'Validation (2025)': X_clean.loc[val_mask],
        'Test (2026)': X_clean.loc[test_mask]
    }

    results = {}

    # Define overall ranges for histograms
    hist_temp_min = float(X_clean['temp_lag0'].min()) if 'temp_lag0' in X_clean.columns else -15.0
    hist_temp_max = float(X_clean['temp_lag0'].max()) if 'temp_lag0' in X_clean.columns else 45.0
    hist_dew_min = float(X_clean['dew_lag0'].min()) if 'dew_lag0' in X_clean.columns else -25.0
    hist_dew_max = float(X_clean['dew_lag0'].max()) if 'dew_lag0' in X_clean.columns else 35.0

    for split_name, split_df in splits.items():
        if split_df.empty:
            continue

        start_date = split_df.index.min().strftime('%Y-%m-%d %H:%M')
        end_date = split_df.index.max().strftime('%Y-%m-%d %H:%M')

        # Compute column statistics for key variables
        key_cols = ['temp_lag0', 'dew_lag0', 'dew_depression_lag0']
        if 'temp_roll_mean_24h' in split_df.columns:
            key_cols.append('temp_roll_mean_24h')

        col_stats = {}
        for col in key_cols:
            if col in split_df.columns:
                display_name = col.replace('_lag0', '')
                col_stats[display_name] = compute_column_stats(split_df[col])

        # Rename temp_lag0 to temp for bad value detection helper
        temp_df = split_df.rename(columns={'temp_lag0': 'temp', 'dew_lag0': 'dew'})
        bad_val_report = detect_bad_values(temp_df, temp_min=temp_min, temp_max=temp_max, dew_min=dew_min, dew_max=dew_max)

        # Compute Histograms
        hist_temp = compute_histogram_bins(temp_df['temp'], num_bins=30, val_range=(hist_temp_min, hist_temp_max))
        hist_dew = compute_histogram_bins(temp_df['dew'], num_bins=30, val_range=(hist_dew_min, hist_dew_max))

        results[split_name] = {
            'records_count': len(split_df),
            'start_date': start_date,
            'end_date': end_date,
            'statistics': col_stats,
            'bad_values': bad_val_report,
            'histograms': {
                'temp': hist_temp,
                'dew': hist_dew
            }
        }

    return results

def export_data_properties_json(X_clean, output_path, temp_min=-30.0, temp_max=50.0, dew_min=-40.0, dew_max=40.0):
    """
    Generates data properties analysis and saves to JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    analysis_data = analyze_dataset_splits(X_clean, temp_min=temp_min, temp_max=temp_max, dew_min=dew_min, dew_max=dew_max)

    with open(output_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)

    print(f"Exported dataset properties & distribution analysis to {output_path}")
    return analysis_data
