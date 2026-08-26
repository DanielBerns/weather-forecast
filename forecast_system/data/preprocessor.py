import numpy as np
import pandas as pd

def parse_hourly_temp(val_str):
    """
    Parses encoded weather values (e.g., '+0050,1' or '9999,9') into floats in Celsius.
    """
    if pd.isna(val_str):
        return np.nan
    if isinstance(val_str, (int, float)):
        val_float = float(val_str)
        if val_float in [9999, -9999, 999.9, -999.9]:
            return np.nan
        return val_float

    try:
        val_str = str(val_str).strip()
        if ',' in val_str:
            val_part, q_flag = val_str.split(',', 1)
        else:
            val_part, q_flag = val_str, '1'

        val = int(val_part)
        if val in [9999, -9999, 999, -999]:
            return np.nan
        if q_flag in ['3', '7', '9']:
            return np.nan
        return val / 10.0
    except Exception:
        return np.nan

def clean_hourly_dataframe(df, is_modern=False):
    """
    Standardizes hourly DataFrame into a clean time series indexed by hourly timestamps.
    Returns DataFrame with columns: ['temp', 'dew']
    """
    df = df.copy()

    # Parse DATE
    if 'DATE' in df.columns:
        df['timestamp'] = pd.to_datetime(df['DATE'], errors='coerce')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        raise ValueError("DataFrame missing DATE or timestamp column")

    df = df.dropna(subset=['timestamp'])

    if is_modern:
        # CRV2025 / CRV2026 format
        temp_col = 'temperature' if 'temperature' in df.columns else 'TMP'
        dew_col = 'dew_point_temperature' if 'dew_point_temperature' in df.columns else 'DEW'

        df['temp'] = df[temp_col].apply(parse_hourly_temp)
        df['dew'] = df[dew_col].apply(parse_hourly_temp)
    else:
        # tytd_hourly_CR format
        df['temp'] = df['TMP'].apply(parse_hourly_temp)
        df['dew'] = df['DEW'].apply(parse_hourly_temp)

    # Filter out physical extremes for Comodoro Rivadavia
    df.loc[(df['temp'] < -30) | (df['temp'] > 50), 'temp'] = np.nan
    df.loc[(df['dew'] < -40) | (df['dew'] > 40), 'dew'] = np.nan

    # Drop rows where both temp and dew are missing
    df = df.dropna(subset=['temp', 'dew'], how='all')

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Deduplicate sub-hourly records by averaging within the hour
    df['hour_floor'] = df['timestamp'].dt.floor('h')
    hourly_df = df.groupby('hour_floor')[['temp', 'dew']].mean().reset_index()
    hourly_df.rename(columns={'hour_floor': 'timestamp'}, inplace=True)
    hourly_df.set_index('timestamp', inplace=True)

    # Reindex to full hourly date range
    if not hourly_df.empty:
        full_idx = pd.date_range(start=hourly_df.index.min(), end=hourly_df.index.max(), freq='1h')
        hourly_df = hourly_df.reindex(full_idx)
        hourly_df.index.name = 'timestamp'

        # Linear interpolation for gaps up to 3 hours
        hourly_df['temp'] = hourly_df['temp'].interpolate(method='time', limit=3)
        hourly_df['dew'] = hourly_df['dew'].interpolate(method='time', limit=3)

    return hourly_df
