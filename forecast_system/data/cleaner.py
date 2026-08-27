import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("forecast_system.data.cleaner")

def clamp_thermodynamic_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces thermodynamic physical laws: Dew Point (Td) cannot exceed Air Temperature (T).
    When Dew > Temp, clamps Dew to Temp (Relative Humidity = 100%).
    """
    df = df.copy()
    if 'temp' in df.columns and 'dew' in df.columns:
        anomaly_mask = df['dew'] > df['temp']
        anomaly_count = int(anomaly_mask.sum())
        if anomaly_count > 0:
            logger.info(f"Clamping {anomaly_count} thermodynamic anomalies where Dew Point > Temperature.")
            df.loc[anomaly_mask, 'dew'] = df.loc[anomaly_mask, 'temp']
    return df

def filter_climatological_outliers(
    df: pd.DataFrame,
    temp_min: float = -15.0,
    temp_max: float = 42.0,
    dew_min: float = -30.0,
    dew_max: float = 25.0
) -> pd.DataFrame:
    """
    Filters extreme sensor outliers outside realistic climatological bounds for Comodoro Rivadavia.
    Values outside range are set to NaN for subsequent interpolation.
    """
    df = df.copy()
    if 'temp' in df.columns:
        temp_outliers = (df['temp'] < temp_min) | (df['temp'] > temp_max)
        t_cnt = int(temp_outliers.sum())
        if t_cnt > 0:
            logger.info(f"Filtering {t_cnt} temperature values outside climatological range [{temp_min}°C, {temp_max}°C].")
            df.loc[temp_outliers, 'temp'] = np.nan

    if 'dew' in df.columns:
        dew_outliers = (df['dew'] < dew_min) | (df['dew'] > dew_max)
        d_cnt = int(dew_outliers.sum())
        if d_cnt > 0:
            logger.info(f"Filtering {d_cnt} dew point values outside climatological range [{dew_min}°C, {dew_max}°C].")
            df.loc[dew_outliers, 'dew'] = np.nan

    return df

def reindex_and_interpolate_timeline(df: pd.DataFrame, interp_limit: int = 6) -> pd.DataFrame:
    """
    Reindexes dataset to a continuous 1-hour frequency grid and fills gaps using time-based linear interpolation.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        elif 'DATE' in df.columns:
            df['timestamp'] = pd.to_datetime(df['DATE'], errors='coerce')
            df = df.set_index('timestamp')
        else:
            raise ValueError("DataFrame index must be a DatetimeIndex or contain a timestamp/DATE column.")

    df = df.sort_index()

    if not df.empty:
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h')
        df = df.reindex(full_idx)
        df.index.name = 'timestamp'

        for col in ['temp', 'dew']:
            if col in df.columns:
                df[col] = df[col].interpolate(method='time', limit=interp_limit)

    return df

def clean_weather_dataset(
    df: pd.DataFrame,
    temp_min: float = -15.0,
    temp_max: float = 42.0,
    dew_min: float = -30.0,
    dew_max: float = 25.0,
    interp_limit: int = 6
) -> pd.DataFrame:
    """
    Full weather dataset cleaning pipeline:
    1. Filters climatological extreme outliers outside local bounds.
    2. Clamps thermodynamic anomalies (Dew > Temp).
    3. Reindexes to continuous 1-hour timeline and interpolates short gaps up to interp_limit.
    """
    logger.info("Executing comprehensive weather data cleaning recommendations...")
    df_clean = filter_climatological_outliers(df, temp_min=temp_min, temp_max=temp_max, dew_min=dew_min, dew_max=dew_max)
    df_clean = clamp_thermodynamic_anomalies(df_clean)
    df_clean = reindex_and_interpolate_timeline(df_clean, interp_limit=interp_limit)
    df_clean = clamp_thermodynamic_anomalies(df_clean)  # Re-verify after interpolation
    return df_clean
