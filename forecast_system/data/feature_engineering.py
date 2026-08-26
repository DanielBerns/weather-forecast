import numpy as np
import pandas as pd

def create_features_and_targets(df, input_window=48, forecast_horizon=24):
    """
    Creates input feature matrix X and target matrix Y for forecasting next-day temperature.

    Args:
        df: pd.DataFrame with index='timestamp', columns=['temp', 'dew']
        input_window: past hours lookback window (default 48 hours)
        forecast_horizon: future hours forecast horizon (default 24 hours)

    Returns:
        X: DataFrame of engineered input features (evaluated at forecast reference time t)
        Y_hourly: DataFrame of 24 hourly target values (temp_{t+1} to temp_{t+24})
        Y_summary: DataFrame of summary targets (tmax, tmin, tavg for next 24 hours)
    """
    df = df.copy()

    # 1. Basic temporal calendar features
    hour = df.index.hour
    doy = df.index.dayofyear

    df['sin_hour'] = np.sin(2 * np.pi * hour / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * hour / 24.0)
    df['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)

    # 2. Dew point depression
    df['dew_depression'] = df['temp'] - df['dew']

    # 3. Temperature & Dew Lags
    lag_hours = [0, 1, 2, 3, 4, 6, 12, 18, 24, 36, 48, 72, 168]
    feature_dict = {}

    for col in ['temp', 'dew', 'dew_depression']:
        for lag in lag_hours:
            if lag == 0:
                feature_dict[f'{col}_lag0'] = df[col]
            else:
                feature_dict[f'{col}_lag{lag}'] = df[col].shift(lag)

    # 4. Rolling Statistics
    for w in [6, 12, 24, 48, 168]:
        feature_dict[f'temp_roll_mean_{w}h'] = df['temp'].shift(1).rolling(w, min_periods=max(1, w//2)).mean()
        feature_dict[f'temp_roll_min_{w}h'] = df['temp'].shift(1).rolling(w, min_periods=max(1, w//2)).min()
        feature_dict[f'temp_roll_max_{w}h'] = df['temp'].shift(1).rolling(w, min_periods=max(1, w//2)).max()
        feature_dict[f'temp_roll_std_{w}h'] = df['temp'].shift(1).rolling(w, min_periods=max(1, w//2)).std()
        feature_dict[f'temp_roll_range_{w}h'] = feature_dict[f'temp_roll_max_{w}h'] - feature_dict[f'temp_roll_min_{w}h']

        feature_dict[f'dew_roll_mean_{w}h'] = df['dew'].shift(1).rolling(w, min_periods=max(1, w//2)).mean()

    # Add Calendar features to feature_dict
    for col in ['sin_hour', 'cos_hour', 'sin_doy', 'cos_doy']:
        feature_dict[col] = df[col]

    X = pd.DataFrame(feature_dict, index=df.index)

    # 5. Targets (Future 24 hours: t+1 to t+24)
    target_dict = {}
    for step in range(1, forecast_horizon + 1):
        target_dict[f'target_h{step}'] = df['temp'].shift(-step)

    Y_hourly = pd.DataFrame(target_dict, index=df.index)

    # Summary Targets: TMAX, TMIN, TAVG over next 24 hours
    Y_summary = pd.DataFrame({
        'target_tmax': Y_hourly.max(axis=1),
        'target_tmin': Y_hourly.min(axis=1),
        'target_tavg': Y_hourly.mean(axis=1)
    }, index=df.index)

    # Drop NaNs across X and Y
    valid_mask = X.notnull().all(axis=1) & Y_hourly.notnull().all(axis=1)

    X_clean = X.loc[valid_mask]
    Y_hourly_clean = Y_hourly.loc[valid_mask]
    Y_summary_clean = Y_summary.loc[valid_mask]

    return X_clean, Y_hourly_clean, Y_summary_clean
