import pandas as pd
import numpy as np
from pathlib import Path
import json

HERE = Path(".")
DATA_DIR = HERE / "data"
PREDICTION_DIR = HERE / "prediction" / "baseline"

def parse_hourly_temp(val_str):
    if pd.isna(val_str):
        return np.nan
    try:
        val_part, q_flag = val_str.split(',')
        val = int(val_part)
        if val == 9999 or val == -9999:
            return np.nan
        if q_flag in ['3', '7', '9']:
            return np.nan
        return val / 10.0
    except Exception:
        return np.nan

def main():
    print("Loading data...")
    df_hourly = pd.read_csv(DATA_DIR / "tytd_hourly_CR.csv", parse_dates=["DATE"], dtype={"CALL_SIGN": str}, low_memory=False)
    
    print("Parsing temperatures...")
    df_hourly['TMP_C'] = df_hourly['TMP'].apply(parse_hourly_temp)
    df_hourly['DEW_C'] = df_hourly['DEW'].apply(parse_hourly_temp)

    # Drop NaNs
    df = df_hourly[['DATE', 'TMP_C', 'DEW_C']].dropna().copy()

    # Drop Leap Years (Feb 29)
    df = df[~((df['DATE'].dt.month == 2) & (df['DATE'].dt.day == 29))]

    # Get Min/Max
    tmp_min = np.floor(df['TMP_C'].min())
    tmp_max = np.ceil(df['TMP_C'].max())
    dew_min = np.floor(df['DEW_C'].min())
    dew_max = np.ceil(df['DEW_C'].max())
    
    # 50x50 grid
    grid_size = 50
    
    print(f"TMP range: {tmp_min} to {tmp_max}")
    print(f"DEW range: {dew_min} to {dew_max}")

    # mapping from '%m-%d' to 0-364
    dates_1999 = pd.date_range('1999-01-01', '1999-12-31')
    md_to_doy = {d.strftime('%m-%d'): i for i, d in enumerate(dates_1999)}
    
    df['doy'] = df['DATE'].dt.strftime('%m-%d').map(md_to_doy)
    df['hour'] = df['DATE'].dt.hour
    df['hour_of_year'] = df['doy'] * 24 + df['hour']
    
    print("Binning data...")
    # Binning
    df['tmp_bin'] = np.clip(np.floor((df['TMP_C'] - tmp_min) / (tmp_max - tmp_min) * grid_size).astype(int), 0, grid_size - 1)
    df['dew_bin'] = np.clip(np.floor((df['DEW_C'] - dew_min) / (dew_max - dew_min) * grid_size).astype(int), 0, grid_size - 1)
    
    # Group by hour_of_year, tmp_bin, dew_bin
    counts = df.groupby(['hour_of_year', 'tmp_bin', 'dew_bin']).size().reset_index(name='count')
    
    # Format output as sparse array of arrays:
    output = [[] for _ in range(8760)]
    for _, row in counts.iterrows():
        h = int(row['hour_of_year'])
        x = int(row['tmp_bin']) # X is TMP
        y = int(row['dew_bin']) # Y is DEW
        c = int(row['count'])
        output[h].append([x, y, c])
        
    meta = {
        'grid_size': grid_size,
        'tmp_min': tmp_min,
        'tmp_max': tmp_max,
        'dew_min': dew_min,
        'dew_max': dew_max,
        'histograms': output
    }
    
    print("Saving to JSON...")
    with open(PREDICTION_DIR / "histogram_data.json", "w") as f:
        json.dump(meta, f)
    
    print("Done!")

if __name__ == '__main__':
    main()
