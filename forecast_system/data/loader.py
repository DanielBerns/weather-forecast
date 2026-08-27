from pathlib import Path
import pandas as pd
from .preprocessor import clean_hourly_dataframe
from .cleaner import clean_weather_dataset

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

def load_combined_dataset(data_dir=DATA_DIR):
    """
    Loads all datasets from data_dir, standardizes them, and merges into a continuous hourly dataframe.
    Applies data quality cleaning recommendations.
    Returns:
        combined_df: pd.DataFrame with index='timestamp' and columns=['temp', 'dew']
    """
    data_dir = Path(data_dir)
    frames = []

    # 1. Load tytd_hourly_CR.csv
    tytd_path = data_dir / "tytd_hourly_CR.csv"
    if tytd_path.exists():
        print(f"Loading historical dataset: {tytd_path.name}...")
        df_hist = pd.read_csv(tytd_path, usecols=['DATE', 'TMP', 'DEW'], low_memory=False)
        clean_hist = clean_hourly_dataframe(df_hist, is_modern=False)
        frames.append(clean_hist)

    # 2. Load CRV2025.csv
    crv2025_path = data_dir / "CRV2025.csv"
    if crv2025_path.exists():
        print(f"Loading 2025 dataset: {crv2025_path.name}...")
        df_2025 = pd.read_csv(crv2025_path, sep=';', low_memory=False)
        clean_2025 = clean_hourly_dataframe(df_2025, is_modern=True)
        frames.append(clean_2025)

    # 3. Load CRV2026.csv
    crv2026_path = data_dir / "CRV2026.csv"
    if crv2026_path.exists():
        print(f"Loading 2026 dataset: {crv2026_path.name}...")
        df_2026 = pd.read_csv(crv2026_path, sep=';', low_memory=False)
        clean_2026 = clean_hourly_dataframe(df_2026, is_modern=True)
        frames.append(clean_2026)

    if not frames:
        raise FileNotFoundError(f"No weather datasets found in {data_dir}")

    # Combine and deduplicate across datasets
    combined_df = pd.concat(frames, axis=0)
    combined_df = combined_df.groupby(combined_df.index).mean()
    combined_df = combined_df.sort_index()

    # Apply data quality cleaner recommendations
    combined_df = clean_weather_dataset(combined_df, temp_min=-15.0, temp_max=42.0, dew_min=-30.0, dew_max=25.0, interp_limit=6)

    return combined_df

def get_train_val_test_splits(df=None, data_dir=DATA_DIR):
    """
    Splits the combined hourly dataset into Train (<=2024), Val (2025), and Test (2026).
    Returns:
        train_df, val_df, test_df
    """
    if df is None:
        df = load_combined_dataset(data_dir=data_dir)

    train_df = df[df.index < '2025-01-01'].copy()
    val_df = df[(df.index >= '2025-01-01') & (df.index < '2026-01-01')].copy()
    test_df = df[df.index >= '2026-01-01'].copy()

    print(f"Train period: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} hours)")
    print(f"Val period:   {val_df.index.min()} to {val_df.index.max()} ({len(val_df)} hours)")
    print(f"Test period:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} hours)")

    return train_df, val_df, test_df
