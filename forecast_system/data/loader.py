import logging
from pathlib import Path
import pandas as pd
from .preprocessor import clean_hourly_dataframe
from .cleaner import clean_weather_dataset

logger = logging.getLogger("forecast_system.data.loader")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

def load_combined_dataset(data_dir=DATA_DIR, prefer_cleaned_csv=True):
    """
    Loads weather datasets, standardizes them, and returns a continuous hourly dataframe.
    If 'cleaned_combined_CR.csv' is present in data_dir and prefer_cleaned_csv is True,
    loads the pre-cleaned dataset directly. Otherwise, loads raw CSV files and applies
    the full data cleaning pipeline (filtering extremes, clamping Dew <= Temp, timeline reindexing).

    Returns:
        combined_df: pd.DataFrame with index='timestamp' and columns=['temp', 'dew']
    """
    data_dir = Path(data_dir)
    cleaned_path = data_dir / "cleaned_combined_CR.csv"

    if prefer_cleaned_csv and cleaned_path.exists():
        msg = f"✓ Found pre-cleaned dataset file: {cleaned_path.name}. Loading cleaned dataset directly..."
        logger.info(msg)
        print(msg)
        df_clean = pd.read_csv(cleaned_path, parse_dates=['timestamp'], index_col='timestamp')
        df_clean = df_clean.sort_index()

        # Re-apply cleaner verification to guarantee physical constraints (Dew <= Temp, local bounds)
        df_clean = clean_weather_dataset(df_clean, temp_min=-15.0, temp_max=42.0, dew_min=-30.0, dew_max=25.0, interp_limit=6)

        bad_dew_cnt = (df_clean['dew'] > df_clean['temp']).sum()
        summary_msg = (
            f"✓ Pre-cleaned dataset loaded successfully.\n"
            f"  - File: {cleaned_path.name}\n"
            f"  - Total Hourly Records: {len(df_clean):,}\n"
            f"  - Date Range: {df_clean.index.min()} to {df_clean.index.max()}\n"
            f"  - Temperature Range: [{df_clean['temp'].min():.2f}°C, {df_clean['temp'].max():.2f}°C]\n"
            f"  - Dew Point Range: [{df_clean['dew'].min():.2f}°C, {df_clean['dew'].max():.2f}°C]\n"
            f"  - Dew > Temp Violation Count: {bad_dew_cnt}"
        )
        logger.info(summary_msg)
        print(summary_msg)
        return df_clean

    msg = f"Pre-cleaned file '{cleaned_path.name}' not found or prefer_cleaned_csv=False. Loading raw datasets from '{data_dir}'..."
    logger.info(msg)
    print(msg)
    frames = []

    # 1. Load tytd_hourly_CR.csv
    tytd_path = data_dir / "tytd_hourly_CR.csv"
    if tytd_path.exists():
        l_msg = f"Loading historical dataset: {tytd_path.name}..."
        logger.info(l_msg)
        print(l_msg)
        df_hist = pd.read_csv(tytd_path, usecols=['DATE', 'TMP', 'DEW'], low_memory=False)
        clean_hist = clean_hourly_dataframe(df_hist, is_modern=False)
        frames.append(clean_hist)

    # 2. Load CRV2025.csv
    crv2025_path = data_dir / "CRV2025.csv"
    if crv2025_path.exists():
        l_msg = f"Loading 2025 dataset: {crv2025_path.name}..."
        logger.info(l_msg)
        print(l_msg)
        df_2025 = pd.read_csv(crv2025_path, sep=';', low_memory=False)
        clean_2025 = clean_hourly_dataframe(df_2025, is_modern=True)
        frames.append(clean_2025)

    # 3. Load CRV2026.csv
    crv2026_path = data_dir / "CRV2026.csv"
    if crv2026_path.exists():
        l_msg = f"Loading 2026 dataset: {crv2026_path.name}..."
        logger.info(l_msg)
        print(l_msg)
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
    logger.info("Applying full data quality cleaning pipeline (filtering extremes, clamping Dew <= Temp, timeline reindexing)...")
    print("Applying full data quality cleaning pipeline (filtering extremes, clamping Dew <= Temp, timeline reindexing)...")
    combined_df = clean_weather_dataset(combined_df, temp_min=-15.0, temp_max=42.0, dew_min=-30.0, dew_max=25.0, interp_limit=6)

    bad_dew_cnt = (combined_df['dew'] > combined_df['temp']).sum()
    summary_msg = (
        f"✓ Raw dataset loading and in-memory cleaning complete.\n"
        f"  - Total Hourly Records: {len(combined_df):,}\n"
        f"  - Date Range: {combined_df.index.min()} to {combined_df.index.max()}\n"
        f"  - Temperature Range: [{combined_df['temp'].min():.2f}°C, {combined_df['temp'].max():.2f}°C]\n"
        f"  - Dew Point Range: [{combined_df['dew'].min():.2f}°C, {combined_df['dew'].max():.2f}°C]\n"
        f"  - Dew > Temp Violation Count: {bad_dew_cnt}"
    )
    logger.info(summary_msg)
    print(summary_msg)

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

    split_msg = (
        f"Dataset Splits Summary:\n"
        f"  - Train period: {train_df.index.min()} to {train_df.index.max()} ({len(train_df):,} hours)\n"
        f"  - Val period:   {val_df.index.min()} to {val_df.index.max()} ({len(val_df):,} hours)\n"
        f"  - Test period:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df):,} hours)"
    )
    logger.info(split_msg)
    print(split_msg)

    return train_df, val_df, test_df
