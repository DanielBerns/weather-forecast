import logging
import yaml
from pathlib import Path
import pandas as pd
from .data_handler import load_data, clean_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def preprocess_data(config_path):
    """
    This function orchestrates the data preprocessing pipeline.
    It loads the raw data, cleans it, splits it into training,
    validation, and test sets, normalizes the features, and
    then saves these processed datasets to disk. This separation
    of concerns ensures that the main training script receives
    clean, ready-to-use data, making the training process
    more streamlined and reproducible.
    """
    logger.info(f"Step 0: Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("No config")

    # --- Load and Prepare Data ---
    logger.info("Step 1: Loading and cleaning data...")
    file_list = list(Path(config['raw_data_path']).glob("*.csv"))
    raw_df = load_data(file_list)
    cleaned_df = clean_data(raw_df.copy())
    station_df = cleaned_df[cleaned_df['station_id'] == 'station_1'].copy()
    station_df.drop('station_id', axis=1, inplace=True)

    # --- Split the Data ---
    logger.info("\nStep 2: Splitting data...")
    n = len(station_df)
    train_df = station_df[0:int(n * 0.7)]
    val_df = station_df[int(n * 0.7):int(n * 0.9)]
    test_df = station_df[int(n * 0.9):]

    # --- Normalize the Data ---
    logger.info("\nStep 3: Normalizing features...")
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # --- Save Processed Data ---
    processed_data_path = Path(config['processed_data_path'])
    processed_data_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_data_path / 'train.csv')
    val_df.to_csv(processed_data_path / 'val.csv')
    test_df.to_csv(processed_data_path / 'test.csv')
    logger.info(f"Processed data saved to {processed_data_path}")
