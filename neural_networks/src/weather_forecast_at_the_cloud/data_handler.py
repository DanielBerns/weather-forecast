import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

def load_data(file_paths: List[str]) -> pd.DataFrame:
    """
    Loads weather data from one or more CSV files into a single pandas DataFrame.

    It assumes the station ID can be inferred from the filename
    (e.g., 'weather_1.csv' -> 'station_1').

    Args:
        file_paths: A list of string paths to the CSV files.

    Returns:
        A pandas DataFrame containing the combined data from all files,
        with a new 'station_id' column.
    """
    all_data = []
    for file_path in file_paths:
        try:
            # Read the specific CSV file
            df = pd.read_csv(Path(file_path).absolute(), encoding="latin-1")

            # Infer station_id from filename. e.g., "weather_1.csv" -> "station_1"
            p = Path(file_path)
            station_id = p.stem.replace('weather_', 'station_')
            df['station_id'] = station_id

            all_data.append(df)
            print(f"Successfully loaded and assigned ID '{station_id}' to {p.name}")

        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    if not all_data:
        # Return an empty DataFrame if no files were loaded
        return pd.DataFrame()

    # Concatenate all dataframes into a single one
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes the weather data DataFrame.

    This function performs several key operations:
    1. Renames columns from Spanish to a standard English format.
    2. Converts the timestamp column to datetime objects.
    3. Handles the cyclical nature of wind direction by converting it to vector components.
    4. Sets the timestamp as the DataFrame index.

    Args:
        df: The raw pandas DataFrame loaded from the CSV files.

    Returns:
        A cleaned and processed pandas DataFrame.
    """
    if df.empty:
        print("Input DataFrame is empty. Skipping cleaning.")
        return df

    # Define a mapping from the original Spanish column names to new English names
    column_mapping = {
        'Fecha/Hora': 'timestamp',
        'Precipitacion (mm)': 'precipitation_mm',
        'Temperatura (ºC)': 'temperature_celsius',
        'Humedad (%)': 'humidity_percent',
        'Presion (hPa)': 'pressure_hpa',
        'Dir. Del Viento (º)': 'wind_direction_deg',
        'Int. del Viento (m/s)': 'wind_speed_ms'
    }

    df.rename(columns=column_mapping, inplace=True)

    # Convert 'timestamp' column to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")

    # --- Handle cyclical features: Wind Direction ---
    # Wind direction is cyclical (360 degrees is the same as 0).
    # Directly using degrees can be misleading for some models.
    # We convert it to vector components (x and y) which models can handle better.
    wd_rad = df.pop('wind_direction_deg') * np.pi / 180
    df['wind_x'] = np.cos(wd_rad)
    df['wind_y'] = np.sin(wd_rad)

    # Set the timestamp as the index for time-series analysis
    df.set_index('timestamp', inplace=True)

    # Sort the dataframe by the index to ensure chronological order
    df.sort_index(inplace=True)

    print("Data cleaning complete. Columns renamed and wind direction converted.")
    return df


def preprocess_for_modeling(df: pd.DataFrame, feature_keys: List[str]):
    """
    Normalizes the feature data for modeling.

    Note: In a real-world pipeline, the scaler should be fit ONLY on the
    training data to prevent data leakage from the validation and test sets.
    This function is a general utility. The fitted scaler should be saved
    and reused for transforming validation, test, and future prediction data.

    Args:
        df: The DataFrame to process.
        feature_keys: A list of column names to normalize.

    Returns:
        A tuple containing:
        - The normalized DataFrame.
        - The dictionary of scalers used for each column.
    """
    # This is a placeholder for a more robust preprocessing step.
    # For now, we will just return the dataframe as is.
    # In the training pipeline, we will implement feature scaling.
    print("Preprocessing placeholder: Returning original data.")
    return df


