#!/usr/bin/env python3
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd

# Add current directory to path for package imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecast_system.data.loader import load_combined_dataset
from forecast_system.data.cleaner import clean_weather_dataset
from forecast_system.data.quality_analysis import export_data_properties_json
from forecast_system.data.feature_engineering import create_features_and_targets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("clean_dataset")

def main():
    parser = argparse.ArgumentParser(description="Clean weather datasets according to physical and climatological recommendations.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing raw weather CSV files")
    parser.add_argument("--output-csv", type=str, default="data/cleaned_combined_CR.csv", help="Path to save cleaned output CSV")
    parser.add_argument("--report-properties", type=str, default="report/outputs/data_properties.json", help="Path to export data properties JSON")
    parser.add_argument("--temp-min", type=float, default=-15.0, help="Climatological min temperature (°C)")
    parser.add_argument("--temp-max", type=float, default=42.0, help="Climatological max temperature (°C)")
    parser.add_argument("--dew-min", type=float, default=-30.0, help="Climatological min dew point (°C)")
    parser.add_argument("--dew-max", type=float, default=25.0, help="Climatological max dew point (°C)")

    args = parser.parse_args()

    logger.info(f"Loading raw datasets from '{args.data_dir}'...")
    df_raw = load_combined_dataset(data_dir=args.data_dir)

    logger.info("Applying data quality cleaning recommendations...")
    df_clean = clean_weather_dataset(
        df_raw,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        dew_min=args.dew_min,
        dew_max=args.dew_max,
        interp_limit=6
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_csv)
    logger.info(f"✓ Saved cleaned dataset ({len(df_clean)} hourly records) to '{output_csv}'")

    if args.report_properties:
        logger.info("Re-evaluating feature engineering and dataset properties after cleaning...")
        X_clean, _, _ = create_features_and_targets(df_clean)
        report_path = Path(args.report_properties)
        export_data_properties_json(
            X_clean,
            report_path,
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            dew_min=args.dew_min,
            dew_max=args.dew_max
        )
        logger.info(f"✓ Clean data properties exported to '{report_path}'")

if __name__ == "__main__":
    main()
