import argparse
from pathlib import Path
from weather_forecast_at_the_cloud.train import next_step

def main():
    """
    Runs the model training pipeline.
    This script uses a configuration file to define the model,
    hyperparameters, and data paths for training the weather forecast model.
    """
    # --- 1. Set up argument parser ---
    # This allows you to specify the config file from the command line.
    parser = argparse.ArgumentParser(description="Run training for the weather forecast model.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    # --- 2. Check if the config file exists ---
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Configuration file not found at '{config_path}'")
        return

    # --- 3. Run the training function from the package ---
    # This function loads the preprocessed data, creates data windows,
    # and trains the specified model.
    try:
        next_step(config_path)
        print("\nModel training completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

if __name__ == "__main__":
    main()
