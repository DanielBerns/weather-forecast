import argparse
from pathlib import Path
from weather_forecast_at_the_cloud.preprocess_data import preprocess_data

def main():
    """
    Runs the data preprocessing pipeline.
    This script takes the path to a configuration file, which specifies
    the locations for raw data and where to save the processed data.
    """
    # --- 1. Set up argument parser ---
    # This allows you to specify the config file from the command line.
    parser = argparse.ArgumentParser(description="Run data preprocessing for the weather forecast model.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="local_parameters.yaml",
        help="Path to the parameters YAML file."
    )
    args = parser.parse_args()

    # --- 2. Check if the parameters_yaml file exists ---
    parameters_yaml = Path(args.config)
    if not parameters_yaml.is_file():
        print(f"Error: Parameters file not found at '{parameters_yaml}'")
        return
    with open(parameters_yaml, "r") as f:
        parameters = yaml.safe_load(f)

    # --- 3. Get the config_yaml file ---
    base = Path.home() / "Info" / parameters['base']
    config_yaml = get_config_yaml(base, parameters)

    # --- 4. Run the preprocessing function from the package ---
    # This function will load, clean, split, normalize, and save the data.
    try:
        preprocess_data(config_yaml)
        print("\nData preprocessing completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred during preprocessing: {e}")

if __name__ == "__main__":
    main()
