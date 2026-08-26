
RAW_DATA = "raw_data"
PROCESSED_DATA: "processed_data"
MODELS = "models"
RESULTS = "results"
LOGS = "logs"

def get_config_yaml(base: Path, parameters_yaml: Path) -> Path:
    with open(parameters_yaml, "r") as f:
        parameters = yaml.safe_load(f)

    raw_data_path = base / RAW_DATA
    processed_data_path = base / PROCESSED_DATA
    models_path = base / MODELS
    results_path= base / RESULTS
    logs_path = base / LOGS
    config_yaml = base / 'config.yaml'

    # Create directories if they don't exist
    raw_data_path.mkdir(parents=True, exist_ok=True)
    processed_data_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    config = {
        "raw_data_path": str(raw_data_path),
        "processed_data_path": str(processed_data_path),
        "models_path": str(models_path),
        "results_path": str(results_path),
        "logs_path": str(logs_path)
    }
    config.update(parameters)

    with open(config_yaml, "w") as f:
        yaml.dump(config, f)

    return config_yaml
