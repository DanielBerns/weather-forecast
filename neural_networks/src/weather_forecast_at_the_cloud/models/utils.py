import json
from pathlib import Path
import tensorflow as tf
from typing import Dict, Any, Type

def save_model(model_instance: Any, base_path: Path, config: Dict[str, Any]):
    """
    Saves a Keras model and its configuration to a specified path.

    This function serializes the model's parameters (like `out_steps`) into a
    .json file and saves the Keras model's architecture and weights into a
    .keras file.

    Args:
        model_instance: The instance of the model wrapper class (e.g., LinearWeatherForecast).
                        It is expected to have a `model` attribute which is the Keras model.
        base_path (Path): The base path for saving the files (e.g., 'saved_models/Linear').
                          The function will append '.keras' and '.json' suffixes.
        config (Dict[str, Any]): A dictionary containing the parameters needed to
                                 re-initialize the model wrapper class.
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save the Keras model file
    keras_model_path = base_path.with_suffix(".keras")
    model_instance.model.save(keras_model_path)

    # 2. Save the configuration file
    config_path = base_path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Model '{model_instance.name}' and config saved to {base_path}.[keras|json]")


def load_model(model_class: Type, base_path: Path) -> Any:
    """
    Loads a Keras model and its configuration from a specified path.

    This function first reads the .json configuration file to get the necessary
    parameters, initializes the model wrapper class with these parameters, and then
    loads the saved Keras model's architecture and weights.

    Args:
        model_class (Type): The class of the model to be loaded (e.g., LinearWeatherForecast).
        base_path (Path): The base path from where to load the model files.

    Returns:
        An instance of the model_class with the loaded Keras model.
    """
    # 1. Load the configuration from the JSON file
    config_path = base_path.with_suffix(".json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Create the instance with the correct parameters
    instance = model_class(**config)

    # 3. Load the Keras model's weights and architecture
    keras_model_path = base_path.with_suffix(".keras")
    instance.model = tf.keras.models.load_model(keras_model_path)

    print(f"Model '{instance.name}' loaded from {base_path}.[keras|json]")
    return instance
