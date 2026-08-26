import logging
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .utils.window_generator import WindowGenerator
from .models.linear_weather_forecast import LinearWeatherForecast
from .models.dense_weather_forecast import DenseWeatherForecast
from .models.cnn_weather_forecast import CNNWeatherForecast
from .models.rnn_weather_forecast import RNNWeatherForecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def plot_metrics(history, session):
    """
    This function visualizes the training and validation loss across epochs
    for each training session. Plotting these metrics is crucial for
    diagnosing the model's learning progress, helping to identify issues
    like overfitting or underfitting. A well-performing model will show
    both training and validation losses decreasing and converging.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'], label=f'Session {session} Train Loss')
    plt.plot(history.history['val_loss'], label=f'Session {session} Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def next_step(config_path):
    logger.info(f"Step 0: Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("No config")
    # --- Load Processed Data ---
    logger.info("Step 1: Loading pre-processed data...")
    processed_data_path = Path(config['processed_data_path'])
    train_df = pd.read_csv(processed_data_path / 'train.csv', index_col='timestamp')
    val_df = pd.read_csv(processed_data_path / 'val.csv', index_col='timestamp')
    test_df = pd.read_csv(processed_data_path / 'test.csv', index_col='timestamp')
    num_features = train_df.shape[1]

    # --- Create Data Windows ---
    logger.info("\nStep 2: Creating data windows...")
    multi_output_window = WindowGenerator(
        input_width=config['input_width'],
        label_width=config['out_steps'],
        shift=config['out_steps'],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=[config['label_column']]
    )

    # --- Train Model ---
    models = {
        "Linear": LinearWeatherForecast,
        "Dense": DenseWeatherForecast,
        "CNN": CNNWeatherForecast,
        "LSTM": RNNWeatherForecast,
    }

    model_class = models.get(config['model_name'])
    if model_class is None:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    this_model_path = Path(config['models_path'], config['model_name']).with_suffix('.keras')
    if this_model_path.exists():
        logger.info(f"Loading existing model: {config['model_name']}")
        model = model_class.load(this_model_path)
    else:
        logger.info(f"Creating new model: {config['model_name']}")
        if config['model_name'] == 'CNN':
            model = model_class(out_steps=config['out_steps'], num_features=num_features, conv_width=config['conv_width'])
        else:
            model = model_class(out_steps=config['out_steps'], num_features=num_features)

    for session in range(1, config.get('sessions', 3) + 1):
        logger.info(f"\n--- Training Session {session} ---")
        history = model.train(
            multi_output_window,
            epochs=config['epochs_per_session'],
            patience=config['patience'],
            verbose=1
        )
        model.save(this_model_path)
        plot_metrics(history, session)

    logger.info("\n--- Final Evaluation ---")
    performance = model.model.evaluate(multi_output_window.test, verbose=0)
    logger.info(f"Test Loss (MSE): {performance[0]}, MAE: {performance[1]}")
