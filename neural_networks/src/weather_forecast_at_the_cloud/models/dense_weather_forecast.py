from pathlib import Path
import tensorflow as tf
import numpy as np

from weather_forecast_at_the_cloud.utils.window_generator import WindowGenerator
from . import utils

class DenseWeatherForecast:
    """
    A multi-layer dense neural network for weather forecasting.

    This model implements the WeatherForecast protocol. It uses two dense hidden
    layers with ReLU activation to learn non-linear relationships in the data.
    """
    name = "Dense"

    def __init__(self, out_steps: int, num_features: int):
        """
        Initializes the DenseWeatherForecast.

        Args:
            out_steps (int): The number of output steps to predict.
            num_features (int): The number of features in the input data.
        """
        self.out_steps = out_steps
        self.num_features = num_features
        self.model = tf.keras.models.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, 64]
            tf.keras.layers.Dense(64, activation='relu'),
            # Shape => [batch, 1, 64]
            tf.keras.layers.Dense(64, activation='relu'),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(out_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([out_steps, num_features])
        ])

    def train(
        self,
        window_generator: WindowGenerator,
        epochs: int = 20,
        patience: int = 2,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Compiles and trains the Dense model.

        Args:
            window_generator: The WindowGenerator instance providing the data.
            epochs: The number of epochs for training.
            patience: The patience for the EarlyStopping callback.
            verbose: Verbosity mode for training.

        Returns:
            The trained Keras model history.
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            restore_best_weights=True
        )

        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

        history = self.model.fit(
            window_generator.train,
            epochs=epochs,
            verbose=verbose,
            validation_data=window_generator.val,
            callbacks=[early_stopping]
        )
        return history

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Makes a prediction using the trained model.

        Args:
            input_data: A numpy array of input features.

        Returns:
            A numpy array of predictions.
        """
        return self.model.predict(input_data)

    def save(self, base_path: Path) -> None:
        """Saves the model and its configuration."""
        config = {
            'out_steps': self.out_steps,
            'num_features': self.num_features
        }
        utils.save_model(self, base_path, config)

    @classmethod
    def load(cls, base_path: Path) -> "DenseWeatherForecast":
        """Loads a model using its configuration file."""
        return utils.load_model(cls, base_path)
