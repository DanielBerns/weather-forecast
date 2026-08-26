from typing import List

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

class WindowGenerator:
    """
    A utility class to generate batches of time-series data windows.

    This class takes a time-series DataFrame and generates windows of data
    suitable for training sequence models. Each window consists of a set of
    input time steps and a set of label time steps.
    """
    def __init__(self,
                 input_width: int,
                 label_width: int,
                 shift: int,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 label_columns: List[str] | None = None
    ) -> None:
        """
        Initializes the WindowGenerator.

        Args:
            input_width (int): The number of time steps in the input window.
            label_width (int): The number of time steps in the label window.
            shift (int): The number of time steps separating the input and label windows.
            train_df (pd.DataFrame): DataFrame for training.
            val_df (pd.DataFrame): DataFrame for validation.
            test_df (pd.DataFrame): DataFrame for testing.
            label_columns (list[str], optional): A list of column names to be used as labels.
                                                 If None, all columns are used.
        """
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])

    def split_window(self, features):
        """
        Splits a window of features into inputs and labels.

        Args:
            features: A batch of sequential data.

        Returns:
            A tuple of (inputs, labels).
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set it manually.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='temperature_celsius', max_subplots=3):
        """
        Visualizes the data windows and model predictions.

        Args:
            model (tf.keras.Model, optional): A trained model to generate predictions.
            plot_col (str): The name of the column to plot.
            max_subplots (int): The maximum number of example windows to plot.
        """
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()

    def make_dataset(self, data):
        """
        Creates a tf.data.Dataset from a DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to convert.

        Returns:
            A tf.data.Dataset instance.
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

