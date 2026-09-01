import os
import json
import logging
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers

from forecast_system.models.lr_policy import PerformanceLRDecayPolicy, PerformanceLRDecayCallback

logger = logging.getLogger("forecast_system.models.cnn_weather_forecast")

class CNNForecastModel:
    """
    1D Convolutional Neural Network Model for 24h temperature forecasting.
    Supports restartable (resume/update) and reset (scratch) chunked training modes
    with configurable performance-based learning rate decay policy.
    """
    def __init__(
        self,
        filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        lr_decay_enabled: bool = True,
        lr_decay_policy: str = 'plateau',
        lr_decay_factor: float = 0.5,
        lr_decay_patience: int = 2,
        lr_min: float = 1e-6,
        lr_decay_threshold: float = 1e-4,
        lr_cooldown: int = 0,
        lr_restart_patience: int = 6,
        lr_restart_factor: float = 0.5,
        max_lr: Optional[float] = None,
        cycle_step_size: int = 5,
        cyclic_mode: str = 'triangular'
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.mean_X = None
        self.std_X = None
        self.mean_Y = None
        self.std_Y = None
        self.training_history = {}

        self.lr_policy = PerformanceLRDecayPolicy(
            enabled=lr_decay_enabled,
            policy=lr_decay_policy,
            factor=lr_decay_factor,
            patience=lr_decay_patience,
            min_lr=lr_min,
            min_delta=lr_decay_threshold,
            cooldown=lr_cooldown,
            restart_patience=lr_restart_patience,
            restart_factor=lr_restart_factor,
            max_lr=max_lr,
            cycle_step_size=cycle_step_size,
            cyclic_mode=cyclic_mode,
            monitor='val_loss',
            mode='min',
            initial_lr=learning_rate
        )

    def _build_model(self, input_dim, output_dim=24):
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Reshape((input_dim, 1)),
            layers.Conv1D(filters=self.filters, kernel_size=min(self.kernel_size, input_dim), padding='same', activation='relu'),
            layers.Conv1D(filters=self.filters, kernel_size=min(self.kernel_size, input_dim), padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(output_dim)
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model

    def fit_restartable(
        self,
        X_train,
        Y_hourly_train,
        Y_summary_train=None,
        X_val=None,
        Y_hourly_val=None,
        epochs_per_iter=5,
        target_mae=1.0,
        max_iters=4,
        batch_size=64,
        patience=10,
        verbose=0,
        checkpoint_path=None,
        history_path=None,
        reset=False
    ):
        if self.mean_X is None:
            self.mean_X = X_train.mean(axis=0)
            self.std_X = X_train.std(axis=0).replace(0, 1.0)
            self.mean_Y = Y_hourly_train.values.mean()
            self.std_Y = Y_hourly_train.values.std()

        X_scaled = (X_train - self.mean_X) / self.std_X
        Y_scaled = (Y_hourly_train.values - self.mean_Y) / self.std_Y

        if X_val is not None and Y_hourly_val is not None:
            X_val_scaled = (X_val - self.mean_X) / self.std_X
            Y_val_scaled = (Y_hourly_val.values - self.mean_Y) / self.std_Y
            val_data = (X_val_scaled.values, Y_val_scaled)
        else:
            val_data = None

        epoch_logs = []
        iteration_logs = []
        total_epochs = 0
        best_val_mae_c = float('inf')
        best_weights = None
        best_epoch = 0

        self.lr_policy.reset(initial_lr=self.learning_rate)

        if reset:
            logger.info("🔄 [RESET MODE] Wiping past CNN model weights and training history. Starting clean...")
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                except Exception:
                    pass
            if history_path and os.path.exists(history_path):
                try:
                    os.remove(history_path)
                except Exception:
                    pass
            self.model = self._build_model(input_dim=X_train.shape[1], output_dim=24)
        else:
            logger.info("🔁 [RESUME MODE] Checking for existing saved weights and training history for CNN model...")
            if self.model is None:
                self.model = self._build_model(input_dim=X_train.shape[1], output_dim=24)
                if checkpoint_path and os.path.exists(checkpoint_path):
                    try:
                        if checkpoint_path.endswith('.keras'):
                            self.model = keras.models.load_model(checkpoint_path)
                        else:
                            self.model.load_weights(checkpoint_path)
                        if hasattr(self.model.optimizer, 'learning_rate'):
                            if hasattr(self.model.optimizer.learning_rate, 'assign'):
                                self.model.optimizer.learning_rate.assign(self.learning_rate)
                            else:
                                tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.learning_rate)
                        logger.info(f"✓ Loaded saved CNN model from {checkpoint_path} (Active LR set to {self.learning_rate})")
                    except Exception as e:
                        logger.info(f"Note: Could not load existing checkpoint ({e}). Training clean model.")

            if history_path and os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        past_hist = json.load(f)
                    epoch_logs = past_hist.get('epochs', [])
                    iteration_logs = past_hist.get('iterations', [])
                    total_epochs = past_hist.get('total_epochs', len(epoch_logs))
                    best_val_mae_c = past_hist.get('best_val_mae', float('inf'))
                    logger.info(f"✓ Resuming from past training history ({total_epochs} epochs executed, past best MAE: {best_val_mae_c}°C)")
                except Exception as e:
                    logger.info(f"Note: Could not read past history file ({e}).")

        final_status = "In Progress"
        is_adequate = False
        overfitting_detected = False
        underfitting_detected = False

        start_iter_idx = len(iteration_logs) + 1
        end_iter_idx = start_iter_idx + max_iters - 1

        logger.info(
            f"--- Running CNN Training (Target MAE < {target_mae}°C | Epochs/iter: {epochs_per_iter} | "
            f"Iterations: {start_iter_idx} to {end_iter_idx} | LR Decay Policy: '{self.lr_policy.policy}') ---"
        )

        for iter_idx in range(start_iter_idx, end_iter_idx + 1):
            lr_cb = PerformanceLRDecayCallback(
                policy=self.lr_policy,
                model_name="CNNModel",
                epoch_offset=total_epochs
            )

            hist = self.model.fit(
                X_scaled.values,
                Y_scaled,
                epochs=epochs_per_iter,
                batch_size=batch_size,
                verbose=verbose,
                validation_data=val_data,
                validation_split=0.1 if val_data is None else 0.0,
                callbacks=[lr_cb]
            )

            iter_train_loss = hist.history.get('loss', [])
            iter_val_loss = hist.history.get('val_loss', [])
            iter_train_mae = hist.history.get('mae', [])
            iter_val_mae = hist.history.get('val_mae', [])

            opt_lr = self.model.optimizer.learning_rate
            current_lr_val = float(opt_lr.numpy()) if hasattr(opt_lr, 'numpy') else float(opt_lr)

            for ep_in_iter in range(len(iter_train_loss)):
                total_epochs += 1
                tr_l = float(iter_train_loss[ep_in_iter])
                va_l = float(iter_val_loss[ep_in_iter]) if iter_val_loss else tr_l
                tr_m = float(iter_train_mae[ep_in_iter]) * self.std_Y
                va_m = float(iter_val_mae[ep_in_iter]) * self.std_Y if iter_val_mae else tr_m

                if va_m < best_val_mae_c:
                    best_val_mae_c = va_m
                    best_epoch = total_epochs
                    best_weights = self.model.get_weights()

                epoch_logs.append({
                    'epoch': total_epochs,
                    'train_loss': round(tr_l, 5),
                    'val_loss': round(va_l, 5),
                    'train_mae': round(tr_m, 3),
                    'val_mae': round(va_m, 3),
                    'lr': round(current_lr_val, 7),
                    'iteration': iter_idx
                })

            current_val_mae_c = epoch_logs[-1]['val_mae']
            current_val_loss = epoch_logs[-1]['val_loss']
            current_train_loss = epoch_logs[-1]['train_loss']

            logger.info(
                f"Iteration {iter_idx}/{end_iter_idx} (Total Epochs: {total_epochs}) -> "
                f"Val MAE: {current_val_mae_c:.2f}°C (Best: {best_val_mae_c:.2f}°C) | Current LR: {current_lr_val:.6e}"
            )

            if best_val_mae_c <= target_mae:
                is_adequate = True
                final_status = f"Adequate Level Reached (Val MAE {best_val_mae_c:.2f}°C <= {target_mae}°C)"
                logger.info(f"✓ {final_status}")
                iteration_logs.append({
                    'iteration': iter_idx,
                    'epochs_run': total_epochs,
                    'val_mae_c': current_val_mae_c,
                    'status': final_status
                })
                break

            if len(epoch_logs) >= 6:
                min_recent_val = min([e['val_loss'] for e in epoch_logs])
                if current_val_loss > 1.25 * min_recent_val and current_train_loss < epoch_logs[0]['train_loss']:
                    overfitting_detected = True
                    final_status = f"Overfitting Detected (Val Loss {current_val_loss:.4f} > min {min_recent_val:.4f}). Restoring best weights."
                    logger.info(f"⚠️ {final_status}")
                    iteration_logs.append({
                        'iteration': iter_idx,
                        'epochs_run': total_epochs,
                        'val_mae_c': current_val_mae_c,
                        'status': final_status
                    })
                    break

            iteration_logs.append({
                'iteration': iter_idx,
                'epochs_run': total_epochs,
                'val_mae_c': current_val_mae_c,
                'status': f"Iteration {iter_idx} completed. Continuing training..."
            })

        if not is_adequate and not overfitting_detected:
            if best_val_mae_c > target_mae:
                underfitting_detected = True
                final_status = f"Underfitting / Limit Reached (Best Val MAE {best_val_mae_c:.2f}°C > target {target_mae}°C after {total_epochs} epochs)"
                logger.info(f"ℹ️ {final_status}")

        if best_weights is not None:
            self.model.set_weights(best_weights)

        if checkpoint_path is not None:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            self.model.save(checkpoint_path)
            logger.info(f"Saved updated CNN model checkpoint to {checkpoint_path}")

        self.training_history = {
            'epochs': epoch_logs,
            'iterations': iteration_logs,
            'total_epochs': total_epochs,
            'best_epoch': best_epoch,
            'best_val_mae': round(best_val_mae_c, 3),
            'target_mae': target_mae,
            'final_status': final_status,
            'is_adequate': is_adequate,
            'overfitting_detected': overfitting_detected,
            'underfitting_detected': underfitting_detected,
            'lr_decay_events': self.lr_policy.decay_events
        }

        if history_path is not None:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            logger.info(f"Saved updated CNN training history log to {history_path}")

        return self

    def fit(self, X_train, Y_hourly_train, Y_summary_train=None, epochs=15, batch_size=64, verbose=0):
        epochs_per_iter = max(1, epochs // 3)
        return self.fit_restartable(
            X_train=X_train,
            Y_hourly_train=Y_hourly_train,
            Y_summary_train=Y_summary_train,
            epochs_per_iter=epochs_per_iter,
            target_mae=1.0,
            max_iters=3,
            batch_size=batch_size,
            verbose=verbose
        )

    def predict(self, X):
        X_scaled = (X - self.mean_X) / self.std_X
        X_arr = np.asarray(X_scaled.values, dtype=np.float32)
        pred_scaled = self.model(X_arr, training=False).numpy()
        pred_hourly = pred_scaled * self.std_Y + self.mean_Y

        pred_summary = np.column_stack([
            np.max(pred_hourly, axis=1),
            np.min(pred_hourly, axis=1),
            np.mean(pred_hourly, axis=1)
        ])

        return pred_hourly, pred_summary
