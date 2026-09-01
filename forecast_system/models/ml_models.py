import os
import joblib
import logging
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.multioutput import MultiOutputRegressor
from forecast_system.models.lr_policy import PerformanceLRDecayPolicy

logger = logging.getLogger("forecast_system.models.ml_models")

class GradientBoostingForecast:
    """
    Gradient Boosted Trees forecasting model for 24-hour hourly temperatures and daily summary targets.
    Supports stage-wise iterative training with configurable performance-based learning rate decay.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 40,
        learning_rate: float = 0.1,
        random_state: int = 42,
        lr_decay_enabled: bool = True,
        lr_decay_policy: str = 'plateau',
        lr_decay_factor: float = 0.5,
        lr_decay_patience: int = 2,
        min_lr: float = 1e-5,
        min_delta: float = 1e-4,
        lr_cooldown: int = 0,
        lr_restart_patience: int = 6,
        lr_restart_factor: float = 0.5,
        max_lr: Optional[float] = None,
        cycle_step_size: int = 5,
        cyclic_mode: str = 'triangular'
    ):
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        self.lr_policy = PerformanceLRDecayPolicy(
            enabled=lr_decay_enabled,
            policy=lr_decay_policy,
            factor=lr_decay_factor,
            patience=lr_decay_patience,
            min_lr=min_lr,
            min_delta=min_delta,
            cooldown=lr_cooldown,
            restart_patience=lr_restart_patience,
            restart_factor=lr_restart_factor,
            max_lr=max_lr,
            cycle_step_size=cycle_step_size,
            cyclic_mode=cyclic_mode,
            monitor='val_mae',
            mode='min',
            initial_lr=learning_rate
        )

        base_estimator = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            warm_start=True
        )
        self.hourly_model = MultiOutputRegressor(base_estimator, n_jobs=-1)

        self.tmax_model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            warm_start=True
        )
        self.tmin_model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            warm_start=True
        )
        self.tavg_model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            warm_start=True
        )

        self.training_history: dict = {}

    def fit(self, X_train, Y_hourly_train, Y_summary_train, X_val=None, Y_hourly_val=None, Y_summary_val=None):
        """
        Fits multi-output regressors on training set, applying performance-based LR decay across stage iterations.
        """
        logger.info(f"\n--- Training Gradient Boosted Trees (Initial LR: {self.learning_rate}, Max Iter: {self.max_iter}) ---")
        self.lr_policy.reset(initial_lr=self.learning_rate)

        if not self.lr_policy.enabled or X_val is None or Y_hourly_val is None:
            logger.info("Fitting GBDT models directly without stage-wise validation LR decay tracking.")
            self.hourly_model.fit(X_train, Y_hourly_train.values)
            self.tmax_model.fit(X_train, Y_summary_train['target_tmax'].values)
            self.tmin_model.fit(X_train, Y_summary_train['target_tmin'].values)
            self.tavg_model.fit(X_train, Y_summary_train['target_tavg'].values)

            self.training_history = {
                'final_learning_rate': self.learning_rate,
                'lr_decay_events': [],
                'status': 'Completed'
            }
            return self

        # Stage-wise training with performance-based LR decay policy evaluation
        stage_step = max(5, self.max_iter // 5)
        current_lr = self.learning_rate
        stage_logs = []

        for stage_max_iter in range(stage_step, self.max_iter + 1, stage_step):
            # Update max_iter & learning_rate safely before/after fit
            try:
                estimators = getattr(self.hourly_model, 'estimators_', None)
                if estimators is not None:
                    for est in estimators:
                        est.max_iter = stage_max_iter
                        est.learning_rate = current_lr
                else:
                    self.hourly_model.estimator.max_iter = stage_max_iter
                    self.hourly_model.estimator.learning_rate = current_lr
            except AttributeError:
                self.hourly_model.estimator.max_iter = stage_max_iter
                self.hourly_model.estimator.learning_rate = current_lr

            self.tmax_model.max_iter = stage_max_iter
            self.tmax_model.learning_rate = current_lr

            self.tmin_model.max_iter = stage_max_iter
            self.tmin_model.learning_rate = current_lr

            self.tavg_model.max_iter = stage_max_iter
            self.tavg_model.learning_rate = current_lr

            self.hourly_model.fit(X_train, Y_hourly_train.values)
            self.tmax_model.fit(X_train, Y_summary_train['target_tmax'].values)
            self.tmin_model.fit(X_train, Y_summary_train['target_tmin'].values)
            self.tavg_model.fit(X_train, Y_summary_train['target_tavg'].values)

            # Evaluate on validation set
            pred_hourly_val = self.hourly_model.predict(X_val)
            val_mae = float(np.mean(np.abs(pred_hourly_val - Y_hourly_val.values)))

            stage_idx = stage_max_iter // stage_step
            new_lr, changed, _ = self.lr_policy.step(
                current_metric=val_mae,
                current_lr=current_lr,
                model_name="GradientBoostedTrees",
                step_idx=stage_idx,
                step_type="Stage"
            )

            stage_logs.append({
                'stage': stage_idx,
                'trees_fitted': stage_max_iter,
                'val_mae': round(val_mae, 4),
                'learning_rate': current_lr
            })

            if changed:
                current_lr = new_lr

        self.training_history = {
            'stages': stage_logs,
            'final_learning_rate': current_lr,
            'lr_decay_events': self.lr_policy.decay_events,
            'status': 'Completed'
        }

        logger.info(f"✓ Gradient Boosted Trees training completed. Final LR: {current_lr:.6e}")
        return self

    def predict(self, X):
        """
        Predicts future 24-hour profile and summary targets.
        """
        pred_hourly = self.hourly_model.predict(X)

        pred_tmax = self.tmax_model.predict(X)
        pred_tmin = self.tmin_model.predict(X)
        pred_tavg = self.tavg_model.predict(X)

        pred_summary = np.column_stack([pred_tmax, pred_tmin, pred_tavg])
        return pred_hourly, pred_summary

    def save(self, filepath):
        """Saves GBDT model artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """Loads GBDT model artifact from disk."""
        return joblib.load(filepath)


class RidgeLinearForecast:
    """
    Regularized Ridge Regression model with optional iterative performance-based learning rate decay.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        learning_rate: float = 0.01,
        lr_decay_enabled: bool = True,
        lr_decay_policy: str = 'plateau',
        lr_decay_factor: float = 0.5,
        lr_decay_patience: int = 2,
        min_lr: float = 1e-6,
        min_delta: float = 1e-4,
        cooldown: int = 0,
        lr_restart_patience: int = 6,
        lr_restart_factor: float = 0.5,
        max_lr: Optional[float] = None,
        cycle_step_size: int = 5,
        cyclic_mode: str = 'triangular'
    ):
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.mean_X = None
        self.std_X = None
        self.mean_Y_h = None
        self.std_Y_h = None
        self.mean_Y_s = None
        self.std_Y_s = None

        self.lr_policy = PerformanceLRDecayPolicy(
            enabled=lr_decay_enabled,
            policy=lr_decay_policy,
            factor=lr_decay_factor,
            patience=lr_decay_patience,
            min_lr=min_lr,
            min_delta=min_delta,
            cooldown=cooldown,
            restart_patience=lr_restart_patience,
            restart_factor=lr_restart_factor,
            max_lr=max_lr,
            cycle_step_size=cycle_step_size,
            cyclic_mode=cyclic_mode,
            monitor='val_mae',
            mode='min',
            initial_lr=learning_rate
        )

        if lr_decay_enabled:
            # Iterative SGD Ridge Regressor with tol=None to allow custom epoch stepping without ConvergenceWarning
            self.hourly_model = MultiOutputRegressor(
                SGDRegressor(penalty='l2', alpha=alpha, eta0=learning_rate, learning_rate='constant', warm_start=True, max_iter=20, tol=None, random_state=42)
            )
            self.summary_model = MultiOutputRegressor(
                SGDRegressor(penalty='l2', alpha=alpha, eta0=learning_rate, learning_rate='constant', warm_start=True, max_iter=20, tol=None, random_state=42)
            )
        else:
            # Analytical Ridge
            self.hourly_model = MultiOutputRegressor(Ridge(alpha=alpha))
            self.summary_model = MultiOutputRegressor(Ridge(alpha=alpha))

        self.training_history: dict = {}

    def fit(self, X_train, Y_hourly_train, Y_summary_train, X_val=None, Y_hourly_val=None, Y_summary_val=None, max_epochs: int = 15):
        logger.info(f"\n--- Training Ridge Linear Model (Alpha: {self.alpha}, Initial LR: {self.learning_rate}) ---")
        self.lr_policy.reset(initial_lr=self.learning_rate)

        # Store feature scalers
        self.mean_X = X_train.mean(axis=0)
        self.std_X = X_train.std(axis=0).replace(0, 1.0)
        self.mean_Y_h = Y_hourly_train.values.mean()
        self.std_Y_h = Y_hourly_train.values.std()
        self.mean_Y_s = Y_summary_train.values.mean(axis=0)
        self.std_Y_s = Y_summary_train.values.std(axis=0)
        self.std_Y_s = np.where(self.std_Y_s == 0, 1.0, self.std_Y_s)

        X_tr_sc = (X_train - self.mean_X) / self.std_X
        Y_h_tr_sc = (Y_hourly_train.values - self.mean_Y_h) / self.std_Y_h
        Y_s_tr_sc = (Y_summary_train.values - self.mean_Y_s) / self.std_Y_s

        if not self.lr_policy.enabled or X_val is None or Y_hourly_val is None:
            logger.info("Fitting closed-form Ridge regression model directly.")
            self.hourly_model.fit(X_tr_sc, Y_h_tr_sc)
            self.summary_model.fit(X_tr_sc, Y_s_tr_sc)

            self.training_history = {
                'final_learning_rate': self.learning_rate,
                'lr_decay_events': [],
                'status': 'Completed'
            }
            return self

        # Iterative SGD training with performance LR decay
        current_lr = self.learning_rate
        epoch_logs = []

        X_va_sc = (X_val - self.mean_X) / self.std_X

        for epoch in range(1, max_epochs + 1):
            # Update learning rate safely before/after fit
            try:
                estimators_h = getattr(self.hourly_model, 'estimators_', None)
                if estimators_h is not None:
                    for est in estimators_h:
                        est.eta0 = current_lr
                else:
                    self.hourly_model.estimator.eta0 = current_lr
            except AttributeError:
                self.hourly_model.estimator.eta0 = current_lr

            try:
                estimators_s = getattr(self.summary_model, 'estimators_', None)
                if estimators_s is not None:
                    for est in estimators_s:
                        est.eta0 = current_lr
                else:
                    self.summary_model.estimator.eta0 = current_lr
            except AttributeError:
                self.summary_model.estimator.eta0 = current_lr

            self.hourly_model.fit(X_tr_sc, Y_h_tr_sc)
            self.summary_model.fit(X_tr_sc, Y_s_tr_sc)

            # Evaluate val MAE
            pred_h_scaled = self.hourly_model.predict(X_va_sc)
            pred_h_orig = pred_h_scaled * self.std_Y_h + self.mean_Y_h
            val_mae = float(np.mean(np.abs(pred_h_orig - Y_hourly_val.values)))

            new_lr, changed, _ = self.lr_policy.step(
                current_metric=val_mae,
                current_lr=current_lr,
                model_name="RidgeRegression",
                step_idx=epoch,
                step_type="Epoch"
            )

            epoch_logs.append({
                'epoch': epoch,
                'val_mae': round(val_mae, 4),
                'learning_rate': current_lr
            })

            if changed:
                current_lr = new_lr

        self.training_history = {
            'epochs': epoch_logs,
            'final_learning_rate': current_lr,
            'lr_decay_events': self.lr_policy.decay_events,
            'status': 'Completed'
        }

        logger.info(f"✓ Ridge Regression training completed. Final LR: {current_lr:.6e}")
        return self

    def predict(self, X):
        if hasattr(self, 'mean_X') and self.mean_X is not None:
            X_sc = (X - self.mean_X) / self.std_X
            pred_h_sc = self.hourly_model.predict(X_sc)
            pred_s_sc = self.summary_model.predict(X_sc)
            pred_hourly = pred_h_sc * self.std_Y_h + self.mean_Y_h
            pred_summary = pred_s_sc * self.std_Y_s + self.mean_Y_s
        else:
            pred_hourly = self.hourly_model.predict(X)
            pred_summary = self.summary_model.predict(X)

        return pred_hourly, pred_summary
