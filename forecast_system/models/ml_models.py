import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

class GradientBoostingForecast:
    """
    Gradient Boosted Trees forecasting model for 24-hour hourly temperatures and daily summary targets.
    """
    def __init__(self, n_estimators=100, max_iter=100, learning_rate=0.1, random_state=42):
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        # Multi-output regressor for 24 hourly steps
        base_estimator = HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=self.learning_rate, random_state=self.random_state)
        self.hourly_model = MultiOutputRegressor(base_estimator, n_jobs=-1)

        # Separate regressors for TMAX, TMIN, TAVG
        self.tmax_model = HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=self.learning_rate, random_state=self.random_state)
        self.tmin_model = HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=self.learning_rate, random_state=self.random_state)
        self.tavg_model = HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=self.learning_rate, random_state=self.random_state)

    def fit(self, X_train, Y_hourly_train, Y_summary_train):
        """
        Fits multi-output regressors on training set.
        """
        print("Training Gradient Boosted Trees for 24-hour sequence...")
        self.hourly_model.fit(X_train, Y_hourly_train.values)

        print("Training Gradient Boosted Trees for daily summary metrics...")
        self.tmax_model.fit(X_train, Y_summary_train['target_tmax'].values)
        self.tmin_model.fit(X_train, Y_summary_train['target_tmin'].values)
        self.tavg_model.fit(X_train, Y_summary_train['target_tavg'].values)

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
    Regularized Ridge Regression baseline/ML model.
    """
    def __init__(self, alpha=1.0):
        self.hourly_model = MultiOutputRegressor(Ridge(alpha=alpha))
        self.summary_model = MultiOutputRegressor(Ridge(alpha=alpha))

    def fit(self, X_train, Y_hourly_train, Y_summary_train):
        self.hourly_model.fit(X_train, Y_hourly_train.values)
        self.summary_model.fit(X_train, Y_summary_train.values)
        return self

    def predict(self, X):
        pred_hourly = self.hourly_model.predict(X)
        pred_summary = self.summary_model.predict(X)
        return pred_hourly, pred_summary
