"""
Forecasting models package: Baselines, ML Regressors, and Deep Learning / Neural Network models.
"""
from forecast_system.models.baselines import PersistenceForecast, ClimatologyForecast
from forecast_system.models.ml_models import RidgeLinearForecast, GradientBoostingForecast
from forecast_system.models.deep_learning import LSTMForecastModel
from forecast_system.models.cnn_weather_forecast import CNNForecastModel
from forecast_system.models.dense_weather_forecast import DenseForecastModel
from forecast_system.models.linear_weather_forecast import LinearForecastModel

__all__ = [
    'PersistenceForecast',
    'ClimatologyForecast',
    'RidgeLinearForecast',
    'GradientBoostingForecast',
    'LSTMForecastModel',
    'CNNForecastModel',
    'DenseForecastModel',
    'LinearForecastModel'
]
