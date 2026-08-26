import numpy as np
import pandas as pd

class PersistenceForecast:
    """
    Persistence baseline model:
    Predicts that future temperatures for the next 24 hours will equal the current temperature T_t.
    """
    def __init__(self):
        pass

    def fit(self, X, Y_hourly, Y_summary=None):
        return self

    def predict(self, X):
        """
        Returns:
            pred_hourly: np.ndarray (n_samples, 24)
            pred_summary: np.ndarray (n_samples, 3) -> [tmax, tmin, tavg]
        """
        t_current = X['temp_lag0'].values
        # Repeat t_current across 24 hours
        pred_hourly = np.tile(t_current[:, np.newaxis], (1, 24))

        pred_summary = np.column_stack([t_current, t_current, t_current])
        return pred_hourly, pred_summary


class ClimatologyForecast:
    """
    Climatology baseline model:
    Calculates historical mean temperature for each (day_of_year, hour_of_day) in training data.
    """
    def __init__(self):
        self.climatology_map = {}
        self.global_mean = 13.0

    def fit(self, df_train):
        """
        df_train: DataFrame with index=timestamp, column 'temp'
        """
        df = df_train.copy()
        df['doy'] = df.index.dayofyear
        df['hour'] = df.index.hour
        self.global_mean = float(df['temp'].mean())

        grouped = df.groupby(['doy', 'hour'])['temp'].mean()
        self.climatology_map = grouped.to_dict()
        return self

    def predict(self, X):
        """
        Predicts using historical average for the 24 target timestamps corresponding to index of X.
        """
        timestamps = X.index
        n_samples = len(timestamps)

        pred_hourly = np.zeros((n_samples, 24))

        for i, ts in enumerate(timestamps):
            # Target timestamps are ts + 1h to ts + 24h
            future_times = pd.date_range(ts + pd.Timedelta(hours=1), periods=24, freq='1h')
            for h, fts in enumerate(future_times):
                key = (fts.dayofyear, fts.hour)
                pred_hourly[i, h] = self.climatology_map.get(key, self.global_mean)

        pred_summary = np.column_stack([
            np.max(pred_hourly, axis=1),
            np.min(pred_hourly, axis=1),
            np.mean(pred_hourly, axis=1)
        ])

        return pred_hourly, pred_summary
