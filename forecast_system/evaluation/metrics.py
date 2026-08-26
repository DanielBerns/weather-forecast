import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculates MAE, RMSE, MAPE, and R2 across flattened arrays or target matrices.
    """
    yt = np.asarray(y_true).flatten()
    yp = np.asarray(y_pred).flatten()

    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))

    # Avoid division by zero in MAPE (temperatures near 0°C)
    nonzero_mask = np.abs(yt) > 0.5
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs((yt[nonzero_mask] - yp[nonzero_mask]) / yt[nonzero_mask])) * 100.0)
    else:
        mape = 0.0

    r2 = float(r2_score(yt, yp))

    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'mape': round(mape, 2),
        'r2': round(r2, 4)
    }

def calculate_tolerance_accuracy(y_true, y_pred):
    """
    Calculates percentage of predictions within ±1.0°C and ±2.0°C tolerances.
    """
    yt = np.asarray(y_true).flatten()
    yp = np.asarray(y_pred).flatten()
    abs_errors = np.abs(yt - yp)

    acc_1c = float(np.mean(abs_errors <= 1.0) * 100.0)
    acc_2c = float(np.mean(abs_errors <= 2.0) * 100.0)

    return {
        'acc_1.0C': round(acc_1c, 2),
        'acc_2.0C': round(acc_2c, 2)
    }

def calculate_directional_metrics(y_true_sequence, y_pred_sequence, x_current_temp):
    """
    Calculates Precision, Recall, and F1-score for temperature trend direction over 24 hours.
    Trend = sign(T_{t+24} - T_t)
    """
    y_true_24 = np.asarray(y_true_sequence)[:, -1]
    y_pred_24 = np.asarray(y_pred_sequence)[:, -1]
    t0 = np.asarray(x_current_temp).flatten()

    actual_rise = (y_true_24 > t0).astype(int)
    pred_rise = (y_pred_24 > t0).astype(int)

    precision = float(precision_score(actual_rise, pred_rise, zero_division=0))
    recall = float(recall_score(actual_rise, pred_rise, zero_division=0))
    f1 = float(f1_score(actual_rise, pred_rise, zero_division=0))

    return {
        'directional_precision': round(precision * 100.0, 2),
        'directional_recall': round(recall * 100.0, 2),
        'directional_f1': round(f1 * 100.0, 2)
    }

def evaluate_model_performance(model_name, y_true_hourly, y_pred_hourly, y_true_summary, y_pred_summary, x_current_temp):
    """
    Computes a comprehensive performance dictionary for a model.
    """
    reg_hourly = calculate_regression_metrics(y_true_hourly, y_pred_hourly)
    tol_hourly = calculate_tolerance_accuracy(y_true_hourly, y_pred_hourly)
    dir_hourly = calculate_directional_metrics(y_true_hourly, y_pred_hourly, x_current_temp)

    # Summary metrics (TMAX, TMIN, TAVG)
    reg_tmax = calculate_regression_metrics(y_true_summary['target_tmax'], y_pred_summary[:, 0])
    reg_tmin = calculate_regression_metrics(y_true_summary['target_tmin'], y_pred_summary[:, 1])
    reg_tavg = calculate_regression_metrics(y_true_summary['target_tavg'], y_pred_summary[:, 2])

    return {
        'model_name': model_name,
        'hourly_24h': {
            **reg_hourly,
            **tol_hourly,
            **dir_hourly
        },
        'tmax': reg_tmax,
        'tmin': reg_tmin,
        'tavg': reg_tavg
    }
