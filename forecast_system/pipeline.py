import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from forecast_system.config import load_config
from forecast_system.data.loader import load_combined_dataset, get_train_val_test_splits
from forecast_system.data.feature_engineering import create_features_and_targets
from forecast_system.data.quality_analysis import export_data_properties_json
from forecast_system.models.baselines import PersistenceForecast, ClimatologyForecast
from forecast_system.models.ml_models import GradientBoostingForecast, RidgeLinearForecast
from forecast_system.models.deep_learning import LSTMForecastModel
from forecast_system.evaluation.evaluator import ForecastEvaluator

def run_pipeline(config_path=None):
    # 1. Load YAML Configuration
    cfg = load_config(config_path)

    mode = cfg['pipeline']['mode']
    output_dir = Path(cfg['pipeline']['output_dir'])
    outputs_dir = output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    epochs_per_iter = cfg['optimization']['epochs_per_iter']
    max_iters = cfg['optimization']['max_iters']
    target_mae = cfg['optimization']['target_mae']

    dl_cfg = cfg['deep_learning']
    ml_cfg = cfg['machine_learning']
    dq_cfg = cfg['data_quality']

    is_reset = (mode.lower() == "reset")

    print("========================================================")
    print("STEP 1: LOADING & PREPROCESSING WEATHER DATASETS")
    print("========================================================")
    df_combined = load_combined_dataset()

    print("\n========================================================")
    print("STEP 2: FEATURE ENGINEERING & DATASET PROPERTIES ANALYSIS")
    print("========================================================")
    X, Y_hourly, Y_summary = create_features_and_targets(df_combined)

    # Export Data Properties & Split Analysis into report/outputs/
    export_data_properties_json(
        X,
        outputs_dir / "data_properties.json",
        temp_min=dq_cfg.get('temp_min', -30.0),
        temp_max=dq_cfg.get('temp_max', 50.0),
        dew_min=dq_cfg.get('dew_min', -40.0),
        dew_max=dq_cfg.get('dew_max', 40.0)
    )
    export_data_properties_json(
        X,
        output_dir / "data_properties.json",
        temp_min=dq_cfg.get('temp_min', -30.0),
        temp_max=dq_cfg.get('temp_max', 50.0),
        dew_min=dq_cfg.get('dew_min', -40.0),
        dew_max=dq_cfg.get('dew_max', 40.0)
    )  # fallback copy

    # Splits
    train_mask = X.index < '2025-01-01'
    val_mask = (X.index >= '2025-01-01') & (X.index < '2026-01-01')
    test_mask = X.index >= '2026-01-01'

    X_train, Y_h_train, Y_s_train = X.loc[train_mask], Y_hourly.loc[train_mask], Y_summary.loc[train_mask]
    X_val, Y_h_val, Y_s_val = X.loc[val_mask], Y_hourly.loc[val_mask], Y_summary.loc[val_mask]
    X_test, Y_h_test, Y_s_test = X.loc[test_mask], Y_hourly.loc[test_mask], Y_summary.loc[test_mask]

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    # Combine Train + Val for baseline models training
    X_train_full = pd.concat([X_train, X_val])
    Y_h_train_full = pd.concat([Y_h_train, Y_h_val])
    Y_s_train_full = pd.concat([Y_s_train, Y_s_val])

    # Parse models selection filter
    selected_models = cfg['pipeline'].get('models', ['all'])
    if isinstance(selected_models, str):
        selected_models = [selected_models]
    selected_models = [m.lower() for m in selected_models]
    run_all = 'all' in selected_models

    print("\n========================================================")
    print(f"STEP 3: TRAINING & MODEL FITTING (MODE: {mode.upper()})")
    print("========================================================")

    models_dict = {}

    # 1. Persistence Baseline
    if run_all or 'persistence' in selected_models:
        models_dict['Persistence Baseline'] = PersistenceForecast().fit(X_train_full, Y_h_train_full, Y_s_train_full)

    # 2. Climatology Baseline
    if run_all or 'climatology' in selected_models:
        models_dict['Climatology Baseline'] = ClimatologyForecast().fit(df_combined.loc[df_combined.index < '2026-01-01'])

    # 3. Ridge Linear
    if run_all or 'ridge' in selected_models:
        models_dict['Ridge Regression'] = RidgeLinearForecast(
            alpha=ml_cfg.get('ridge_alpha', 1.0)
        ).fit(X_train_full, Y_h_train_full, Y_s_train_full)

    # 4. HistGradientBoosting (GBDT) - Resumable checkpointing
    if run_all or 'gbdt' in selected_models or 'gradient_boosting' in selected_models or 'gbt' in selected_models:
        gbdt_checkpoint_file = outputs_dir / "gbdt_checkpoint.joblib"
        if not is_reset and gbdt_checkpoint_file.exists():
            print(f"✓ [RESUME MODE] Loading pre-trained Gradient Boosted Trees model from {gbdt_checkpoint_file}")
            gbdt_model = GradientBoostingForecast.load(gbdt_checkpoint_file)
        else:
            gbdt_model = GradientBoostingForecast(
                max_iter=ml_cfg.get('gbdt_max_iter', 40),
                learning_rate=ml_cfg.get('gbdt_learning_rate', 0.1),
                random_state=ml_cfg.get('gbdt_random_state', 42)
            ).fit(X_train_full, Y_h_train_full, Y_s_train_full)
            gbdt_model.save(gbdt_checkpoint_file)
            print(f"Saved Gradient Boosted Trees checkpoint to {gbdt_checkpoint_file}")
        models_dict['Gradient Boosted Trees'] = gbdt_model

    # 5. Deep Learning (LSTM) - Resumable/Resetable Training
    if run_all or 'lstm' in selected_models or 'deep_learning' in selected_models:
        lstm_model = LSTMForecastModel(
            units=dl_cfg.get('units', 64),
            dropout=dl_cfg.get('dropout', 0.2),
            learning_rate=dl_cfg.get('learning_rate', 0.001)
        )
        checkpoint_file = outputs_dir / "lstm_checkpoint.keras"
        history_file = outputs_dir / "training_history.json"

        lstm_model.fit_restartable(
            X_train=X_train,
            Y_hourly_train=Y_h_train,
            Y_summary_train=Y_s_train,
            X_val=X_val,
            Y_hourly_val=Y_h_val,
            epochs_per_iter=epochs_per_iter,
            target_mae=target_mae,
            max_iters=max_iters,
            batch_size=dl_cfg.get('batch_size', 64),
            patience=dl_cfg.get('patience', 10),
            verbose=0,
            checkpoint_path=str(checkpoint_file),
            history_path=str(history_file),
            reset=is_reset
        )

        # Fallback copy for root report directory
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(lstm_model.training_history, f, indent=2)

        models_dict['Deep Learning (LSTM)'] = lstm_model

    print("\n========================================================")
    print("STEP 4: EVALUATING OUT-OF-SAMPLE PRECISION & ACCURACY")
    print("========================================================")
    evaluator = ForecastEvaluator(models_dict)
    eval_results = evaluator.evaluate(X_test, Y_h_test, Y_s_test)

    evaluator.save_summary_json(outputs_dir / "model_evaluation_metrics.json")
    evaluator.save_summary_json(output_dir / "model_evaluation_metrics.json")

    # Export a slice of test predictions for interactive visualization in report
    sample_size = min(dq_cfg.get('sample_test_size', 200), len(X_test))
    sample_indices = X_test.index[-sample_size:]

    time_series_export = {
        'timestamps': [ts.strftime('%Y-%m-%d %H:%M') for ts in sample_indices],
        'actual_current_temp': df_combined.loc[sample_indices, 'temp'].tolist(),
        'actual_next24_tmax': Y_s_test.loc[sample_indices, 'target_tmax'].tolist(),
        'actual_next24_tmin': Y_s_test.loc[sample_indices, 'target_tmin'].tolist(),
        'actual_next24_tavg': Y_s_test.loc[sample_indices, 'target_tavg'].tolist(),
        'predictions': {}
    }

    for name, res in eval_results.items():
        time_series_export['predictions'][name] = {
            'tmax': res['predictions_summary'][-sample_size:, 0].tolist(),
            'tmin': res['predictions_summary'][-sample_size:, 1].tolist(),
            'tavg': res['predictions_summary'][-sample_size:, 2].tolist(),
            'sample_24h_profile': res['predictions_hourly'][-1, :].tolist()
        }

    time_series_export['actual_last_24h_profile'] = Y_h_test.iloc[-1].tolist()
    time_series_export['last_timestamp'] = sample_indices[-1].strftime('%Y-%m-%d %H:%M')

    with open(outputs_dir / "test_predictions.json", 'w') as f:
        json.dump(time_series_export, f, indent=2)

    with open(output_dir / "test_predictions.json", 'w') as f:
        json.dump(time_series_export, f, indent=2)

    print(f"Exported test time series predictions to {outputs_dir / 'test_predictions.json'}")
    print("\nPipeline execution complete!")

def main():
    parser = argparse.ArgumentParser(description="Run weather forecast pipeline with YAML configuration.")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML configuration file (e.g. config.yaml)")

    args = parser.parse_args()
    run_pipeline(config_path=args.config)

if __name__ == "__main__":
    main()
