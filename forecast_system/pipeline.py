import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import logging
from datetime import datetime
from forecast_system.config import load_config, get_git_repo_root
from forecast_system.data.loader import load_combined_dataset, get_train_val_test_splits
from forecast_system.data.feature_engineering import create_features_and_targets
from forecast_system.data.quality_analysis import export_data_properties_json
from forecast_system.models.baselines import PersistenceForecast, ClimatologyForecast
from forecast_system.models.ml_models import GradientBoostingForecast, RidgeLinearForecast
from forecast_system.models.deep_learning import (
    LSTMForecastModel,
    CNNForecastModel,
    DenseForecastModel,
    LinearForecastModel
)
from forecast_system.evaluation.evaluator import ForecastEvaluator


def sync_report_ui_files(output_dir, config_dir):
    """
    Syncs report UI files (index.html, script.js, style.css) between output_dir
    and the Git repository's report directory.
    """
    git_root = get_git_repo_root(config_dir)
    if not git_root:
        git_root = get_git_repo_root(Path.cwd())
    if not git_root:
        return

    git_report_dir = git_root / "report"
    git_report_dir.mkdir(parents=True, exist_ok=True)
    ui_files = ["index.html", "script.js", "style.css"]

    output_dir = Path(output_dir)

    # 1. Copy template files from git repo report/ to output_dir if missing
    if output_dir != git_report_dir:
        for fname in ui_files:
            src = git_report_dir / fname
            dst = output_dir / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    # 2. Copy latest version from output_dir to git repo report/
    for fname in ui_files:
        src = output_dir / fname
        dst = git_report_dir / fname
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
            print(f"✓ Copied latest version of {fname} -> {dst}")


def setup_file_logging(logs_dir="logs"):
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "forecast_pipeline.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not any(isinstance(h, logging.FileHandler) and Path(getattr(h, 'baseFilename', '')).name == "forecast_pipeline.log" for h in root_logger.handlers):
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(file_formatter)
        root_logger.addHandler(fh)

    logging.info("========================================================")
    logging.info(f"PIPELINE EXECUTION LOG STARTED AT {datetime.now().isoformat()}")
    logging.info("========================================================")
    return log_file

def run_pipeline(config_path=None):
    # 1. Load YAML Configuration (and enforce git repo check)
    cfg = load_config(config_path)
    config_dir = Path(cfg['_config_dir'])

    mode = cfg['pipeline']['mode']
    is_reset = (mode.lower() == "reset")

    raw_output_dir = Path(cfg['pipeline']['output_dir'])
    if raw_output_dir.is_absolute():
        output_dir = raw_output_dir
    else:
        output_dir = config_dir / raw_output_dir

    outputs_dir = output_dir / "outputs"
    logs_dir = config_dir / "logs"

    # Reset mode: Erase and recreate logs and report/outputs directories
    if is_reset:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

        if logs_dir.exists():
            shutil.rmtree(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ [RESET MODE] Erased and recreated logs directory at '{logs_dir}'")

        if outputs_dir.exists():
            shutil.rmtree(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ [RESET MODE] Erased and recreated outputs directory at '{outputs_dir}'")
    else:
        outputs_dir.mkdir(parents=True, exist_ok=True)

    # 2. Set up logging inside the config directory's logs/ folder
    log_file = setup_file_logging(logs_dir=logs_dir)
    print(f"Logging extensive pipeline diagnostics to '{log_file}'")

    # Sync UI template files to output_dir
    sync_report_ui_files(output_dir, config_dir)

    epochs_per_iter = cfg['optimization']['epochs_per_iter']
    max_iters = cfg['optimization']['max_iters']
    target_mae = cfg['optimization']['target_mae']

    dl_cfg = cfg.get('deep_learning', {})
    cnn_cfg = cfg.get('cnn', {})
    dense_cfg = cfg.get('dense', {})
    linear_cfg = cfg.get('linear', {})
    ml_cfg = cfg.get('machine_learning', {})
    dq_cfg = cfg.get('data_quality', {})

    is_reset = (mode.lower() == "reset")

    logging.info(f"Pipeline execution mode: {mode.upper()} | Output Dir: {outputs_dir}")
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

    # Parse models selection filter & enabled flags
    opt_cfg = cfg.get('optimization', {})
    default_decay_enabled = opt_cfg.get('lr_decay_enabled', True)
    default_decay_policy = opt_cfg.get('lr_decay_policy', 'plateau')
    default_decay_factor = opt_cfg.get('lr_decay_factor', 0.5)
    default_decay_patience = opt_cfg.get('lr_decay_patience', 2)
    default_lr_min = opt_cfg.get('lr_min', 1e-6)
    default_lr_threshold = opt_cfg.get('lr_decay_threshold', 1e-4)
    default_cooldown = opt_cfg.get('lr_cooldown', 0)

    lstm_cfg = {**cfg.get('deep_learning', {}), **cfg.get('lstm', {})}
    cnn_cfg = cfg.get('cnn', {})
    dense_cfg = cfg.get('dense', {})
    linear_cfg = cfg.get('linear', {})
    ridge_cfg = {**cfg.get('machine_learning', {}), **cfg.get('ridge', {})}
    gbdt_cfg = {**cfg.get('machine_learning', {}), **cfg.get('gbdt', {})}
    dq_cfg = cfg.get('data_quality', {})

    def is_enabled(model_name):
        sec = cfg.get(model_name, {})
        if not isinstance(sec, dict):
            if model_name == 'lstm' and isinstance(cfg.get('deep_learning'), dict):
                sec = cfg.get('deep_learning')
            else:
                sec = {}
        return bool(sec.get('enabled', True))


    print("\n========================================================")
    print(f"STEP 3: TRAINING & MODEL FITTING (MODE: {mode.upper()})")
    print("========================================================")

    models_dict = {}

    # 1. Persistence Baseline
    if is_enabled('persistence'):
        models_dict['Persistence Baseline'] = PersistenceForecast().fit(X_train_full, Y_h_train_full, Y_s_train_full)

    # 2. Climatology Baseline
    if is_enabled('climatology'):
        models_dict['Climatology Baseline'] = ClimatologyForecast().fit(df_combined.loc[df_combined.index < '2026-01-01'])

    # 3. Ridge Linear
    if is_enabled('ridge'):
        ridge_model = RidgeLinearForecast(
            alpha=ridge_cfg.get('alpha', ridge_cfg.get('ridge_alpha', 1.0)),
            learning_rate=ridge_cfg.get('learning_rate', 0.01),
            lr_decay_enabled=ridge_cfg.get('lr_decay_enabled', default_decay_enabled),
            lr_decay_policy=ridge_cfg.get('lr_decay_policy', default_decay_policy),
            lr_decay_factor=ridge_cfg.get('lr_decay_factor', default_decay_factor),
            lr_decay_patience=ridge_cfg.get('lr_decay_patience', default_decay_patience),
            min_lr=ridge_cfg.get('lr_min', default_lr_min),
            min_delta=ridge_cfg.get('lr_decay_threshold', default_lr_threshold),
            cooldown=ridge_cfg.get('lr_cooldown', default_cooldown)
        ).fit(X_train, Y_h_train, Y_s_train, X_val=X_val, Y_hourly_val=Y_h_val, Y_summary_val=Y_s_val)
        models_dict['Ridge Regression'] = ridge_model
        if hasattr(ridge_model, 'training_history') and ridge_model.training_history:
            with open(outputs_dir / "ridge_training_history.json", 'w') as f:
                json.dump(ridge_model.training_history, f, indent=2)

    # 4. HistGradientBoosting (GBDT) - Resumable checkpointing
    if is_enabled('gbdt'):
        gbdt_checkpoint_file = outputs_dir / "gbdt_checkpoint.joblib"
        if not is_reset and gbdt_checkpoint_file.exists():
            print(f"✓ [RESUME MODE] Loading pre-trained Gradient Boosted Trees model from {gbdt_checkpoint_file}")
            gbdt_model = GradientBoostingForecast.load(gbdt_checkpoint_file)
        else:
            gbdt_model = GradientBoostingForecast(
                max_iter=gbdt_cfg.get('max_iter', gbdt_cfg.get('gbdt_max_iter', 40)),
                learning_rate=gbdt_cfg.get('learning_rate', gbdt_cfg.get('gbdt_learning_rate', 0.1)),
                random_state=gbdt_cfg.get('random_state', gbdt_cfg.get('gbdt_random_state', 42)),
                lr_decay_enabled=gbdt_cfg.get('lr_decay_enabled', default_decay_enabled),
                lr_decay_policy=gbdt_cfg.get('lr_decay_policy', default_decay_policy),
                lr_decay_factor=gbdt_cfg.get('lr_decay_factor', default_decay_factor),
                lr_decay_patience=gbdt_cfg.get('lr_decay_patience', default_decay_patience),
                min_lr=gbdt_cfg.get('lr_min', 1e-5),
                min_delta=gbdt_cfg.get('lr_decay_threshold', default_lr_threshold),
                cooldown=gbdt_cfg.get('lr_cooldown', default_cooldown)
            ).fit(X_train, Y_h_train, Y_s_train, X_val=X_val, Y_hourly_val=Y_h_val, Y_summary_val=Y_s_val)
            gbdt_model.save(gbdt_checkpoint_file)
            print(f"Saved Gradient Boosted Trees checkpoint to {gbdt_checkpoint_file}")
        if hasattr(gbdt_model, 'training_history') and gbdt_model.training_history:
            with open(outputs_dir / "gbdt_training_history.json", 'w') as f:
                json.dump(gbdt_model.training_history, f, indent=2)
        models_dict['Gradient Boosted Trees'] = gbdt_model

    # 5. Deep Learning (LSTM) - Resumable/Resetable Training
    if is_enabled('lstm'):
        lstm_model = LSTMForecastModel(
            units=lstm_cfg.get('units', 64),
            dropout=lstm_cfg.get('dropout', 0.2),
            learning_rate=lstm_cfg.get('learning_rate', 0.001),
            lr_decay_enabled=lstm_cfg.get('lr_decay_enabled', default_decay_enabled),
            lr_decay_policy=lstm_cfg.get('lr_decay_policy', default_decay_policy),
            lr_decay_factor=lstm_cfg.get('lr_decay_factor', default_decay_factor),
            lr_decay_patience=lstm_cfg.get('lr_decay_patience', default_decay_patience),
            lr_min=lstm_cfg.get('lr_min', default_lr_min),
            lr_decay_threshold=lstm_cfg.get('lr_decay_threshold', default_lr_threshold),
            lr_cooldown=lstm_cfg.get('lr_cooldown', default_cooldown)
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
            batch_size=lstm_cfg.get('batch_size', 64),
            patience=lstm_cfg.get('patience', 10),
            verbose=0,
            checkpoint_path=str(checkpoint_file),
            history_path=str(history_file),
            reset=is_reset
        )

        with open(outputs_dir / "training_history.json", 'w') as f:
            json.dump(lstm_model.training_history, f, indent=2)
        with open(outputs_dir / "lstm_training_history.json", 'w') as f:
            json.dump(lstm_model.training_history, f, indent=2)

        models_dict['Deep Learning (LSTM)'] = lstm_model

    # 6. Convolutional Neural Network (CNN)
    if is_enabled('cnn'):
        cnn_model = CNNForecastModel(
            filters=cnn_cfg.get('filters', 64),
            kernel_size=cnn_cfg.get('kernel_size', 3),
            dropout=cnn_cfg.get('dropout', 0.2),
            learning_rate=cnn_cfg.get('learning_rate', 0.001),
            lr_decay_enabled=cnn_cfg.get('lr_decay_enabled', default_decay_enabled),
            lr_decay_policy=cnn_cfg.get('lr_decay_policy', default_decay_policy),
            lr_decay_factor=cnn_cfg.get('lr_decay_factor', default_decay_factor),
            lr_decay_patience=cnn_cfg.get('lr_decay_patience', default_decay_patience),
            lr_min=cnn_cfg.get('lr_min', default_lr_min),
            lr_decay_threshold=cnn_cfg.get('lr_decay_threshold', default_lr_threshold),
            lr_cooldown=cnn_cfg.get('lr_cooldown', default_cooldown)
        )
        checkpoint_file = outputs_dir / "cnn_checkpoint.keras"
        history_file = outputs_dir / "cnn_training_history.json"

        cnn_model.fit_restartable(
            X_train=X_train,
            Y_hourly_train=Y_h_train,
            Y_summary_train=Y_s_train,
            X_val=X_val,
            Y_hourly_val=Y_h_val,
            epochs_per_iter=epochs_per_iter,
            target_mae=target_mae,
            max_iters=max_iters,
            batch_size=cnn_cfg.get('batch_size', 64),
            patience=cnn_cfg.get('patience', 10),
            verbose=0,
            checkpoint_path=str(checkpoint_file),
            history_path=str(history_file),
            reset=is_reset
        )
        models_dict['Convolutional Neural Network (CNN)'] = cnn_model

    # 7. Dense Neural Network
    if is_enabled('dense'):
        dense_model = DenseForecastModel(
            hidden_units=dense_cfg.get('hidden_units', (128, 64)),
            dropout=dense_cfg.get('dropout', 0.2),
            learning_rate=dense_cfg.get('learning_rate', 0.001),
            lr_decay_enabled=dense_cfg.get('lr_decay_enabled', default_decay_enabled),
            lr_decay_policy=dense_cfg.get('lr_decay_policy', default_decay_policy),
            lr_decay_factor=dense_cfg.get('lr_decay_factor', default_decay_factor),
            lr_decay_patience=dense_cfg.get('lr_decay_patience', default_decay_patience),
            lr_min=dense_cfg.get('lr_min', default_lr_min),
            lr_decay_threshold=dense_cfg.get('lr_decay_threshold', default_lr_threshold),
            lr_cooldown=dense_cfg.get('lr_cooldown', default_cooldown)
        )
        checkpoint_file = outputs_dir / "dense_checkpoint.keras"
        history_file = outputs_dir / "dense_training_history.json"

        dense_model.fit_restartable(
            X_train=X_train,
            Y_hourly_train=Y_h_train,
            Y_summary_train=Y_s_train,
            X_val=X_val,
            Y_hourly_val=Y_h_val,
            epochs_per_iter=epochs_per_iter,
            target_mae=target_mae,
            max_iters=max_iters,
            batch_size=dense_cfg.get('batch_size', 64),
            patience=dense_cfg.get('patience', 10),
            verbose=0,
            checkpoint_path=str(checkpoint_file),
            history_path=str(history_file),
            reset=is_reset
        )
        models_dict['Dense Neural Network'] = dense_model

    # 8. Linear Neural Network
    if is_enabled('linear'):
        linear_model = LinearForecastModel(
            learning_rate=linear_cfg.get('learning_rate', 0.001),
            lr_decay_enabled=linear_cfg.get('lr_decay_enabled', default_decay_enabled),
            lr_decay_policy=linear_cfg.get('lr_decay_policy', default_decay_policy),
            lr_decay_factor=linear_cfg.get('lr_decay_factor', default_decay_factor),
            lr_decay_patience=linear_cfg.get('lr_decay_patience', default_decay_patience),
            lr_min=linear_cfg.get('lr_min', default_lr_min),
            lr_decay_threshold=linear_cfg.get('lr_decay_threshold', default_lr_threshold),
            lr_cooldown=linear_cfg.get('lr_cooldown', default_cooldown)
        )
        checkpoint_file = outputs_dir / "linear_checkpoint.keras"
        history_file = outputs_dir / "linear_training_history.json"

        linear_model.fit_restartable(
            X_train=X_train,
            Y_hourly_train=Y_h_train,
            Y_summary_train=Y_s_train,
            X_val=X_val,
            Y_hourly_val=Y_h_val,
            epochs_per_iter=epochs_per_iter,
            target_mae=target_mae,
            max_iters=max_iters,
            batch_size=linear_cfg.get('batch_size', 64),
            patience=linear_cfg.get('patience', 10),
            verbose=0,
            checkpoint_path=str(checkpoint_file),
            history_path=str(history_file),
            reset=is_reset
        )
        models_dict['Linear Neural Network'] = linear_model

    # Export Consolidated Training Histories for All Models
    all_histories = {}
    for name, model_inst in models_dict.items():
        if hasattr(model_inst, 'training_history') and model_inst.training_history:
            all_histories[name] = model_inst.training_history

    with open(outputs_dir / "all_training_histories.json", 'w') as f:
        json.dump(all_histories, f, indent=2)

    print("\n========================================================")
    print("STEP 4: EVALUATING OUT-OF-SAMPLE PRECISION & ACCURACY")
    print("========================================================")
    evaluator = ForecastEvaluator(models_dict)
    eval_results = evaluator.evaluate(X_test, Y_h_test, Y_s_test)

    evaluator.save_summary_json(outputs_dir / "model_evaluation_metrics.json")

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

    print(f"Exported test time series predictions to {outputs_dir / 'test_predictions.json'}")

    # Copy latest version of report/index.html, script.js, style.css to git repo report/
    sync_report_ui_files(output_dir, config_dir)

    print("\nPipeline execution complete!")

def main():
    parser = argparse.ArgumentParser(description="Run weather forecast pipeline with YAML configuration.")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML configuration file (e.g. config.yaml)")

    args = parser.parse_args()
    run_pipeline(config_path=args.config)

if __name__ == "__main__":
    main()
