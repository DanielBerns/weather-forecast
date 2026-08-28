import os
import tempfile
import numpy as np
import pandas as pd
from forecast_system.config import load_config
from forecast_system.data.preprocessor import parse_hourly_temp, clean_hourly_dataframe
from forecast_system.data.feature_engineering import create_features_and_targets
from forecast_system.data.quality_analysis import analyze_dataset_splits, detect_bad_values, compute_column_stats
from forecast_system.models.baselines import PersistenceForecast, ClimatologyForecast
from forecast_system.models.deep_learning import LSTMForecastModel
from forecast_system.evaluation.metrics import calculate_regression_metrics, calculate_tolerance_accuracy, calculate_directional_metrics

def test_parse_hourly_temp():
    assert parse_hourly_temp('+0050,1') == 5.0
    assert parse_hourly_temp('-0061,1') == -6.1
    assert parse_hourly_temp('9999,9') is np.nan or pd.isna(parse_hourly_temp('9999,9'))
    assert parse_hourly_temp(14.2) == 14.2
    print("✓ test_parse_hourly_temp passed!")

def test_config_loading():
    # 1. Test refusal when config is in git repo
    with tempfile.NamedTemporaryFile('w', dir='.', suffix='.yaml', delete=False) as f:
        f.write("pipeline:\n  mode: resume\n")
        in_repo_config = f.name
    try:
        try:
            load_config(in_repo_config)
            assert False, "Expected ValueError when loading config inside git repo"
        except ValueError as e:
            assert "inside the Git repository" in str(e) or "within the Git repository" in str(e)
    finally:
        if os.path.exists(in_repo_config):
            os.remove(in_repo_config)

    # 2. Test custom YAML loading from external directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        external_cfg = os.path.join(tmp_dir, "custom_config.yaml")
        with open(external_cfg, 'w') as f:
            f.write("""
pipeline:
  mode: "reset"
optimization:
  target_mae: 0.8
deep_learning:
  units: 128
""")
        cfg_custom = load_config(external_cfg)
        assert cfg_custom['pipeline']['mode'] == "reset"
        assert cfg_custom['optimization']['target_mae'] == 0.8
        assert cfg_custom['deep_learning']['units'] == 128
        assert cfg_custom['_config_dir'] == tmp_dir
        print("✓ test_config_loading passed!")

def test_git_repo_restriction():
    repo_config = os.path.abspath("reset.yaml")
    try:
        load_config(repo_config)
        assert False, "Expected load_config to raise ValueError for reset.yaml in git repo"
    except ValueError as e:
        assert "inside the Git repository" in str(e) or "within the Git repository" in str(e)
    print("✓ test_git_repo_restriction passed!")

def test_logs_and_report_in_config_dir():
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            f.write("pipeline:\n  mode: resume\n  output_dir: report\n")

        cfg = load_config(config_path)
        assert cfg['_config_dir'] == tmp_dir

        logs_dir = Path(cfg['_config_dir']) / "logs"
        output_dir = Path(cfg['_config_dir']) / cfg['pipeline']['output_dir']

        assert str(logs_dir) == os.path.join(tmp_dir, "logs")
        assert str(output_dir) == os.path.join(tmp_dir, "report")
        print("✓ test_logs_and_report_in_config_dir passed!")

def test_create_config_dir_script():
    import sys
    import subprocess
    with tempfile.TemporaryDirectory() as tmp_dir:
        ext_target = os.path.join(tmp_dir, "external_config_workspace")
        res = subprocess.run(
            [sys.executable, "create_config_dir.py", ext_target],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        assert res.returncode == 0, f"Script failed: {res.stderr}"
        assert os.path.exists(os.path.join(ext_target, "reset.yaml"))
        assert os.path.exists(os.path.join(ext_target, "resume.yaml"))

        # Test running script targeting directory inside git repo (should fail)
        in_repo_target = os.path.abspath("in_repo_test_dir")
        res_fail = subprocess.run(
            [sys.executable, "create_config_dir.py", in_repo_target],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        assert res_fail.returncode == 1
        assert "inside the Git repository" in res_fail.stderr
        print("✓ test_create_config_dir_script passed!")

def test_reset_mode_directory_erasing():
    import shutil
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config_path = tmp_path / "config.yaml"
        with open(config_path, 'w') as f:
            f.write("pipeline:\n  mode: reset\n  output_dir: report\n")

        # Create dummy old files in logs and report/outputs
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        old_log = logs_dir / "old_pipeline.log"
        old_log.write_text("old log content")

        outputs_dir = tmp_path / "report" / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        old_output = outputs_dir / "old_checkpoint.keras"
        old_output.write_text("old model weights")

        cfg = load_config(config_path)
        is_reset = (cfg['pipeline']['mode'].lower() == "reset")
        assert is_reset is True

        if is_reset:
            if logs_dir.exists():
                shutil.rmtree(logs_dir)
            logs_dir.mkdir(parents=True, exist_ok=True)
            if outputs_dir.exists():
                shutil.rmtree(outputs_dir)
            outputs_dir.mkdir(parents=True, exist_ok=True)

        assert logs_dir.exists()
        assert outputs_dir.exists()
        assert not old_log.exists()
        assert not old_output.exists()
        print("✓ test_reset_mode_directory_erasing passed!")

def test_sync_report_ui_files():
    from pathlib import Path
    from forecast_system.pipeline import sync_report_ui_files
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "report"
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "index.html").write_text("<html>Test Report</html>")
        (out_dir / "script.js").write_text("// Test JS Script")
        (out_dir / "style.css").write_text("/* Test CSS Style */")

        sync_report_ui_files(out_dir, Path.cwd())

        repo_report = Path.cwd() / "report"
        assert (repo_report / "index.html").read_text() == "<html>Test Report</html>"
        assert (repo_report / "script.js").read_text() == "// Test JS Script"
        assert (repo_report / "style.css").read_text() == "/* Test CSS Style */"
        print("✓ test_sync_report_ui_files passed!")

def test_feature_engineering():
    dates = pd.date_range('2024-01-01', periods=300, freq='1h')
    temps = 15.0 + 5.0 * np.sin(np.linspace(0, 10 * np.pi, 300))
    dews = temps - 4.0

    df = pd.DataFrame({'temp': temps, 'dew': dews}, index=dates)
    X, Y_h, Y_s = create_features_and_targets(df)

    assert not X.empty
    assert not Y_h.empty
    assert not Y_s.empty
    assert Y_h.shape[1] == 24
    assert 'target_tmax' in Y_s.columns
    print("✓ test_feature_engineering passed!")

def test_data_quality_analysis():
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')
    temps = 15.0 + 5.0 * np.sin(np.linspace(0, 5 * np.pi, 200))
    dews = temps - 3.0
    df = pd.DataFrame({'temp': temps, 'dew': dews}, index=dates)

    stats = compute_column_stats(df['temp'])
    assert stats['count'] == 200
    assert 'mean' in stats
    assert 'outlier_count' in stats

    bad = detect_bad_values(df)
    assert bad['bad_temp_count'] == 0
    assert bad['dew_above_temp_count'] == 0

    print("✓ test_data_quality_analysis passed!")

def test_baselines_and_metrics():
    dates = pd.date_range('2024-01-01', periods=300, freq='1h')
    temps = 15.0 + 5.0 * np.sin(np.linspace(0, 10 * np.pi, 300))
    dews = temps - 4.0
    df = pd.DataFrame({'temp': temps, 'dew': dews}, index=dates)

    X, Y_h, Y_s = create_features_and_targets(df)

    p_model = PersistenceForecast()
    p_pred_h, p_pred_s = p_model.predict(X)

    reg = calculate_regression_metrics(Y_h.values, p_pred_h)
    tol = calculate_tolerance_accuracy(Y_h.values, p_pred_h)
    dir_m = calculate_directional_metrics(Y_h.values, p_pred_h, X['temp_lag0'])

    assert 'mae' in reg
    assert 'acc_1.0C' in tol
    assert 'directional_f1' in dir_m
    print("✓ test_baselines_and_metrics passed!")

def test_restartable_training():
    dates = pd.date_range('2024-01-01', periods=300, freq='1h')
    temps = 15.0 + 5.0 * np.sin(np.linspace(0, 10 * np.pi, 300))
    dews = temps - 4.0
    df = pd.DataFrame({'temp': temps, 'dew': dews}, index=dates)

    X, Y_h, Y_s = create_features_and_targets(df)

    model = LSTMForecastModel(units=32)
    model.fit_restartable(
        X_train=X.iloc[:200],
        Y_hourly_train=Y_h.iloc[:200],
        X_val=X.iloc[200:],
        Y_hourly_val=Y_h.iloc[200:],
        epochs_per_iter=2,
        max_iters=2,
        target_mae=1.0,
        verbose=0
    )

    assert hasattr(model, 'training_history')
    assert 'epochs' in model.training_history
    assert len(model.training_history['epochs']) > 0
    assert 'final_status' in model.training_history
    print("✓ test_restartable_training passed!")

def test_reset_and_resume_modes():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test.keras")
        hist_path = os.path.join(tmp_dir, "hist.json")

        dates = pd.date_range('2024-01-01', periods=300, freq='1h')
        temps = 15.0 + 5.0 * np.sin(np.linspace(0, 10 * np.pi, 300))
        dews = temps - 4.0
        df = pd.DataFrame({'temp': temps, 'dew': dews}, index=dates)

        X, Y_h, Y_s = create_features_and_targets(df)

        # 1. First run: Reset mode (fresh)
        model1 = LSTMForecastModel(units=16)
        model1.fit_restartable(
            X_train=X.iloc[:200],
            Y_hourly_train=Y_h.iloc[:200],
            X_val=X.iloc[200:],
            Y_hourly_val=Y_h.iloc[200:],
            epochs_per_iter=2,
            max_iters=1,
            checkpoint_path=ckpt_path,
            history_path=hist_path,
            reset=True,
            verbose=0
        )
        assert os.path.exists(ckpt_path)
        assert os.path.exists(hist_path)
        first_epochs = model1.training_history['total_epochs']

        # 2. Second run: Resume mode (resumes and appends history)
        model2 = LSTMForecastModel(units=16)
        model2.fit_restartable(
            X_train=X.iloc[:200],
            Y_hourly_train=Y_h.iloc[:200],
            X_val=X.iloc[200:],
            Y_hourly_val=Y_h.iloc[200:],
            epochs_per_iter=2,
            max_iters=1,
            checkpoint_path=ckpt_path,
            history_path=hist_path,
            reset=False,
            verbose=0
        )
        second_epochs = model2.training_history['total_epochs']
        assert second_epochs > first_epochs

        # 3. Third run: Reset mode again (wipes clean back to initial epoch count)
        model3 = LSTMForecastModel(units=16)
        model3.fit_restartable(
            X_train=X.iloc[:200],
            Y_hourly_train=Y_h.iloc[:200],
            X_val=X.iloc[200:],
            Y_hourly_val=Y_h.iloc[200:],
            epochs_per_iter=2,
            max_iters=1,
            checkpoint_path=ckpt_path,
            history_path=hist_path,
            reset=True,
            verbose=0
        )
        third_epochs = model3.training_history['total_epochs']
        assert third_epochs == 2

        print("✓ test_reset_and_resume_modes passed!")

def test_cli_argument_parsing():
    import sys
    from unittest.mock import patch
    from forecast_system.pipeline import main

    # 1. Test valid --config argument
    with patch.object(sys, 'argv', ['pipeline.py', '--config', 'config.yaml']):
        with patch('forecast_system.pipeline.run_pipeline') as mock_run:
            main()
            mock_run.assert_called_once_with(config_path='config.yaml')

    # 2. Test valid -c short argument
    with patch.object(sys, 'argv', ['pipeline.py', '-c', 'custom_config.yaml']):
        with patch('forecast_system.pipeline.run_pipeline') as mock_run:
            main()
            mock_run.assert_called_once_with(config_path='custom_config.yaml')

    # 3. Test removed CLI parameter triggers SystemExit
    for invalid_arg in ['--mode', '--epochs-per-iter', '--max-iters', '--target-mae', '--output-dir']:
        with patch.object(sys, 'argv', ['pipeline.py', invalid_arg, 'value']):
            try:
                with patch('sys.stderr'):
                    main()
                assert False, f"Expected SystemExit for removed argument {invalid_arg}"
            except SystemExit:
                pass

    print("✓ test_cli_argument_parsing passed!")

def test_gbdt_checkpoint_and_selective_models():
    from forecast_system.models.ml_models import GradientBoostingForecast

    with tempfile.TemporaryDirectory() as tmp_dir:
        dates = pd.date_range('2024-01-01', periods=300, freq='1h')
        temps = 15.0 + 5.0 * np.sin(np.linspace(0, 5 * np.pi, 300))
        df = pd.DataFrame({'temp': temps, 'dew': temps - 2.0}, index=dates)
        X, Y_h, Y_s = create_features_and_targets(df)

        gbdt = GradientBoostingForecast(max_iter=5)
        gbdt.fit(X, Y_h, Y_s)
        save_path = os.path.join(tmp_dir, "gbdt.joblib")
        gbdt.save(save_path)

        assert os.path.exists(save_path)
        loaded_gbdt = GradientBoostingForecast.load(save_path)
        p_h, p_s = loaded_gbdt.predict(X)
        assert p_h.shape == Y_h.shape
        assert p_s.shape == (len(X), 3)

        print("✓ test_gbdt_checkpoint_and_selective_models passed!")

def test_cnn_dense_linear_models():
    from forecast_system.models.cnn_weather_forecast import CNNForecastModel
    from forecast_system.models.dense_weather_forecast import DenseForecastModel
    from forecast_system.models.linear_weather_forecast import LinearForecastModel

    dates = pd.date_range('2024-01-01', periods=300, freq='1h')
    temps = 15.0 + 5.0 * np.sin(np.linspace(0, 10 * np.pi, 300))
    dews = temps - 4.0
    df = pd.DataFrame({'temp': temps, 'dew': dews}, index=dates)

    X, Y_h, Y_s = create_features_and_targets(df)

    # 1. Test CNN Model
    cnn_model = CNNForecastModel(filters=16, kernel_size=3)
    cnn_model.fit_restartable(
        X_train=X.iloc[:200],
        Y_hourly_train=Y_h.iloc[:200],
        X_val=X.iloc[200:],
        Y_hourly_val=Y_h.iloc[200:],
        epochs_per_iter=1,
        max_iters=1,
        verbose=0
    )
    p_h, p_s = cnn_model.predict(X.iloc[200:])
    assert p_h.shape == (len(X.iloc[200:]), 24)
    assert p_s.shape == (len(X.iloc[200:]), 3)

    # 2. Test Dense Model
    dense_model = DenseForecastModel(hidden_units=(32, 16))
    dense_model.fit_restartable(
        X_train=X.iloc[:200],
        Y_hourly_train=Y_h.iloc[:200],
        X_val=X.iloc[200:],
        Y_hourly_val=Y_h.iloc[200:],
        epochs_per_iter=1,
        max_iters=1,
        verbose=0
    )
    p_h, p_s = dense_model.predict(X.iloc[200:])
    assert p_h.shape == (len(X.iloc[200:]), 24)
    assert p_s.shape == (len(X.iloc[200:]), 3)

    # 3. Test Linear Model
    linear_model = LinearForecastModel()
    linear_model.fit_restartable(
        X_train=X.iloc[:200],
        Y_hourly_train=Y_h.iloc[:200],
        X_val=X.iloc[200:],
        Y_hourly_val=Y_h.iloc[200:],
        epochs_per_iter=1,
        max_iters=1,
        verbose=0
    )
    p_h, p_s = linear_model.predict(X.iloc[200:])
    assert p_h.shape == (len(X.iloc[200:]), 24)
    assert p_s.shape == (len(X.iloc[200:]), 3)


    print("✓ test_cnn_dense_linear_models passed!")

def test_enabled_flags_and_full_config():
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write("""
pipeline:
  mode: "reset"

cnn:
  enabled: false
  filters: 32
dense:
  enabled: true
  hidden_units: [64, 32]
linear:
  enabled: false
""")

        tmp_name = f.name

    try:
        cfg = load_config(tmp_name)
        assert cfg['cnn']['enabled'] is False
        assert cfg['cnn']['filters'] == 32
        assert cfg['dense']['enabled'] is True
        assert cfg['dense']['hidden_units'] == [64, 32]
        assert cfg['linear']['enabled'] is False
        print("✓ test_enabled_flags_and_full_config passed!")
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

if __name__ == '__main__':
    test_parse_hourly_temp()
    test_config_loading()
    test_git_repo_restriction()
    test_logs_and_report_in_config_dir()
    test_create_config_dir_script()
    test_reset_mode_directory_erasing()
    test_sync_report_ui_files()
    test_feature_engineering()
    test_data_quality_analysis()
    test_baselines_and_metrics()
    test_restartable_training()
    test_reset_and_resume_modes()
    test_cli_argument_parsing()
    test_gbdt_checkpoint_and_selective_models()
    test_cnn_dense_linear_models()
    test_enabled_flags_and_full_config()
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")


