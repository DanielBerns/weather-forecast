# Next-Day Weather Forecast & Precision Evaluation System

Weather analysis, multi-step 24-hour time-series temperature forecasting, resumable/resetable neural network training pipeline, and precision/accuracy evaluation system for **Comodoro Rivadavia, Chubut, Argentina** (Station `SAVC / 87860`).

---

## 🌟 Key Features & Pipeline Capabilities

1. **Multi-Dataset Ingestion & Cleaning**:
   - Ingests historical and modern weather datasets (`tytd_hourly_CR.csv`, `CRV2025.csv`, `CRV2026.csv`).
2. **Feature Engineering**:
   - Multi-period lag features (1h to 168h), rolling stats (mean, min, max, std, range), dew point depression ($T - T_{\text{dew}}$), and sine/cosine cyclical time encodings.
3. **Data Quality & Split Analysis**:
   - Automated quality analysis across **Train ($\le$2024)**, **Validation (2025)**, and **Test (2026)** splits.
   - Detects NaNs, physical temperature/dew point anomalies (ranges [-30°C to 50°C], [-40°C to 40°C], dew &le; temp), and statistical IQR outliers.
   - Generates comparative distribution histograms and statistical summaries.
4. **Configurable Learning Rate Decay Engine**:
   - **Performance-Based Decay Policies**: Dynamically decays learning rate as a function of validation learning performance (monitoring `val_loss` or `val_mae`). Applies to all machine learning (`HistGradientBoosting`, `Ridge`) and deep learning (`LSTM`, `CNN`, `Dense`, `Linear`) models in `forecast_system`.
   - **Policy Strategies**: Supports `"plateau"` (multiplicative factor), `"exponential_plateau"` (exponential scaling), and `"step_plateau"` (step subtraction).
   - **Extensive Logging & Diagnostics**: Logs detailed epoch/stage metrics, stagnant counts, learning rate adjustments, and exports structured `lr_decay_events` to JSON training histories for post-execution analysis.
5. **Resumable & Resetable Neural Network Training Engine**:
   - **Reset Mode (`--mode reset`)**: Erases all existing saved model checkpoints and training history, starting a clean training run from scratch.
   - **Resume Mode (`--mode resume`)**: Loads existing saved model weights (`lstm_checkpoint.keras`) and training history (`training_history.json`), continuing incremental training to further refine and improve model accuracy.
   - **Stopping Criteria**: Incremental epoch-block training (`epochs_per_iter`, default `5`) with target precision threshold (`target_mae < 1.0°C`).
   - **Fit Diagnostics**: Automated detection for **Overfitting** (halting early and restoring best weights) and **Underfitting** (triggering additional iteration passes).
6. **Separation of Static Web Assets & Pipeline Outputs**:
   - Web application source files (`index.html`, `style.css`, `script.js`) remain in `report/`.
   - Generated model checkpoints and JSON metrics are saved separately into `report/outputs/`.
7. **Dark-Mode Glassmorphic Tabbed Web Dashboard**:
   - **Tab 1 (📊 Model Evaluation & Live Forecast Engine)**: Out-of-Sample Leaderboard, KPI cards, sequence profiles, and interactive forecast generator.
   - **Tab 2 (🔍 Data Quality & Split Analysis)**: Split breakdown cards, bad values/anomaly summary, statistics table, and Plotly comparative histograms.
   - **Tab 3 (🔄 Training Evolution & Fit Diagnostics)**: Fit status banner, MSE loss evolution curve per epoch, and MAE accuracy convergence vs target threshold.
   - **Markdown Link (🖼️ Static Figures Gallery)**: Direct top-nav link to [`figures.md`](report/figures.md) containing all high-resolution static figure visualizations.

---

## 🚀 Quickstart & Pipeline Execution Guide (Running via `uv`)

All commands use the `uv` package manager. Because the project is configured as a `uv` package, **no manual `PYTHONPATH` configuration is required**.

### 1. Run Unit Tests

Execute unit tests covering data preprocessing, feature engineering, data quality analysis, baseline models, metrics, and reset/resume pipeline execution modes:

```bash
uv run python tests/test_forecast_system.py
```

---

### 2. Pipeline Execution (`-c` / `--config`)

The pipeline CLI accepts `--config` (or `-c`) as its sole command-line parameter. All pipeline settings (execution mode, iteration counts, target MAE, output directories) are configured directly in `config.yaml`.

```bash
uv run forecast-pipeline --config config.yaml
```

*Alternative explicit module invocation:*
```bash
uv run python -m forecast_system.pipeline -c config.yaml
```

#### Pipeline CLI Configuration Parameters:
| Option | Choices / Default | Description |
| :--- | :--- | :--- |
| `-c`, `--config` | `config.yaml` (default if omitted) | Path to YAML configuration file |

---

## 📈 Learning Rate Decay Policy & Post-Execution Logging

The system provides a configurable performance-based learning rate decay engine (`PerformanceLRDecayPolicy`) across all machine learning (`HistGradientBoosting`, `RidgeLinear`) and deep learning (`LSTM`, `CNN`, `Dense`, `Linear`) models in `forecast_system`.

### Configuration Options & Parameters

Learning rate decay behavior can be configured globally under the `optimization` block or customized for individual models in your YAML config file (`config.yaml`, `reset.yaml`, `resume.yaml`):

| Parameter | Options / Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lr_decay_enabled` | `boolean` | `true` | Enables or disables performance-based learning rate decay |
| `lr_decay_policy` | `"plateau"`, `"exponential_plateau"`, `"step_plateau"` | `"plateau"` | Policy strategy: multiplicative factor, exponential decay, or step reduction |
| `lr_decay_factor` | `float` (0.0 to 1.0) | `0.5` | Decay factor applied to learning rate upon performance plateau |
| `lr_decay_patience` | `integer` ($\ge 1$) | `2` | Number of epochs/iterations without improvement before decaying LR |
| `lr_min` | `float` | `0.000001` | Minimum lower bound floor for learning rate |
| `lr_decay_threshold` | `float` | `0.0001` | Minimum delta change in monitored metric to qualify as an improvement |
| `lr_cooldown` | `integer` ($\ge 0$) | `0` | Cooldown iterations after an LR reduction before resuming evaluation |

### YAML Configuration Example (`config.yaml`)

```yaml
optimization:
  target_mae: 1.0
  epochs_per_iter: 5
  max_iters: 3
  lr_decay_enabled: true
  lr_decay_policy: "plateau"   # Options: "plateau", "exponential_plateau", "step_plateau"
  lr_decay_factor: 0.5         # Halves the learning rate on plateau
  lr_decay_patience: 2         # Decays after 2 stagnant epochs/iterations
  lr_min: 0.000001             # Minimum LR limit
  lr_decay_threshold: 0.0001   # Minimum metric improvement threshold
  lr_cooldown: 0

lstm:
  enabled: true
  units: 64
  learning_rate: 0.001
  lr_decay_enabled: true
  lr_decay_policy: "plateau"
  lr_decay_factor: 0.5
  lr_decay_patience: 2

gbdt:
  enabled: true
  max_iter: 40
  learning_rate: 0.1
  lr_decay_enabled: true
  lr_decay_policy: "plateau"
  lr_decay_factor: 0.5
  lr_decay_patience: 2
```

### Extensive Logging & Post-Execution Analysis

The training engine outputs detailed diagnostic logs via standard Python `logging`:
1. **Console & Log Traces**: Logs monitored metrics per epoch/iteration pass. When performance plateaus, a prominent log entry is emitted:
   ```text
   🔻 [LR DECAY TRIGGERED] [LSTMModel] Epoch 4: Learning rate reduced from 1.000000e-03 to 5.000000e-04! Reason: Performance metric 'val_loss' failed to improve by >=0.0001 for 2 consecutive epochs. Policy: 'plateau'.
   ```
2. **JSON History Export**: Structured training history files (`training_history.json`, `cnn_training_history.json`, `dense_training_history.json`, `linear_training_history.json`) record:
   - Per-epoch active learning rates (`lr`).
   - Structured `lr_decay_events` log containing timestamps, model names, step indices, old LR, new LR, and trigger reasons for post-execution visual and quantitative analysis.

---

### 3. Open the Interactive Web Dashboard

Launch a local HTTP server using `uv` to view the interactive glassmorphic web dashboard:

```bash
uv run python -m http.server 8000 --directory report
```

Then open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🖥️ High-Resource / External Machine Testing Guide

If you are executing the system on a computer with higher hardware resources (16GB+ RAM, multi-core CPU, or GPU acceleration), follow this guide to perform full-scale, deep model training:

### Recommended Hardware Requirements:
- **RAM**: 16 GB or higher (to comfortably handle multi-year hourly feature matrices and Keras models without OS OOM halts).
- **CPU / GPU**: Multi-core CPU or NVIDIA CUDA GPU.

### Step-by-Step High-Resource Execution Workflow:

1. **Clone Repository & Install Dependencies**:
   ```bash
   git clone <repository_url>
   cd weather_forecast
   uv sync
   ```

2. **Run Unit Verification**:
   ```bash
   uv run python tests/test_forecast_system.py
   ```

3. **Step 3A: Clean Initial Training Run (Reset Mode)**:
   Set `pipeline.mode: "reset"` in `config.yaml` to start clean, then execute:
   ```bash
   uv run forecast-pipeline --config config.yaml
   ```

4. **Step 3B: Continuous Improvement Run (Resume Mode)**:
   Set `pipeline.mode: "resume"` in `config.yaml` to continue training saved checkpoints, then execute:
   ```bash
   uv run forecast-pipeline --config config.yaml
   ```

5. **Resource Control & Memory Best Practices**:
   - On RAM-constrained machines, keep `epochs_per_iter` between `3` and `5` and `max_iters` between `2` and `4` in `config.yaml`.
   - On high-capacity machines with dedicated GPUs, TensorFlow will automatically leverage CUDA acceleration.

---

## 📁 Repository Structure

```
weather_forecast/
├── data/                             # Raw weather datasets (CSV)
│   ├── CRV2025.csv                   # 2025 detailed hourly weather data
│   ├── CRV2026.csv                   # 2026 detailed hourly weather data
│   ├── typ_daily_CR.csv              # Historical daily weather summary
│   └── tytd_hourly_CR.csv            # Historical multi-year hourly dataset
├── forecast_system/                  # Core Forecasting Package
│   ├── data/
│   │   ├── loader.py                 # Multi-file dataset loader & temporal splitter
│   │   ├── preprocessor.py           # Temperature string parser & gap interpolation
│   │   ├── feature_engineering.py    # Lag features, rolling stats, cyclical encodings
│   │   └── quality_analysis.py       # Data split stats, bad values & histogram generator
│   ├── models/
│   │   ├── baselines.py              # Persistence & Climatology baseline models
│   │   ├── ml_models.py              # HistGradientBoosting & Ridge regressors
│   │   └── deep_learning.py          # Keras Multi-layer / LSTM Resumable & Resetable Model
│   ├── evaluation/
│   │   ├── metrics.py                # MAE, RMSE, Tolerance Accuracy, Directional F1
│   │   └── evaluator.py              # Test set benchmarking & JSON exporter
│   └── pipeline.py                   # Main end-to-end execution pipeline with --mode flag
├── report/                           # Interactive Web Dashboard
│   ├── index.html                    # Glassmorphic dashboard UI with Tab Navigation
│   ├── style.css                     # Dark-mode glassmorphism & tab styling
│   ├── script.js                     # Plotly chart logic, tab switcher & JSON loaders
│   └── outputs/                      # Generated Pipeline Output Artifacts
│       ├── data_properties.json      # Dataset split properties, stats, and histograms
│       ├── training_history.json     # Per-epoch loss/accuracy evolution & fit status
│       ├── lstm_checkpoint.keras     # Saved deep learning model checkpoint
│       ├── model_evaluation_metrics.json # Generated out-of-sample test metrics
│       └── test_predictions.json     # Generated test prediction time series
├── tests/
│   └── test_forecast_system.py       # Comprehensive unit tests (includes reset/resume tests)
└── pyproject.toml                    # UV build system & dependency configuration
```

---

## 📊 Evaluation Metrics Summary

| Metric | Description |
| :--- | :--- |
| **MAE (°C)** | Mean Absolute Error in degrees Celsius |
| **RMSE (°C)** | Root Mean Squared Error |
| **Acc (±1.0°C) %** | Percentage of forecasts within $1.0^\circ\text{C}$ of actual temperature |
| **Acc (±2.0°C) %** | Percentage of forecasts within $2.0^\circ\text{C}$ of actual temperature |
| **Directional F1 %** | Precision/Recall F1 score for predicting temperature trend direction over 24 hours |
| **$R^2$ Score** | Coefficient of determination |
