import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def generate_all_visualizations(outputs_dir="report/outputs"):
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    # 1. Load History Files
    all_histories_path = outputs_path / "all_training_histories.json"
    all_histories = {}
    if all_histories_path.exists():
        with open(all_histories_path, 'r') as f:
            all_histories = json.load(f)
    else:
        history_files = {
            "Deep Learning (LSTM)": outputs_path / "training_history.json",
            "Convolutional Neural Network (CNN)": outputs_path / "cnn_training_history.json",
            "Dense Neural Network": outputs_path / "dense_training_history.json",
            "Linear Neural Network": outputs_path / "linear_training_history.json",
            "Gradient Boosted Trees": outputs_path / "gbdt_training_history.json",
            "Ridge Regression": outputs_path / "ridge_training_history.json",
        }
        for name, p in history_files.items():
            if p.exists():
                with open(p, 'r') as f:
                    all_histories[name] = json.load(f)

    # Set dark/sleek matplotlib theme for figures
    plt.style.use('dark_background')
    fig_color = '#0f172a'
    ax_color = '#1e293b'
    text_color = '#f8fafc'
    grid_color = '#334155'

    colors = {
        "Deep Learning (LSTM)": "#38bdf8",
        "Convolutional Neural Network (CNN)": "#a855f7",
        "Dense Neural Network": "#f43f5e",
        "Linear Neural Network": "#fbbf24",
        "Gradient Boosted Trees": "#34d399",
        "Ridge Regression": "#c084fc",
        "Persistence Baseline": "#94a3b8",
        "Climatology Baseline": "#64748b"
    }

    file_slugs = {
        "Deep Learning (LSTM)": "lstm",
        "Convolutional Neural Network (CNN)": "cnn",
        "Dense Neural Network": "dense",
        "Linear Neural Network": "linear",
        "Gradient Boosted Trees": "gbdt",
        "Ridge Regression": "ridge",
        "Persistence Baseline": "persistence",
        "Climatology Baseline": "climatology"
    }

    # =========================================================================
    # 1. OVERALL COMBINED TRAINING EVOLUTION FIGURE
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), facecolor=fig_color)
    fig.suptitle("Training Evolution Across All Models\n(Loss, Accuracy MAE, Learning Rate)", color=text_color, fontsize=16, fontweight='bold', y=0.98)

    # Subplot 1: Loss Evolution (MSE)
    ax1 = axes[0]
    ax1.set_facecolor(ax_color)
    ax1.set_title("1. Loss Evolution (Validation MSE)", color=text_color, fontsize=12, fontweight='bold', pad=10)

    for model_name, hist in all_histories.items():
        epochs_data = hist.get('epochs', [])
        if epochs_data and 'val_loss' in epochs_data[0]:
            epochs = [e['epoch'] for e in epochs_data]
            val_loss = [e['val_loss'] for e in epochs_data]
            color = colors.get(model_name, '#38bdf8')
            ax1.plot(epochs, val_loss, label=f"{model_name} (Val)", color=color, linewidth=2, marker='o', markersize=4)

    ax1.set_xlabel("Epoch / Stage", color=text_color)
    ax1.set_ylabel("Validation MSE Loss", color=text_color)
    ax1.grid(True, color=grid_color, linestyle='--', alpha=0.5)
    ax1.tick_params(colors=text_color)
    ax1.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color)

    # Subplot 2: Accuracy Evolution (Val MAE °C vs Target)
    ax2 = axes[1]
    ax2.set_facecolor(ax_color)
    ax2.set_title("2. Accuracy Evolution (Validation MAE °C vs Target Threshold 1.0°C)", color=text_color, fontsize=12, fontweight='bold', pad=10)

    max_steps = 0
    for model_name, hist in all_histories.items():
        color = colors.get(model_name, '#38bdf8')
        if 'epochs' in hist and hist['epochs']:
            steps = [e['epoch'] for e in hist['epochs']]
            val_mae = [e['val_mae'] for e in hist['epochs']]
            max_steps = max(max_steps, max(steps))
            ax2.plot(steps, val_mae, label=model_name, color=color, linewidth=2, marker='s', markersize=4)
        elif 'stages' in hist and hist['stages']:
            steps = [s['stage'] for s in hist['stages']]
            val_mae = [s['val_mae'] for s in hist['stages']]
            max_steps = max(max_steps, max(steps))
            ax2.plot(steps, val_mae, label=model_name, color=color, linewidth=2, linestyle='--', marker='^', markersize=5)

    if max_steps > 0:
        ax2.axhline(1.0, color='#f43f5e', linestyle=':', linewidth=2, label='Target MAE Threshold (1.0°C)')

    ax2.set_xlabel("Epoch / Stage Step", color=text_color)
    ax2.set_ylabel("Validation MAE (°C)", color=text_color)
    ax2.grid(True, color=grid_color, linestyle='--', alpha=0.5)
    ax2.tick_params(colors=text_color)
    ax2.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color)

    # Subplot 3: Learning Rate Evolution
    ax3 = axes[2]
    ax3.set_facecolor(ax_color)
    ax3.set_title("3. Performance-Based Learning Rate Decay Evolution", color=text_color, fontsize=12, fontweight='bold', pad=10)

    for model_name, hist in all_histories.items():
        color = colors.get(model_name, '#38bdf8')
        if 'epochs' in hist and hist['epochs'] and 'lr' in hist['epochs'][0]:
            steps = [e['epoch'] for e in hist['epochs']]
            lrs = [e['lr'] for e in hist['epochs']]
            ax3.plot(steps, lrs, label=model_name, color=color, linewidth=2, marker='o', markersize=4)
        elif 'stages' in hist and hist['stages'] and 'learning_rate' in hist['stages'][0]:
            steps = [s['stage'] for s in hist['stages']]
            lrs = [s['learning_rate'] for s in hist['stages']]
            ax3.plot(steps, lrs, label=model_name, color=color, linewidth=2, linestyle='--', marker='d', markersize=5)

    ax3.set_yscale('log')
    ax3.set_xlabel("Epoch / Stage Step", color=text_color)
    ax3.set_ylabel("Active Learning Rate (Log Scale)", color=text_color)
    ax3.grid(True, color=grid_color, linestyle='--', alpha=0.5)
    ax3.tick_params(colors=text_color)
    ax3.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_hist_path = outputs_path / "training_evolution_all_models.png"
    plt.savefig(fig_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved overall training evolution visualization to {fig_hist_path}")

    # =========================================================================
    # 2. SEPARATE TRAINING EVOLUTION FIGURES FOR EACH INDIVIDUAL MODEL
    # =========================================================================
    for model_name, hist in all_histories.items():
        slug = file_slugs.get(model_name, model_name.lower().replace(' ', '_'))
        color = colors.get(model_name, '#38bdf8')

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), facecolor=fig_color)
        fig.suptitle(f"Training Evolution: {model_name}\n(Loss, Accuracy, Learning Rate Decay)", color=text_color, fontsize=14, fontweight='bold', y=0.98)

        # 1. Loss
        ax_l = axes[0]
        ax_l.set_facecolor(ax_color)
        ax_l.set_title("1. Loss Evolution (MSE)", color=text_color, fontsize=11, fontweight='bold')
        if 'epochs' in hist and hist['epochs'] and 'train_loss' in hist['epochs'][0]:
            epochs = [e['epoch'] for e in hist['epochs']]
            tr_l = [e['train_loss'] for e in hist['epochs']]
            va_l = [e['val_loss'] for e in hist['epochs']]
            ax_l.plot(epochs, tr_l, label="Train Loss (MSE)", color='#38bdf8', linewidth=2, marker='o')
            ax_l.plot(epochs, va_l, label="Val Loss (MSE)", color='#a855f7', linewidth=2, marker='s')
            ax_l.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color)
        else:
            ax_l.text(0.5, 0.5, "Closed-form / Stage-wise Fit (No Epoch MSE Loss)", color=text_color, ha='center', va='center')

        ax_l.set_xlabel("Epoch Step", color=text_color)
        ax_l.set_ylabel("MSE Loss", color=text_color)
        ax_l.grid(True, color=grid_color, linestyle='--', alpha=0.5)
        ax_l.tick_params(colors=text_color)

        # 2. Accuracy
        ax_a = axes[1]
        ax_a.set_facecolor(ax_color)
        ax_a.set_title("2. Accuracy Evolution (MAE °C vs Target Threshold 1.0°C)", color=text_color, fontsize=11, fontweight='bold')
        if 'epochs' in hist and hist['epochs']:
            steps = [e['epoch'] for e in hist['epochs']]
            tr_m = [e.get('train_mae', 0) for e in hist['epochs']]
            va_m = [e['val_mae'] for e in hist['epochs']]
            if any(tr_m):
                ax_a.plot(steps, tr_m, label="Train MAE (°C)", color='#34d399', linewidth=2, marker='o')
            ax_a.plot(steps, va_m, label="Val MAE (°C)", color='#fbbf24', linewidth=2, marker='s')
            ax_a.axhline(1.0, color='#f43f5e', linestyle=':', linewidth=2, label="Target MAE Threshold (1.0°C)")
            ax_a.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color)
        elif 'stages' in hist and hist['stages']:
            steps = [s['stage'] for s in hist['stages']]
            va_m = [s['val_mae'] for s in hist['stages']]
            ax_a.plot(steps, va_m, label="Stage Val MAE (°C)", color='#fbbf24', linewidth=2, marker='^')
            ax_a.axhline(1.0, color='#f43f5e', linestyle=':', linewidth=2, label="Target MAE Threshold (1.0°C)")
            ax_a.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color)

        ax_a.set_xlabel("Epoch / Stage Step", color=text_color)
        ax_a.set_ylabel("MAE (°C)", color=text_color)
        ax_a.grid(True, color=grid_color, linestyle='--', alpha=0.5)
        ax_a.tick_params(colors=text_color)

        # 3. Learning Rate Decay
        ax_lr = axes[2]
        ax_lr.set_facecolor(ax_color)
        ax_lr.set_title("3. Performance-Based Learning Rate Decay", color=text_color, fontsize=11, fontweight='bold')
        if 'epochs' in hist and hist['epochs'] and 'lr' in hist['epochs'][0]:
            steps = [e['epoch'] for e in hist['epochs']]
            lrs = [e['lr'] for e in hist['epochs']]
            ax_lr.plot(steps, lrs, label="Active LR", color=color, linewidth=2.5, marker='d')
            ax_lr.set_yscale('log')
            ax_lr.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color)
        elif 'stages' in hist and hist['stages'] and 'learning_rate' in hist['stages'][0]:
            steps = [s['stage'] for s in hist['stages']]
            lrs = [s['learning_rate'] for s in hist['stages']]
            ax_lr.plot(steps, lrs, label="Stage Active LR", color=color, linewidth=2.5, marker='d')
            ax_lr.set_yscale('log')
            ax_lr.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color)
        else:
            ax_lr.text(0.5, 0.5, f"Constant LR: {hist.get('final_learning_rate', 'N/A')}", color=text_color, ha='center', va='center')

        ax_lr.set_xlabel("Epoch / Stage Step", color=text_color)
        ax_lr.set_ylabel("Learning Rate", color=text_color)
        ax_lr.grid(True, color=grid_color, linestyle='--', alpha=0.5)
        ax_lr.tick_params(colors=text_color)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        model_fig_path = outputs_path / f"{slug}_training_evolution.png"
        plt.savefig(model_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved separate training evolution figure to {model_fig_path}")

    # =========================================================================
    # 3. OVERALL COMBINED PREDICTIONS FIGURE
    # =========================================================================
    pred_path = outputs_path / "test_predictions.json"
    if not pred_path.exists():
        print("Note: test_predictions.json not found for prediction plots.")
        return

    with open(pred_path, 'r') as f:
        pred_data = json.load(f)

    fig, axes = plt.subplots(3, 1, figsize=(14, 15), facecolor=fig_color)
    fig.suptitle("Out-of-Sample Test Predictions Across All Models (2026 Test Set)", color=text_color, fontsize=16, fontweight='bold', y=0.98)

    timestamps = pred_data.get('timestamps', [])
    x_axis = range(len(timestamps)) if len(timestamps) > 0 else range(100)

    # Subplot 1: Next-24h TMAX Predictions (All Models vs Actual)
    ax_tmax = axes[0]
    ax_tmax.set_facecolor(ax_color)
    ax_tmax.set_title("1. Next-Day TMAX Forecast Comparison (All Models)", color=text_color, fontsize=12, fontweight='bold', pad=10)

    if 'actual_next24_tmax' in pred_data:
        ax_tmax.plot(x_axis, pred_data['actual_next24_tmax'], label="Actual TMAX Ground Truth", color="#ffffff", linewidth=3, linestyle='-')

    predictions = pred_data.get('predictions', {})
    for model_name, pdict in predictions.items():
        if 'tmax' in pdict:
            color = colors.get(model_name, '#38bdf8')
            linestyle = '--' if 'Baseline' in model_name else '-'
            alpha = 0.6 if 'Baseline' in model_name else 0.85
            ax_tmax.plot(x_axis, pdict['tmax'], label=model_name, color=color, linewidth=1.5, linestyle=linestyle, alpha=alpha)

    ax_tmax.set_ylabel("Temperature (°C)", color=text_color)
    ax_tmax.grid(True, color=grid_color, linestyle='--', alpha=0.5)
    ax_tmax.tick_params(colors=text_color)
    ax_tmax.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color, loc='upper right', fontsize=8, ncol=2)

    # Subplot 2: Next-24h TMIN Predictions (All Models vs Actual)
    ax_tmin = axes[1]
    ax_tmin.set_facecolor(ax_color)
    ax_tmin.set_title("2. Next-Day TMIN Forecast Comparison (All Models)", color=text_color, fontsize=12, fontweight='bold', pad=10)

    if 'actual_next24_tmin' in pred_data:
        ax_tmin.plot(x_axis, pred_data['actual_next24_tmin'], label="Actual TMIN Ground Truth", color="#ffffff", linewidth=3, linestyle='-')

    for model_name, pdict in predictions.items():
        if 'tmin' in pdict:
            color = colors.get(model_name, '#38bdf8')
            linestyle = '--' if 'Baseline' in model_name else '-'
            alpha = 0.6 if 'Baseline' in model_name else 0.85
            ax_tmin.plot(x_axis, pdict['tmin'], label=model_name, color=color, linewidth=1.5, linestyle=linestyle, alpha=alpha)

    ax_tmin.set_ylabel("Temperature (°C)", color=text_color)
    ax_tmin.grid(True, color=grid_color, linestyle='--', alpha=0.5)
    ax_tmin.tick_params(colors=text_color)
    ax_tmin.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color, loc='upper right', fontsize=8, ncol=2)

    # Subplot 3: 24-Hour Sequence Forecast Profiles (All Models vs Actual)
    ax_24h = axes[2]
    ax_24h.set_facecolor(ax_color)
    last_ts = pred_data.get('last_timestamp', 'Latest Test Window')
    ax_24h.set_title(f"3. 24-Hour Forecast Sequence Profile vs Actual ({last_ts})", color=text_color, fontsize=12, fontweight='bold', pad=10)

    hours = range(1, 25)
    if 'actual_last_24h_profile' in pred_data:
        ax_24h.plot(hours, pred_data['actual_last_24h_profile'], label="Actual Observed 24h Profile", color="#34d399", linewidth=3.5, marker='o')

    for model_name, pdict in predictions.items():
        if 'sample_24h_profile' in pdict:
            color = colors.get(model_name, '#38bdf8')
            linestyle = '--' if 'Baseline' in model_name else '-'
            ax_24h.plot(hours, pdict['sample_24h_profile'], label=f"{model_name} Forecast", color=color, linewidth=1.8, linestyle=linestyle, marker='s', markersize=3)

    ax_24h.set_xlabel("Forecast Ahead Horizon (Hours)", color=text_color)
    ax_24h.set_ylabel("Temperature (°C)", color=text_color)
    ax_24h.set_xticks(hours)
    ax_24h.set_xticklabels([f"+{h}h" for h in hours], fontsize=8, color=text_color)
    ax_24h.grid(True, color=grid_color, linestyle='--', alpha=0.5)
    ax_24h.tick_params(colors=text_color)
    ax_24h.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color, loc='lower right', fontsize=8, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_pred_path = outputs_path / "predictions_all_models.png"
    plt.savefig(fig_pred_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved overall predictions visualization to {fig_pred_path}")

    # =========================================================================
    # 4. SEPARATE PREDICTION FIGURES FOR EACH INDIVIDUAL MODEL
    # =========================================================================
    actual_tmax = np.array(pred_data.get('actual_next24_tmax', []))
    actual_tmin = np.array(pred_data.get('actual_next24_tmin', []))

    for model_name, pdict in predictions.items():
        slug = file_slugs.get(model_name, model_name.lower().replace(' ', '_'))
        color = colors.get(model_name, '#38bdf8')

        fig, axes = plt.subplots(3, 1, figsize=(12, 13), facecolor=fig_color)
        fig.suptitle(f"Model Predictions: {model_name}\n(Out-of-Sample Test Set 2026)", color=text_color, fontsize=14, fontweight='bold', y=0.98)

        # 1. TMAX & TMIN Predictions vs Actual
        ax1 = axes[0]
        ax1.set_facecolor(ax_color)
        ax1.set_title("1. Out-of-Sample Daily TMAX & TMIN Forecasts", color=text_color, fontsize=11, fontweight='bold')
        if len(actual_tmax) > 0:
            ax1.plot(x_axis, actual_tmax, label="Actual TMAX", color='#f43f5e', linewidth=2)
            ax1.plot(x_axis, actual_tmin, label="Actual TMIN", color='#34d399', linewidth=2)
        if 'tmax' in pdict:
            ax1.plot(x_axis, pdict['tmax'], label=f"{model_name} TMAX", color=color, linewidth=2, linestyle='--')
        if 'tmin' in pdict:
            ax1.plot(x_axis, pdict['tmin'], label=f"{model_name} TMIN", color='#fbbf24', linewidth=2, linestyle='--')

        ax1.set_ylabel("Temperature (°C)", color=text_color)
        ax1.grid(True, color=grid_color, linestyle='--', alpha=0.5)
        ax1.tick_params(colors=text_color)
        ax1.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color, loc='upper right')

        # 2. 24-Hour Sequence Profile
        ax2 = axes[1]
        ax2.set_facecolor(ax_color)
        ax2.set_title(f"2. 24-Hour Sequence Forecast Profile vs Actual ({last_ts})", color=text_color, fontsize=11, fontweight='bold')
        if 'actual_last_24h_profile' in pred_data:
            ax2.plot(hours, pred_data['actual_last_24h_profile'], label="Actual Observed 24h Profile", color="#34d399", linewidth=3, marker='o')
        if 'sample_24h_profile' in pdict:
            ax2.plot(hours, pdict['sample_24h_profile'], label=f"{model_name} Forecast Profile", color=color, linewidth=2.5, linestyle='--', marker='s')

        ax2.set_xlabel("Forecast Ahead Horizon (Hours)", color=text_color)
        ax2.set_ylabel("Temperature (°C)", color=text_color)
        ax2.set_xticks(hours)
        ax2.set_xticklabels([f"+{h}h" for h in hours], fontsize=8, color=text_color)
        ax2.grid(True, color=grid_color, linestyle='--', alpha=0.5)
        ax2.tick_params(colors=text_color)
        ax2.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color)

        # 3. Residual Error Distribution (Histogram)
        ax3 = axes[2]
        ax3.set_facecolor(ax_color)
        ax3.set_title("3. Out-of-Sample Prediction Residual Error Distribution (y_pred - y_actual)", color=text_color, fontsize=11, fontweight='bold')

        if 'tmax' in pdict and len(actual_tmax) == len(pdict['tmax']):
            residuals = np.array(pdict['tmax']) - actual_tmax
            ax3.hist(residuals, bins=25, color=color, alpha=0.7, edgecolor=grid_color, label=f"Residual Error (Mean={residuals.mean():.2f}°C, Std={residuals.std():.2f}°C)")
            ax3.axvline(0, color='#ffffff', linestyle='--', linewidth=1.5, label="Zero Bias Line")
            ax3.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color)
        else:
            ax3.text(0.5, 0.5, "No residual data available", color=text_color, ha='center', va='center')

        ax3.set_xlabel("Prediction Error (°C)", color=text_color)
        ax3.set_ylabel("Frequency Count", color=text_color)
        ax3.grid(True, color=grid_color, linestyle='--', alpha=0.5)
        ax3.tick_params(colors=text_color)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        model_pred_path = outputs_path / f"{slug}_predictions.png"
        plt.savefig(model_pred_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved separate predictions figure to {model_pred_path}")

if __name__ == "__main__":
    generate_all_visualizations()
