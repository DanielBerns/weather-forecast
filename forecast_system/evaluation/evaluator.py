import json
from pathlib import Path
import pandas as pd
from .metrics import evaluate_model_performance

class ForecastEvaluator:
    """
    Evaluates multiple forecasting models on out-of-sample test data.
    """
    def __init__(self, models_dict):
        self.models_dict = models_dict
        self.results = {}

    def evaluate(self, X_test, Y_hourly_test, Y_summary_test):
        """
        Runs predictions for all models and calculates performance metrics.
        """
        x_current_temp = X_test['temp_lag0'].values

        print("\n========================================================")
        print("EVALUATING MODELS ON OUT-OF-SAMPLE TEST SET (CRV2026)")
        print("========================================================\n")

        for name, model in self.models_dict.items():
            print(f"Evaluating {name}...")
            pred_hourly, pred_summary = model.predict(X_test)

            perf = evaluate_model_performance(
                model_name=name,
                y_true_hourly=Y_hourly_test.values,
                y_pred_hourly=pred_hourly,
                y_true_summary=Y_summary_test,
                y_pred_summary=pred_summary,
                x_current_temp=x_current_temp
            )

            self.results[name] = {
                'metrics': perf,
                'predictions_hourly': pred_hourly,
                'predictions_summary': pred_summary
            }

            h = perf['hourly_24h']
            print(f"[{name}] MAE: {h['mae']}°C | RMSE: {h['rmse']}°C | Acc(±1°C): {h['acc_1.0C']}% | Acc(±2°C): {h['acc_2.0C']}% | Dir F1: {h['directional_f1']}% | R2: {h['r2']}")

        return self.results

    def save_summary_json(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary_data = {name: data['metrics'] for name, data in self.results.items()}
        with open(output_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"\nSaved evaluation metrics JSON to {output_path}")

    def to_dataframe(self):
        records = []
        for name, data in self.results.items():
            m = data['metrics']['hourly_24h']
            records.append({
                'Model': name,
                'MAE (°C)': m['mae'],
                'RMSE (°C)': m['rmse'],
                'Acc (±1.0°C) %': m['acc_1.0C'],
                'Acc (±2.0°C) %': m['acc_2.0C'],
                'Directional F1 %': m['directional_f1'],
                'R² Score': m['r2']
            })
        return pd.DataFrame(records)
