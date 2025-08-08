import os
import hopsworks
import pandas as pd
import joblib

from aqi_preprocessor import AQIDataPreprocessor
from forecasting_system import AQIForecastingSystem

def get_hopsworks_project():
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT"]
    )
    return project

def filter_numeric_metrics(metrics_dict):
    return {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}

def run_initial_training_for_horizon(horizon):
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    horizon_configs = {
        24: {'max_features': 10},
        48: {'max_features': 15},
        72: {'max_features': 8}
    }

    if horizon not in horizon_configs:
        print(f"No configuration found for {horizon}h horizon.")
        return

    config = horizon_configs[horizon]
    print(f"\n--- Starting Initial Training for {horizon}h Horizon ---")
    print(f"   Configuration: {config}")

    try:
        fg = fs.get_feature_group(name=f"aqi_features_{horizon}h_prod", version=1)
        raw_df = fg.read()
        print(f"Raw data shape from feature group: {raw_df.shape}")
    except Exception as e:
        print(f"Could not read data from feature group: {e}")
        return

    preprocessor = AQIDataPreprocessor(dataframe=raw_df)
    success = preprocessor.run_full_preprocessing(
        correlation_threshold=0.85,
        variance_threshold=1e-6,
        vif_threshold=8,
        dataframe=raw_df,
        forecast_horizon=horizon
    )

    if not success:
        print(f"Preprocessing failed for {horizon}h.")
        return

    processed_data = preprocessor.get_processed_data()
    if not processed_data or 'full_data' not in processed_data:
        print(f"Processed data missing for {horizon}h.")
        return

    final_df = processed_data['full_data']
    print(f"Processed data shape for {horizon}h: {final_df.shape}")

    forecasting_system = AQIForecastingSystem(
        horizon_df=final_df,
        target_column=f'aqi_{horizon}h',
        horizon_hours=horizon,
        model_save_path=f"./aqi_models_{horizon}h"
    )

    results = forecasting_system.run_complete_pipeline(
        max_features=config['max_features'] )

    if not results:
        print(f"Modeling failed for {horizon}h. Skipping registration.")
        return

    eval_results = results['evaluation_results']
    best_model_name = None
    best_model_metrics = None
    best_r2 = -float('inf')

    for model_name, metrics in eval_results.items():
        r2 = metrics.get("avg_test_r2", -float('inf'))
        if r2 > best_r2:
            best_model_name = model_name
            best_model_metrics = metrics
            best_r2 = r2

    if not best_model_name:
        print(f"No valid model found for registration.")
        return

    print(f"Best model for {horizon}h: {best_model_name} with R² = {best_r2:.3f}")

    def filter_numeric_metrics(metrics_dict):
        return {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}

    combined_metrics = {
        **best_model_metrics,
        "train_size": config['train_size'],
        "test_size": config['test_size'],
        "step_size": config['step_size']
    }

    cleaned_metrics = filter_numeric_metrics(combined_metrics)
    model_description = f"The best model for {horizon}h is {best_model_name}"

    model = mr.python.create_model(
        name=f"aqi_forecast_model_{horizon}h",
        description=model_description,
        metrics=cleaned_metrics
    )

    model.save(results['model_save_path'])
    print(f"Model registered: {model.name} (version {model.version})")


if __name__ == "__main__":
    for horizon in [24, 48, 72]:
        run_initial_training_for_horizon(horizon)
