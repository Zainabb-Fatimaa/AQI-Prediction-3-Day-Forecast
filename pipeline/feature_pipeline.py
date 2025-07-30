import os
import hopsworks
import pandas as pd
<<<<<<< HEAD
import joblib

from aqi_preprocessor import AQIDataPreprocessor
from forecasting_system import AQIForecastingSystem

Optional: for local development/testing without Hopsworks
from local_data import existing_df  
=======
from aqi_preprocessor import AQIDataPreprocessor 
>>>>>>> c5ced96a4c2a4a817202b9e66a3d2c613f0bdf74

def get_hopsworks_project():
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT"]
    )
    return project

<<<<<<< HEAD
def filter_numeric_metrics(metrics_dict):
    """Returns a dictionary with only numeric (int/float) values."""
    return {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}
=======
def get_or_create_feature_group(fs, horizon):
    """Gets or creates a single feature group in Hopsworks."""
    fg_name = f"aqi_features_{horizon}h_prod"
    try:
        print(f"Getting feature group '{fg_name}'...")
        fg = fs.get_feature_group(name=fg_name, version=1)
        print(f"✅ Feature group '{fg_name}' found.")
        return fg
    except:
        print(f"Creating feature group: {fg_name}")
        fg = fs.get_or_create_feature_group(
            name=fg_name,
            version=1,
            description=f"Preprocessed AQI features for {horizon}h forecast",
            primary_key=["unique_id"],
            event_time="event_time",
            online_enabled=True,
        )
        return fg
>>>>>>> c5ced96a4c2a4a817202b9e66a3d2c613f0bdf74

def run_initial_training_for_horizon(horizon):
    """
    Runs preprocessing, training, and model registration for a given AQI forecast horizon.
    """
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

<<<<<<< HEAD
    horizon_configs = {
        24: {'max_features': 17, 'train_size': 6*7*24, 'test_size': 3*7*24, 'step_size': 5*24},
        48: {'max_features': 17, 'train_size': 6*7*24, 'test_size': 3*7*24, 'step_size': 5*24},
        72: {'max_features': 11, 'train_size': 6*7*24, 'test_size': 3*7*24, 'step_size': 5*24}
    }

    if horizon not in horizon_configs:
        print(f"❌ No configuration found for {horizon}h horizon.")
        return

    config = horizon_configs[horizon]
    print(f"\n--- Starting Initial Training for {horizon}h Horizon ---")
    print(f"   Configuration: {config}")

    # 1. Get Raw Data from Feature Store
    try:
        fg = fs.get_feature_group(name=f"aqi_features_{horizon}h_prod", version=1)
        raw_df = fg.read()
        print(f"✅ Raw data shape from feature group: {raw_df.shape}")
    except Exception as e:
        print(f"❌ Could not read data from feature group: {e}")
        return

    # 2. Preprocess Data
    preprocessor = AQIDataPreprocessor(dataframe=raw_df)
    success = preprocessor.run_full_preprocessing(
        correlation_threshold=0.85,
        variance_threshold=1e-6,
        vif_threshold=8,
        dataframe=raw_df,
        forecast_horizon=horizon
    )

    if not success:
        print(f"❌ Preprocessing failed for {horizon}h.")
        return

    processed_data = preprocessor.get_processed_data()
    if not processed_data or 'full_data' not in processed_data:
        print(f"❌ Processed data missing for {horizon}h.")
        return

    final_df = processed_data['full_data']
    print(f"✅ Processed data shape for {horizon}h: {final_df.shape}")

    # 3. Run Modeling Pipeline
    forecasting_system = AQIForecastingSystem(
        horizon_df=final_df,
        target_column=f'aqi_{horizon}h',
        horizon_hours=horizon,
        model_save_path=f"./aqi_models_{horizon}h"
    )

    results = forecasting_system.run_pipeline(
        max_features=config['max_features'],
        train_size=config['train_size'],
        test_size=config['test_size'],
        step_size=config['step_size']
    )

    if not results:
        print(f"❌ Modeling failed for {horizon}h. Skipping registration.")
        return

    # 4. Select Best Model
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
        print(f"❌ No valid model found for registration.")
        return

    print(f"🏆 Best model for {horizon}h: {best_model_name} with R² = {best_r2:.3f}")

    # 5. Clean and Add Metrics
    def filter_numeric_metrics(metrics_dict):
        return {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}

    combined_metrics = {
        **best_model_metrics,
        "train_size": config['train_size'],
        "test_size": config['test_size'],
        "step_size": config['step_size']
    }

    cleaned_metrics = filter_numeric_metrics(combined_metrics)

    # 6. Register the Model
    model = mr.python.create_model(
        name=f"aqi_forecast_model_{horizon}h",
        description=f"AQI forecast model for {horizon}-hour horizon.",
        metrics=cleaned_metrics
    )

    model.save(results['model_save_path'])
    print(f"✅ Model registered: {model.name} (version {model.version})")


if __name__ == "__main__":
    for horizon in [24, 48, 72]:
        run_initial_training_for_horizon(horizon)
=======
    fg_old = fs.get_feature_group(name="karachi_raw_data_store", version=1)
    raw_df = fg_old.read()

    for horizon in [24, 48, 72]:
        print(f"\n--- Processing for {horizon}h horizon ---")

        # Get or create the feature group for the current horizon
        fg = get_or_create_feature_group(fs, horizon)

        preprocessor = AQIDataPreprocessor(dataframe=raw_df)
        success = preprocessor.run_full_preprocessing(
            dataframe=raw_df,
            forecast_horizon=horizon
        )

        if success:
            processed_data = preprocessor.get_processed_data()
            if processed_data and 'full_data' in processed_data:
                features_df = processed_data['full_data'].copy()
                features_df["event_time"] = pd.to_datetime(features_df.index)
                features_df['unique_id'] = range(len(features_df))

                fg.insert(features_df, write_options={"wait_for_job": True})
                print(f"✅ Inserted {len(features_df)} rows into '{fg.name}'")

if __name__ == "__main__":
    run_feature_pipeline()
>>>>>>> c5ced96a4c2a4a817202b9e66a3d2c613f0bdf74
