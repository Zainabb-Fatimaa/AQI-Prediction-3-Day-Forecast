# scripts/download_models.py
import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import hopsworks

def ensure_directories():
    """Create necessary directories"""
    directories = [
        'backend/models',
        'backend/data',
        'backend/config'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_latest_model_version(mr, model_name):
    """Get the latest version of a model, preferring production-tagged versions."""
    try:
        # First try to get production version
        try:
            model_info = mr.get_model(name=model_name, tag="production")
            print(f"Using production-tagged version {model_info.version} for {model_name}")
            return model_info
        except:
            pass

        # If no production tag, try "latest" tag
        try:
            model_info = mr.get_model(name=model_name, tag="latest")
            print(f"Using latest-tagged version {model_info.version} for {model_name}")
            return model_info
        except:
            pass

        # If no tags, get all versions and pick the highest
        models_list = mr.get_models(name=model_name)
        if not models_list:
            raise Exception(f"No models found with name {model_name}")

        # Sort by version number (descending) and get the first one
        latest_model = max(models_list, key=lambda x: x.version)
        print(f"Using highest version {latest_model.version} for {model_name}")
        return latest_model

    except Exception as e:
        print(f"Error getting latest model version for {model_name}: {e}")
        return None

def parse_best_model_from_description(description):
    """Parse the best model name from description."""
    if not description:
        return None

    pattern = r"The best model for \d+h is ([A-Za-z]+)"
    match = re.search(pattern, description, re.IGNORECASE)
    if match:
        return match.group(1)

    print(f"Could not parse model name from description: {description}")
    return None

def find_model_file(model_dir, best_model_name, horizon):
    """Find the correct model file path."""
    possible_paths = [
        os.path.join(model_dir, "individual_models", best_model_name, f"{best_model_name}_{horizon}h.pkl"),
        os.path.join(model_dir, "individual_models", best_model_name, f"{best_model_name}_{horizon}h.cbm"),
        os.path.join(model_dir, "individual_models", best_model_name, f"{best_model_name}_{horizon}h.json"),
        os.path.join(model_dir, "individual_models", best_model_name, f"{best_model_name}_{horizon}h.joblib"),
        os.path.join(model_dir, "individual_models", best_model_name, f"{best_model_name}_{horizon}h.txt"),
        os.path.join(model_dir, "individual_models", f"{best_model_name}_{horizon}h.pkl"),
        os.path.join(model_dir, "individual_models", f"{best_model_name}_{horizon}h.cbm"),
        os.path.join(model_dir, "individual_models", f"{best_model_name}_{horizon}h.json"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found model file at: {path}")
            return path
    return None

def load_selected_features(model_dir, horizon):
    """Load selected features from the JSON file in the model directory."""
    features_file = os.path.join(model_dir, "selected_features.json")

    if not os.path.exists(features_file):
        print(f"Selected features file not found at: {features_file}")
        return None

    try:
        with open(features_file, 'r') as f:
            features = json.load(f)
        print(f"Loaded {len(features)} selected features for {horizon}h model")
        return features
    except Exception as e:
        print(f"Error loading selected features: {e}")
        return None

def convert_model_to_standard_format(model_obj, model_type, output_path):
    """Convert different model types to a standard joblib format when possible."""
    try:
        if model_type in ['CatBoost', 'catboost']:
            # For CatBoost, save as cbm format (native format)
            output_path = output_path.replace('.pkl', '.cbm')
            model_obj.save_model(output_path)
            return output_path, 'cbm'
        elif model_type in ['XGBoost', 'xgboost']:
            # For XGBoost, save as json format (native format)
            output_path = output_path.replace('.pkl', '.json')
            model_obj.save_model(output_path)
            return output_path, 'json'
        elif model_type in ['LightGBM', 'lightgbm']:
            # For LightGBM, save as txt format (native format)
            output_path = output_path.replace('.pkl', '.txt')
            model_obj.save_model(output_path)
            return output_path, 'txt'
        else:
            # For sklearn models, use joblib
            joblib.dump(model_obj, output_path)
            return output_path, 'pkl'
    except Exception as e:
        print(f"Error converting model: {e}")
        # Fallback to joblib
        joblib.dump(model_obj, output_path)
        return output_path, 'pkl'

def main():
    try:
        # Connect to Hopsworks
        project = hopsworks.login(
            api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
            project=os.environ.get("HOPSWORKS_PROJECT")
        )
        mr = project.get_model_registry()
        print("Successfully connected to Hopsworks.")

        ensure_directories()

        model_info = {}
        
        for horizon in [24, 48, 72]:
            try:
                model_name = f"aqi_forecast_model_{horizon}h"
                print(f"\nProcessing {model_name}...")

                # Get the latest model version
                model_info_obj = get_latest_model_version(mr, model_name)
                if not model_info_obj:
                    print(f"Could not find any version of {model_name}")
                    continue

                # Download model
                model_dir = model_info_obj.download()

                # Load selected features
                features = load_selected_features(model_dir, horizon)
                if not features:
                    print(f"Could not load selected features for {horizon}h model")
                    continue

                # Save selected features to backend/config
                features_output_path = f"backend/config/selected_features_{horizon}h.json"
                with open(features_output_path, 'w') as f:
                    json.dump(features, f, indent=2)
                print(f"Saved selected features to {features_output_path}")

                # Parse best model name from description
                description = model_info_obj.description
                best_model_name = parse_best_model_from_description(description)

                if not best_model_name:
                    print(f"Could not find best model name for {horizon}h model")
                    continue

                print(f"Best model for {horizon}h: {best_model_name}")

                # Find model file
                model_file_path = find_model_file(model_dir, best_model_name, horizon)
                if not model_file_path:
                    print(f"Could not find model file for {best_model_name}_{horizon}h")
                    continue

                # Load the model
                model_obj = None
                if best_model_name in ['CatBoost', 'catboost']:
                    from catboost import CatBoostRegressor
                    model_obj = CatBoostRegressor()
                    model_obj.load_model(model_file_path)
                elif best_model_name in ['XGBoost', 'xgboost']:
                    from xgboost import XGBRegressor
                    model_obj = XGBRegressor()
                    model_obj.load_model(model_file_path)
                elif best_model_name in ['LightGBM', 'lightgbm']:
                    import lightgbm as lgb
                    model_obj = lgb.Booster(model_file=model_file_path)
                else:
                    # For sklearn models
                    model_obj = joblib.load(model_file_path)

                # Save model to backend/models
                output_model_path = f"backend/models/model_{horizon}h.pkl"
                final_path, model_format = convert_model_to_standard_format(
                    model_obj, best_model_name, output_model_path
                )

                # Store model metadata
                model_info[horizon] = {
                    "model_type": best_model_name,
                    "model_file": os.path.basename(final_path),
                    "model_format": model_format,
                    "version": model_info_obj.version,
                    "features_count": len(features),
                    "updated_at": datetime.now().isoformat()
                }

                print(f"Successfully processed {horizon}h model")

            except Exception as e:
                print(f"Failed to process {horizon}h model: {e}")
                import traceback
                traceback.print_exc()

        # Save model metadata
        metadata_path = "backend/config/model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"\nModel download completed. Metadata saved to {metadata_path}")
        print(f"Total models processed: {len(model_info)}")

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()